#!/usr/bin/env python3
"""
3D-PAWS PM2.5 cleaning and calibration pipeline (generalized)
=============================================================
Generalization of pm25_3dpaws_pipeline.py, which reproduces Pham (NSF SOARS
2026), "Characterizing PM2.5 Data in Low-Cost Air Quality Sensors". The goal of
this version is to run on ANY 3D-PAWS instrument with a co-located reference,
without the user having studied the data first.

Workflow
--------
  Loading and alignment
    1. Put both datasets on local standard time: 3D-PAWS from UTC, the
       reference from whatever clock it was logged in (--ref-tz)
    2. Keep only the concurrent period
    3. Regularize the 3D-PAWS record: infer the nominal sampling interval and
       insert empty (NaN) rows wherever a step is a missed sample or longer.
       Every point-based window below therefore spans a consistent amount of
       time, and gaps are represented explicitly instead of being skipped over.

  Cleaning (paper Sec. 3a, Fig. 3)
    4. Remove values at/above a plausibility ceiling
    5. Temporal sequence analysis: neighborhood anomaly (Eq. 1); flagged points
       are REPLACED by the mean of their nearest neighbors (at least one on
       each side required).
    6. Oscillation filter; points above the threshold are REMOVED. The paper
       used the rolling std; the default here is the rolling point-to-point
       roughness, which spares smooth real plumes (see rolling_roughness).
    7. Rolling mean to smooth integer (quantization) steps

  Thresholds for steps 5 and 6 are data-driven by default: the sensor's
  typical spread is fitted as a + b x concentration, and points beyond
  k x that spread are acted on (see "Data-driven thresholds" below).
  --anomaly-threshold / --rolling-std-threshold give the paper's fixed values.

  Calibration (paper Sec. 3b, Fig. 4)
    8. Delta (Eq. 2) and relative difference (Eq. 3) vs. the reference
    9. Regression of reference (y) on cleaned 3D-PAWS (x). By default two
       models compete: a robust (Huber) straight line, and a robust "hinge"
       (two joined segments, bend point searched from the data). They are
       scored by time-blocked cross-validation: the record is cut into
       consecutive blocks, each block is predicted by a model fitted on the
       others, and the simpler model wins unless the hinge's held-out RMSE is
       lower, block by block, by more than one standard error (and >= 2%).
       The winner is refit on all data.
   10. Apply it to the 3D-PAWS data; report R^2, RMSE, MAE both in-sample
       (comparable to the paper) and held-out (what to expect on new data)

Status of the generalization
----------------------------
  Done:  site table removed; time zone, names, sampling interval, pairing
         tolerance and header rows come from arguments or the data; gaps are
         explicit NaN rows; window lengths are in time or sample counts;
         concentration-dependent, data-driven cleaning thresholds;
         calibration model chosen by time-blocked cross-validation.
  Paper methods remain available for comparison: --anomaly-threshold,
  --rolling-std-threshold, --calibration linear|piecewise|low_bias_linear.

Usage
-----
  # Synthetic data:
  python pm25_3dpaws_pipeline_general.py --demo

  # Boulder 3D-PAWS 153 vs. Marine St reference (the paper's CO data):
  python pm25_3dpaws_pipeline_general.py \\
      --instrument "data/raw/3D-PAWS_Instrument-153_2025-10-23_2025-12-09.csv" \\
      --reference "data/raw/Marine Street 1 minute data_102325-120825.csv" \\
      --tz America/Denver --name "Boulder, CO" --instrument-id 153 \\
      --inst-time-col time --inst-value-col pm1e25 \\
      --ref-time-col Date --ref-value-col PMFine --ref-tz LST --outdir results_CO
      # -> writes to results/results_CO/

  # Same, with the paper's hand-picked CO choices instead:
  #   add  --anomaly-threshold 5 --rolling-std-threshold 10 --calibration low_bias_linear
"""

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

GAP_FACTOR = 1.5   # a step >= 1.5 nominal intervals contains a missed sample


@dataclass
class RunConfig:
    name: str
    instrument_id: str
    utc_offset_hours: float        # local STANDARD time (no DST)
    outlier_ceiling: float         # ug/m3
    anomaly_threshold: float       # ug/m3 fixed threshold, or None = data-driven
    anomaly_k: float               # data-driven: multiple of the noise scale
    rolling_std_threshold: float   # ug/m3 fixed rolling-std threshold (paper), or None
    oscillation_metric: str        # "roughness" | "std" (used when data-driven)
    oscillation_k: float           # data-driven: multiple of the metric's typical value
    oscillation_window: str        # time span, e.g. "30min"
    smooth_window: str             # "30" = 30 samples, or a time span
    min_coverage: float            # fraction of a window that must hold data
    calibration: str               # "linear" | "piecewise" | "low_bias_linear"
    piecewise_split: float = 70.0  # ug/m3
    low_bias_cutoff: float = 2.0   # ug/m3


def standard_utc_offset(tz_name):
    """Standard-time UTC offset (hours) of an IANA zone. The smaller of the
    January and July offsets is standard time in either hemisphere."""
    z = ZoneInfo(tz_name)
    offs = [datetime(2025, m, 1, tzinfo=z).utcoffset().total_seconds() / 3600
            for m in (1, 7)]
    return min(offs)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def find_header_row(path, col, max_lines=50):
    """Number of title rows above the line that contains column `col`
    (e.g. some EPA exports have site-name rows above the real header)."""
    with open(path, newline="", encoding="utf-8-sig") as f:
        for i, row in enumerate(csv.reader(f)):
            if col in [c.strip() for c in row]:
                return i
            if i >= max_lines:
                break
    with open(path, newline="", encoding="utf-8-sig") as f:
        first = next(csv.reader(f), [])
    raise ValueError(f"column '{col}' not found in the first {max_lines} lines of "
                     f"{path}; first line has columns {first}")


def read_table(path, time_col, value_col, skiprows=None):
    if skiprows is None:
        skiprows = find_header_row(path, time_col)
    df = pd.read_csv(path, skiprows=skiprows, encoding="utf-8-sig")
    df.columns = [str(c).strip() for c in df.columns]
    for c in (time_col, value_col):
        if c not in df.columns:
            raise ValueError(f"column '{c}' not in {path}; columns are {list(df.columns)}")
    return df


def _to_series(t, values):
    s = pd.Series(pd.to_numeric(values, errors="coerce").to_numpy(dtype=float),
                  index=pd.DatetimeIndex(t).astype("datetime64[ns]"), name="pm25")
    s.index.name = "time"
    s = s[s.index.notna()]
    return s[~s.index.duplicated()].sort_index().dropna()


def load_instrument(path, time_col, value_col, utc_offset_hours, skiprows=None):
    """Load 3D-PAWS (CHORDS) data. Timestamps are UTC -> local standard time."""
    df = read_table(path, time_col, value_col, skiprows)
    t_utc = pd.to_datetime(df[time_col], utc=True).dt.tz_convert(None)
    return _to_series(t_utc + pd.Timedelta(hours=utc_offset_hours), df[value_col])


def reference_to_standard_time(t, ref_tz, std_offset_hours):
    """Convert naive reference timestamps to local STANDARD time.

    ref_tz describes the clock the reference file was recorded in:
      "LST"            already local standard time (EPA reference sites)
      -6, "-6", -6.0   fixed UTC offset in hours
      "UTC"            UTC
      IANA name        e.g. "America/Denver": local clock time WITH DST
                       switches, for files that span a DST change
    """
    if str(ref_tz).upper() == "LST":
        return t
    if str(ref_tz).upper() == "UTC":
        return t + pd.Timedelta(hours=std_offset_hours)
    try:
        off = float(ref_tz)
    except ValueError:
        off = None
    if off is not None:
        # clock = UTC + off  ->  UTC = clock - off  ->  LST = UTC + std_offset
        return t + pd.Timedelta(hours=std_offset_hours - off)
    try:
        aware = t.dt.tz_localize(ref_tz, ambiguous="infer", nonexistent="NaT")
    except Exception:  # repeated fall-back hour could not be inferred
        print("  [warn] could not infer ambiguous DST times; marking them NaT")
        aware = t.dt.tz_localize(ref_tz, ambiguous="NaT", nonexistent="NaT")
    return aware.dt.tz_convert("UTC").dt.tz_localize(None) + pd.Timedelta(hours=std_offset_hours)


def load_reference(path, time_col, value_col, skiprows=None, time_format=None,
                   ref_tz="LST", std_offset_hours=0):
    """Load reference-monitor data and put it on local standard time."""
    df = read_table(path, time_col, value_col, skiprows)
    t = pd.to_datetime(df[time_col], format=time_format)
    if getattr(t.dt, "tz", None) is not None:   # tz-aware strings: trust them
        t = t.dt.tz_convert("UTC").dt.tz_localize(None)
        ref_tz = "UTC"
    t = reference_to_standard_time(t, ref_tz, std_offset_hours)
    return _to_series(t, df[value_col])


# ---------------------------------------------------------------------------
# Sampling interval and gap handling
# ---------------------------------------------------------------------------

def infer_interval(index):
    """Nominal sampling interval: the mean of the ordinary steps (those
    shorter than GAP_FACTOR x the median step). The mean, not the median,
    because CHORDS steps alternate e.g. 61/62 s and the median would drift."""
    d = pd.Series(index).diff().dropna()
    med = d.median()
    return d[d < GAP_FACTOR * med].mean()


def insert_gap_rows(s, interval):
    """Insert empty (NaN) rows wherever samples are missing, so that the k-th
    neighbor of a point is always ~k intervals away. Real timestamps are kept
    as they are; inserted rows are spaced evenly inside each gap."""
    if len(s) < 2:
        return s.copy(), {"n_gaps": 0, "n_inserted": 0}
    t = s.index
    steps = np.diff(t.asi8)                                  # ns
    n_missing = np.maximum(np.round(steps / interval.value) - 1, 0).astype(int)
    new_times = []
    for i in np.flatnonzero(n_missing):
        m = n_missing[i]
        step = (t[i + 1] - t[i]) / (m + 1)
        new_times.append(t[i] + step * np.arange(1, m + 1))
    if new_times:
        filler = pd.Series(np.nan, index=pd.DatetimeIndex(np.concatenate(new_times)))
        s = pd.concat([s, filler]).sort_index()
        s.index.name = "time"
        s.name = "pm25"
    return s, {"n_gaps": int((n_missing > 0).sum()), "n_inserted": int(n_missing.sum())}


def window_points(window, interval):
    """Window length in samples: '30' means 30 samples, '30min' a time span."""
    w = str(window).strip()
    if re.fullmatch(r"\d+", w):
        return max(1, int(w))
    return max(1, int(round(pd.Timedelta(w) / interval)))


def min_periods(n, coverage, floor=1):
    return max(floor, math.ceil(coverage * n))


# ---------------------------------------------------------------------------
# Data-driven thresholds
#
# Optical PM sensors have a small fixed noise floor plus noise proportional to
# concentration, so one fixed threshold either flags real structure during
# enhancement events or misses spikes at background levels. Instead, the
# typical spread of a quantity (neighborhood anomaly, rolling std) is fitted
# as a function of the local concentration L:
#
#     scale(L) = a + b * L        threshold(L) = k * scale(L)
#
# a and b are fitted per dataset from concentration bins, using statistics
# that a few spikes cannot inflate; k is a fixed multiple chosen once for all
# sites. a is floored at the noise that integer reporting alone produces.
# ---------------------------------------------------------------------------

def detect_resolution(s):
    """Reporting step of the sensor: 1.0 if every value is a whole number
    (Plantower integer output), else 0.0."""
    v = s.dropna().to_numpy()
    if len(v) and np.all(np.abs(v - np.round(v)) < 1e-9):
        return 1.0
    return 0.0


def core_std(x):
    """Std of the central 98% of values. Unlike the MAD it is not zero when
    most values are tied (integer data with many identical readings), and
    unlike the plain std it is not inflated by the spikes being hunted."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 3:
        return np.nan
    lo, hi = np.percentile(x, [1, 99])
    return float(x[(x >= lo) & (x <= hi)].std())


@dataclass
class NoiseModel:
    what: str
    a: float          # ug/m3, spread at zero concentration
    b: float          # spread per ug/m3 of concentration
    k: float          # threshold multiple
    bins: list = field(default_factory=list, repr=False)  # (level, n, spread)

    def scale(self, level):
        return self.a + self.b * np.clip(level, 0, None)

    def threshold(self, level):
        return self.k * self.scale(level)

    def describe(self):
        return f"{self.k:g} x ({self.a:.2f} + {self.b:.3f} x L)"


def fit_noise_model(level, values, stat, floor, k, what, nbins=15, min_n=200):
    """Fit scale(L) = a + b*L to `stat` (e.g. core_std) of `values` in
    concentration bins of equal population. Weighted by sqrt(bin size)."""
    d = pd.DataFrame({"L": np.asarray(level, float),
                      "v": np.asarray(values, float)}).dropna()
    nb = min(nbins, len(d) // min_n)
    if nb < 2:
        a = stat(d["v"]) if len(d) >= 3 else floor
        a = max(a if np.isfinite(a) else floor, floor)
        return NoiseModel(what, float(a), 0.0, k)
    d["bin"] = pd.qcut(d["L"].rank(method="first"), nb, labels=False)
    t = d.groupby("bin").agg(L=("L", "median"), n=("v", "size"),
                             sc=("v", stat)).dropna()
    b, a = np.polyfit(t["L"], t["sc"], 1, w=np.sqrt(t["n"]))
    bins = [(float(r.L), int(r.n), float(r.sc)) for r in t.itertuples()]
    return NoiseModel(what, float(max(a, floor)), float(max(b, 0.0)), k, bins)


def local_level(s, n, coverage):
    """Concentration level for the rolling-std model: centered rolling median."""
    return s.rolling(n, center=True, min_periods=min_periods(n, coverage)).median()


# ---------------------------------------------------------------------------
# Cleaning. All functions take the regularized series (NaN = no sample) and
# return series on the same index. Thresholds may be a number or a Series
# (one threshold per point).
# ---------------------------------------------------------------------------

def remove_outliers(s, ceiling):
    return s.where(~(s >= ceiling))


def neighborhood_anomaly(s):
    """Eq. 1: x_i minus the mean of x_{i-2}, x_{i-1}, x_{i+1}, x_{i+2}.
    (The paper's printed equation lists x_{i+2} twice; x_{i+1} is intended.)
    Empty neighbor slots are left out of the mean, but at least one neighbor
    is required on EACH side; otherwise (e.g. at the edge of an outage) the
    anomaly is NaN and the point is never flagged or replaced. This keeps a
    single missed sample from disabling the check for four points."""
    nb_mean, _ = _neighbors(s)
    return s - nb_mean, nb_mean


def _neighbors(s):
    """Mean and median of the available neighbors x_{i-2}, x_{i-1}, x_{i+1},
    x_{i+2}; NaN unless at least one neighbor exists on each side."""
    before = pd.concat([s.shift(2), s.shift(1)], axis=1)
    after = pd.concat([s.shift(-1), s.shift(-2)], axis=1)
    nbrs = pd.concat([before, after], axis=1)
    ok = before.notna().any(axis=1) & after.notna().any(axis=1)
    return nbrs.mean(axis=1).where(ok), nbrs.median(axis=1).where(ok)


def neighborhood_level(s):
    """Concentration level used to scale the anomaly threshold: the MEDIAN of
    the neighbors. The mean would be pulled up by a multi-point spike, which
    would raise the threshold and let that spike through."""
    return _neighbors(s)[1]


def temporal_sequence_filter(s, threshold):
    """Flag |anomaly| > threshold and replace flagged points with the
    neighbor MEDIAN, preserving continuity of the record. (The paper uses the
    mean; for a spike lasting 2 samples the mean still contains the other
    spike sample, e.g. 300 on a background of 3 is replaced by ~77, while the
    median gives ~3. For single-sample spikes the two agree closely.)"""
    anomaly, _ = neighborhood_anomaly(s)
    nb_median = neighborhood_level(s)
    flagged = anomaly.abs() > threshold
    out = s.copy()
    out[flagged] = nb_median[flagged]
    return out, anomaly, flagged


def threshold_sweep(s, model=None, fixed=None):
    """How many points alternative anomaly thresholds would flag (paper Fig. 5
    table). Data-driven mode sweeps the multiple k; fixed mode sweeps ug/m3
    values. The setting in use is marked chosen=True. Thresholds are also
    reported at the median and 95th-percentile neighbor level, since in
    data-driven mode they vary with concentration."""
    anomaly, _ = neighborhood_anomaly(s)
    nb_level = neighborhood_level(s)
    n_valid = int(s.notna().sum())
    lv = nb_level.dropna()
    l50, l95 = (float(lv.quantile(.5)), float(lv.quantile(.95))) if len(lv) else (0.0, 0.0)
    rows = []
    if model is not None:
        for k in sorted({4, 5, 6, 8, 10, model.k}):
            th = k * model.scale(nb_level)
            n = int((anomaly.abs() > th).sum())
            rows.append({"setting": f"k={k:g}", "thr_at_median_L": k * model.scale(l50),
                         "thr_at_p95_L": k * model.scale(l95), "flagged_points": n,
                         "percent_flagged": 100 * n / max(n_valid, 1),
                         "chosen": k == model.k})
    else:
        for th in sorted({5, 10, 15, 20, 25, fixed}):
            n = int((anomaly.abs() > th).sum())
            rows.append({"setting": f"{th:g}", "thr_at_median_L": th, "thr_at_p95_L": th,
                         "flagged_points": n, "percent_flagged": 100 * n / max(n_valid, 1),
                         "chosen": th == fixed})
    df = pd.DataFrame(rows)
    df["total_points"] = n_valid
    return df.round({"thr_at_median_L": 2, "thr_at_p95_L": 2, "percent_flagged": 3})


def rolling_std(s, n, coverage):
    return s.rolling(n, center=True, min_periods=min_periods(n, coverage, 2)).std()


def rolling_roughness(s, n, coverage):
    """Point-to-point variability: sqrt(rolling mean of (x_i - x_{i-1})^2 / 2).
    For white noise it equals the std, but a smooth real plume that rises and
    falls over the window has a large std and a SMALL roughness. A sensor
    flipping between high and low values has a large roughness. This is what
    separates sensor oscillation from real short events."""
    d2 = s.diff() ** 2 / 2.0
    return np.sqrt(d2.rolling(n, center=True, min_periods=min_periods(n, coverage, 2)).mean())


OSCILLATION_METRICS = {"roughness": rolling_roughness, "std": rolling_std}


def oscillation_filter(s, metric_values, threshold):
    """Remove (set to NaN) points where the signal oscillates heavily."""
    removed = s.notna() & (metric_values > threshold)
    return s.where(~removed), removed


def rolling_mean_smooth(s, n, coverage):
    """Centered rolling mean, reported only where a sample survived cleaning
    and the window holds enough data. Gaps are never filled in."""
    m = s.rolling(n, center=True, min_periods=min_periods(n, coverage)).mean()
    return m.where(s.notna())


WARN_PCT = 5.0   # warn if a filter acts on more than this % of the record


def clean_instrument(grid, cfg, interval):
    """Full Fig. 3 workflow on the regularized series."""
    n_osc = window_points(cfg.oscillation_window, interval)
    n_mean = window_points(cfg.smooth_window, interval)
    q = detect_resolution(grid)
    step1 = remove_outliers(grid, cfg.outlier_ceiling)

    # Temporal sequence analysis
    anomaly, _ = neighborhood_anomaly(step1)
    if cfg.anomaly_threshold is None:
        # Rounding to integers alone gives the anomaly a std of ~0.32 x step
        nb_level = neighborhood_level(step1)
        anom_model = fit_noise_model(nb_level, anomaly, core_std,
                                     floor=max(0.32 * q, 0.05), k=cfg.anomaly_k,
                                     what="neighborhood anomaly")
        anom_thr = anom_model.threshold(nb_level)
    else:
        anom_model, anom_thr = None, cfg.anomaly_threshold
    step2, anomaly, flagged = temporal_sequence_filter(step1, anom_thr)

    # Oscillation filter (paper: rolling standard deviation)
    metric = "std" if cfg.rolling_std_threshold is not None else cfg.oscillation_metric
    osc = OSCILLATION_METRICS[metric](step2, n_osc, cfg.min_coverage)
    olevel = local_level(step2, n_osc, cfg.min_coverage)
    if cfg.rolling_std_threshold is None:
        # Rounding to integers alone gives either metric ~0.29 x step
        osc_model = fit_noise_model(olevel, osc, np.median,
                                    floor=max(0.29 * q, 0.05), k=cfg.oscillation_k,
                                    what=f"rolling {metric}")
        osc_thr = osc_model.threshold(olevel)
    else:
        osc_model, osc_thr = None, cfg.rolling_std_threshold
    step3, removed = oscillation_filter(step2, osc, osc_thr)

    step4 = rolling_mean_smooth(step3, n_mean, cfg.min_coverage)
    n_obs = int(grid.notna().sum())
    n1 = int(step1.notna().sum())
    info = {
        "n_raw": n_obs,
        "reporting_step": q,
        "n_outliers_removed": n_obs - n1,
        "anomaly_threshold": (anom_model.describe() if anom_model
                              else f"{cfg.anomaly_threshold:g} (fixed)"),
        "n_anomaly_replaced": int(flagged.sum()),
        "pct_anomaly_replaced": round(100 * flagged.sum() / max(n1, 1), 3),
        "n_skipped_near_gaps": int((step1.notna() & anomaly.isna()).sum()),
        "oscillation_metric": metric,
        "oscillation_window_points": n_osc,
        "oscillation_threshold": (osc_model.describe() if osc_model
                                  else f"{cfg.rolling_std_threshold:g} (fixed)"),
        "n_oscillation_removed": int(removed.sum()),
        "pct_oscillation_removed": round(100 * removed.sum() / max(n1, 1), 3),
        "smooth_window_points": n_mean,
        "n_dropped_low_coverage": int((step3.notna() & step4.isna()).sum()),
        "n_clean": int(step4.notna().sum()),
    }
    for key, label in [("pct_anomaly_replaced", "anomaly filter replaced"),
                       ("pct_oscillation_removed", "oscillation filter removed")]:
        if info[key] > WARN_PCT:
            print(f"  [warn] {label} {info[key]:.1f}% of the record; check the "
                  f"diagnostic plots, and raise the k value if real features are hit")
    stages = {"no_outliers": step1, "anomaly": anomaly, "flagged": flagged,
              "anomaly_threshold": anom_thr, "replaced": step2,
              "osc": osc, "osc_metric": metric, "osc_level": olevel,
              "osc_threshold": osc_thr, "removed": removed,
              "std_filtered": step3, "cleaned": step4}
    models = {"anomaly": anom_model, "oscillation": osc_model}
    return step4, stages, info, models


# ---------------------------------------------------------------------------
# Alignment and difference metrics
# ---------------------------------------------------------------------------

def estimate_time_lag(inst, ref, max_lag_min=180, step_min=5):
    """Find the shift of the 3D-PAWS series that best correlates with the
    reference. A best lag near +/-60 min means one dataset's time zone or DST
    setting is probably wrong. Positive lag = 3D-PAWS timestamps would need to
    move LATER to match."""
    a = inst.resample("1min").mean().rolling(15, min_periods=5, center=True).mean()
    b = ref.resample("1min").mean()
    out = {}
    for lag in range(-max_lag_min, max_lag_min + 1, step_min):
        j = pd.concat([a.shift(freq=f"{lag}min"), b], axis=1, sort=True).dropna()
        if len(j) > 100:
            out[lag] = j.corr().iloc[0, 1]
    r = pd.Series(out)
    if r.empty:
        return {"best_lag_min": 0, "r_best": np.nan, "r_at_zero": np.nan}
    return {"best_lag_min": int(r.idxmax()), "r_best": float(r.max()),
            "r_at_zero": float(r.get(0, np.nan))}


def align(inst, ref, tolerance):
    """Pair each 3D-PAWS observation with the nearest reference observation
    within `tolerance`. Only concurrent periods survive."""
    a = inst.dropna().rename("instrument").reset_index()
    b = ref.dropna().rename("reference").reset_index()
    m = pd.merge_asof(a.sort_values("time"), b.sort_values("time"), on="time",
                      direction="nearest", tolerance=pd.Timedelta(tolerance))
    return m.dropna().set_index("time")


def add_differences(df):
    """Eq. 2 (delta) and Eq. 3 (relative difference)."""
    df = df.copy()
    df["delta"] = df["instrument"] - df["reference"]
    with np.errstate(divide="ignore", invalid="ignore"):
        df["rel_diff"] = np.where(df["reference"] > 0,
                                  df["delta"] / df["reference"], np.nan)
    return df


def metrics(pred, ref):
    mask = np.isfinite(pred) & np.isfinite(ref)
    p, r = np.asarray(pred)[mask], np.asarray(ref)[mask]
    if len(p) < 3:
        return {"n": int(len(p)), "R2": np.nan, "R2_det": np.nan, "RMSE": np.nan,
                "MAE": np.nan, "mean_bias": np.nan}
    # R2: squared Pearson r (variance explained). Unchanged by any linear
    #     calibration, so it only moves because of cleaning or nonlinear steps.
    # R2_det: coefficient of determination 1 - SS_res/SS_tot of the sensor
    #     values as a prediction of the reference. Penalizes bias, so it DOES
    #     reflect what a calibration fixes. Can be negative.
    r2 = np.corrcoef(p, r)[0, 1] ** 2
    r2_det = 1 - np.sum((r - p) ** 2) / np.sum((r - r.mean()) ** 2)
    return {"n": int(len(p)),
            "R2": float(r2),
            "R2_det": float(r2_det),
            "RMSE": float(np.sqrt(np.mean((p - r) ** 2))),
            "MAE": float(np.mean(np.abs(p - r))),
            "mean_bias": float(np.mean(p - r))}


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

@dataclass
class Calibration:
    kind: str
    segments: list = field(default_factory=list)  # (lo, hi, slope, intercept)
    low_cutoff: float = None
    low_bias: float = 0.0

    def apply(self, x):
        x = np.asarray(x, dtype=float).copy()
        if self.low_cutoff is not None and self.low_bias:
            x = np.where(x < self.low_cutoff, x + self.low_bias, x)
        y = np.full_like(x, np.nan)
        for lo, hi, m, b in self.segments:
            sel = (x >= lo) & (x < hi)
            y[sel] = m * x[sel] + b
        return y

    def split_points(self):
        return [lo for lo, _, _, _ in self.segments if np.isfinite(lo)]

    def describe(self):
        lines = []
        if self.low_cutoff is not None and self.low_bias:
            lines.append(f"x -> x + {self.low_bias:.3f} for x < {self.low_cutoff:g}")
        for lo, hi, m, b in self.segments:
            rng = ""
            if np.isfinite(lo) or np.isfinite(hi):
                rng = f"  ({lo:g} <= x < {hi:g})"
            lines.append(f"y = {m:.3f}x {'+' if b >= 0 else '-'} {abs(b):.3f}{rng}")
        return lines


def _linfit(x, y):
    m, b = np.polyfit(x, y, 1)
    return float(m), float(b)


# ---- Robust regression (Huber M-estimator) --------------------------------

HUBER_C = 1.345   # standard tuning constant: 95% efficiency for normal errors


def _robust_scale(r):
    s = 1.4826 * np.median(np.abs(r - np.median(r)))
    if s > 1e-9:
        return s
    sd = np.std(r)
    return sd if sd > 1e-9 else 1.0


def huber_irls(X, y, scale=None, c=HUBER_C, max_iter=50, tol=1e-6):
    """Huber regression by iteratively reweighted least squares. Residuals
    larger than c robust standard deviations are down-weighted (weight
    c*s/|r| < 1), so a minority of badly-off points cannot drag the line the
    way they drag ordinary least squares. This replaces the paper's ad-hoc
    'drop points outside 1 std of the relative difference' filter.
    Returns (coefficients, final weights, residual scale)."""
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    for _ in range(max_iter):
        r = y - X @ beta
        s = scale or _robust_scale(r)
        u = np.abs(r) / (c * s)
        w = np.where(u <= 1, 1.0, 1.0 / np.maximum(u, 1e-12))
        sw = np.sqrt(w)
        new = np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)[0]
        done = np.max(np.abs(new - beta)) < tol * (1 + np.max(np.abs(beta)))
        beta = new
        if done:
            break
    r = y - X @ beta
    s = scale or _robust_scale(r)
    u = np.abs(r) / (c * s)
    w = np.where(u <= 1, 1.0, 1.0 / np.maximum(u, 1e-12))
    return beta, w, s


def _huber_loss(r, s, c=HUBER_C):
    u = np.abs(r) / s
    return float(np.sum(np.where(u <= c, 0.5 * u ** 2, c * u - 0.5 * c ** 2)))


def _hinge_design(x, s):
    return np.column_stack([np.ones_like(x), x, np.maximum(x - s, 0.0)])


# ---- Candidate models. Each takes (x, y) arrays and returns
#      (Calibration, weights in [0, 1] showing how much each point counted).

def fit_robust_linear(x, y):
    """y = a + b*x, Huber fit."""
    beta, w, _ = huber_irls(np.column_stack([np.ones_like(x), x]), y)
    return Calibration("robust_linear", [(-np.inf, np.inf, beta[1], beta[0])]), w


def fit_hinge(x, y, max_search=20000, min_frac=0.02, seed=0):
    """Continuous two-segment line y = a + b*x + c*max(x - s, 0), Huber fit.
    The bend point s is searched over quantiles of x (each side must hold at
    least min_frac of the data). A bend near the top covers the paper's VA
    case (split at 70); a bend near the bottom covers the CO case (bias
    correction below 2). Unlike the paper's split, the two segments meet, so
    a small change in x never causes a jump in the calibrated value."""
    n = len(x)
    min_n = max(50, int(min_frac * n))
    qs = np.r_[np.arange(0.05, 0.951, 0.05), 0.97, 0.98, 0.99]
    cands = [s for s in np.unique(np.round(np.quantile(x, qs), 3))
             if (x < s).sum() >= min_n and (x >= s).sum() >= min_n]
    if not cands:
        cal, w = fit_robust_linear(x, y)
        cal.kind = "hinge"
        return cal, w
    # Search on a subsample (1-minute data are highly autocorrelated, so this
    # loses little), with one common residual scale so losses are comparable.
    idx = np.random.default_rng(seed).choice(n, min(n, max_search), replace=False)
    xs, ys = x[idx], y[idx]
    _, _, s0 = huber_irls(np.column_stack([np.ones_like(xs), xs]), ys)
    losses = []
    for s in cands:
        H = _hinge_design(xs, s)
        beta, _, _ = huber_irls(H, ys, scale=s0)
        losses.append(_huber_loss(ys - H @ beta, s0))
    s = float(cands[int(np.argmin(losses))])
    (a, b, c), w, _ = huber_irls(_hinge_design(x, s), y)
    return Calibration("hinge", [(-np.inf, s, b, a), (s, np.inf, b + c, a - c * s)]), w


def fit_ols_linear(x, y):
    """Paper Eq. 5 style: ordinary least squares line."""
    m, b = _linfit(x, y)
    return Calibration("linear", [(-np.inf, np.inf, m, b)]), np.ones(len(x))


def fit_paper_piecewise(x, y, split):
    """Paper Eq. 4: split at a fixed concentration, then drop points whose
    relative difference lies outside 1 std of that segment's own
    distribution, and fit each segment by OLS."""
    with np.errstate(divide="ignore", invalid="ignore"):
        rd = np.where(y > 0, (x - y) / y, np.nan)
    keep = np.zeros(len(x), bool)
    segs = []
    for lo, hi in [(-np.inf, split), (split, np.inf)]:
        part = (x >= lo) & (x < hi) & np.isfinite(rd)
        if part.sum() < 10:
            continue
        r = rd[part]
        ok = part.copy()
        ok[part] = np.abs(r - r.mean()) <= r.std()
        if ok.sum() >= 10:
            segs.append((lo, hi, *_linfit(x[ok], y[ok])))
            keep |= ok
    if len(segs) < 2:   # not enough data on one side: one filtered line
        ok = np.isfinite(rd)
        ok[ok] = np.abs(rd[ok] - np.nanmean(rd)) <= np.nanstd(rd)
        return (Calibration("piecewise", [(-np.inf, np.inf, *_linfit(x[ok], y[ok]))]),
                ok.astype(float))
    return Calibration("piecewise", segs), keep.astype(float)


def fit_paper_low_bias(x, y, cutoff):
    """Paper Eq. 6: shift x < cutoff by the mean bias there, then OLS."""
    low = x < cutoff
    bias = float(y[low].mean() - x[low].mean()) if low.any() else 0.0
    m, b = _linfit(np.where(low, x + bias, x), y)
    return (Calibration("low_bias_linear", [(-np.inf, np.inf, m, b)],
                        low_cutoff=cutoff, low_bias=bias), np.ones(len(x)))


AUTO_MODELS = ["robust_linear", "hinge"]   # in order of increasing complexity


def make_fitter(name, cfg):
    return {
        "robust_linear": fit_robust_linear,
        "hinge": fit_hinge,
        "linear": fit_ols_linear,
        "piecewise": lambda x, y: fit_paper_piecewise(x, y, cfg.piecewise_split),
        "low_bias_linear": lambda x, y: fit_paper_low_bias(x, y, cfg.low_bias_cutoff),
    }[name]


# ---- Time-blocked cross-validation ----------------------------------------

def time_folds(index, k, buffer):
    """Split the record into k consecutive blocks of equal duration. For each
    block, train on everything else except a buffer on both sides of it, so
    autocorrelated minutes right next to the test block do not leak in."""
    edges = pd.date_range(index.min(), index.max(), periods=k + 1)
    t = index.to_numpy()
    folds = []
    for i in range(k):
        lo, hi = edges[i], edges[i + 1]
        test = (t >= lo.to_datetime64()) & ((t < hi.to_datetime64()) if i < k - 1
                                            else (t <= hi.to_datetime64()))
        train = (t < (lo - buffer).to_datetime64()) | (t > (hi + buffer).to_datetime64())
        folds.append((i + 1, lo, hi, test, train))
    return folds


def cross_validate(df, fitter, k, buffer, min_test=100, min_train=500):
    """Fit on k-1 blocks, score on the held-out block, for every block.
    Returns per-fold metrics and the out-of-fold predictions."""
    x, y = df["instrument"].to_numpy(), df["reference"].to_numpy()
    oof = np.full(len(df), np.nan)
    rows = []
    for i, lo, hi, test, train in time_folds(df.index, k, buffer):
        if test.sum() < min_test or train.sum() < min_train:
            continue
        cal, _ = fitter(x[train], y[train])
        p = cal.apply(x[test])
        oof[test] = p
        rows.append({"fold": i, "start": lo, "end": hi, **metrics(p, y[test])})
    return pd.DataFrame(rows), pd.Series(oof, index=df.index, name="calibrated_cv")


def cv_baseline(df, k, buffer, min_test=100):
    """Per-fold metrics of the cleaned, UNcalibrated series, for comparison."""
    x, y = df["instrument"].to_numpy(), df["reference"].to_numpy()
    rows = [{"fold": i, "start": lo, "end": hi, **metrics(x[test], y[test])}
            for i, lo, hi, test, _ in time_folds(df.index, k, buffer)
            if test.sum() >= min_test]
    return pd.DataFrame(rows)


def summarize_cv(folds_by_model):
    rows = []
    for name, f in folds_by_model.items():
        if f.empty:
            continue
        nf = len(f)
        rows.append({"model": name, "folds": nf,
                     "RMSE_mean": f["RMSE"].mean(),
                     "RMSE_se": f["RMSE"].std(ddof=1) / np.sqrt(nf) if nf > 1 else np.nan,
                     "MAE_mean": f["MAE"].mean(), "R2_mean": f["R2"].mean(),
                     "R2_det_mean": f["R2_det"].mean(), "bias_mean": f["mean_bias"].mean()})
    return pd.DataFrame(rows)


MIN_REL_GAIN = 0.02   # a more complex model must cut held-out RMSE by >= 2%


def select_model(folds_by_model, order):
    """Start from the simplest model; switch to the next more complex one
    only if it is CLEARLY better on held-out data: its RMSE, compared block
    by block (paired, since some blocks are harder for every model), must be
    lower on average by more than one standard error of that difference, and
    by at least MIN_REL_GAIN. Returns (chosen model, list of reasons)."""
    avail = [m for m in order if m in folds_by_model and not folds_by_model[m].empty]
    if not avail:
        return order[0], ["no cross-validation folds; using the simplest model"]
    chosen, reasons = avail[0], []
    for m in avail[1:]:
        a = folds_by_model[chosen].set_index("fold")["RMSE"]
        b = folds_by_model[m].set_index("fold")["RMSE"]
        d = (a - b).dropna()
        if len(d) < 2:
            reasons.append(f"{m} vs {chosen}: fewer than 2 common blocks; keeping {chosen}")
            continue
        gain, se = d.mean(), d.std(ddof=1) / np.sqrt(len(d))
        better = gain > se and gain >= MIN_REL_GAIN * a.loc[d.index].mean()
        reasons.append(f"{m} vs {chosen}: held-out RMSE lower by {gain:+.3f} ± {se:.3f} "
                       f"(paired over {len(d)} blocks, {int((d > 0).sum())} won) -> "
                       f"{'switch to ' + m if better else 'keep ' + chosen}")
        if better:
            chosen = m
    return chosen, reasons


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _zero_floor(fig):
    """Concentration axes start exactly at 0 with no padding below it.
    Axes that can legitimately go negative (anomaly, relative difference)
    and log-scaled count axes are left alone."""
    for ax in fig.axes:
        if not ax.get_visible() or ax.axison is False:
            continue
        yl, xl = ax.get_ylabel(), ax.get_xlabel()
        if ax.get_yscale() == "linear" and not yl.startswith(("Anomaly", "Relative")) \
                and ("µg" in yl or yl in ("Density", "Count")):
            ax.set_ylim(bottom=0)
        if "µg" in xl:
            ax.set_xlim(left=0)


def _save(fig, path):
    _zero_floor(fig)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _window_label(window):
    w = str(window).strip()
    return f"{w}-point" if re.fullmatch(r"\d+", w) else w


def plot_raw_vs_ref(raw, ref, title, path):
    fig, ax = plt.subplots(figsize=(12, 3.5))
    ax.plot(raw.index, raw, color="tab:blue", lw=0.5, label="3D-PAWS raw")
    ax.plot(ref.index, ref, color="black", lw=0.6, label="Reference")
    ax.set_ylabel("PM$_{2.5}$ (µg m$^{-3}$)")
    ax.set_title(title)
    ax.legend(loc="upper right")
    _save(fig, path)


def _plot_threshold_band(ax, thr):
    """Shade the +/- threshold band: fixed (number) or per point (Series)."""
    if isinstance(thr, pd.Series):
        t = thr.dropna()
        ax.fill_between(t.index, -t, t, color="red", alpha=0.18, lw=0,
                        label="±threshold", zorder=1)
    else:
        for sign in (1, -1):
            ax.axhline(sign * thr, color="red", ls="--", lw=1, zorder=1)


def _thr_label(model, fixed):
    return model.describe() if model else f"{fixed:g}"


def plot_anomaly(stages, model, fixed, sweep, title, path):
    s, an, fl, rep = (stages["no_outliers"], stages["anomaly"],
                      stages["flagged"], stages["replaced"])
    thr = stages["anomaly_threshold"]
    fig = plt.figure(figsize=(12, 10))
    gs = fig.add_gridspec(4, 1, height_ratios=[1, 1, 1, 0.45])
    axes = [fig.add_subplot(gs[0])]
    axes += [fig.add_subplot(gs[i], sharex=axes[0]) for i in (1, 2)]
    tax = fig.add_subplot(gs[3])
    tax.axis("off")
    axes[0].plot(s.index, s, color="tab:blue", lw=0.5)
    axes[0].scatter(s.index[fl], s[fl], color="red", s=6, zorder=3, label="flagged")
    axes[0].set_title(f"{title}: raw PM$_{{2.5}}$ with flagged points")
    axes[0].legend(loc="upper right")
    _plot_threshold_band(axes[1], thr)
    axes[1].plot(an.index, an, color="purple", lw=0.5, zorder=2)
    axes[1].set_title(f"Neighborhood anomaly (threshold = ±{_thr_label(model, fixed)}"
                      + (", L = neighbor median)" if model else ")"))
    axes[2].plot(rep.index, rep, color="green", lw=0.5)
    axes[2].set_title("After replacing flagged points")
    for ax in axes:
        ax.set_ylabel("µg m$^{-3}$")
    axes[1].set_ylabel("Anomaly (µg m$^{-3}$)")
    cols = ["setting", "thr_at_median_L", "thr_at_p95_L", "flagged_points",
            "total_points", "percent_flagged"]
    cells = [[r.setting, f"{r.thr_at_median_L:.1f}", f"{r.thr_at_p95_L:.1f}",
              f"{r.flagged_points:d}", f"{r.total_points:d}", f"{r.percent_flagged:.2f}%"]
             for r in sweep.itertuples()]
    tab = tax.table(cellText=cells, colLabels=cols, loc="center", cellLoc="center")
    for i, chosen in enumerate(sweep["chosen"], start=1):
        if chosen:
            for j in range(len(cols)):
                tab[(i, j)].set_facecolor("#ffe9a8")
    _save(fig, path)


def plot_oscillation(stages, model, cfg, interval, title, path):
    """Left: oscillation metric against concentration with the threshold
    actually used (removed points in red). Right: its distribution for the
    chosen window and one twice as long (paper Fig. 6)."""
    series, metric = stages["replaced"], stages["osc_metric"]
    fn = OSCILLATION_METRICS[metric]
    name = "rolling std" if metric == "std" else "rolling roughness"
    n = window_points(cfg.oscillation_window, interval)
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(13, 4.5))
    d = pd.DataFrame({"L": stages["osc_level"], "r": stages["osc"],
                      "rm": stages["removed"]}).dropna()
    ax0.scatter(d.loc[~d.rm, "L"], d.loc[~d.rm, "r"], s=1, color="tab:blue",
                alpha=0.15, rasterized=True, label="kept")
    ax0.scatter(d.loc[d.rm, "L"], d.loc[d.rm, "r"], s=2, color="red",
                rasterized=True, label="removed")
    xx = np.linspace(0, max(d["L"].max(), 1) if len(d) else 1, 200)
    if model:
        ax0.plot(xx, model.scale(xx), color="black", lw=1, label="typical (a + b·L)")
        if model.bins:
            bl, _, bs = zip(*model.bins)
            ax0.scatter(bl, bs, color="black", s=12, zorder=3, label="bin medians")
        ax0.plot(xx, model.threshold(xx), color="red", ls="--", lw=1,
                 label=f"threshold {model.describe()}")
    else:
        ax0.axhline(cfg.rolling_std_threshold, color="red", ls="--", lw=1,
                    label=f"threshold {cfg.rolling_std_threshold:g}")
    ax0.set_xlabel("Local concentration L, rolling median (µg m$^{-3}$)")
    ax0.set_ylabel(f"{name.capitalize()} (µg m$^{{-3}}$)")
    ax0.set_title(f"{title}: {n}-point {name} vs. concentration", fontsize=10)
    ax0.legend(loc="upper left", fontsize=7, markerscale=4)
    for k, c in [(1, "tab:blue"), (2, "tab:orange")]:
        v = fn(series, n * k, cfg.min_coverage).dropna()
        minutes = (interval * n * k).total_seconds() / 60
        ax1.hist(v, bins=100, alpha=0.5, color=c,
                 label=f"{n * k}-point (~{minutes:.0f} min) {name}")
    if not model:
        ax1.axvline(cfg.rolling_std_threshold, color="red", ls="--", lw=1, label="threshold")
    ax1.set_yscale("log")
    ax1.set_xlabel(f"{name.capitalize()} (µg m$^{{-3}}$)")
    ax1.set_ylabel("Count")
    ax1.set_title("Distribution for two window lengths", fontsize=10)
    ax1.legend(fontsize=8)
    _save(fig, path)


def plot_cleaned(raw, cleaned, ref, cfg, title, path):
    fig, ax = plt.subplots(figsize=(12, 3.5))
    ax.plot(raw.index, raw, color="tab:blue", lw=0.4, alpha=0.6, label="3D-PAWS raw")
    ax.plot(cleaned.index, cleaned, color="green", lw=0.8,
            label=f"Cleaned ({_window_label(cfg.smooth_window)} mean)")
    ax.plot(ref.index, ref, color="black", lw=0.6, label="Reference")
    ax.set_ylabel("PM$_{2.5}$ (µg m$^{-3}$)")
    ax.set_title(title)
    ax.legend(loc="upper right")
    _save(fig, path)


def plot_reldiff(df, title, path):
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    axes[0].plot(df.index, df["instrument"], color="tab:blue", lw=0.6, label="Cleaned 3D-PAWS")
    axes[0].plot(df.index, df["reference"], color="black", lw=0.6, label="Reference")
    axes[0].set_ylabel("µg m$^{-3}$")
    axes[0].legend(loc="upper right")
    axes[0].set_title(title)
    axes[1].plot(df.index, df["rel_diff"], color="tab:orange", lw=0.6)
    axes[1].axhline(0, color="grey", lw=0.8)
    axes[1].set_ylabel("Relative difference")
    _save(fig, path)


def plot_scatter(df, weights, cal, title, path):
    """Reference vs. cleaned 3D-PAWS with the calibration. Points the fit
    down-weighted below 0.5 (robust fits) or excluded (paper methods) are grey."""
    fig, ax = plt.subplots(figsize=(6, 6))
    used = np.asarray(weights) >= 0.5
    ax.scatter(df["instrument"][~used], df["reference"][~used], s=2,
               color="lightgrey", label="down-weighted / excluded", rasterized=True)
    ax.scatter(df["instrument"][used], df["reference"][used], s=2,
               color="tab:blue", alpha=0.4, label="full weight in fit", rasterized=True)
    lim = max(df["instrument"].max(), df["reference"].max()) * 1.05
    ax.plot([0, lim], [0, lim], color="grey", ls="--", label="1:1")
    xx = np.linspace(0, df["instrument"].max(), 500)
    ax.plot(xx, cal.apply(xx), color="red", lw=1.5, label="calibration")
    for xs in cal.split_points():
        ax.axvline(xs, color="tab:orange", ls=":", lw=1)
    ax.set_xlabel("Cleaned 3D-PAWS PM$_{2.5}$ (µg m$^{-3}$)")
    ax.set_ylabel("Reference PM$_{2.5}$ (µg m$^{-3}$)")
    ax.set_title(f"{title} — {cal.kind}\n" + "\n".join(cal.describe()), fontsize=9)
    ax.legend(loc="upper left", fontsize=8)
    _save(fig, path)


def plot_cv(table, folds, title, path):
    """Held-out RMSE per model: mean ± s.e. (bars) and each fold (dots)."""
    fig, ax = plt.subplots(figsize=(7, 4))
    for i, r in enumerate(table.itertuples()):
        c = "tab:orange" if r.chosen else ("grey" if r.model.startswith("cleaned") else "tab:blue")
        ax.bar(i, r.RMSE_mean, yerr=r.RMSE_se if np.isfinite(r.RMSE_se) else 0,
               color=c, alpha=0.6, capsize=4)
        f = folds[folds["model"] == r.model]
        ax.scatter(np.full(len(f), i), f["RMSE"], color="black", s=10, zorder=3)
    ax.set_xticks(range(len(table)))
    ax.set_xticklabels([m + ("\n(chosen)" if ch else "")
                        for m, ch in zip(table["model"], table["chosen"])], fontsize=8)
    ax.set_ylabel("Held-out RMSE (µg m$^{-3}$)")
    ax.set_title(f"{title}: time-blocked cross-validation", fontsize=10)
    _save(fig, path)


def plot_calibrated(out, m_raw, m_cal, m_cv, title, path):
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True, sharey=True)
    axes[0].plot(out.index, out["raw"], color="tab:blue", lw=0.5, label="3D-PAWS raw")
    axes[0].plot(out.index, out["reference"], color="black", lw=0.6, label="Reference")
    axes[0].set_title(f"{title} — raw: R² = {100*m_raw['R2']:.1f}% "
                      f"(1−SS: {m_raw['R2_det']:.2f}), RMSE = {m_raw['RMSE']:.2f} µg m⁻³")
    axes[1].plot(out.index, out["calibrated"], color="purple", lw=0.5, label="3D-PAWS calibrated")
    axes[1].plot(out.index, out["reference"], color="black", lw=0.6, label="Reference")
    axes[1].set_title(f"Calibrated: R² = {100*m_cal['R2']:.1f}% "
                      f"(1−SS: {m_cal['R2_det']:.2f}), RMSE = {m_cal['RMSE']:.2f} µg m⁻³"
                      f"  |  held-out (CV): 1−SS {m_cv['R2_det']:.2f}, "
                      f"RMSE {m_cv['RMSE']:.2f} µg m⁻³", fontsize=10)
    for ax in axes:
        ax.set_ylabel("µg m$^{-3}$")
        ax.legend(loc="upper right")
    _save(fig, path)


def plot_workflow(ref, stages, calibrated, cfg, models, cal_kind, title, path):
    """Every stage of the workflow, top to bottom, against the reference."""
    s, fl = stages["no_outliers"], stages["flagged"]
    anom = _thr_label(models["anomaly"], cfg.anomaly_threshold)
    rthr = _thr_label(models["oscillation"], cfg.rolling_std_threshold)
    osc_name = "rolling std" if stages["osc_metric"] == "std" else "roughness"
    panels = [
        (f"1. Raw 3D-PAWS (values ≥{cfg.outlier_ceiling:g} removed)", s, "tab:blue"),
        (f"2. Temporal sequence analysis: flagged |anomaly| > {anom}", s, "tab:blue"),
        ("3. Flagged points replaced by 4-neighbor mean", stages["replaced"], "green"),
        (f"4. After {_window_label(cfg.oscillation_window)} {osc_name} filter "
         f"(> {rthr} removed)", stages["std_filtered"], "teal"),
        (f"5. After {_window_label(cfg.smooth_window)} rolling mean", stages["cleaned"],
         "darkgreen"),
        (f"6. Calibrated ({cal_kind})", calibrated, "purple"),
    ]
    fig, axes = plt.subplots(len(panels), 1, figsize=(12, 15), sharex=True, sharey=True)
    for ax, (lab, ser, c) in zip(axes, panels):
        ax.plot(ref.index, ref, color="black", lw=0.5, label="Reference")
        ax.plot(ser.index, ser, color=c, lw=0.5, label="3D-PAWS")
        ax.set_title(lab, fontsize=10, loc="left")
        ax.set_ylabel("µg m$^{-3}$")
    axes[1].scatter(s.index[fl], s[fl], color="red", s=6, zorder=3, label="flagged")
    axes[1].legend(loc="upper right", fontsize=8)
    axes[0].legend(loc="upper right", fontsize=8)
    fig.suptitle(f"{title}: complete processing and calibration workflow")
    _save(fig, path)


def plot_distributions(out, title, path):
    fig, ax = plt.subplots(figsize=(7, 4))
    hi = np.nanpercentile(out[["raw", "reference"]].to_numpy(), 99.5)
    bins = np.linspace(0, max(hi, 5), 80)
    for col, c, lab in [("raw", "tab:blue", "Raw 3D-PAWS"),
                        ("calibrated", "purple", "Calibrated 3D-PAWS"),
                        ("reference", "black", "Reference")]:
        ax.hist(out[col].dropna(), bins=bins, histtype="step", lw=1.3,
                color=c, label=lab, density=True)
    ax.set_xlabel("PM$_{2.5}$ (µg m$^{-3}$)")
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.legend()
    _save(fig, path)


# ---------------------------------------------------------------------------
# Synthetic demo data (for testing the pipeline without real files)
# ---------------------------------------------------------------------------

RESULTS_ROOT = Path(__file__).resolve().parent / "results"


def resolve_outdir(outdir, instrument_id, demo):
    """Where to write this run's outputs.

    Every run gets its own folder inside RESULTS_ROOT (``results/`` next to this
    script), whatever directory the script is launched from:
      no --outdir         results/results_<instrument-id>  (results/results_demo for --demo)
      --outdir NAME       results/NAME   (a leading "results/" is not doubled)
      --outdir /abs/path  used exactly as given
    """
    if outdir is None:
        outdir = "results_demo" if demo else f"results_{_slug(str(instrument_id or 'run'))}"
    path = Path(outdir).expanduser()
    if path.is_absolute():
        return path
    if path.parts and path.parts[0] == RESULTS_ROOT.name:
        path = Path(*path.parts[1:])
    return RESULTS_ROOT / path


def make_demo_data(utc_offset_hours, outdir, days=30, seed=0, two_cluster=False):
    rng = np.random.default_rng(seed)
    t = pd.date_range("2025-11-01", periods=days * 1440, freq="1min")
    hrs = np.asarray(t.hour + t.minute / 60, dtype=float)
    day = np.arange(len(t)) / 1440
    ref = 8 + 2 * np.sin(2 * np.pi * day / 7) + 3 * np.sin(2 * np.pi * (hrs - 6) / 24)
    for c in rng.uniform(1, days - 1, 6):               # enhancement events
        ref += rng.uniform(15, 40) * np.exp(-0.5 * ((day - c) / 0.15) ** 2)
    ref = np.clip(ref + rng.normal(0, 0.5, len(t)), 0, None)

    inst = 1.6 * ref - 4 + rng.normal(0, 2, len(t))     # under low, over high
    if two_cluster:
        # second bias cluster like the paper's VA site (Fig. 9): episodes where
        # the sensor reads far higher while the reference stays moderate
        for c in rng.uniform(2, days - 2, 4):
            w = np.exp(-0.5 * ((day - c) / 0.4) ** 2)
            inst += 4.0 * ref * w
    spikes = rng.random(len(t)) < 0.005
    inst[spikes] += rng.uniform(20, 80, spikes.sum())
    for s0 in rng.integers(0, len(t) - 120, 8):          # oscillating periods
        inst[s0:s0 + 120] += 15 * np.where(np.arange(120) % 2, 1, -1)
    inst[rng.integers(0, len(t), 5)] = 400               # implausible outliers
    inst = np.clip(np.round(inst), 0, None)              # integer reporting

    t_utc = t - pd.Timedelta(hours=utc_offset_hours)
    t_utc = t_utc + pd.to_timedelta(rng.integers(-5, 5, len(t)), unit="s")
    keep = np.ones(len(t), bool)                          # transmission outages
    for g0 in rng.integers(0, len(t) - 600, 3):
        keep[g0:g0 + rng.integers(60, 600)] = False
    t_utc, inst = t_utc[keep], inst[keep]
    ipath, rpath = outdir / "demo_instrument_utc.csv", outdir / "demo_reference_local.csv"
    pd.DataFrame({"time": t_utc.strftime("%Y-%m-%dT%H:%M:%SZ"), "pm1e25": inst}).to_csv(ipath, index=False)
    pd.DataFrame({"time": t, "pm25": np.round(ref, 2)}).to_csv(rpath, index=False)
    return ipath, rpath


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def _slug(text):
    return re.sub(r"[^A-Za-z0-9]+", "_", str(text)).strip("_") or "run"


def run(args):
    if args.tz:
        utc_offset = standard_utc_offset(args.tz)
    elif args.utc_offset is not None:
        utc_offset = args.utc_offset
    else:  # demo only (parse_args enforces this)
        utc_offset = -5.0
    cfg = RunConfig(
        name=args.name, instrument_id=str(args.instrument_id or ""),
        utc_offset_hours=utc_offset,
        outlier_ceiling=args.outlier_ceiling,
        anomaly_threshold=args.anomaly_threshold,
        anomaly_k=args.anomaly_k,
        rolling_std_threshold=args.rolling_std_threshold,
        oscillation_metric=args.oscillation_metric,
        oscillation_k=args.oscillation_k,
        oscillation_window=args.oscillation_window,
        smooth_window=args.smooth_window,
        min_coverage=args.min_coverage,
        calibration=args.calibration,
        piecewise_split=args.piecewise_split,
        low_bias_cutoff=args.low_bias_cutoff,
    )
    outdir = resolve_outdir(args.outdir, cfg.instrument_id, args.demo)
    outdir.mkdir(parents=True, exist_ok=True)
    tag = _slug(f"{cfg.name}_{cfg.instrument_id}" if cfg.instrument_id else cfg.name)
    title = f"{cfg.name} (Instrument {cfg.instrument_id})" if cfg.instrument_id else cfg.name

    if args.demo:
        args.instrument, args.reference = make_demo_data(
            utc_offset, outdir, two_cluster=args.demo_two_cluster)
        args.inst_time_col, args.inst_value_col = "time", "pm1e25"
        args.ref_time_col, args.ref_value_col = "time", "pm25"

    print(f"== {title} ==")
    raw = load_instrument(args.instrument, args.inst_time_col, args.inst_value_col,
                          utc_offset, args.inst_skiprows)
    ref = load_reference(args.reference, args.ref_time_col, args.ref_value_col,
                         args.ref_skiprows, args.ref_time_format,
                         args.ref_tz, utc_offset)
    print(f"  time base: local standard time (UTC{utc_offset:+g}); "
          f"3D-PAWS from UTC, reference from '{args.ref_tz}'")

    if args.drop_zeros:
        nz_i, nz_r = int((raw <= 0).sum()), int((ref <= 0).sum())
        raw, ref = raw[raw > 0], ref[ref > 0]
        print(f"  dropped zero readings: 3D-PAWS {nz_i} "
              f"({100*nz_i/(len(raw)+nz_i):.1f}%), reference {nz_r}")

    # Keep only the concurrent observation period
    start, end = max(raw.index.min(), ref.index.min()), min(raw.index.max(), ref.index.max())
    if start >= end:
        raise SystemExit("  [error] the instrument and reference records do not overlap; "
                         "check the time columns and time zone settings")
    raw, ref = raw[start:end], ref[start:end]
    print(f"  concurrent period: {start} -> {end}")

    # Sampling intervals and gap handling
    interval = (pd.Timedelta(args.interval) if args.interval != "auto"
                else infer_interval(raw.index))
    ref_interval = infer_interval(ref.index)
    tolerance = (pd.Timedelta(args.tolerance) if args.tolerance != "auto"
                 else ref_interval)
    grid, gap_info = insert_gap_rows(raw, interval)
    print(f"  3D-PAWS interval: {interval.total_seconds():.1f} s; "
          f"{gap_info['n_gaps']} gaps, {gap_info['n_inserted']} empty rows inserted "
          f"({100*gap_info['n_inserted']/len(grid):.1f}% of the record)")
    print(f"  reference interval: {ref_interval.total_seconds():.1f} s; "
          f"pairing tolerance {tolerance.total_seconds():.1f} s")

    lag = estimate_time_lag(raw, ref)
    print(f"  alignment check: best lag {lag['best_lag_min']:+d} min "
          f"(r={lag['r_best']:.3f}; r at 0 lag={lag['r_at_zero']:.3f})")
    if abs(lag["best_lag_min"]) >= 30:
        print("  [warn] series look misaligned by ~"
              f"{abs(lag['best_lag_min'])} min; check time zone / DST settings")

    # ---- Cleaning ----
    cleaned, stages, clean_info, models = clean_instrument(grid, cfg, interval)
    no_outliers = stages["no_outliers"]
    sweep = threshold_sweep(no_outliers, models["anomaly"], cfg.anomaly_threshold)
    sweep.to_csv(outdir / f"{tag}_anomaly_threshold_sweep.csv", index=False)
    for k, v in clean_info.items():
        print(f"  {k}: {v}")

    # ---- Calibration: model choice by time-blocked cross-validation ----
    clean_df = add_differences(align(cleaned, ref, tolerance))
    names = AUTO_MODELS if args.calibration == "auto" else [args.calibration]
    buffer = pd.Timedelta(args.cv_buffer)
    cv_folds, cv_oof = {}, {}
    for nm in names:
        cv_folds[nm], cv_oof[nm] = cross_validate(clean_df, make_fitter(nm, cfg),
                                                  args.cv_folds, buffer)
    base_folds = cv_baseline(clean_df, args.cv_folds, buffer)
    cv_table = summarize_cv({"cleaned (no calibration)": base_folds, **cv_folds})
    chosen, reasons = select_model(cv_folds, names)
    cv_table["chosen"] = cv_table["model"] == chosen
    n_folds = int(cv_table.loc[cv_table.model == chosen, "folds"].sum()) if len(cv_table) else 0
    if n_folds < 2:
        print("  [warn] too little data for cross-validation; held-out scores are "
              "unavailable and the simplest model is used")
    print(f"  cross-validation: {args.cv_folds} consecutive time blocks, "
          f"{args.cv_buffer} buffer; scores on HELD-OUT blocks (mean ± s.e.):")
    for r in cv_table.itertuples():
        print(f"    {r.model:<26} RMSE={r.RMSE_mean:.2f} ± {r.RMSE_se:.2f}  "
              f"MAE={r.MAE_mean:.2f}  R2(1-SS)={r.R2_det_mean:.3f}  bias={r.bias_mean:+.2f}"
              + ("   <- chosen" if r.chosen else ""))
    for line in reasons:
        print(f"    selection: {line}")

    x_all, y_all = clean_df["instrument"].to_numpy(), clean_df["reference"].to_numpy()
    cal, fit_weights = make_fitter(chosen, cfg)(x_all, y_all)
    print(f"  calibration ({chosen}, refit on all data):")
    for line in cal.describe():
        print(f"    {line}")
    if any(m <= 0 for _, _, m, _ in cal.segments):
        print("  [warn] a calibration segment has a non-positive slope; "
              "inspect the regression plot")

    base = no_outliers if args.apply_to == "raw" else cleaned
    calibrated = pd.Series(cal.apply(base.to_numpy()), index=base.index)

    out = align(no_outliers, ref, tolerance).rename(columns={"instrument": "raw"})
    out = out.join(cleaned.rename("cleaned"), how="left")
    out = out.join(calibrated.rename("calibrated"), how="left")
    if chosen in cv_oof:
        out = out.join(cv_oof[chosen], how="left")
    else:
        out["calibrated_cv"] = np.nan
    out["delta_raw"] = out["raw"] - out["reference"]
    out["delta_calibrated"] = out["calibrated"] - out["reference"]

    ev = out
    if args.eval_start or args.eval_end:
        ev = out.loc[args.eval_start:args.eval_end]
    m_raw = metrics(ev["raw"].to_numpy(), ev["reference"].to_numpy())
    m_cln = metrics(ev["cleaned"].to_numpy(), ev["reference"].to_numpy())
    m_cal = metrics(ev["calibrated"].to_numpy(), ev["reference"].to_numpy())
    m_cv = metrics(ev["calibrated_cv"].to_numpy(), ev["reference"].to_numpy())
    print("  overall scores (calibrated = fitted and scored on the same data; "
          "calibrated_cv = each block predicted by a model that never saw it):")
    for lab, m in [("raw", m_raw), ("cleaned", m_cln), ("calibrated", m_cal),
                   ("calibrated_cv", m_cv)]:
        print(f"    {lab:<14} R2(pearson)={m['R2']:.3f}  R2(1-SS)={m['R2_det']:.3f}  "
              f"RMSE={m['RMSE']:.2f}  MAE={m['MAE']:.2f}  bias={m['mean_bias']:+.2f}  n={m['n']}")

    # ---- Outputs ----
    out.to_csv(outdir / f"{tag}_calibrated_timeseries.csv")
    cv_table.to_csv(outdir / f"{tag}_cv_summary.csv", index=False)
    pd.concat([f.assign(model=nm) for nm, f in
               {"cleaned (no calibration)": base_folds, **cv_folds}.items() if not f.empty]
              or [pd.DataFrame()]).to_csv(outdir / f"{tag}_cv_folds.csv", index=False)
    summary = {
        "config": asdict(cfg), "time_zone_input": args.tz,
        "period": [str(start), str(end)],
        "time_base": f"local standard time (UTC{utc_offset:+g})",
        "sampling": {"instrument_interval_s": interval.total_seconds(),
                     "reference_interval_s": ref_interval.total_seconds(),
                     "pairing_tolerance_s": tolerance.total_seconds(), **gap_info},
        "reference_tz_input": str(args.ref_tz), "drop_zeros": bool(args.drop_zeros),
        "alignment_check": lag,
        "cleaning": clean_info,
        "threshold_models": {k: (None if m is None else
                                 {"a": m.a, "b": m.b, "k": m.k, "bins_level_n_spread": m.bins})
                             for k, m in models.items()},
        "calibration": {"requested": args.calibration, "chosen": chosen,
                        "equations": cal.describe(),
                        "cross_validation": {"folds": args.cv_folds, "buffer": args.cv_buffer,
                                             "selection": reasons,
                                             "summary": cv_table.to_dict("records")}},
        "applied_to": args.apply_to,
        "eval_window": [args.eval_start, args.eval_end],
        "metrics": {"raw": m_raw, "cleaned": m_cln, "calibrated": m_cal,
                    "calibrated_cv": m_cv},
    }
    with open(outdir / f"{tag}_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    plot_raw_vs_ref(grid, ref, title, outdir / f"{tag}_01_raw_vs_reference.png")
    plot_anomaly(stages, models["anomaly"], cfg.anomaly_threshold, sweep, title,
                 outdir / f"{tag}_02_neighborhood_anomaly.png")
    plot_oscillation(stages, models["oscillation"], cfg, interval, title,
                     outdir / f"{tag}_03_oscillation.png")
    plot_cleaned(grid, cleaned, ref, cfg, title, outdir / f"{tag}_04_cleaned.png")
    plot_reldiff(clean_df, title, outdir / f"{tag}_05_relative_difference.png")
    plot_scatter(clean_df, fit_weights, cal, title, outdir / f"{tag}_06_regression.png")
    plot_calibrated(ev, m_raw, m_cal, m_cv, title, outdir / f"{tag}_07_calibrated.png")
    plot_distributions(out, title, outdir / f"{tag}_08_distributions.png")
    plot_workflow(ref, stages, calibrated, cfg, models, chosen, title,
                  outdir / f"{tag}_09_workflow_all_stages.png")
    plot_cv(cv_table, pd.concat([f.assign(model=nm) for nm, f in
                                 {"cleaned (no calibration)": base_folds, **cv_folds}.items()
                                 if not f.empty] or [pd.DataFrame(columns=["model"])]),
            title, outdir / f"{tag}_10_cross_validation.png")
    print(f"  outputs written to {outdir.resolve()}")
    return summary


def parse_args():
    p = argparse.ArgumentParser(description="3D-PAWS PM2.5 cleaning + calibration (generalized)")
    io = p.add_argument_group("input files")
    io.add_argument("--instrument", help="3D-PAWS CSV (CHORDS export, UTC timestamps)")
    io.add_argument("--reference", help="Co-located reference CSV")
    io.add_argument("--demo", action="store_true", help="Run on synthetic data")
    io.add_argument("--demo-two-cluster", action="store_true",
                    help="Demo data with a second high-bias cluster (like the paper's VA site)")
    io.add_argument("--inst-time-col", default="time")
    io.add_argument("--inst-value-col", default="pm1e25")
    io.add_argument("--inst-skiprows", type=int, default=None,
                    help="Title rows above the header (default: detect)")
    io.add_argument("--ref-time-col", default="time")
    io.add_argument("--ref-value-col", default="pm25")
    io.add_argument("--ref-skiprows", type=int, default=None,
                    help="Title rows above the header (default: detect)")
    io.add_argument("--ref-time-format", default=None,
                    help="strftime format for reference times (default: infer), "
                         "e.g. '%%m/%%d/%%Y %%H:%%M'")

    site = p.add_argument_group("site and time")
    site.add_argument("--name", default="3D-PAWS", help="Site name for titles and file names")
    site.add_argument("--instrument-id", help="3D-PAWS instrument number")
    site.add_argument("--tz", help="IANA time zone of the site, e.g. America/Denver. "
                                   "Its standard-time offset is used year-round.")
    site.add_argument("--utc-offset", type=float,
                      help="Local standard time offset from UTC in hours (alternative to --tz)")
    site.add_argument("--ref-tz", default="LST",
                      help="Clock the reference file is in: LST (default; EPA sites), "
                           "a UTC offset in hours, UTC, or an IANA name for local "
                           "clock time with DST")
    site.add_argument("--interval", default="auto",
                      help="3D-PAWS sampling interval, e.g. 60s (default: infer)")
    site.add_argument("--tolerance", default="auto",
                      help="Max time gap when pairing with the reference "
                           "(default: the reference sampling interval)")

    cl = p.add_argument_group("cleaning")
    cl.add_argument("--outlier-ceiling", type=float, default=350.0,
                    help="Values at/above this are removed (ug/m3)")
    cl.add_argument("--anomaly-k", type=float, default=6.0,
                    help="Data-driven anomaly threshold: flag |anomaly| > k x the "
                         "sensor's fitted noise at that concentration")
    cl.add_argument("--anomaly-threshold", type=float, default=None,
                    help="Fixed anomaly threshold in ug/m3 instead (paper: VA/GA 10, CO 5)")
    cl.add_argument("--oscillation-metric", choices=["roughness", "std"], default="roughness",
                    help="What the oscillation filter measures: point-to-point "
                         "roughness (default; spares smooth real plumes) or the "
                         "rolling std (paper)")
    cl.add_argument("--oscillation-k", type=float, default=6.0,
                    help="Data-driven oscillation threshold: remove where the metric "
                         "> k x its typical value at that concentration")
    cl.add_argument("--rolling-std-threshold", type=float, default=None,
                    help="Use the paper's fixed rolling std threshold in ug/m3 instead "
                         "(VA/GA 7, CO 10); implies --oscillation-metric std")
    cl.add_argument("--oscillation-window", default="30min",
                    help="Oscillation filter window: a time span (30min) or sample count (30)")
    cl.add_argument("--smooth-window", default="30",
                    help="Rolling mean window: sample count (30, as in the paper) or a time span")
    cl.add_argument("--min-coverage", type=float, default=0.5,
                    help="Fraction of a rolling window that must contain data")
    cl.add_argument("--drop-zeros", action="store_true",
                    help="Treat PM2.5 readings of 0 as missing in both datasets")

    ca = p.add_argument_group("calibration")
    ca.add_argument("--calibration", default="auto",
                    choices=["auto", "robust_linear", "hinge",
                             "linear", "piecewise", "low_bias_linear"],
                    help="auto (default): choose between robust_linear and hinge by "
                         "cross-validation. Or force one model; linear, piecewise and "
                         "low_bias_linear are the paper's Eqs. 5, 4 and 6")
    ca.add_argument("--piecewise-split", type=float, default=70.0,
                    help="Split for --calibration piecewise (paper VA: 70)")
    ca.add_argument("--low-bias-cutoff", type=float, default=2.0,
                    help="Cutoff for --calibration low_bias_linear (paper CO: 2)")
    ca.add_argument("--cv-folds", type=int, default=5,
                    help="Number of consecutive time blocks for cross-validation")
    ca.add_argument("--cv-buffer", default="1h",
                    help="Data within this time of a test block are left out of training")
    ca.add_argument("--apply-to", choices=["cleaned", "raw"], default="cleaned",
                    help="Apply the calibration to the cleaned (default) or raw series")

    ev = p.add_argument_group("evaluation and output")
    ev.add_argument("--eval-start", help="Metrics window start, e.g. 2025-12-01")
    ev.add_argument("--eval-end", help="Metrics window end, e.g. 2025-12-25")
    ev.add_argument("--outdir", default=None,
                    help="Output folder name, created inside results/ next to this script "
                         "(default: results_<instrument-id>, or results_demo with --demo). "
                         "An absolute path is used as given.")

    a = p.parse_args()
    if not a.demo and not (a.instrument and a.reference):
        p.error("provide --instrument and --reference, or use --demo")
    if not a.demo and not (a.tz or a.utc_offset is not None):
        p.error("provide --tz (e.g. America/Denver) or --utc-offset")
    if a.tz and a.utc_offset is not None:
        p.error("give --tz or --utc-offset, not both")
    if not 0 < a.min_coverage <= 1:
        p.error("--min-coverage must be in (0, 1]")
    if a.anomaly_k <= 0 or a.oscillation_k <= 0:
        p.error("--anomaly-k and --oscillation-k must be positive")
    if a.cv_folds < 2:
        p.error("--cv-folds must be at least 2")
    return a


if __name__ == "__main__":
    run(parse_args())
