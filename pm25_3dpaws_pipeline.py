#!/usr/bin/env python3
"""
3D-PAWS PM2.5 cleaning and calibration pipeline
================================================
Implements the workflow in Pham (NSF SOARS 2026), "Characterizing PM2.5 Data in
Low-Cost Air Quality Sensors":

  Section 3a - Data pre-processing and cleaning (Fig. 3)
    1. Put both datasets on local standard time: 3D-PAWS from UTC, the
       reference from whatever clock it was logged in (--ref-tz)
    2. Remove outliers >= 350 ug/m3
    3. Temporal sequence analysis: neighborhood anomaly (Eq. 1); flagged points
       are REPLACED by the mean of their four nearest neighbors
    4. 30-minute rolling standard deviation; points above the site threshold are
       REMOVED
    5. 30-measurement rolling mean to smooth integer (quantization) steps

  Section 3b - Calibration (Fig. 4)
    6. Delta (Eq. 2) and relative difference (Eq. 3) vs. the reference
    7. Site-specific regression of reference (y) on cleaned 3D-PAWS (x)
         VA: piecewise linear split at 70 ug/m3, after removing points whose
             relative difference falls outside 1 standard deviation (Eq. 4)
         GA: single linear fit (Eq. 5)
         CO: mean-bias correction for x < 2 ug/m3, then linear fit (Eq. 6)
    8. Apply the regression to the 3D-PAWS data; evaluate R^2, RMSE, MAE

Usage
-----
  # Try it end-to-end on synthetic data:
  python pm25_3dpaws_pipeline.py --site VA --demo

  # Boulder 3D-PAWS 153 vs. Marine St reference:
  python pm25_3dpaws_pipeline.py --site CO \
      --instrument AQ_Comparison_2_CU_153_May2026-Oct2026.csv \
      --reference PM2_5_data_from_Marine_St__May2026-Oct2026_1.csv \
      --inst-time-col timestamp --inst-value-col pm1e25 \
      --ref-skiprows 2 --ref-time-col Date --ref-value-col UG/M3 \
      --ref-time-format "%m/%d/%Y %H:%M" --ref-tz LST --drop-zeros --outdir results_CO
      # Marine St logs in MST year-round (no DST), even May-Oct: the alignment
      # check finds the best match at UTC-7, and MDT leaves a ~1 h offset.

  # Real data (CHORDS export + Teledyne T640x reference):
  python pm25_3dpaws_pipeline.py --site GA \
      --instrument chords_167.csv --reference athens_t640x.csv \
      --inst-time-col time --inst-value-col pm1e25 \
      --ref-time-col datetime --ref-value-col pm25 \
      --outdir results_GA
"""

import argparse
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Site configuration (values from Sections 2 and 4a of the paper)
# ---------------------------------------------------------------------------

OUTLIER_CEILING = 350.0       # ug/m3; points at/above are treated as sensor error
ROLLING_STD_WINDOW = "30min"  # 30-minute window chosen over 1 hour (Fig. 6)
ROLLING_MEAN_WINDOW = 30      # 30 measurements
MAX_STEP_GAP = "3min"         # neighbors further apart than this are not "neighbors"


@dataclass
class SiteConfig:
    name: str
    instrument_id: int
    utc_offset_hours: int          # local STANDARD time (no DST)
    anomaly_threshold: float       # ug/m3, neighborhood anomaly (Sec. 4a)
    rolling_std_threshold: float   # ug/m3, 30-min rolling std (Sec. 4a)
    calibration: str               # "piecewise" | "linear" | "low_bias_linear"
    piecewise_split: float = 70.0  # ug/m3 (VA)
    low_bias_cutoff: float = 2.0   # ug/m3 (CO)


SITES = {
    "VA": SiteConfig("Lee District Park, VA", 156, -5, 10.0, 7.0, "piecewise"),
    "GA": SiteConfig("Athens, GA", 167, -5, 10.0, 7.0, "linear"),
    "CO": SiteConfig("Boulder, CO", 153, -7, 5.0, 10.0, "low_bias_linear"),
}

# Coefficients reported in the paper (Eqs. 4-6), usable with --use-paper-coeffs.
# Each segment is (lower_bound, upper_bound, slope, intercept) on raw x.
PAPER_COEFFS = {
    "VA": [(-np.inf, 70.0, 0.32, 2.43), (70.0, np.inf, 0.21, -11.81)],
    "GA": [(-np.inf, np.inf, 0.56, 5.0)],
    "CO": [(-np.inf, np.inf, 0.82, 0.92)],
}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_instrument(path, time_col, value_col, utc_offset_hours):
    """Load 3D-PAWS (CHORDS) data. Timestamps are UTC -> local standard time."""
    df = pd.read_csv(path)
    t_utc = pd.to_datetime(df[time_col], utc=True).dt.tz_convert(None)
    t_local = t_utc + pd.Timedelta(hours=utc_offset_hours)
    s = pd.Series(pd.to_numeric(df[value_col], errors="coerce").to_numpy(dtype=float),
                  index=pd.DatetimeIndex(t_local).astype("datetime64[ns]"),
                  name="pm25")
    s.index.name = "time"
    return s[~s.index.duplicated()].sort_index().dropna()


def reference_to_standard_time(t, ref_tz, std_offset_hours):
    """Convert naive reference timestamps to the site's local STANDARD time,
    the common time base used throughout the paper (Sec. 2).

    ref_tz describes the clock the reference file was recorded in:
      "LST"            already local standard time (Kaylena's 2025 CO data)
      -6, "-6", -6.0   fixed UTC offset in hours (Boulder MDT = -6)
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


def load_reference(path, time_col, value_col, skiprows=0, time_format=None,
                   ref_tz="LST", std_offset_hours=0):
    """Load Teledyne T640x reference data and put it on local standard time.
    skiprows drops title rows above the real header, e.g. the Marine St export
    has 'Marine Street' and 'PMFine' rows above 'Date,UG/M3' (skiprows=2)."""
    df = pd.read_csv(path, skiprows=skiprows)
    t = pd.to_datetime(df[time_col], format=time_format)
    if getattr(t.dt, "tz", None) is not None:   # tz-aware strings: trust them
        t = t.dt.tz_convert("UTC").dt.tz_localize(None)
        ref_tz = "UTC"
    t = reference_to_standard_time(t, ref_tz, std_offset_hours)
    s = pd.Series(pd.to_numeric(df[value_col], errors="coerce").to_numpy(dtype=float),
                  index=pd.DatetimeIndex(t).astype("datetime64[ns]"), name="pm25")
    s.index.name = "time"
    s = s[s.index.notna()]
    return s[~s.index.duplicated()].sort_index().dropna()


def estimate_time_lag(inst, ref, max_lag_min=180, step_min=5):
    """Diagnostic (not part of the paper's workflow): find the shift of the
    3D-PAWS series that best correlates with the reference. A best lag near
    +/-60 min means one dataset's time zone or DST setting is probably wrong.
    Positive lag = 3D-PAWS timestamps would need to move LATER to match."""
    a = inst.resample("1min").mean().rolling(15, min_periods=5, center=True).mean()
    b = ref.resample("1min").mean()
    out = {}
    for lag in range(-max_lag_min, max_lag_min + 1, step_min):
        j = pd.concat([a.shift(freq=f"{lag}min"), b], axis=1, sort=True).dropna()
        if len(j) > 100:
            out[lag] = j.corr().iloc[0, 1]
    r = pd.Series(out)
    return {"best_lag_min": int(r.idxmax()), "r_best": float(r.max()),
            "r_at_zero": float(r.get(0, np.nan))}


# ---------------------------------------------------------------------------
# Section 3a: cleaning
# ---------------------------------------------------------------------------

def remove_outliers(s, ceiling=OUTLIER_CEILING):
    return s[s < ceiling]


def neighborhood_anomaly(s, max_step_gap=MAX_STEP_GAP):
    """Eq. 1: x_i minus the mean of x_{i-2}, x_{i-1}, x_{i+1}, x_{i+2}.
    (The paper's printed equation lists x_{i+2} twice; x_{i+1} is intended.)

    Gap-aware: if any step between x_i and one of its neighbors exceeds
    max_step_gap (e.g. a transmission outage), that point's neighborhood is
    invalid, its anomaly is NaN, and it is never flagged or replaced."""
    nb_mean = (s.shift(2) + s.shift(1) + s.shift(-1) + s.shift(-2)) / 4.0
    if max_step_gap is not None:
        step = pd.Series(s.index, index=s.index).diff()      # gap before each point
        big = step > pd.Timedelta(max_step_gap)
        # neighborhood i-2..i+2 spans the gaps before i-1, i, i+1, i+2
        bad = big | big.shift(1, fill_value=False) | big.shift(-1, fill_value=False) \
            | big.shift(-2, fill_value=False)
        nb_mean = nb_mean.mask(bad)
    return s - nb_mean, nb_mean


def temporal_sequence_filter(s, threshold, max_step_gap=MAX_STEP_GAP):
    """Flag |anomaly| > threshold and replace flagged points with the
    four-neighbor mean, preserving continuity of the record."""
    anomaly, nb_mean = neighborhood_anomaly(s, max_step_gap)
    flagged = anomaly.abs() > threshold
    out = s.copy()
    out[flagged] = nb_mean[flagged]
    return out, anomaly, flagged


def threshold_sweep(s, thresholds=(5, 10, 15, 20, 25), max_step_gap=MAX_STEP_GAP):
    """Reproduces the threshold table embedded in Fig. 5."""
    anomaly, _ = neighborhood_anomaly(s, max_step_gap)
    rows = []
    for th in thresholds:
        n = int((anomaly.abs() > th).sum())
        rows.append({"threshold": th, "flagged_points": n,
                     "total_points": len(s),
                     "percent_flagged": round(100 * n / len(s), 3)})
    return pd.DataFrame(rows)


def rolling_std(s, window=ROLLING_STD_WINDOW):
    return s.rolling(window, center=True, min_periods=2).std()


def rolling_std_filter(s, threshold, window=ROLLING_STD_WINDOW):
    """Remove points where the signal oscillates heavily."""
    rstd = rolling_std(s, window)
    keep = ~(rstd > threshold)
    return s[keep], rstd, ~keep


def rolling_mean_smooth(s, window=ROLLING_MEAN_WINDOW):
    return s.rolling(window, center=True, min_periods=1).mean()


def clean_instrument(raw, cfg, max_step_gap=MAX_STEP_GAP):
    """Full Fig. 3 workflow. Returns the cleaned series plus intermediates."""
    step1 = remove_outliers(raw)
    step2, anomaly, flagged = temporal_sequence_filter(step1, cfg.anomaly_threshold,
                                                       max_step_gap)
    step3, rstd, removed = rolling_std_filter(step2, cfg.rolling_std_threshold)
    step4 = rolling_mean_smooth(step3)
    info = {
        "n_raw": int(len(raw)),
        "n_outliers_removed": int(len(raw) - len(step1)),
        "n_anomaly_replaced": int(flagged.sum()),
        "pct_anomaly_replaced": round(100 * flagged.mean(), 3),
        "n_skipped_near_gaps": int(anomaly.isna().sum()),
        "n_oscillation_removed": int(removed.sum()),
        "pct_oscillation_removed": round(100 * removed.mean(), 3),
        "n_clean": int(len(step4)),
    }
    stages = {"no_outliers": step1, "anomaly": anomaly, "flagged": flagged,
              "replaced": step2, "rstd": rstd, "std_filtered": step3,
              "cleaned": step4}
    return step4, stages, info


# ---------------------------------------------------------------------------
# Alignment and difference metrics
# ---------------------------------------------------------------------------

def align(inst, ref, tolerance="60s"):
    """Pair each 3D-PAWS observation with the nearest reference minute.
    Only concurrent periods survive (Sec. 2)."""
    a = inst.rename("instrument").reset_index()
    b = ref.rename("reference").reset_index()
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
# Section 3b: calibration
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

    def describe(self):
        lines = []
        if self.low_cutoff is not None and self.low_bias:
            lines.append(f"x -> x + {self.low_bias:.3f} for x < {self.low_cutoff}")
        for lo, hi, m, b in self.segments:
            rng = ""
            if np.isfinite(lo) or np.isfinite(hi):
                rng = f"  ({lo:g} <= x < {hi:g})"
            lines.append(f"y = {m:.3f}x {'+' if b >= 0 else '-'} {abs(b):.3f}{rng}")
        return lines


def _linfit(x, y):
    m, b = np.polyfit(x, y, 1)
    return float(m), float(b)


def fit_calibration(clean_df, cfg):
    """Regress reference (y) on cleaned 3D-PAWS (x), site-specifically."""
    df = add_differences(clean_df)
    x, y = df["instrument"].to_numpy(), df["reference"].to_numpy()

    if cfg.calibration == "linear":
        m, b = _linfit(x, y)
        return Calibration("linear", [(-np.inf, np.inf, m, b)]), df.index

    if cfg.calibration == "piecewise":
        # Split at 70 ug/m3 FIRST, then drop points whose relative difference
        # lies outside 1 std of that segment's own distribution. Filtering on
        # the pooled distribution would discard the whole overestimation
        # cluster that the >=70 segment is meant to fit (Figs. 9-10, Eq. 4).
        segs, kept = [], []
        for lo, hi in [(-np.inf, cfg.piecewise_split), (cfg.piecewise_split, np.inf)]:
            part = df[(df["instrument"] >= lo) & (df["instrument"] < hi)]
            rd = part["rel_diff"]
            part = part[(rd - rd.mean()).abs() <= rd.std()]
            if len(part) >= 10:
                segs.append((lo, hi, *_linfit(part["instrument"], part["reference"])))
                kept.append(part.index)
        if len(segs) < 2:  # not enough high-concentration data: fall back
            print("  [warn] too few points on one side of the split; "
                  "using a single linear fit instead")
            rd = df["rel_diff"]
            sub = df[(rd - rd.mean()).abs() <= rd.std()]
            return (Calibration("piecewise", [(-np.inf, np.inf,
                    *_linfit(sub["instrument"], sub["reference"]))]), sub.index)
        return Calibration("piecewise", segs), kept[0].append(kept[1])

    if cfg.calibration == "low_bias_linear":
        low = x < cfg.low_bias_cutoff
        bias = float(y[low].mean() - x[low].mean()) if low.any() else 0.0
        xc = np.where(low, x + bias, x)
        m, b = _linfit(xc, y)
        return (Calibration("low_bias_linear", [(-np.inf, np.inf, m, b)],
                            low_cutoff=cfg.low_bias_cutoff, low_bias=bias),
                df.index)

    raise ValueError(f"Unknown calibration type: {cfg.calibration}")


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


def plot_raw_vs_ref(raw, ref, title, path):  # Fig. 2
    fig, ax = plt.subplots(figsize=(12, 3.5))
    ax.plot(raw.index, raw, color="tab:blue", lw=0.5, label="3D-PAWS raw")
    ax.plot(ref.index, ref, color="black", lw=0.6, label="Reference")
    ax.set_ylabel("PM$_{2.5}$ (µg m$^{-3}$)")
    ax.set_title(title)
    ax.legend(loc="upper right")
    _save(fig, path)


def plot_anomaly(stages, threshold, sweep, title, path):  # Fig. 5
    s, an, fl, rep = (stages["no_outliers"], stages["anomaly"],
                      stages["flagged"], stages["replaced"])
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
    axes[1].plot(an.index, an, color="purple", lw=0.5)
    for sign in (1, -1):
        axes[1].axhline(sign * threshold, color="red", ls="--", lw=1)
    axes[1].set_title(f"Neighborhood anomaly (threshold = ±{threshold})")
    axes[2].plot(rep.index, rep, color="green", lw=0.5)
    axes[2].set_title("After replacing flagged points")
    for ax in axes:
        ax.set_ylabel("µg m$^{-3}$")
    axes[1].set_ylabel("Anomaly (µg m$^{-3}$)")
    cells = [[f"{r.threshold:g}", f"{r.flagged_points:d}", f"{r.total_points:d}",
              f"{r.percent_flagged:.2f}%"] for r in sweep.itertuples()]
    tax.table(cellText=cells, colLabels=list(sweep.columns), loc="center",
              cellLoc="center")
    _save(fig, path)


def plot_rstd_hist(series, title, path):  # Fig. 6
    fig, ax = plt.subplots(figsize=(7, 4))
    for w, c in [("30min", "tab:blue"), ("60min", "tab:orange")]:
        v = rolling_std(series, w).dropna()
        ax.hist(v, bins=100, alpha=0.5, color=c, label=f"{w} rolling std")
    ax.set_yscale("log")
    ax.set_xlabel("Rolling standard deviation (µg m$^{-3}$)")
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.legend()
    _save(fig, path)


def plot_cleaned(raw, cleaned, ref, title, path):  # Fig. 7 / Fig. 17 style
    fig, ax = plt.subplots(figsize=(12, 3.5))
    ax.plot(raw.index, raw, color="tab:blue", lw=0.4, alpha=0.6, label="3D-PAWS raw")
    ax.plot(cleaned.index, cleaned, color="green", lw=0.8, label="Cleaned (30-pt mean)")
    ax.plot(ref.index, ref, color="black", lw=0.6, label="Reference")
    ax.set_ylabel("PM$_{2.5}$ (µg m$^{-3}$)")
    ax.set_title(title)
    ax.legend(loc="upper right")
    _save(fig, path)


def plot_reldiff(df, title, path):  # Fig. 8
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


def plot_scatter(df, fit_index, cal, cfg, title, path):  # Figs. 9-12
    fig, ax = plt.subplots(figsize=(6, 6))
    used = df.index.isin(fit_index)
    ax.scatter(df["instrument"][~used], df["reference"][~used], s=2,
               color="lightgrey", label="excluded from fit")
    ax.scatter(df["instrument"][used], df["reference"][used], s=2,
               color="tab:blue", alpha=0.4, label="used in fit")
    lim = max(df["instrument"].max(), df["reference"].max()) * 1.05
    ax.plot([0, lim], [0, lim], color="grey", ls="--", label="1:1")
    xx = np.linspace(0, df["instrument"].max(), 500)
    ax.plot(xx, cal.apply(xx), color="red", lw=1.5, label="calibration")
    if cfg.calibration == "piecewise":
        ax.axvline(cfg.piecewise_split, color="tab:orange", ls=":", lw=1)
    ax.set_xlabel("Cleaned 3D-PAWS PM$_{2.5}$ (µg m$^{-3}$)")
    ax.set_ylabel("Reference PM$_{2.5}$ (µg m$^{-3}$)")
    ax.set_title(title + "\n" + "; ".join(cal.describe()), fontsize=9)
    ax.legend(loc="upper left", fontsize=8)
    _save(fig, path)


def plot_calibrated(out, m_raw, m_cal, title, path):  # Figs. 13-16
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True, sharey=True)
    axes[0].plot(out.index, out["raw"], color="tab:blue", lw=0.5, label="3D-PAWS raw")
    axes[0].plot(out.index, out["reference"], color="black", lw=0.6, label="Reference")
    axes[0].set_title(f"{title} — raw: R² = {100*m_raw['R2']:.1f}% "
                      f"(1−SS: {m_raw['R2_det']:.2f}), RMSE = {m_raw['RMSE']:.2f} µg m⁻³")
    axes[1].plot(out.index, out["calibrated"], color="purple", lw=0.5, label="3D-PAWS calibrated")
    axes[1].plot(out.index, out["reference"], color="black", lw=0.6, label="Reference")
    axes[1].set_title(f"Calibrated: R² = {100*m_cal['R2']:.1f}% "
                      f"(1−SS: {m_cal['R2_det']:.2f}), RMSE = {m_cal['RMSE']:.2f} µg m⁻³")
    for ax in axes:
        ax.set_ylabel("µg m$^{-3}$")
        ax.legend(loc="upper right")
    _save(fig, path)


def plot_workflow(raw, ref, stages, calibrated, threshold, title, path):  # Fig. 17
    """Every stage of the workflow, top to bottom, against the reference."""
    s, fl = stages["no_outliers"], stages["flagged"]
    panels = [
        ("1. Raw 3D-PAWS (outliers ≥350 removed)", s, "tab:blue"),
        (f"2. Temporal sequence analysis: flagged |anomaly| > {threshold:g}", s, "tab:blue"),
        ("3. Flagged points replaced by 4-neighbor mean", stages["replaced"], "green"),
        ("4. After 30-min rolling standard deviation filter", stages["std_filtered"], "teal"),
        ("5. After 30-measurement rolling mean", stages["cleaned"], "darkgreen"),
        ("6. Calibrated (site-specific regression)", calibrated, "purple"),
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


def plot_distributions(out, title, path):  # Fig. 18
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

def make_demo_data(cfg, outdir, days=30, seed=0, site="GA"):
    rng = np.random.default_rng(seed)
    t = pd.date_range("2025-11-01", periods=days * 1440, freq="1min")
    hrs = np.asarray(t.hour + t.minute / 60, dtype=float)
    day = np.arange(len(t)) / 1440
    ref = 8 + 2 * np.sin(2 * np.pi * day / 7) + 3 * np.sin(2 * np.pi * (hrs - 6) / 24)
    for c in rng.uniform(1, days - 1, 6):               # enhancement events
        ref += rng.uniform(15, 40) * np.exp(-0.5 * ((day - c) / 0.15) ** 2)
    ref = np.clip(ref + rng.normal(0, 0.5, len(t)), 0, None)

    inst = 1.6 * ref - 4 + rng.normal(0, 2, len(t))     # under low, over high
    if site == "VA":
        # VA-like second bias cluster (Fig. 9): episodes where the sensor reads
        # far above 70 ug/m3 while the reference stays near 10-20 ug/m3
        for c in rng.uniform(2, days - 2, 4):
            w = np.exp(-0.5 * ((day - c) / 0.4) ** 2)
            inst += 4.0 * ref * w
    spikes = rng.random(len(t)) < 0.005
    inst[spikes] += rng.uniform(20, 80, spikes.sum())
    for s0 in rng.integers(0, len(t) - 120, 8):          # oscillating periods
        inst[s0:s0 + 120] += 15 * np.where(np.arange(120) % 2, 1, -1)
    inst[rng.integers(0, len(t), 5)] = 400               # >350 outliers
    inst = np.clip(np.round(inst), 0, None)              # integer reporting

    t_utc = t - pd.Timedelta(hours=cfg.utc_offset_hours)
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

def run(args):
    cfg = SITES[args.site]
    if args.anomaly_threshold is not None:
        cfg.anomaly_threshold = args.anomaly_threshold
    if args.rolling_std_threshold is not None:
        cfg.rolling_std_threshold = args.rolling_std_threshold
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    tag = f"{args.site}_{cfg.instrument_id}"
    title = f"{cfg.name} (Instrument {cfg.instrument_id})"

    if args.demo:
        args.instrument, args.reference = make_demo_data(cfg, outdir, site=args.site)
        args.inst_time_col, args.inst_value_col = "time", "pm1e25"
        args.ref_time_col, args.ref_value_col = "time", "pm25"

    print(f"== {title} ==")
    raw = load_instrument(args.instrument, args.inst_time_col,
                          args.inst_value_col, cfg.utc_offset_hours)
    ref = load_reference(args.reference, args.ref_time_col, args.ref_value_col,
                         args.ref_skiprows, args.ref_time_format,
                         args.ref_tz, cfg.utc_offset_hours)
    print(f"  time base: local standard time (UTC{cfg.utc_offset_hours:+d}); "
          f"3D-PAWS from UTC, reference from '{args.ref_tz}'")

    if args.drop_zeros:
        nz_i, nz_r = int((raw <= 0).sum()), int((ref <= 0).sum())
        raw, ref = raw[raw > 0], ref[ref > 0]
        print(f"  dropped zero readings: 3D-PAWS {nz_i} "
              f"({100*nz_i/(len(raw)+nz_i):.1f}%), reference {nz_r}")

    # Keep only the concurrent observation period
    start, end = max(raw.index.min(), ref.index.min()), min(raw.index.max(), ref.index.max())
    raw, ref = raw[start:end], ref[start:end]
    print(f"  concurrent period: {start} -> {end}")
    lag = estimate_time_lag(raw, ref)
    print(f"  alignment check: best lag {lag['best_lag_min']:+d} min "
          f"(r={lag['r_best']:.3f}; r at 0 lag={lag['r_at_zero']:.3f})")
    if abs(lag["best_lag_min"]) >= 30:
        print("  [warn] series look misaligned by ~"
              f"{abs(lag['best_lag_min'])} min; check time zone / DST settings")

    # ---- Section 3a: cleaning ----
    max_gap = None if str(args.max_gap).lower() == "none" else args.max_gap
    sweep = threshold_sweep(remove_outliers(raw), max_step_gap=max_gap)
    sweep.to_csv(outdir / f"{tag}_anomaly_threshold_sweep.csv", index=False)
    cleaned, stages, clean_info = clean_instrument(raw, cfg, max_gap)
    for k, v in clean_info.items():
        print(f"  {k}: {v}")

    # ---- Section 3b: calibration ----
    clean_df = add_differences(align(cleaned, ref, args.tolerance))
    if args.use_paper_coeffs:
        cal = Calibration(cfg.calibration, PAPER_COEFFS[args.site])
        fit_index = clean_df.index
        print("  using paper coefficients (Eqs. 4-6)")
    else:
        cal, fit_index = fit_calibration(clean_df, cfg)
    for line in cal.describe():
        print(f"  calibration: {line}")

    base = remove_outliers(raw) if args.apply_to == "raw" else cleaned
    calibrated = pd.Series(cal.apply(base.to_numpy()), index=base.index)

    out = align(remove_outliers(raw), ref, args.tolerance).rename(columns={"instrument": "raw"})
    out = out.join(cleaned.rename("cleaned"), how="left")
    out = out.join(calibrated.rename("calibrated"), how="left")
    out["delta_raw"] = out["raw"] - out["reference"]
    out["delta_calibrated"] = out["calibrated"] - out["reference"]

    ev = out
    if args.eval_start or args.eval_end:
        ev = out.loc[args.eval_start:args.eval_end]
    m_raw = metrics(ev["raw"].to_numpy(), ev["reference"].to_numpy())
    m_cln = metrics(ev["cleaned"].to_numpy(), ev["reference"].to_numpy())
    m_cal = metrics(ev["calibrated"].to_numpy(), ev["reference"].to_numpy())
    for lab, m in [("raw", m_raw), ("cleaned", m_cln), ("calibrated", m_cal)]:
        print(f"  {lab:<10} R2(pearson)={m['R2']:.3f}  R2(1-SS)={m['R2_det']:.3f}  "
              f"RMSE={m['RMSE']:.2f}  MAE={m['MAE']:.2f}  bias={m['mean_bias']:+.2f}  n={m['n']}")

    # ---- Outputs ----
    out.to_csv(outdir / f"{tag}_calibrated_timeseries.csv")
    summary = {
        "site": asdict(cfg), "period": [str(start), str(end)],
        "time_base": f"local standard time (UTC{cfg.utc_offset_hours:+d})",
        "reference_tz_input": str(args.ref_tz), "drop_zeros": bool(args.drop_zeros), "alignment_check": lag,
        "cleaning": clean_info, "calibration": cal.describe(),
        "applied_to": args.apply_to,
        "eval_window": [args.eval_start, args.eval_end],
        "metrics": {"raw": m_raw, "cleaned": m_cln, "calibrated": m_cal},
    }
    with open(outdir / f"{tag}_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    plot_raw_vs_ref(raw, ref, title, outdir / f"{tag}_01_raw_vs_reference.png")
    plot_anomaly(stages, cfg.anomaly_threshold, sweep, title, outdir / f"{tag}_02_neighborhood_anomaly.png")
    plot_rstd_hist(stages["replaced"], title, outdir / f"{tag}_03_rolling_std_hist.png")
    plot_cleaned(raw, cleaned, ref, title, outdir / f"{tag}_04_cleaned.png")
    plot_reldiff(clean_df, title, outdir / f"{tag}_05_relative_difference.png")
    plot_scatter(clean_df, fit_index, cal, cfg, title, outdir / f"{tag}_06_regression.png")
    plot_calibrated(ev, m_raw, m_cal, title, outdir / f"{tag}_07_calibrated.png")
    plot_distributions(out, title, outdir / f"{tag}_08_distributions.png")
    plot_workflow(raw, ref, stages, calibrated, cfg.anomaly_threshold, title,
                  outdir / f"{tag}_09_workflow_all_stages.png")
    print(f"  outputs written to {outdir.resolve()}")
    return summary


def parse_args():
    p = argparse.ArgumentParser(description="3D-PAWS PM2.5 cleaning + calibration")
    p.add_argument("--site", required=True, choices=SITES.keys())
    p.add_argument("--instrument", help="3D-PAWS CSV (CHORDS export, UTC timestamps)")
    p.add_argument("--reference", help="Reference CSV (local standard time)")
    p.add_argument("--demo", action="store_true", help="Run on synthetic data")
    p.add_argument("--inst-time-col", default="time")
    p.add_argument("--inst-value-col", default="pm1e25")
    p.add_argument("--ref-time-col", default="time")
    p.add_argument("--ref-value-col", default="pm25")
    p.add_argument("--ref-skiprows", type=int, default=0,
                   help="Title rows above the reference header (Marine St: 2)")
    p.add_argument("--ref-time-format", default=None,
                   help="strftime format for reference times, e.g. '%%m/%%d/%%Y %%H:%%M'")
    p.add_argument("--ref-tz", default="LST",
                   help="Clock the reference file is in: LST (default, as in the paper), "
                        "a UTC offset in hours (Boulder MDT: -6), UTC, or an IANA name "
                        "such as America/Denver for files that cross a DST change. "
                        "Everything is converted to local standard time.")
    p.add_argument("--drop-zeros", action="store_true",
                   help="Treat PM2.5 readings of 0 as missing in both datasets")
    p.add_argument("--anomaly-threshold", type=float, help="Override site default")
    p.add_argument("--rolling-std-threshold", type=float, help="Override site default")
    p.add_argument("--apply-to", choices=["cleaned", "raw"], default="cleaned",
                   help="Apply calibration to the cleaned series (default; Fig. 17 and "
                        "consistent with the paper's reported R2/RMSE) or to raw data")
    p.add_argument("--use-paper-coeffs", action="store_true",
                   help="Use Eqs. 4-6 instead of fitting new coefficients")
    p.add_argument("--max-gap", default=MAX_STEP_GAP,
                   help="Max time step between anomaly-filter neighbors (e.g. 3min); "
                        "'none' disables the gap check")
    p.add_argument("--tolerance", default="60s", help="Max time gap when pairing with reference")
    p.add_argument("--eval-start", help="Metrics window start, e.g. 2025-12-01")
    p.add_argument("--eval-end", help="Metrics window end, e.g. 2025-12-25")
    p.add_argument("--outdir", default="results")
    a = p.parse_args()
    if not a.demo and not (a.instrument and a.reference):
        p.error("provide --instrument and --reference, or use --demo")
    return a


if __name__ == "__main__":
    run(parse_args())
