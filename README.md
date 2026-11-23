# Generalized PM2.5 Cleaning and Calibration Pipeline

`pm25_3dpaws_pipeline_general.py` compares PM2.5 from a 3D-PAWS sensor with a reference-grade
monitor at the same site. It then:

- cleans the 3D-PAWS data;
- works out a correction equation (a *calibration*) that brings the 3D-PAWS readings in line with
  the reference;
- reports how much the correction can be trusted on data it has not seen.

It follows the method in Kaylena Pham's SOARS paper (`docs/`), with one difference: it makes no
assumption that anyone has already studied the data. Every setting Kaylena chose by inspecting
plots is now worked out from the data itself, each time the script runs.

---

## Contents

1. [What changed from the original script](#1-what-changed-from-the-original-script)
2. [What the pipeline does, step by step](#2-what-the-pipeline-does-step-by-step)
3. [How to run it](#3-how-to-run-it)
4. [What it produces](#4-what-it-produces)
5. [How to read the results](#5-how-to-read-the-results)
6. [Limitations and open questions](#6-limitations-and-open-questions)
7. [Optional settings](#7-optional-settings)

---

## 1. What changed from the original script

`pm25_3dpaws_pipeline.py` (the original) reproduces Kaylena's analysis for the three sites in her
paper. It is unchanged and still works. The general script is a separate file built from it.

| | Original (`pm25_3dpaws_pipeline.py`) | Generalized (`pm25_3dpaws_pipeline_general.py`) |
|---|---|---|
| **Which data it handles** | Only her three sites (VA, GA, CO), chosen with `--site`. | Any 3D-PAWS file paired with any reference file. You give the site's time zone instead of a site name. |
| **Spike filter threshold** | A fixed number per site that Kaylena chose (e.g. 5 µg/m³ in Boulder). | Calculated from each dataset's own noise level. It allows larger swings at higher concentrations, so real smoke events are not mistaken for spikes. |
| **Oscillation filter threshold** | A fixed rolling-standard-deviation limit per site (e.g. 10 in Boulder). In Boulder this removed parts of a real wildfire event. | Calculated from the data, using a measure of point-to-point jumpiness that leaves real, steady rises and falls alone. |
| **Missing data** | Rolling windows counted rows, so a window could silently span a multi-hour outage. | Missing timestamps are detected and filled with blank rows, so every window covers the time it says it does. |
| **Choice of regression** | Kaylena chose the shape for each site by looking at plots (straight line, two-piece line, or a line fitted only above 2 µg/m³). | The script tries a straight line and a line with one bend, and keeps the bend only if it clearly predicts better on held-out data. |
| **Fitting method** | Ordinary least squares, which a handful of bad points can pull off course. | A robust fit that gives less weight to points far from the line. |
| **How accuracy is reported** | Only on the same data the equation was fitted to, which flatters the result. | Also on held-out data: the record is split into 5 time blocks, and each block is predicted by an equation fitted without it (*cross-validation*). |
| **Paper's settings** | Built in. | Still available as options (Section 7), so the paper can be reproduced for comparison. |

On the Boulder data from Oct–Dec 2025 (instrument 153), the new script's automatic choices
predicted held-out data with an average error (RMSE) of **1.59 µg/m³**. Kaylena's hand-picked
settings, scored the same way, gave **2.01 µg/m³**.

---

## 2. What the pipeline does, step by step

1. **Load and line up the clocks.** 3D-PAWS timestamps are in UTC. Reference monitors usually
   record in local *standard* time (no daylight saving). Both are converted to local standard
   time. The script then checks the two records really line up by sliding one against the other
   and reporting which shift matches best. That shift should be close to 0.
2. **Trim to the overlap.** Only the period covered by both files is kept.
3. **Mark missing data.** Gaps in the 3D-PAWS record get blank rows so later steps know data is
   missing rather than assuming the readings are continuous.
4. **Remove implausible values.** Readings above 350 µg/m³ (the cutoff carried over from the
   original script) are discarded.
5. **Fix isolated spikes.** Each reading is compared with its neighbors a minute or two either
   side. If it jumps too far from them, it is replaced with the neighbors' typical value. "Too
   far" is learned from the dataset itself.
6. **Remove unstable stretches.** In 30-minute windows, the script measures how much the reading
   bounces up and down from one sample to the next. Windows that bounce far more than is normal
   for that concentration are removed.
7. **Smooth.** A 30-point (about 30 minutes at 1-minute sampling) rolling average evens out the
   whole-number steps the sensor reports in.
8. **Calibrate.** The cleaned 3D-PAWS data is paired with the reference minute by minute. Two
   candidate equations are fitted and compared on held-out time blocks, and the better one is
   refitted on all the data and applied.
9. **Report.** Results, plots and the final equation are written to the output folder.

---

## 3. How to run it

### One-time setup

From the project folder, in a terminal:

```zsh
cd /path/to/Kathryns-AQI
python3 -m venv .venv                                    # only if .venv doesn't exist yet
source .venv/bin/activate
pip install -r pm25_3dpaws_pipeline_requirements.txt
```

### Each time

Activate the environment, then run the script. This example is the May–Oct 2026 Boulder data:

```zsh
source .venv/bin/activate

python pm25_3dpaws_pipeline_general.py \
    --instrument data/raw/AQ_Comparison_2_CU_153_May2026-Oct2026.csv \
    --reference "data/raw/PM2.5 data from Marine St. May2026-Oct2026 (1).csv" \
    --inst-time-col timestamp \
    --tz America/Denver --ref-time-col Date --ref-value-col UG/M3 \
    --name Boulder --instrument-id 153 --outdir results_153_2026
```

### What you must tell it

| Option | What it is | Example |
|---|---|---|
| `--instrument` | The 3D-PAWS CSV file. | `data/raw/AQ_Comparison_2_CU_153_May2026-Oct2026.csv` |
| `--reference` | The reference monitor's CSV file. Put it in quotes if the name has spaces. | `"data/raw/PM2.5 data from Marine St. ... .csv"` |
| `--tz` | The site's time zone. | `America/Denver` (Colorado), `America/New_York` (Virginia, Georgia) |
| `--inst-time-col`, `--inst-value-col` | Column names for time and PM2.5 in the 3D-PAWS file. Defaults: `time` and `pm1e25`. | Newer CHORDS exports call the time column `timestamp`. |
| `--ref-time-col`, `--ref-value-col` | Column names for time and PM2.5 in the reference file. Defaults: `time` and `pm25`. | Marine St: `Date` and `UG/M3` |
| `--ref-tz` | The clock the reference file uses. Default: local standard time, which is normal for regulatory monitors. | `UTC` for the Erie/AJAX file |
| `--name`, `--instrument-id` | Labels for the output file names and plot titles. | `Boulder`, `153` |
| `--outdir` | Name of the folder for this run's results. It is always created inside the `results/` folder, so `results_153_2026` becomes `results/results_153_2026`. Optional: if left out, the folder is named `results_<instrument-id>`. | `results_153_2026` |

**Finding column names:** open the CSV in a text editor or Excel and look at the header row. If
the file has a few label lines above the real header (as the Marine St files do), the script finds
the header row automatically. Use the names from that row. If you name a column that doesn't
exist, the script stops and lists the columns it did find.

### More examples

Erie, instrument 16 against the AJAX reference (reference clock is UTC):

```zsh
python pm25_3dpaws_pipeline_general.py \
    --instrument data/raw/3D-PAWS_Instrument-16_2024-12-06_2025-05-12.csv \
    --reference data/reformatted/UplandS_1min_S12.6.24_E5.12.25.csv \
    --ref-value-col "PM 2.5" --ref-tz UTC \
    --tz America/Denver --name Erie --instrument-id 16 --outdir results_16
```

A self-test on made-up data with known problems built in (no files needed):

```zsh
python pm25_3dpaws_pipeline_general.py --demo                # writes to results/results_demo
```

For a list of every option: `python pm25_3dpaws_pipeline_general.py --help`

---

## 4. What it produces

Each run gets its own folder inside `results/`, which sits next to the script, e.g.
`results/results_153_2026/`. This is the same wherever the script is run from. Re-running with
the same `--outdir` overwrites that folder's files. Every file name starts with
`<name>_<instrument-id>_` (e.g. `Boulder_153_`).

```
results/
├── results_153/          Boulder 153, Oct–Dec 2025
├── results_153_2026/     Boulder 153, May–Oct 2026
└── results_demo/         self-test on made-up data
```

### The calibration equation

The final equation appears in the terminal output and in `_summary.json` under
`calibration → equations`. For example:

```
y = 0.569x + 2.749
```

`x` is the cleaned 3D-PAWS reading and `y` is the corrected PM2.5 in µg/m³. If the script chose
the model with a bend, the equation has two parts, one each side of the bend point.

### Data files

| File | Contents |
|---|---|
| `_summary.json` | Everything about the run in one place: settings used, the time period, how many gaps and corrected points there were, the thresholds the script worked out, the chosen equation and why it was chosen, and accuracy scores. Opens in any text editor. |
| `_calibrated_timeseries.csv` | The data minute by minute. Columns: `time` (local standard time), `raw` (3D-PAWS as recorded), `reference`, `cleaned` (after steps 4–7), `calibrated` (equation applied), `calibrated_cv` (each time block predicted by an equation that never saw it), and `delta_raw` / `delta_calibrated` (3D-PAWS minus reference, before and after calibration). Blank cells mean the point was removed or missing. |
| `_cv_summary.csv` | One row per candidate equation, with its average held-out accuracy across the 5 time blocks. The `chosen` column marks the winner. |
| `_cv_folds.csv` | The same comparison broken out by time block, with each block's start and end date. Useful for seeing whether accuracy changes over the season. |
| `_anomaly_threshold_sweep.csv` | How many points the spike filter would flag at stricter and looser settings than the one used. If the count changes a lot between neighboring rows, the setting matters for this dataset and is worth a look. |

### Plots

| File | What it shows | What to look for |
|---|---|---|
| `_01_raw_vs_reference.png` | Raw 3D-PAWS and reference over the full period. | Long outages, obvious sensor faults, and whether the two broadly track each other. |
| `_02_neighborhood_anomaly.png` | The spike check: each point's jump from its neighbors, with the threshold band. | Flagged points should be clear one-off spikes, not the start of real events. |
| `_03_oscillation.png` | The oscillation check: jumpiness against concentration, with the threshold. | Removed points should sit clearly above the main cloud. |
| `_04_cleaned.png` | Raw vs. cleaned 3D-PAWS. | Real events (smoke, inversions) should survive cleaning. |
| `_05_relative_difference.png` | Percent difference from the reference over time. | Drift or periods where the sensor behaves differently. |
| `_06_regression.png` | Scatter of 3D-PAWS against reference, with the fitted equation. Grey points were given little weight by the robust fit. | Whether the line follows the bulk of the points. |
| `_07_calibrated.png` | Calibrated 3D-PAWS against reference over time. | The two lines should now sit on top of each other. |
| `_08_distributions.png` | Histograms of reference, raw and calibrated values. | The calibrated distribution should resemble the reference. |
| `_09_workflow_all_stages.png` | Every stage of the pipeline stacked in one figure. | A single overview suitable for a presentation. |
| `_10_cross_validation.png` | Held-out accuracy of each candidate equation in each time block. | Whether the winner wins consistently or just narrowly. |

---

## 5. How to read the results

The terminal prints a short report. The key lines from the May–Oct 2026 Boulder run:

```
alignment check: best lag +5 min (r=0.962; r at 0 lag=0.961)
anomaly_threshold: 6 x (0.39 + 0.028 x L)
pct_anomaly_replaced: 0.134
pct_oscillation_removed: 0.041
selection: hinge vs robust_linear: ... (paired over 5 blocks, 1 won) -> keep robust_linear
    y = 0.569x + 2.749
calibrated_cv  R2(pearson)=0.920  R2(1-SS)=0.920  RMSE=1.56  ...
```

- **Alignment check.** The best shift should be within a few minutes of 0. A shift near ±60
  minutes means a time zone setting is wrong (usually `--ref-tz`), and the results should not be
  used.
- **Thresholds.** `6 x (0.39 + 0.028 x L)` means a reading is flagged as a spike if it is more
  than 6 × (0.39 + 0.028 × the local concentration) away from its neighbors: about 2.5 µg/m³ in
  clean air, more during smoke.
- **Percent replaced / removed.** These are normally well under 1%. Above 5%, the script prints a
  `[warn]` line: either the sensor had a genuinely bad period, or the filter is too aggressive for
  this data. Check plots 02–04.
- **Selection.** Which equation was chosen and why. `robust_linear` is a straight line. `hinge` is
  a line with one bend, used when the sensor's response changes at higher concentrations.
- **Accuracy scores.** There are four rows: `raw`, `cleaned`, `calibrated` and `calibrated_cv`.
  **`calibrated_cv` is the honest one:** it is scored only on data the equation never saw, so it
  is the best estimate of accuracy on new data. `calibrated` is scored on the data used to fit it
  and will always look slightly better.
  - **RMSE / MAE:** typical size of the error, in µg/m³. Smaller is better.
  - **R2(pearson):** how well the two series move together (0 to 1). A straight-line calibration
    cannot change this; it only fixes scale and offset.
  - **R2(1-SS):** how close the values actually are. It can be negative before calibration when
    the sensor reads consistently high or low.
  - **bias:** average over- or under-reading. Near 0 after calibration.

---

## 6. Limitations and open questions

**What the results apply to.** The equation applies to *this* sensor, at *this* site, over *this*
period and range of concentrations. It is not automatically valid for other units, other seasons
or much higher smoke levels. For example, instrument 153 gave `0.51x + 1.95` (straight line) in
Oct–Dec 2025 and `0.569x + 2.749` in May–Oct 2026. The difference may be seasonal or sensor drift.

**A co-located reference monitor is required.** The script derives a calibration from a period
when the sensor sat beside a reference. It does not calibrate sensors that have no reference.

**Known weak spots:**
- Sustained false readings lasting 3–15 minutes (seen at Erie instrument 16, 150–340 µg/m³) are
  too long to count as spikes and too smooth to count as oscillation, so they survive cleaning.
- If a sensor switches between two distinct behaviors during the period (as at Kaylena's Virginia
  site), neither candidate equation fits well. Neither did the paper's two-piece fit when scored
  on held-out data.
- The held-out scores are very slightly optimistic: the cleaning thresholds are learned from the
  whole record before the time blocks are split. Reference data is never used for cleaning, so the
  effect should be small.

**Open questions for review:**
- **Paper Eq. 6.** Neither the original script nor this one reproduces Eq. 6 (`0.82x + 0.92` for
  Boulder) from the Oct–Dec 2025 files in `data/raw/`. Both get about `0.51x + 1.95`. The files
  or reference column Kaylena used may differ from these.
- **Erie/AJAX offset.** Calibrating against the AJAX reference needs an offset of about
  7 µg/m³, which is unusually large. Worth confirming the `PM 2.5` column in that file is the
  quantity we expect.

---

## 7. Optional settings

The defaults are designed to work without adjustment. These options exist for special cases and
for comparison with the paper.

| Option | Default | Purpose |
|---|---|---|
| `--anomaly-threshold N` | worked out from data | Use a fixed spike threshold instead (Kaylena used 5 for Boulder). |
| `--rolling-std-threshold N` | worked out from data | Use Kaylena's fixed rolling-standard-deviation filter instead (she used 10 for Boulder). |
| `--calibration` | `auto` | Force one equation: `robust_linear`, `hinge`, `linear` (ordinary least squares), `piecewise` (paper's two-piece split at `--piecewise-split`, default 70), or `low_bias_linear` (paper's Boulder method, fitted only above `--low-bias-cutoff`, default 2). |
| `--anomaly-k`, `--oscillation-k` | 6 | Strictness of the automatic filters. Lower values flag more points. |
| `--oscillation-window` | `30min` | Window length for the oscillation check. |
| `--smooth-window` | `30` | Rolling-average length. A plain number means samples; `30min` means time. |
| `--min-coverage` | 0.5 | A smoothed value is only kept if at least this fraction of its window has data. |
| `--drop-zeros` | off | Discard exact-zero readings before anything else. |
| `--cv-folds` | 5 | Number of time blocks for cross-validation. |
| `--apply-to` | `cleaned` | Apply the equation to the cleaned data (default) or the raw data. |
| `--eval-start`, `--eval-end` | whole period | Score accuracy only within a date range (e.g. a smoke event). |
| `--outlier-ceiling` | 350 | Readings above this are discarded. |

To reproduce Kaylena's Boulder choices for comparison, add:

```zsh
--anomaly-threshold 5 --rolling-std-threshold 10 --calibration low_bias_linear
```
