# Seismic First Break Picking — Master Plan

I'll deliver this in phases. This first response covers **Phase 0 (Environment & Storage)** and **Phase 1 (EDA)**. Reply and I'll continue with Phases 2–5.

---

## PHASE 0 — Environment, Storage & Repository Architecture

Before touching any data, you need your infrastructure solid. Colab sessions die, drives get messy, and 4 large compressed HDF5 files need careful handling.

---

### 0.1 — Google Drive Folder Structure

Design this before downloading anything. A clean hierarchy prevents chaos later. Proposed layout:

```
MyDrive/
└── seismic_fbp/
    ├── raw/                  ← compressed .xz files live here, never touched again after extraction
    │   ├── brunswick.hdf5.xz
    │   ├── halfmile.hdf5.xz
    │   ├── lalor.hdf5.xz
    │   └── sudbury.hdf5.xz
    ├── extracted/            ← decompressed .hdf5 files
    │   ├── brunswick.hdf5
    │   ├── halfmile.hdf5
    │   ├── lalor.hdf5
    │   └── sudbury.hdf5
    ├── processed/            ← shot gathers as numpy arrays or HDF5 shards, post-preprocessing
    │   ├── brunswick/
    │   ├── halfmile/
    │   ├── lalor/
    │   └── sudbury/
    ├── datasets/             ← final train/val/test splits (combined or per-asset)
    │   ├── train/
    │   ├── val/
    │   └── test/
    ├── eda/                  ← EDA outputs: plots, stats CSVs, notebooks
    ├── models/               ← saved checkpoints per model
    │   ├── unet/
    │   ├── resnet/
    │   └── ...
    ├── results/              ← benchmark tables, prediction CSVs, metric logs
    └── repo/                 ← your code repository (cloned from GitHub)
```

**Critical note:** The `.xz` files are large. Extracting them on Colab is fine but keep both the compressed and extracted versions on Drive. You do NOT want to re-download if something breaks.

---

### 0.2 — GitHub Repository Structure

A single notebook is indeed catastrophic for a project this size. Structure your repo modularly from day one:

```
seismic_fbp/
├── configs/
│   ├── datasets.yaml          ← paths, sampling rates, scale factors per asset
│   ├── preprocessing.yaml     ← normalization, windowing, augmentation toggles
│   ├── model_unet.yaml
│   ├── model_resnet.yaml
│   ├── model_1dcnn.yaml
│   └── model_classical.yaml
├── src/
│   ├── data/
│   │   ├── hdf5_reader.py     ← reads raw HDF5, applies scale factors
│   │   ├── shot_gather_builder.py  ← groups traces → 2D gathers
│   │   ├── dataset.py         ← PyTorch Dataset class
│   │   └── transforms.py      ← normalization, augmentation
│   ├── models/
│   │   ├── unet.py
│   │   ├── resnet_regressor.py
│   │   ├── cnn_1d.py
│   │   └── classical.py
│   ├── training/
│   │   ├── trainer.py
│   │   └── losses.py
│   ├── evaluation/
│   │   ├── metrics.py         ← MAE, RMSE, within-N-ms accuracy
│   │   └── visualizer.py      ← plot predictions vs ground truth
│   └── utils/
│       ├── config_loader.py
│       └── logger.py
├── notebooks/
│   ├── 00_download_and_extract.ipynb
│   ├── 01_eda_brunswick.ipynb
│   ├── 01_eda_halfmile.ipynb
│   ├── 01_eda_lalor.ipynb
│   ├── 01_eda_sudbury.ipynb
│   ├── 01_eda_combined.ipynb  ← cross-dataset analysis
│   ├── 02_preprocessing_pipeline.ipynb
│   ├── 03_train_classical.ipynb
│   ├── 03_train_1dcnn.ipynb
│   ├── 03_train_unet.ipynb
│   ├── 03_train_resnet.ipynb
│   └── 04_benchmark_and_compare.ipynb
├── requirements.txt
└── README.md
```

This gives you isolated notebooks per model, shared helper code via `src/`, and YAML configs so you can change hyperparameters without touching code.

---

### 0.3 — Colab-Specific Infrastructure Notes

- Always mount Drive at the top of every notebook. Keep a standard header cell you copy across notebooks.
- Colab free tier has ~12GB RAM and ~15GB GPU VRAM (T4). Plan your batch sizes and array loading accordingly — you will NOT be able to load all 4 full HDF5 files into RAM simultaneously.
- Use Colab's persistent storage via Drive for all outputs. Never save only to `/content/` — it disappears when the session ends.
- For long training runs, use Colab's "Keep awake" extensions or write checkpoint saves every N epochs so you can resume.
- Install your `src/` package at the top of each notebook with `pip install -e .` or by appending to `sys.path`.

---

### 0.4 — Clarification on Task Nature (Critical — Read Before EDA)

This is one of the most important things to nail down before you write a single line of analysis. The task is **NOT binary classification**.

Each seismic trace is a 1D time-series signal. The first break is a single **time value in milliseconds** — it is a **regression target**. The label stored in `SPARE1` is a continuous value (e.g., 124.5 ms).

However, the way you frame the **model input** changes everything:

- **Trace-level framing:** Input = one 1D trace (shape: `[n_samples]`), Output = one float (first break time in ms). This is pure regression on 1D signals.
- **Shot-gather framing:** Input = one 2D image (shape: `[n_traces, n_samples]`), Output = a vector of first break times, one per trace column. This can be treated as a 1D regression output per column, or reframed as a **semantic segmentation problem** where you predict a binary mask (above/below the first break curve).

Both framings are valid. The 2D/segmentation framing is more powerful because it lets the model see neighboring traces and exploit the **spatial coherence** of the first break curve across a shot gather. Keep both in mind during EDA.

---

## PHASE 1 — Exploratory Data Analysis

EDA must be done **four times independently** (once per asset) and then once **jointly** across all four. Do not skip the joint analysis — it is critical for deciding your combination strategy.

---

### 1.1 — Step 1: Download and Decompress

Before any EDA, all four files must be on your Drive. The files are `.hdf5.xz`, meaning they are XZ-compressed HDF5 files.

- Download each URL directly to Drive from within Colab using `wget` or `gdown`-style commands. Do not download to your local machine and re-upload — that is painfully slow.
- Decompress each `.xz` file using the `lzma` Python module or the `xz` command-line utility available in Colab's Linux environment.
- Verify file integrity after decompression by checking that the HDF5 can be opened and the `/TRACE_DATA/DEFAULT` group is accessible.
- Record the file sizes of both compressed and decompressed versions.

---

### 1.2 — Step 2: Per-Dataset Structural Audit

For each of the four assets, before looking at a single waveform, audit the raw HDF5 structure:

**Keys and shapes:**
- List all keys in `/TRACE_DATA/DEFAULT`
- Record the exact shape of `data_array` — this tells you total trace count and sample count
- Check for any undocumented keys. Assets from different acquisition campaigns sometimes contain extra metadata not listed in the task description. Catalog everything.

**Scalar/constant fields verification:**
- `SAMP_RATE`: Is it truly constant across all traces in this asset? Extract it, check min/max, flag if not.
- `COORD_SCALE`: Same check. This is critical — coordinates need to be divided by this value (or multiplied, depending on sign convention in SEG-Y/HDF5 practice).
- `HT_SCALE`: Same check.
- `SAMP_NUM`: Verify it matches the second dimension of `data_array`.

**Why this matters:** If `SAMP_RATE` varies across traces within one asset, your time-axis calculations break. If `COORD_SCALE` is inconsistent, your shot gather reconstruction will be wrong.

---

### 1.3 — Step 3: Label Audit Per Asset

For each asset, analyze the `SPARE1` field:

- Total trace count
- Count of labeled traces (SPARE1 ≠ 0 and SPARE1 ≠ -1)
- Count of unlabeled traces (SPARE1 = 0 or SPARE1 = -1) — record both separately, as 0 and -1 may have different semantic meanings
- Percentage labeled
- Distribution of label values: histogram of first break times in ms, min, max, mean, median, std, percentiles (5th, 25th, 75th, 95th)
- Check for suspicious values: are there labels beyond the trace duration? (label_ms > SAMP_RATE_in_ms × SAMP_NUM would be physically impossible)
- Check for duplicate label values that repeat suspiciously (could indicate placeholder values)

**This directly answers your question about label distribution.** You need this per-asset AND combined before you can design a stratified split.

---

### 1.4 — Step 4: Coordinate Analysis and Shot Gather Reconstruction Logic

This is where you figure out how to build your 2D images. The task says to use receiver coordinates. Here is the full logic to work out during EDA:

**Coordinate scaling:**
- Apply `COORD_SCALE` to `REC_X`, `REC_Y`, `SOURCE_X`, `SOURCE_Y`. The scale factor in SEG-Y convention is often a power of 10 and may be negative (meaning divide). Verify this produces sensible geographic coordinates.
- Apply `HT_SCALE` to `SOURCE_HT` and `REC_HT`.

**Shot gather identification:**
- A "shot gather" is all traces that share the same shot. You have `SHOTID` and `SHOT_PEG` — check if they are redundant or carry different information. Use whichever uniquely identifies shots.
- Group traces by shot ID. For each shot, count how many traces it has. Plot the distribution of traces-per-shot. Are all shots the same size? Irregular shot sizes complicate batching.
- Within each shot gather, sort traces by **offset** (distance from source to receiver: `sqrt((REC_X - SOURCE_X)^2 + (REC_Y - SOURCE_Y)^2)`). This is the standard ordering for shot gathers and is what produces the recognizable V-shape first break curve seen in Figure 3.

**What defines a "2D image":**
The task says to use receiver coordinates to split. This means each unique shot location = one 2D image. The traces within it are ordered by offset, forming the x-axis. Time is the y-axis.

**Per-asset shot statistics to compute:**
- Total number of unique shots
- Distribution of shot sizes (traces per shot)
- Min/max offset range per shot
- Geographic spread of shots (plot SOURCE_X vs SOURCE_Y to see acquisition geometry)
- Are there shots with 0 labeled traces? With partial labeling? With 100% labeling?

---

### 1.5 — Step 5: Waveform-Level EDA

Now look at the actual signals:

**Signal properties:**
- Compute the time axis: `time_ms = sample_index × (SAMP_RATE / 1000)` (SAMP_RATE is in microseconds, divide by 1000 for ms)
- Plot 5–10 individual traces from each asset. Visually inspect the signal character — is the first break a sharp onset or gradual?
- Compute per-trace SNR proxies: variance before and after the first break time. High variance after / low variance before = clean first break.
- Check for dead traces (near-zero amplitude across all samples). These should be flagged and excluded.
- Check for clipped traces (amplitude rail hitting a hard maximum). Flag these.
- Check for traces where the first break label falls at sample 0 or sample 1 — this is suspicious and likely a bad label.

**Shot-gather-level visual inspection:**
- Reconstruct 3–5 shot gathers per asset and visualize them as 2D images (imshow with wiggle or variable density display).
- Overlay the first break picks as a red line (as in Figure 3).
- Visually assess: Does the first break curve look physically coherent (smooth, consistent with offset)? Are there obvious mispicks in the ground truth labels?

---

### 1.6 — Step 6: Cross-Dataset Comparability Analysis

This is the step most people forget. Before you decide to combine all four datasets, you need to verify they are actually compatible.

**Compare across assets:**

| Property | Brunswick | Halfmile | Lalor | Sudbury |
|---|---|---|---|---|
| SAMP_RATE (µs) | ? | ? | ? | ? |
| SAMP_NUM | ? | ? | ? | ? |
| Total duration (ms) | ? | ? | ? | ? |
| FB time range (ms) | ? | ? | ? | ? |
| Avg traces/shot | ? | ? | ? | ? |
| % labeled | ? | ? | ? | ? |

**Critical questions to answer:**
- Do all four assets have the same sample rate? If not, you need resampling before combining.
- Do all four have the same `SAMP_NUM`? If not, your 2D images will have different heights — you need padding/cropping strategies.
- Is the first break time range similar across assets? If Brunswick has FBs clustered at 50–200ms and Sudbury has them at 400–900ms, a model trained on one may generalize poorly to the other. You need to know this before deciding on a combined model vs. per-asset models.
- Are the coordinate systems in the same units and projection? (Less critical for the ML task but matters for sanity checking.)

**Research the datasets online:**
The four datasets (Brunswick 3D, Halfmile 3D, Lalor 3D, Sudbury 3D) are real Canadian mining seismic surveys. They are part of a publicly known benchmark dataset sometimes called the FBP benchmark. Search for any published papers or GitHub repos that used these exact datasets. This gives you: expected sample rates, known challenges per asset, and published baseline MAE scores to compare against. You can find prior art that will save you enormous time.

---

### 1.7 — Step 7: Labeling Quality Assessment

Ground truth labels are manually picked, and human pickers make mistakes. You need to quantify label quality before trusting them for training:

- For each shot gather, fit a smooth curve (e.g., a polynomial or lowess smoother) to the labeled first break times as a function of offset.
- Compute residuals between each label and the smooth fit. Labels with residuals > 2–3× std are likely mispicks.
- Flag these as "suspicious labels" — do not delete them yet, but track them. During training you may want to exclude them or down-weight them.
- Record what percentage of labels per asset appear to be mispicks. This affects your expectation of model accuracy — if ground truth itself is noisy, your model cannot exceed that noise floor.

---

### 1.8 — Step 8: EDA Summary Document

After completing EDA for all four assets and the cross-dataset analysis, write a summary (markdown cell or separate doc) containing:

- A table of key statistics per asset
- Known challenges per asset (noise level, labeling density, signal character)
- Decision on whether to combine all four or keep separate for training
- Decision on whether to use trace-level or shot-gather-level framing
- Identified data quality issues (dead traces, suspicious labels, coordinate anomalies)
- Recommended preprocessing steps flowing from the EDA findings

This document becomes your reference for all subsequent phases.

---

