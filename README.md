# Automated Seismic First Break Picking
## A Deep Learning Pipeline for Hard-Rock Seismic Exploration

Saved sanity plot: /content/drive/MyDrive/seismic-first-break-picking/results/sanity_plots/00_first_traces.png
This plot shows how representative traces from the four datasets look before any model sees them. Repo-relative path: `results/sanity_plots/00_first_traces.png`.

---

## Table of Contents

1. [Project Summary](#1-project-summary)
2. [Current Status and Scope](#2-current-status-and-scope)
3. [The Scientific Problem](#3-the-scientific-problem)
4. [The HardPicks Datasets](#4-the-hardpicks-datasets)
5. [How 3D Surveys Become 2D Images and 1D Traces](#5-how-3d-surveys-become-2d-images-and-1d-traces)
6. [Data Fields Used by the Pipeline](#6-data-fields-used-by-the-pipeline)
7. [EDA Notebook-by-Notebook Findings](#7-eda-notebook-by-notebook-findings)
8. [What the EDA Changed in the Final Pipeline](#8-what-the-eda-changed-in-the-final-pipeline)
9. [Preprocessing and Transformation Pipeline](#9-preprocessing-and-transformation-pipeline)
10. [Model Families and Architecture Details](#10-model-families-and-architecture-details)
11. [Training Strategy and Infrastructure](#11-training-strategy-and-infrastructure)
12. [Evaluation Protocol and Metrics](#12-evaluation-protocol-and-metrics)
13. [Results on the Current Repository Split](#13-results-on-the-current-repository-split)
14. [Result Analysis and Why Some Models Won](#14-result-analysis-and-why-some-models-won)
15. [Latency Analysis](#15-latency-analysis)
16. [Connection to Related Work](#16-connection-to-related-work)
17. [Plots to Include in a Report or Presentation](#17-plots-to-include-in-a-report-or-presentation)
18. [Limitations and Next Steps](#18-limitations-and-next-steps)
19. [Repository Structure](#19-repository-structure)
20. [References](#20-references)

---

## 1. Project Summary

This repository implements an end-to-end machine learning pipeline for automated seismic first break picking on the four public HardPicks hard-rock mining surveys: Brunswick, Halfmile, Lalor, and Sudbury. The project starts from HDF5 seismic archives, verifies and audits them, transforms each survey into harmonized shot-gather tensors and trace-level samples, trains multiple model families, and benchmarks the trained models with common physical metrics and inference-latency measurements.

The central learning problem is: given seismic waveforms and expert-labeled first break times, predict the first arrival time for unseen traces. In this project, the strongest models are gather-level 2D models with a Soft-Argmax regression head, especially a pretrained ResNet-UNet variant that achieved the best internal test performance.

The pipeline is modular. Data verification and preprocessing live in `src/data/`, handcrafted features in `src/features/`, neural architectures in `src/models/`, and training infrastructure in `src/training/`. The notebooks orchestrate the workflow from environment setup to EDA, preprocessing, training, and model comparison.

---

## 2. Current Status and Scope

### What is already implemented and evidenced in the repository

- Raw data verification and sanity plotting are complete.
- Per-asset and cross-asset EDA are complete.
- The preprocessing pipeline has already produced processed gathers, split indices, and a preprocessing report.
- Multiple neural models have already been trained and benchmarked on the repository's internal split.
- Leaderboard CSVs and training curves already exist under `artifacts/plots/`.

### Important evaluation note

These datasets come from the HardPicks benchmark, but the **current repository results are not the official leave-one-survey-out HardPicks benchmark**.

What this repository currently evaluates:
- A deterministic **per-asset stratified 70/15/15 train/val/test split** stored in `data/processed/split_index.csv`.
- A **combined multi-asset training regime** with balanced asset sampling.
- Final metrics reported on the internal `test` split built from all four assets.

What this repository does **not** yet report in the current README:
- The official HardPicks leave-one-survey-out cross-survey benchmark used in the St-Charles et al. evaluation protocol.

That distinction matters scientifically. The current results are useful for comparing architectures **inside this repository**, but they should not be presented as direct cross-survey benchmark replacements.

---

## 3. The Scientific Problem

A seismic trace records ground motion as a function of time after an active source event. The **first break** is the earliest time at which seismic energy clearly arrives at a receiver. Before the first break, the trace is dominated by ambient or acquisition noise. After the first break, the signal becomes physically meaningful for downstream processing.

Accurate first-break picks are used for:
- static corrections,
- near-surface velocity estimation,
- refraction analysis,
- muting and other processing decisions,
- better structural imaging of deep ore deposits.

Manual picking does not scale to modern 3D mining surveys that contain millions of traces. It is also subjective on noisy data. This project therefore treats first-break picking as a supervised learning problem with human expert labels stored in the datasets.

The target is a continuous physical quantity measured in milliseconds, so the problem is fundamentally **regression**, even when some visual representations make it look superficially similar to segmentation.

---

## 4. The HardPicks Datasets

The project uses four real hard-rock seismic surveys from the public HardPicks collection.

### Raw dataset overview from EDA

| Dataset | Total Traces | Labeled Traces | Label % | Sample Rate | Samples | Time Window |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| Brunswick | 4,496,540 | 3,733,221 | 83.02% | 2 ms | 751 | 1500 ms |
| Halfmile | 1,099,559 | 993,189 | 90.33% | 2 ms | 751 | 1500 ms |
| Lalor | 2,424,923 | 1,211,857 | 49.98% | 1 ms | 1501 | 1500 ms |
| Sudbury | 1,810,220 | 200,338 | 11.07% | 2 ms | 1001 | 2000 ms |

### Processed dataset overview after preprocessing, filtering, and masking

These numbers come from `results/preprocessing_report.json` and therefore reflect the data the models actually consume.

| Dataset | Processed Shots | Processed Traces | Valid Labels After Masking | Label Fraction |
| :--- | ---: | ---: | ---: | ---: |
| Brunswick | 1,541 | 4,496,540 | 3,707,228 | 82.45% |
| Halfmile | 690 | 1,099,559 | 975,143 | 88.68% |
| Lalor | 905 | 2,424,921 | 1,203,041 | 49.61% |
| Sudbury | 1,016 | 1,810,220 | 198,805 | 10.98% |
| Total | 4,152 | 9,831,240 | 6,084,217 | 61.89% |

### Asset-specific context

**Brunswick**
- Largest survey in the collection.
- Latest arrivals among the four assets, with first breaks reaching 1358 ms.
- Largest gathers in the project, which makes it the main driver of memory pressure.

**Halfmile**
- Highest label coverage.
- Gather widths are unusually consistent, which makes it the most regular asset geometrically.
- The EDA found the highest dead-trace fraction in the sampled amplitude audit, so dead-channel handling matters here.

**Lalor**
- Only survey sampled at 1 ms instead of 2 ms.
- Highest SNR by the refined shot-level SNR definition.
- Strong amplitude range differences and the highest mispick/outlier fraction among the four assets.

**Sudbury**
- Most label-sparse asset by far.
- Longest raw recording window at 2000 ms.
- Coordinate scaling is awkward enough that offset handling had to prefer precomputed offset fields when available.

### Why these datasets are difficult together

The four surveys are heterogeneous in:
- sample rate,
- trace length,
- label density,
- amplitude scale,
- gather width,
- coordinate scale,
- likely receiver/source characteristics,
- apparent label quality.

That heterogeneity is exactly why EDA had to drive the engineering decisions.

---

## 5. How 3D Surveys Become 2D Images and 1D Traces

These are 3D seismic surveys, but the models in this repository do not ingest full 3D cubes directly.

### What "3D" means here

A 3D survey means the acquisition spans a two-dimensional surface geometry and aims to image a three-dimensional subsurface volume. For first-break picking, however, the data is naturally organized by source event and receiver layout.

### The 2D gather representation used by the main models

For each shot:
1. Group all traces with the same `SHOTID`.
2. Compute or load source-receiver offsets.
3. Sort traces by offset.
4. Stack them side-by-side.
5. Transpose into a matrix of shape `[time, traces]`.

This produces a 2D image where:
- the vertical axis is time,
- the horizontal axis is ordered receiver/offset position,
- each column is one trace,
- the first-break moveout appears as a coherent curve across columns.

This is why U-Net style image models make sense: they can use lateral context that is invisible to single-trace models.

### The 1D trace representation used by the baseline families

The 1D models do not use the full gather image. Instead, the `trace_collate_fn` in `src/data/dataset.py` extracts only valid labeled traces from processed gathers and forms a batch of shape `[P, 1, 751]`.

That framing is much simpler and cheaper, but it throws away the neighboring-trace context that often disambiguates weak onsets.

---

## 6. Data Fields Used by the Pipeline

The most important fields in the HDF5 files are:

- `data_array`: the raw waveform samples.
- `SHOTID`: the grouping key used to build shot gathers.
- `SHOT_PEG`: alternate shot identifier; not always identical to `SHOTID` across assets.
- `SOURCE_X`, `SOURCE_Y`, `SOURCE_HT`: source coordinates and elevation.
- `REC_X`, `REC_Y`, `REC_HT`: receiver coordinates and elevation.
- `REC_PEG`: receiver identifier.
- `SAMP_RATE`: sample interval in microseconds.
- `SAMP_NUM`: number of samples per trace.
- `COORD_SCALE`, `HT_SCALE`: scaling rules inherited from SEG-Y conventions.
- `SPARE1`: the only field actually used as ground-truth first-break label.

### What the supplementary EDA disproved

The supplementary notebook established that several seemingly useful fields are effectively dead placeholders in this dataset release:
- `FIRST_BREAK_TIME`,
- `MODELLED_BREAK_TIME`,
- `FIRST_BREAK_AMPLIT`,
- `FIRST_BREAK_VELOCITY`.

Across the dataset release inspected here, these were all zero and therefore unusable as labels or auxiliary targets. This is why the final pipeline treats `SPARE1` as the sole source of ground truth.

---

## 7. EDA Notebook-by-Notebook Findings

The repository contains five main EDA notebooks:

- `notebooks/01_eda_brunswick.ipynb`
- `notebooks/01_eda_halfmile.ipynb`
- `notebooks/01_eda_lalor.ipynb`
- `notebooks/01_eda_sudbury.ipynb`
- `notebooks/01_eda_combined.ipynb`

A supplementary notebook, `notebooks/01_eda_supplementary.ipynb`, refined several early assumptions and is also important to the final pipeline.

### 7.1 Brunswick notebook

Key findings from `results/eda/brunswick_eda_report.json`:
- 4.50M traces, 83.02% labeled.
- Median first-break time 384 ms and maximum 1358 ms.
- Massive gathers: median width 2975, maximum 3355.
- No sampled dead-trace problem in the amplitude audit.
- Low outlier fraction in the polynomial coherence check: about 0.39%.

Why it mattered:
- Brunswick is the main reason the pipeline needs memory-aware batching and width divisibility.
- Its very late arrivals are why the project kept the full 1500 ms common window instead of trying an aggressive crop.
- Because Brunswick dominates raw trace count, training needed balancing to avoid a Brunswick-biased model.

### 7.2 Halfmile notebook

Key findings from `results/eda/halfmile_eda_report.json`:
- 1.10M traces, 90.33% labeled.
- Median first-break time 344 ms.
- Gather sizes are highly regular: mostly 1575 to 1604 traces.
- Sampled dead-trace fraction estimate around 1.36%, the highest among the four assets.
- Outlier fraction about 0.47%.

Why it mattered:
- Halfmile helped validate the dead-trace masking path in preprocessing.
- Its regular geometry is a useful contrast to Brunswick and Sudbury.
- Its high label coverage makes it one of the cleaner supervision sources in combined training.

### 7.3 Lalor notebook

Key findings from `results/eda/lalor_eda_report.json`:
- 2.42M traces, about 50% labeled.
- Only asset sampled at 1 ms and 1501 samples.
- Median first-break time 248 ms and maximum 881 ms.
- Highest amplitude range in the sample audit.
- Highest mispick/outlier fraction in the quality audit: about 1.38%.
- Very high SNR after the refined shot-level SNR analysis.

Why it mattered:
- Lalor forced the pipeline to implement explicit temporal harmonization by resampling from 1 ms to 2 ms.
- The strong amplitude heterogeneity reinforced the need for robust normalization.
- The higher mispick fraction supported masking suspect labels rather than trusting every annotation equally.

### 7.4 Sudbury notebook

Key findings from `results/eda/sudbury_eda_report.json`:
- 1.81M traces, only 11.07% labeled.
- Raw recording window is 2000 ms with 1001 samples.
- Median first-break time 162 ms and maximum 599 ms in the raw label audit.
- 301 shot gathers contain zero labels.
- Coordinate scaling leads to implausibly large absolute positions if used naively.
- Outlier fraction around 1.12%.

Why it mattered:
- Sudbury is why the pipeline keeps unlabeled traces in 2D inputs but excludes them from loss and metric computation.
- Its longer recording window motivated common-window cropping to 1500 ms.
- The coordinate issue is why the pipeline prefers precomputed offset fields over naive coordinate-derived offsets where possible.

### 7.5 Combined notebook

Key findings from `results/eda/eda_summary.json`:
- The four assets are structurally incompatible without harmonization.
- Label density is highly imbalanced across assets.
- Gather widths vary enough to make fixed-width padding wasteful.
- Cross-asset amplitude and coordinate heterogeneity are real, not cosmetic.
- The benchmark nature of the dataset suggests cross-survey evaluation as an eventual target.

Why it mattered:
- It justified a common temporal target of 751 samples at 2 ms.
- It justified balanced multi-asset sampling.
- It justified building both 1D and 2D pipelines rather than only one framing.
- It made clear that evaluation claims must be separated into internal-split and future cross-survey benchmark claims.

### 7.6 Supplementary notebook

The supplementary notebook added several important corrections:
- it confirmed that `SPARE1` is the only usable label field,
- it validated `resample_poly` as the preferred Lalor resampling method,
- it redefined the shot-level SNR analysis more meaningfully,
- it estimated gather-memory requirements and supported the width cap of 4096,
- it selected 12 benchmark visualization examples by difficulty.

This notebook is the main reason the final implementation differs from some of the earliest planning assumptions.

---

## 8. What the EDA Changed in the Final Pipeline

The final engineering decisions are direct consequences of the notebook findings.

| EDA Finding | Final Pipeline Decision |
| :--- | :--- |
| Lalor is 1 ms, others are mostly 2 ms | Resample Lalor to 2 ms using `scipy.signal.resample_poly` |
| Sudbury is 1001 samples / 2000 ms | Crop Sudbury to the first 751 samples / 1500 ms |
| Asset label density is highly uneven | Use balanced asset sampling during training |
| Sudbury contains many unlabeled shots/traces | Keep unlabeled traces in 2D context, mask them from loss |
| Brunswick gathers are huge | Variable-width batching with width divisibility of 16 |
| Precomputed label-like fields are dead | Use `SPARE1` only |
| Coordinate scaling is inconsistent | Prefer precomputed offset fields and fallback to scaled Euclidean distance |
| Some labels are suspicious | Use a coherence-based label mask instead of deleting whole gathers |
| Amplitude scales vary widely | Use per-trace max-absolute normalization |

### Two especially important clarifications

**No label imputation was used.**
Unlabeled traces were not assigned pseudo-labels in the current benchmark runs. They are either excluded entirely in the 1D framing or preserved as context but masked from loss in the 2D framing.

**No feature imputation was used.**
There is no missing-value imputation step for waveform amplitudes or labels. Dead traces are detected, zero-safe normalized, and masked out when needed.

---

## 9. Preprocessing and Transformation Pipeline

The implemented preprocessing lives primarily in `src/data/shot_gather_builder.py`, `src/data/transforms.py`, and `src/data/dataset.py`.

### 9.1 Verification before transformation

Before any training data is produced, the project verifies:
- archive integrity,
- HDF5 structure,
- presence of required headers,
- coordinate scale behavior,
- waveform-label sanity through diagnostic plotting.

This guards against a common failure mode in scientific ML projects: building a sophisticated model on misaligned labels or misunderstood metadata.

### 9.2 Temporal harmonization

The target representation is:
- sample rate: 2 ms,
- samples per trace: 751,
- total time window: 1500 ms.

How each asset reaches that target:
- Brunswick: already 751 samples at 2 ms, so pass-through.
- Halfmile: already 751 samples at 2 ms, so pass-through.
- Lalor: downsample from 1501 samples at 1 ms to 751 samples at 2 ms with `resample_poly`.
- Sudbury: crop from 1001 samples at 2 ms to the first 751 samples.

### 9.3 Shot-gather construction

Each processed gather is built by:
1. grouping traces by `SHOTID`,
2. ordering traces by offset,
3. harmonizing trace length and sample rate,
4. transposing to `[751, K]`,
5. normalizing amplitudes,
6. validating labels,
7. saving one `.npz` per gather.

### 9.4 Normalization and standardization

The current default is **per-trace max-absolute normalization**:

`trace_norm = trace / (max(abs(trace)) + epsilon)`

Why this was chosen:
- it is robust to the extreme amplitude spread seen in EDA,
- it avoids global-scale dependence across assets,
- it is common in related seismic deep-learning pipelines,
- it keeps every trace on a common amplitude range even when acquisition scales differ.

What the project does **not** do:
- no global z-score standardization across the entire dataset,
- no learned normalization statistics file,
- no per-dataset scalar amplitude calibration,
- no label standardization, since targets stay in physical milliseconds.

### 9.5 Label validation and masking

Labels are created from `SPARE1 > 0` and then filtered by:
- minimum valid label threshold,
- trace-duration validity,
- per-asset extreme-value threshold based on training labels only,
- dead-trace masking,
- shot-level coherence checks.

This is important: the project does not silently trust all labels equally. Instead, it keeps physically useful context while masking traces that are unlabeled or likely suspicious.

### 9.6 Split strategy

The repository currently uses a deterministic **per-asset stratified 70/15/15 split**. Stratification is based on median first-break time per shot gather.

Why that matters:
- it preserves early, medium, and late arrival regimes in all splits,
- it avoids trace-level leakage by splitting at the shot level,
- it supports per-asset representation in train, validation, and test.

Why it still needs to be described carefully:
- it is a useful internal evaluation protocol,
- but it is not the same as the official leave-one-survey-out HardPicks benchmark.

### 9.7 Data imbalance handling

The project faces **asset imbalance** and **label imbalance**.

How the current pipeline addresses them:
- `BalancedAssetSampler` oversamples smaller assets at the gather level.
- 2D models keep unlabeled traces as context but apply losses only where `label_mask` is true.
- 1D models discard invalid/unlabeled traces during batch construction.

This is the main current answer to imbalance. There is no synthetic oversampling of traces, no pseudo-labeling in the benchmark runs, and no class reweighting because the task is regression rather than classification.

### 9.8 Augmentation

The current augmentation module supports physically plausible perturbations:
- amplitude scaling,
- Gaussian noise,
- trace dropout,
- small time shifts,
- polarity reversal.

Explicitly avoided because they would destroy physical meaning:
- horizontal flips,
- vertical flips,
- arbitrary rotations,
- random spatial crops that break offset order.

---

## 10. Model Families and Architecture Details

The codebase implements several tiers of models, from classical signal-processing baselines to deep gather-level architectures.

### 10.1 Classical baselines

Implemented in `src/models/classical.py`:
- STA/LTA picker,
- Modified Energy Ratio picker,
- AIC picker.

These methods do not learn from data. They operate directly on 1D traces and provide a physically interpretable baseline.

### 10.2 Tabular baseline

Implemented in `src/models/tabular.py` with features in `src/features/features.py`.

Feature set includes:
- offset,
- maximum absolute amplitude,
- location of that maximum,
- zero-crossing count,
- RMS energy across five temporal windows,
- max STA/LTA ratio,
- location of max STA/LTA ratio.

This baseline tests whether first-break prediction can be handled by handcrafted summary statistics plus a tree ensemble.

### 10.3 1D neural models

Implemented in `src/models/cnn_1d.py` and `src/models/unet_1d.py`.

**CNN-1D**
- stacked temporal convolutions,
- batch normalization and ReLU,
- pooling for temporal compression,
- adaptive average pooling,
- small MLP regression head.

**ResNet-1D**
- residual blocks with temporal convolutions,
- deeper receptive field than the plain CNN,
- global average pooling,
- linear regression head.

**SoftArgmax 1D U-Net**
- encoder-decoder on a single trace,
- skip connections,
- per-sample logits over time,
- Soft-Argmax expectation to convert the temporal distribution into a millisecond prediction.

Important limitation of the current 1D neural family:
- the deep 1D models in this repository operate on waveform shape only and do **not** explicitly ingest offset as an additional input,
- so they are missing the strongest geometric prior available to the gather-level models and the tabular model.

### 10.4 2D gather-level models

Implemented in `src/models/unet.py`.

**SoftArgmax U-Net**
- input shape `[B, 1, T, W]`, where `T = 751` and `W` is the padded gather width,
- four downsampling stages,
- symmetric pooling in both time and trace dimensions,
- decoder with skip connections,
- final per-pixel logits collapsed with a Soft-Argmax along time for each trace column.

Why the design matters:
- it preserves neighborhood information across traces,
- it predicts one first-break time per trace column,
- it avoids forcing a brittle hard boundary in heavily amplified noise regions.

**ResNet-UNet with ImageNet pretraining**
- wraps a pretrained `segmentation_models_pytorch` U-Net with ResNet-34 encoder,
- repeats the single-channel gather to three channels to match ImageNet pretraining assumptions,
- applies the same Soft-Argmax temporal expectation idea at the output.

Why this model is powerful here:
- the encoder starts from a much stronger feature basis than a randomly initialized custom U-Net,
- skip connections and gather-level context remain intact,
- transfer learning improves optimization efficiency even though seismic images are not natural images.

### 10.5 Why the Soft-Argmax head matters

The Soft-Argmax head is one of the most important design choices in the project.

It allows the network to learn a **distribution over arrival time** for each trace rather than a raw scalar with no temporal structure. That makes the output more aligned with the physical problem: the model is learning where onset energy is most likely to occur.

It also avoids a brittle segmentation framing under per-trace normalization, where pure noise can be visually amplified enough to make a hard above/below boundary unreliable.

---

## 11. Training Strategy and Infrastructure

### 11.1 Combined multi-asset training

The main benchmarked neural models are trained in **combined mode**, meaning all four assets contribute to the same training regime through a balanced sampler.

Why combined training was used here:
- it is practical under a shared preprocessing format,
- it reduces the chance that Brunswick dominates learning simply because it is largest,
- it encourages the models to see multiple acquisition regimes during training.

The codebase also contains a `ProgressiveAssetSampler`, but the current benchmarked configurations are the combined ones.

### 11.2 Losses and masking

The main losses are masked regression losses:
- `MaskedMAELoss`
- `MaskedHuberLoss`

This is essential because:
- not every trace has a valid label,
- 2D batches contain real but unlabeled Sudbury traces,
- padded columns exist in variable-width batches,
- the project must avoid NaN pollution from invalid targets.

### 11.3 Runtime controls

The trainer supports:
- automatic mixed precision on CUDA,
- gradient accumulation,
- gradient clipping,
- cosine-style scheduler settings through config,
- rerun-safe checkpoints,
- MLflow logging.

These are not cosmetic features. They are what make wide-gather training feasible on Colab-class hardware.

### 11.4 Trained vs untrained model status in the current repository

Currently benchmarked and present in the leaderboard:
- CNN-1D
- ResNet-1D
- UNet-1D
- UNet-2D
- ResNet-UNet

Implemented but not benchmarked in the current leaderboard CSVs:
- classical pickers,
- LightGBM baseline,
- TCN,
- BiLSTM,
- Transformer-based variants.

---

## 12. Evaluation Protocol and Metrics

### 12.1 Current protocol used by the repository

All metrics below are computed on the repository's internal validation and test splits, not on leave-one-survey-out benchmark folds.

### 12.2 Metrics used

**MAE (ms)**
- primary metric,
- easiest physical interpretation,
- directly answers: on average, how many milliseconds away is the predicted pick?

**RMSE (ms)**
- more sensitive to catastrophic misses,
- useful when two models have similar MAE but different tail behavior.

**Within-5 ms accuracy**
- strict tolerance metric,
- approximately 2.5 samples at 2 ms sampling.

**Within-10 ms accuracy**
- more forgiving tolerance metric,
- useful for comparing practical acceptability.

**Latency (ms/trace)**
- needed because a deployable picker must be both accurate and computationally usable.

### 12.3 Why these metrics were chosen

The metric suite balances:
- physical interpretability,
- average-case accuracy,
- sensitivity to bad failures,
- usefulness in downstream deployment.

The project intentionally stays in milliseconds rather than normalizing the target, because the downstream geophysical meaning is tied to real time, not abstract units.

---

## 13. Results on the Current Repository Split

### Important caution before reading the table

These are **internal split** results from `notebooks/04_benchmark_and_compare.ipynb`, saved in:
- `artifacts/plots/leaderboard/val_leaderboard.csv`
- `artifacts/plots/leaderboard/test_leaderboard.csv`

They are excellent for comparing models in this repository, but they are **not** a direct substitute for the official cross-survey HardPicks benchmark.

### Validation leaderboard

| Rank | Model | Val MAE (ms) | P90 Error (ms) | <5 ms (%) | <10 ms (%) | Val CPU Latency (ms/trace) |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| 1 | ResNet-UNet | 27.416 | 67.680 | 20.79 | 38.32 | 0.0774 |
| 2 | UNet-2D | 89.503 | 227.194 | 8.51 | 16.20 | 0.1187 |
| 3 | UNet-1D | 151.323 | 323.797 | 2.35 | 4.71 | 0.1917 |
| 4 | CNN-1D | 151.551 | 320.712 | 2.25 | 4.50 | 0.0542 |
| 5 | ResNet-1D | 151.853 | 326.232 | 2.30 | 4.60 | 0.0707 |

### Test leaderboard

| Rank | Model | Test MAE (ms) | P90 Error (ms) | <5 ms (%) | <10 ms (%) | Test CPU Latency (ms/trace) |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| 1 | ResNet-UNet | 26.333 | 67.743 | 22.52 | 40.84 | 0.0764 |
| 2 | UNet-2D | 76.731 | 198.447 | 10.55 | 19.91 | 0.1214 |
| 3 | UNet-1D | 146.866 | 312.733 | 2.43 | 4.88 | 0.2021 |
| 4 | CNN-1D | 148.593 | 313.027 | 2.32 | 4.61 | 0.0578 |
| 5 | ResNet-1D | 148.720 | 318.403 | 2.35 | 4.69 | 0.0694 |

### Key artifact paths for the top two models

- ResNet-UNet training curve: `artifacts/plots/ResNet_UNet/training_curve.png`
- UNet-2D training curve: `artifacts/plots/UNet_2D/training_curve.png`
- ResNet-UNet test scatter: `artifacts/plots/ResNet_UNet/test_scatter.png`
- UNet-2D test scatter: `artifacts/plots/UNet_2D/test_scatter.png`

---

## 14. Result Analysis and Why Some Models Won

### 14.1 Why ResNet-UNet won

The ResNet-UNet is the clear winner on both validation and test splits. The reasons are architectural and data-driven.

1. **It uses gather-level context.**
   First breaks are not just local waveform events. They form smooth moveout patterns across neighboring traces. Gather-level models can exploit this coherence; trace-level models cannot.

2. **It uses a pretrained encoder.**
   The training curve for the winner is steady rather than chaotic. Validation MAE drops quickly from the mid-60 ms range into the high-20 ms range and then stabilizes. That is exactly what you expect from a model with a much stronger feature initializer.

3. **It uses a Soft-Argmax output.**
   The model is not guessing a scalar blindly. It is learning a time-localized distribution for each trace column.

4. **It amortizes computation across the whole gather.**
   One forward pass predicts many traces at once. This matters for both accuracy and latency.

### 14.2 What the winner's training curve says

The ResNet-UNet training curve shows:
- rapid early improvement,
- continued train MAE reduction across epochs,
- validation MAE stabilizing in the high-20 ms range,
- a real but manageable train/validation gap.

Interpretation:
- the model is learning useful structure quickly,
- it is not collapsing into instability,
- some overfitting exists, but validation remains strong and stable enough to support the reported ranking.

### 14.3 Why UNet-2D came second but far behind

The custom UNet-2D still clearly beats the 1D family, which confirms that the gather framing is fundamentally right. However, it is much worse than the pretrained ResNet-UNet.

Likely reasons:
- it trains from scratch rather than from a pretrained encoder,
- its validation curve is visibly noisier and less stable,
- it appears less data-efficient during optimization,
- the model still benefits from 2D context, but it does not extract that context as effectively as the pretrained variant.

The UNet-2D training curve shows a persistent gap between train and validation MAE and much higher epoch-to-epoch variance than the ResNet-UNet. That is consistent with weaker optimization and poorer generalization.

### 14.4 Why all 1D models cluster around 147 to 152 ms MAE

This clustering is one of the clearest results in the repository.

The 1D models all fail in roughly the same way because they all suffer from the same information bottleneck:
- they look at one trace at a time,
- they do not see the neighboring moveout curve,
- they do not explicitly use offset as an auxiliary input,
- per-trace normalization suppresses some absolute-amplitude cues that might otherwise help.

As a result, the 1D models can learn generic onset statistics, but they cannot resolve many ambiguous picks accurately enough.

The CNN-1D and ResNet-1D curves are especially revealing:
- they improve quickly at first,
- then flatten almost immediately near ~151 ms validation MAE,
- which suggests they have reached the limit of what the single-trace framing can explain in this setup.

### 14.5 Why UNet-1D only marginally improves over the simpler 1D models

UNet-1D adds a better temporal localization bias than CNN-1D or ResNet-1D. That likely explains why it edges them out slightly.

But it still operates on one trace at a time, so it still lacks the cross-trace physical context that made the 2D models dramatically better.

This is an important project result: improving temporal architecture alone does not compensate for the loss of gather geometry.

### 14.6 What failed, scientifically speaking

The failed idea was not "deep learning". The failed idea was **treating first-break picking as mostly a single-trace problem under this preprocessing regime**.

The repository evidence strongly supports the opposite conclusion:
- spatial context is essential,
- gather-level structure matters more than deeper single-trace temporal modeling,
- pretraining further amplifies that advantage.

---

## 15. Latency Analysis

### Test latency ranking

| Rank | Model | Test CPU Latency (ms/trace) |
| :--- | :--- | ---: |
| 1 | CNN-1D | 0.0578 |
| 2 | ResNet-1D | 0.0694 |
| 3 | ResNet-UNet | 0.0764 |
| 4 | UNet-2D | 0.1214 |
| 5 | UNet-1D | 0.2021 |

### Why latency behaves this way

**CNN-1D is fastest**
- small model,
- simple temporal convolutions,
- no decoder,
- very cheap per-trace inference.

**ResNet-1D is slightly slower**
- residual blocks add depth and compute,
- but it remains a compact trace-level model.

**ResNet-UNet is surprisingly efficient for a gather model**
- it is more expensive per forward pass,
- but it predicts many traces at once,
- so the cost is amortized over the whole gather,
- and the underlying implementation is likely benefiting from optimized backbone code.

**UNet-2D is slower than ResNet-UNet**
- custom full encoder-decoder over wide gathers,
- one-gather-at-a-time inference in the benchmark notebook for memory safety,
- less efficient overall accuracy-per-millisecond tradeoff than the pretrained variant.

**UNet-1D is slowest per trace**
- unlike the simple 1D CNNs, it carries encoder-decoder cost for each individual trace,
- unlike the 2D models, it cannot amortize that cost across the entire gather.

### The main practical insight

The best model is not only the most accurate. It is also near the top of the latency ranking. That is a very strong result for practical deployment: the project did not have to trade away speed to get better picks.

---

## 16. Connection to Related Work

The repository's qualitative findings line up well with the published literature, even though the current evaluation protocol differs from the official benchmark protocol.

### St-Charles et al.

The benchmark paper established that gather-like image representations and convolutional models are a strong baseline for this family of datasets. This repository agrees with that general direction: gather-based models clearly beat trace-level models.

### DSU-Net and other strong 2D methods

Papers that improve 2D gather architectures tend to outperform simpler baselines by exploiting the geometry of first-break curves more effectively. The success of the ResNet-UNet here is consistent with that broader pattern.

### Mardan et al.

The success of the pretrained ResNet-UNet is especially consistent with transfer-learning-based results in the literature. In this repository, pretraining appears to be the single largest reason the top 2D model separates so strongly from the custom 2D model.

### DGL-FB and graph-informed methods

Graph-based methods in the literature try to encode wider spatial relationships than a simple local gather image. This repository does not yet benchmark that family, but the strong performance gap between 2D and 1D models suggests that exploiting geometry more explicitly is indeed valuable.

### Important qualification

Because the current repository table uses an internal split rather than leave-one-survey-out folds, the connection to related work should be stated as **qualitative agreement in architectural trends**, not as a direct benchmark ranking against those papers.

---

## 17. Plots to Include in a Report or Presentation

### Strong existing plots already generated in the repository

1. `results/sanity_plots/00_first_traces.png`
   - Best opening figure for showing how different the four datasets look at the raw trace level.

2. `results/preprocessing_validation.png`
   - Useful for demonstrating that the harmonized and saved processed outputs are sane.

3. `results/eda_plots/combined/combined_fb_stats.png`
   - Good summary of how first-break statistics differ across assets.

4. `artifacts/plots/leaderboard/val_vs_test_leaderboard.png`
   - Best single plot for communicating model ranking.

5. `artifacts/plots/leaderboard/deployment_pareto.png`
   - Best plot for accuracy-versus-speed discussion.

6. `artifacts/plots/ResNet_UNet/training_curve.png`
   - Should be shown when discussing the winning model's convergence.

7. `artifacts/plots/UNet_2D/training_curve.png`
   - Useful side-by-side with the winner to show what weaker optimization looks like.

8. `artifacts/plots/ResNet_UNet/test_scatter.png`
   - Good for showing calibration and spread of the winning model.

9. `artifacts/plots/UNet_2D/test_error_hist.png`
   - Good for showing long-tail error behavior in a weaker but still gather-aware model.

### Additional plot ideas worth adding later

- Per-asset error tables for the current internal split.
- Error versus offset for the top two models.
- Error versus label density, especially for Sudbury.
- Error versus SNR category using the 12 benchmark visualization examples.
- Confident versus uncertain trace visualizations from the Soft-Argmax probability profiles.

---

## 18. Limitations and Next Steps

### Current limitations

- The current leaderboard is not the official HardPicks leave-one-survey-out benchmark.
- Classical and LightGBM baselines are implemented but not included in the current leaderboard CSVs.
- The current reported results are aggregate, not broken down per asset.
- The README can now summarize the repository-backed evidence, but a future paper-style report should still include explicit cross-survey testing before claiming benchmark competitiveness.

### High-priority next steps

1. Run the official leave-one-survey-out benchmark protocol.
2. Add per-asset test breakdowns for the existing internal split.
3. Benchmark the classical and LightGBM baselines fully.
4. Consider adding offset as an explicit auxiliary feature to the neural 1D models.
5. Explore graph-based or more geometry-aware models if cross-survey generalization becomes the primary goal.
6. Add calibration and uncertainty analysis for the Soft-Argmax outputs.

---

## 19. Repository Structure

```text
configs/
  datasets.yaml
  preprocessing.yaml
  model_*.yaml

data/
  raw/
  extracted/
  processed/

notebooks/
  00_environment_setup.ipynb
  01_eda_*.ipynb
  02_preprocessing_pipeline.ipynb
  03_train_*.ipynb
  04_benchmark_and_compare.ipynb

results/
  sanity_plots/
  eda/
  eda_plots/
  benchmark/

artifacts/plots/
  leaderboard/
  ResNet_UNet/
  UNet_2D/
  UNet_1D/
  CNN_1D/
  ResNet_1D/

src/
  data/
  evaluation/
  features/
  models/
  training/
  utils/
```

### Most important implementation files

- `src/data/shot_gather_builder.py`: preprocessing and split generation.
- `src/data/dataset.py`: dataset wrappers, collate functions, balanced sampling.
- `src/data/transforms.py`: training-time augmentations.
- `src/models/unet.py`: 2D SoftArgmax U-Net and ResNet-UNet.
- `src/models/unet_1d.py`: 1D SoftArgmax U-Net.
- `src/models/cnn_1d.py`: CNN-1D, ResNet-1D, TCN.
- `src/models/classical.py`: classical first-break pickers.
- `src/features/features.py`: handcrafted features for LightGBM.
- `src/training/trainer.py`: masked training loop, AMP, clipping, checkpointing.

---

## 20. References

St-Charles, P.-L., Rousseau, B., Ghosn, J., Bellefleur, G., & Schetselaar, E. (2024). A deep learning benchmark for first break detection from hardrock seismic reflection data. *Geophysics*, 89(1), WA279-WA294.

St-Charles, P.-L., Rousseau, B., Ghosn, J., Bellefleur, G., & Schetselaar, E. (2021). A multi-survey dataset and benchmark for first break picking in hard rock seismic exploration. *NeurIPS 2021 Workshop on Machine Learning for the Physical Sciences (ML4PS)*.

Mardan, A., Blouin, M., & Giroux, B. (2024). A fine-tuning workflow for automatic first-break picking with deep learning. *Near Surface Geophysics*, 22(5), 539-552.

Zwartjes, P., & Yoo, J. (2022). First break picking with deep learning - evaluation of network architectures. *Geophysical Prospecting*, 70(2), 318-342.

Wang, H., Feng, R., Wu, L., Liu, M., Cui, Y., Zhang, C., & Guo, Z. (2024). DSU-Net: Dynamic Snake U-Net for 2-D Seismic First Break Picking. *IEEE Transactions on Geoscience and Remote Sensing*, 62.

Wang, H., Long, L., Zhang, J., Wei, X., Zhang, C., & Guo, Z. (2024). DGL-FB: Seismic First Break Picking in a Higher Dimension Using Deep Graph Learning. arXiv:2404.08408.

Bellefleur, G., Schetselaar, E., White, D., Miah, K., & Dueck, P. (2015). 3D seismic imaging of the Lalor volcanogenic sulphide deposit, Manitoba, Canada. *Geophysical Prospecting*, 63, 813-832.

Malehmir, A., & Bellefleur, G. (2009). 3D seismic reflection imaging of volcanic-hosted massive sulfide deposits: Insights from reprocessing Halfmile Lake data, New Brunswick, Canada. *Geophysics*, 74(6), B209-B219.

Adam, E., Perron, G., Milkereit, B., Wu, J., Calvert, A. J., Salisbury, M., Verpaelst, P., & Dion, D. J. (2000). A review of high-resolution seismic profiling across the Sudbury, Selbaie, Noranda, and Matagami mining camps. *Canadian Journal of Earth Sciences*, 37(2-3), 503-516.

Akaike, H. (1974). A new look at the statistical model identification. *IEEE Transactions on Automatic Control*, 19(6), 716-723.

Allen, R. V. (1978). Automatic earthquake recognition and timing from single traces. *Bulletin of the Seismological Society of America*, 68, 1521-1532.

Allen, R. V. (1982). Automatic phase pickers: Their present use and future prospects. *Bulletin of the Seismological Society of America*, 72(6B), S225-S242.

Sabbione, J. I., & Velis, D. (2010). Automatic first-breaks picking: New strategies and algorithms. *Geophysics*, 75(4), V67-V76.
