# Phase 2 — Preprocessing Pipeline

This phase transforms raw HDF5 data into clean, model-ready inputs. It is the most consequential phase of the entire project — garbage in, garbage out, and every model you train later inherits every mistake made here.

---

## 2.1 — Step 1: Coordinate Scaling and Geometry Reconstruction

Before building any shot gathers, all geometry must be correctly scaled. This is the foundation everything else rests on.

**Applying scale factors:**

The `COORD_SCALE` field follows SEG-Y convention, which has a specific rule: if the value is negative, it is a divisor (divide coordinates by its absolute value); if positive, it is a multiplier. For example, a `COORD_SCALE` of -100 means divide all coordinate values by 100. Apply this to `SOURCE_X`, `SOURCE_Y`, `REC_X`, `REC_Y`. Apply `HT_SCALE` identically to `SOURCE_HT` and `REC_HT`.

Do this before any distance or offset calculation. Using raw unscaled coordinates will produce nonsensical offset values and destroy your shot gather ordering.

**Computing offset:**

For each trace, compute the source-receiver offset as the Euclidean distance in the horizontal plane:

```
offset = sqrt((REC_X - SOURCE_X)^2 + (REC_Y - SOURCE_Y)^2)
```

Store this as a new derived field alongside the trace metadata. You will use it extensively for sorting within shot gathers, for sanity-checking first break labels (first break time should generally increase with offset), and as a potential input feature for some model families.

**Verifying geometry makes physical sense:**

Plot a sample of shot locations (SOURCE_X vs SOURCE_Y) and receiver locations (REC_X vs REC_Y) for each asset. You should see a coherent acquisition grid pattern. If coordinates look random or clustered in a nonsensical way, your scale factor application is wrong. Fix it before proceeding. This one plot can save you days of debugging downstream.

---

## 2.2 — Step 2: Building Shot Gathers (The 2D Image Construction)

This is the core data restructuring step described in the task instructions.

**Grouping traces into shots:**

Group all traces by their shot identifier. Use `SHOTID` as the primary grouping key. If you find during EDA that `SHOTID` is not unique (two different physical shots share an ID), fall back to grouping by unique `(SOURCE_X, SOURCE_Y)` pairs after coordinate scaling. Verify your grouping by checking that all traces within a group share the same SOURCE_X and SOURCE_Y values — if they do not, your grouping key is wrong.

**Sorting traces within each shot:**

Within each shot gather, sort traces by offset in ascending order. This produces the standard shot gather layout where near-offset traces (short source-receiver distance) appear on the left and far-offset traces appear on the right. The first break curve will then form the characteristic V or hyperbolic shape seen in Figure 3. Without this sorting, the first break curve will be jagged and incoherent, and any 2D model will fail to learn it.

**Handling irregular shot sizes:**

From your EDA you will know the distribution of traces-per-shot. You have several options for shots with non-standard sizes:

- **Minimum viable size filter:** Drop shots with fewer than N traces (e.g., fewer than 10 traces). These edge shots contribute little signal and a lot of noise.
- **Maximum size cap:** For shots with unusually many traces, decide whether to keep all of them or truncate to a consistent width. Inconsistent widths make batching harder.
- **Padding:** For shots slightly below your target width, pad with zero-filled traces on the edges and mark those positions as invalid in a mask array.

The cleanest approach is to define a target shot gather width (e.g., the median traces-per-shot rounded to a round number) and pad/crop all gathers to this width. Record this decision in your `preprocessing.yaml` config.

**Data structure for a shot gather:**

Each processed shot gather should consist of:
- `traces`: 2D numpy array of shape `[n_samples, n_traces]` — note: time on the vertical axis, traces on horizontal, consistent with Figure 1
- `labels`: 1D numpy array of shape `[n_traces]` — first break time in ms for each trace (NaN for unlabeled traces)
- `offsets`: 1D numpy array of shape `[n_traces]` — offset in meters for each trace
- `label_mask`: 1D boolean array — True where label is valid, False where trace is unlabeled or label is suspicious
- `metadata`: dict containing shot_id, source coordinates, asset name, sample_rate

Save each shot gather as a separate `.npz` file to your `processed/{asset}/` Drive folder. This avoids reloading the full HDF5 on every Colab session.

---

## 2.3 — Step 3: Label Conversion and Validation

The `SPARE1` field stores first break times in milliseconds, but you need to verify and potentially convert before using them as targets.

**Converting ms labels to sample indices:**

You will need labels in two forms depending on the model:
- **Milliseconds** — for regression models that output a time value directly
- **Sample index** — for segmentation models that output a per-sample classification

The conversion is: `sample_index = round(label_ms / (SAMP_RATE_us / 1000))`

Where `SAMP_RATE_us` is the sampling rate in microseconds. Always store both forms.

**Label validation per trace:**

For every labeled trace, check:
- Label > 0 (after filtering out 0 and -1 sentinel values)
- Label < total trace duration in ms (`SAMP_NUM × SAMP_RATE_us / 1000`)
- Label is not suspiciously early (e.g., < 1ms could indicate a corrupt label)
- Label is not suspiciously late (e.g., > 95th percentile of all labels in that asset — investigate these individually)

Any trace failing these checks should have its label_mask set to False.

**Shot-level label coherence check:**

After converting to sample indices, for each shot gather compute the first difference of the sorted (by offset) label array — i.e., how much the first break time changes from one trace to the next. In a clean shot gather, this should be small and positive (first break time increases with offset). Large jumps between adjacent traces (e.g., > 50ms between neighboring traces) are strong indicators of mispicks. Flag these within the shot but do not automatically discard them — some are real (e.g., a fault crossing the spread). You want to track them, not silently corrupt your labels.

---

## 2.4 — Step 4: Signal Normalization

Raw seismic trace amplitudes vary enormously — between traces within a shot, between shots, and between assets. You must normalize before any model sees the data.

**Per-trace normalization (recommended baseline):**

Normalize each trace independently by dividing by its maximum absolute amplitude. This produces values in `[-1, 1]` per trace. It is the standard in seismic processing and is robust to amplitude variations.

```
normalized_trace = trace / (max(|trace|) + epsilon)
```

Use a small epsilon (1e-10) to prevent division by zero on dead traces.

**Per-gather normalization (alternative):**

Normalize the entire 2D gather by its maximum absolute amplitude. This preserves relative amplitude information between traces within a shot, which can be valuable for models that see the full gather. However it makes different gathers incomparable in amplitude.

**Global normalization (not recommended for this task):**

Computing a global mean/std across all traces of all assets and normalizing by those statistics is theoretically clean but practically problematic — amplitude scales differ so much between assets that global stats are dominated by outliers.

**Recommendation:** Use per-trace normalization as your default. Add a config flag `normalization: per_trace | per_gather` so you can experiment with both without code changes.

**Dead trace handling:**

A dead trace after normalization will be all zeros (or near-zero after the epsilon). Set its `label_mask` to False and fill with actual zeros — do not let the normalization produce NaN or inf values.

---

## 2.5 — Step 5: Handling the Unlabeled Traces

A significant fraction of traces across the four assets will be unlabeled. You have three strategies and you must choose deliberately:

**Strategy A — Exclude entirely from supervised training:**
Simplest approach. Only traces with valid labels enter the training set. The model is purely supervised. This is your baseline strategy.

**Strategy B — Semi-supervised inclusion:**
Use unlabeled traces as additional input context within shot gathers. Even if a trace has no label, it still contributes to the 2D image that the model sees. The loss is simply masked to zero for unlabeled positions. This is natural in the shot-gather framing and should be your default for 2D models — you construct the full shot gather image (including unlabeled traces) but compute loss only on labeled positions.

**Strategy C — Pseudo-labeling (advanced, Phase 4):**
After training an initial model, run inference on unlabeled traces, use high-confidence predictions as pseudo-labels, and retrain. Do not attempt this until you have a working baseline.

**Recommendation:** Use Strategy B for all 2D model training. For trace-level models, use Strategy A. Implement Strategy C as an optional late-stage improvement.

---

## 2.6 — Step 6: Data Augmentation Design

Augmentation must be physically meaningful for seismic data. Standard image augmentations (random crop, horizontal flip, color jitter) are mostly wrong here.

**Valid augmentations for seismic shot gathers:**

- **Amplitude scaling:** Multiply the entire gather by a random scalar (e.g., uniformly sampled from [0.5, 2.0]). This simulates different source energies. Labels are unchanged because they are time values.
- **Additive Gaussian noise:** Add low-amplitude Gaussian noise to simulate higher noise floors. Scale the noise to be a fraction (e.g., 5–15%) of the signal RMS. Labels unchanged.
- **Trace dropout:** Randomly zero out a small fraction (e.g., 5%) of traces within a gather during training. This simulates dead channels and improves robustness. Set the corresponding label_mask entries to False for those positions.
- **Time shift of the whole gather:** Shift all traces in a gather by the same random time offset (e.g., ±10 samples) and shift all labels by the same amount. Only valid if the shifted labels remain within the trace duration. This helps with temporal generalization.
- **Polarity reversal:** Multiply the entire gather by -1. First break physics is polarity-independent (you are detecting arrival time, not polarity). Labels unchanged.

**Invalid augmentations:**

- **Horizontal flip** — this would reverse the offset ordering, producing physically impossible shot gathers where near offset is on the right. Never do this.
- **Vertical flip** — reverses time. Never do this.
- **Random crop of arbitrary size** — destroys the offset structure. If you crop, crop consistently along the trace axis only and adjust labels accordingly.
- **Rotation** — completely inapplicable.

**Implementation:** Write augmentations as composable transform classes in `src/data/transforms.py`, each with a toggle in `preprocessing.yaml`. This lets you ablate augmentation strategies cleanly.

---

## 2.7 — Step 7: Train / Validation / Test Split

This is one of the most critical decisions and the one most commonly botched. There are two levels to get right: the split strategy and the stratification.

**The fundamental rule — no data leakage:**

Traces from the same shot gather must all go into the same split. You cannot put some traces from Shot 47 into training and others into validation. The model would effectively see the answer during training. Split at the **shot gather level**, not the trace level.

**Cross-asset split strategy:**

You have two philosophical options:

Option A — Combined split: Pool all labeled shot gathers from all four assets, then split into train/val/test. The model trains and evaluates on a mixture of all assets. This tests generalization across assets.

Option B — Per-asset split then combine training sets: Split each asset independently into 70/15/15, then combine the four training sets into one, the four val sets into one, and keep four separate test sets. This lets you report per-asset test performance, which is more informative. **This is the recommended approach.**

**Stratification:**

You cannot do naive random splitting because the first break time distribution may be very uneven. To stratify at the shot gather level:

- For each shot gather, compute the **median first break time** across its labeled traces. This is your stratification variable.
- Bin this variable into quantiles (e.g., 5 quantiles).
- Perform stratified split within each asset so that each split contains a proportional representation of early, middle, and late first break shots.

This ensures your validation and test sets are representative of the full difficulty range, not accidentally composed of only easy (early first break) shots.

**Recommended split ratios:**
- Train: 70%
- Validation: 15% — used for hyperparameter tuning and early stopping
- Test: 15% — touched only once per model, at final evaluation

Record the exact shot IDs assigned to each split in a CSV file saved to Drive. This is your reproducibility anchor — if you ever re-run preprocessing, you can verify the same shots went to the same splits.

---

## 2.8 — Step 8: Handling Cross-Asset Differences

From your EDA you may have found that assets differ in sample rate, trace length, or first break time range. Here is how to handle each:

**Different sample rates:**

If two assets have different `SAMP_RATE` values, you cannot combine them directly — a sample at index 50 means 50ms in one asset and 25ms in another. You must resample all assets to a common sample rate. Choose the finest (lowest) sample rate as your target to avoid losing resolution. Resample using `scipy.signal.resample`. After resampling, recompute label sample indices.

**Different trace lengths (SAMP_NUM):**

If assets differ in the number of samples per trace, you have two options. Padding: zero-pad shorter traces to match the longest. Cropping: truncate all traces to the shortest common length. For first break picking, the break almost always occurs in the first portion of the trace — you rarely need the full 1500ms. Inspect your label distributions: if 99% of first breaks occur before 800ms, you can safely crop everything to 800ms, dramatically reducing your data size and model input size.

**Different first break time distributions:**

This is a harder problem. If assets have very different FB time ranges, a combined model must learn to handle all of them. This is actually a strength of the combined approach if you frame it correctly — a robust model should generalize. However, during training you must ensure batches contain a mix of assets. Implement a **balanced sampler** in your DataLoader that samples roughly equally from each asset per batch.

---

## 2.9 — Step 9: Saving the Processed Dataset

After all the above, save your processed dataset in a format efficient for Colab loading:

**Per-shot NPZ files:** As described in 2.2. One `.npz` file per shot gather, named by `{asset}_{shot_id}.npz`. These can be loaded individually during training without loading the full dataset into RAM.

**Master index CSV:** A single CSV file with one row per shot gather containing: asset, shot_id, n_traces, n_labeled, median_fb_ms, split_assignment, file_path. This is your dataset index and gets loaded fully into RAM at training time (it is tiny). The DataLoader uses it to look up which file to load for each sample.

**Normalization statistics file:** A JSON or YAML file recording the normalization parameters computed from the training set only (if you compute any global statistics). This must be saved and reused at inference time — you cannot recompute it from data you have not seen.

---

## 2.10 — Preprocessing Checklist Summary

Before leaving Phase 2, verify every item:

- All four assets decompressed and structurally audited
- Coordinate scaling applied and verified via geometry plot
- Shot gathers reconstructed, sorted by offset, saved as NPZ
- Labels converted to both ms and sample index forms
- Label validation applied, suspicious labels flagged in mask
- Shot-level coherence check done, mispick rate recorded per asset
- Dead and clipped traces identified and masked
- Sample rate and trace length harmonized across assets
- Normalization scheme chosen and implemented
- Augmentation classes written and toggled via config
- Stratified split performed at shot-gather level
- Split index CSV saved to Drive
- Master index CSV saved to Drive

---
