# Phase 4 — Training Pipeline Architecture

This phase covers everything between "model defined" and "benchmark results in hand." It is the engineering backbone of your entire project. A poorly structured training pipeline will cost you days of repeated work, lost checkpoints, and irreproducible results.

---

## 4.1 — Configuration System

Every training run must be fully described by its config file. If you cannot reproduce a run by loading its config, your pipeline is broken.

---

### 4.1.1 — Master Config Structure

Each model gets its own YAML file in `configs/`. Here is the full structure every config file must contain:

```yaml
# configs/model_unet.yaml

experiment:
  name: "unet_seg_brunswick_v1"
  seed: 42
  asset: "all"              # "all" | "brunswick" | "halfmile" | "lalor" | "sudbury"
  framing: "segmentation"   # "trace_regression" | "gather_regression" | "segmentation"

data:
  index_csv: "/content/drive/MyDrive/seismic_fbp/datasets/master_index.csv"
  processed_dir: "/content/drive/MyDrive/seismic_fbp/processed/"
  target_n_samples: 1500    # crop/pad all traces to this length
  target_n_traces: 120      # crop/pad all gathers to this width
  normalization: "per_trace"
  cache_in_ram: false       # set true only if dataset fits in Colab RAM

augmentation:
  enabled: true
  amplitude_scale: [0.5, 2.0]
  additive_noise_std: 0.05
  trace_dropout_prob: 0.05
  time_shift_max_samples: 10
  polarity_flip_prob: 0.5

split:
  train_ratio: 0.70
  val_ratio: 0.15
  test_ratio: 0.15
  stratify_by: "median_fb_ms"
  n_strata: 5

model:
  architecture: "unet"
  encoder_depth: 4
  base_channels: 64
  pretrained_backbone: null  # null | "resnet34" | "efficientnet-b2"
  dropout: 0.3

training:
  epochs: 100
  training_mode: "combined"   # "combined" (default) | "progressive"
                               # combined: all four assets trained simultaneously
                               # progressive: one asset per run, fallback only
  batch_size: 8
  optimizer: "adamw"
  lr: 1e-3
  weight_decay: 1e-4
  scheduler: "cosine_annealing"
  scheduler_params:
    T_max: 100
    eta_min: 1e-6
  early_stopping_patience: 15
  gradient_clip_norm: 1.0

loss:
  primary: "mae"            # "mae" | "mse" | "huber" | "wing"
  huber_delta: 5.0
  segmentation_bce_weight: 0.5
  segmentation_dice_weight: 0.5
  suspicious_label_weight: 0.1

evaluation:
  primary_metric: "mae_ms"
  within_threshold_ms: [5, 10]
  per_asset: true
  per_offset_bins: 10

output:
  checkpoint_dir: "/content/drive/MyDrive/seismic_fbp/models/unet/"
  results_dir: "/content/drive/MyDrive/seismic_fbp/results/"
  save_best_only: true
  log_every_n_steps: 50
```

**Why every field matters:**
The `seed` field ensures reproducibility. The `experiment.name` field becomes the filename prefix for every checkpoint and result file from this run. The `framing` field tells your DataLoader how to format outputs. Nothing in your training code should be hardcoded — it all flows from this file.

---

### 4.1.2 — Config Loader

In `src/utils/config_loader.py`, write a loader that reads YAML, validates required fields are present, and returns a nested namespace object. Also write a function that saves a copy of the active config alongside every checkpoint — so when you load a checkpoint six weeks later you know exactly what produced it.

---

## 4.2 — PyTorch Dataset Implementation

This is the most important single class in your codebase. Everything the model trains on flows through here.

---

### 4.2.1 — SeismicDataset Class Design

```
src/data/dataset.py
```

The dataset class receives the master index CSV and the config. On `__init__`, it loads the index, filters to the requested split and assets, and optionally pre-caches data into RAM if `cache_in_ram` is true. On `__getitem__`, it loads one shot gather NPZ file, applies transforms, and returns the correctly formatted (input, target, mask) tuple.

**What `__getitem__` must return for each framing:**

For trace-level regression (Framing A):
- `trace`: tensor of shape `[1, n_samples]`
- `label`: scalar float (first break time in ms)
- `offset`: scalar float (for models that use it as auxiliary input)
- `valid`: boolean (is this label trustworthy)

For gather-level regression (Framing B):
- `gather`: tensor of shape `[1, n_samples, n_traces]`
- `labels`: tensor of shape `[n_traces]` (first break times, NaN for unlabeled)
- `label_mask`: tensor of shape `[n_traces]` (True where label is valid)
- `offsets`: tensor of shape `[n_traces]`

For segmentation (Framing C):
- `gather`: tensor of shape `[1, n_samples, n_traces]`
- `mask`: tensor of shape `[1, n_samples, n_traces]` (binary, 0 above FB, 1 below)
- `label_mask`: tensor of shape `[n_traces]` (which columns have valid ground truth)
- `fb_samples`: tensor of shape `[n_traces]` (first break sample index, for evaluation only)

**How to construct the segmentation mask from a label vector:**
For each trace column with a valid label, set all samples below the first break sample index to 1, all samples at or above to 0. For unlabeled columns, fill with zeros and mark `label_mask` as False — the loss will be masked out for these columns regardless.

---

### 4.2.2 — Handling Variable Gather Sizes

Your gathers will not all be the same size after shot gather construction. The Dataset class must handle this before returning tensors, because PyTorch's default collate function requires all tensors in a batch to be the same shape.

In `__getitem__`, after loading the NPZ:

**Along the trace axis (n_traces dimension):**
- If `gather.shape[1] > target_n_traces`: crop to center (or to the lowest-offset traces, which have the most consistent first breaks)
- If `gather.shape[1] < target_n_traces`: pad with zeros on the right, mark padded positions False in `label_mask`

**Along the time axis (n_samples dimension):**
- If `gather.shape[0] > target_n_samples`: crop from the top (time zero) to `target_n_samples`. Since first breaks always occur early in the trace, you almost never need the full 1500ms.
- If `gather.shape[0] < target_n_samples`: pad with zeros at the bottom.

After this step, every item returned by `__getitem__` has identical shape, and the default PyTorch collate function works correctly.

---

### 4.2.3 — Balanced Multi-Asset Sampler

If you are training on all four assets combined, a naive random sampler will oversample the largest asset. You need a custom sampler.

In `src/data/dataset.py`, implement a `BalancedAssetSampler` class that:
- At the start of each epoch, computes how many samples each asset contributes
- Oversamples smaller assets (with replacement) and undersamples larger ones so each asset contributes equally to each epoch
- This is implemented as a PyTorch `Sampler` subclass that returns a shuffled list of indices

This is critical for training a model that generalizes across all four assets rather than memorizing the largest one.

---

## 4.3 — Transform Pipeline Implementation

```
src/data/transforms.py
```

Every augmentation from Phase 2 must be a composable class following a consistent interface. Each transform class has an `__init__` that accepts its parameters from the config, and a `__call__` that accepts and returns a dict containing `gather`, `labels`, `label_mask`, and any other relevant fields.

**Key implementation rules:**

- Augmentations are only applied during training. The Dataset class passes a `is_training` flag to the transform pipeline.
- Each augmentation that modifies labels must modify them consistently (e.g., time shift must shift both the trace data AND the label values AND ensure shifted labels remain within bounds).
- Augmentations are applied in a fixed order: amplitude first, then noise, then dropout, then shift. Order matters.
- Use `numpy.random.Generator` seeded from the global seed for all stochastic operations — this ensures augmentation behavior is reproducible given the same seed.
- Never apply augmentations to validation or test data. Add an assertion in your training loop that verifies the dataset's `is_training` flag is False during evaluation.

---

## 4.4 — Model Implementations

```
src/models/
```

A few critical implementation notes for each model family that are not obvious.

---

### 4.4.1 — All Models: Common Interface

Every model class must implement the same interface:

```python
class BaseModel(nn.Module):
    def forward(self, x, offsets=None):
        # x shape depends on framing
        # returns predictions in ms (or logits for segmentation)
        pass
    
    def predict(self, x, offsets=None):
        # wraps forward, converts segmentation mask to ms values
        # always returns first break times in ms
        pass
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
```

The `predict` method is crucial — it gives you a unified interface for evaluation regardless of framing. Every model returns ms values from `predict`, so your evaluation code never needs to know the framing.

---

### 4.4.2 — U-Net: Critical Implementation Details

Several non-obvious implementation decisions significantly affect U-Net performance for this task:

**Aspect ratio awareness:**
Your gather images are NOT square. A 1500-sample × 120-trace gather has an aspect ratio of 12.5:1. Standard U-Net assumes roughly square inputs. If you downsample equally in both dimensions, you will run out of trace-axis resolution after 3–4 pooling steps. 

Solution: Use **asymmetric pooling kernels** — pool with stride 2 along the time axis but stride 1 (or no pooling) along the trace axis in early encoder layers. Only start pooling along the trace axis in deeper layers once the time axis has been sufficiently compressed.

**Batch size constraints:**
U-Net with a full gather input is memory-intensive. On Colab's T4 (15GB VRAM), you can typically fit batch_size=4–8 for a standard U-Net with 64 base channels. If you get OOM errors, reduce base_channels from 64 to 32 before reducing batch size — smaller batch sizes destabilize batch normalization.

**Bilinear vs. transposed convolution upsampling:**
Use bilinear interpolation followed by a Conv2D for upsampling (not transposed convolution). Transposed convolutions produce checkerboard artifacts in the output mask, which create jagged first break curves. Bilinear upsampling is smoother and more appropriate for this task.

---

### 4.4.3 — Pretrained Backbone Models: Loading and Freezing

When using ResNet or EfficientNet as a pretrained encoder:

**Step 1 — Initial frozen training (5 epochs):**
Freeze all pretrained encoder weights. Set `requires_grad = False` for all encoder parameters. Train only the decoder and final head. Use a relatively high learning rate (1e-3) for the new layers. This "warms up" the randomly initialized decoder without disrupting pretrained features.

**Step 2 — Gradual unfreezing (next 10 epochs):**
Unfreeze the last encoder block only. Use a discriminative learning rate: 1e-5 for the newly unfrozen encoder block, 1e-4 for the decoder. Train for 5–10 epochs.

**Step 3 — Full fine-tuning (remaining epochs):**
Unfreeze all layers. Use discriminative learning rates across depth: earliest encoder layers get the smallest LR (1e-6), latest encoder layers get 1e-5, decoder gets 1e-4. This prevents catastrophic forgetting.

**Implementation:** Use PyTorch parameter groups to assign different learning rates. Pass a list of dicts to the optimizer, each dict containing `params` and `lr`.

---

## 4.5 — Loss Function Implementation

```
src/training/losses.py
```

---

### 4.5.1 — Masked MAE Loss

For gather-level and segmentation models, the loss must be masked — you only compute error on traces with valid labels.

```
masked_mae = sum(|pred - target| * mask) / sum(mask)
```

Add a guard for the edge case where `sum(mask) == 0` (a batch where every trace is unlabeled) — return zero loss in this case and log a warning. This can happen with very small batch sizes.

---

### 4.5.2 — Combined BCE + Dice Loss for Segmentation

The Dice loss for a binary mask is:

```
dice = 1 - (2 * sum(pred * target) + smooth) / (sum(pred) + sum(target) + smooth)
```

Use `smooth = 1.0` to prevent division by zero on empty masks. Compute both BCE and Dice losses, then combine as a weighted sum with weights from your config.

Apply this loss only to columns where `label_mask` is True. For unlabeled columns, the mask target is all zeros by construction, which would incorrectly drive the model to predict "noise everywhere." Always mask the loss on unlabeled columns.

---

### 4.5.3 — Label Smoothing for Segmentation

Hard binary masks (exactly 0 and 1) can make segmentation training unstable, because the model is forced to predict exactly 0 or 1 at the first break boundary — which is a physically ambiguous region. Apply **label smoothing**: replace 1.0 values with 0.9 and 0.0 values with 0.1. Additionally, apply a **soft boundary**: the few samples immediately around the first break time use intermediate values (e.g., a Gaussian blob centered at the first break sample). This makes the loss landscape smoother and convergence faster.

---

## 4.6 — Training Loop Design

```
src/training/trainer.py
```

---

### 4.6.1 — Core Training Loop Structure

Your trainer class handles the epoch loop, validation, checkpointing, and early stopping. Here is the complete structure of what each training epoch must do:

**Forward pass:**
Load batch from DataLoader → move to GPU → forward through model → compute masked loss → backward → gradient clipping → optimizer step → scheduler step (if step-based scheduler)

**Gradient clipping:**
Always clip gradients by norm (not by value). Use `torch.nn.utils.clip_grad_norm_` with `max_norm` from config (typically 1.0). This prevents training instability from occasional large gradients caused by noisy labels. Without this, a single mispicked label can cause a catastrophic weight update.

**Mixed precision training:**
Use `torch.cuda.amp.autocast()` and `torch.cuda.amp.GradScaler()`. This halves your VRAM usage and approximately doubles training speed on Colab's T4 GPU with negligible accuracy impact. There is essentially no reason not to use this.

**Logging:**
Every `log_every_n_steps` steps, log: current loss, learning rate, gradient norm, GPU memory usage. Write these to a CSV file on Drive (not just to console — console output disappears when the Colab session ends).

---

### 4.6.2 — Validation Loop

Run validation at the end of every epoch. The validation loop must:

- Set model to `eval()` mode — this disables dropout and switches BatchNorm to use running statistics
- Use `torch.no_grad()` context — saves memory and speeds up validation
- Call `model.predict()` (not `model.forward()`) to get ms predictions regardless of framing
- Compute all metrics defined in the config
- Set model back to `train()` mode at the end

**Track these per validation epoch:**
- Val MAE (primary metric for checkpointing decisions)
- Val loss (may differ from MAE if using Huber or segmentation loss)
- Per-asset val MAE (if training on all assets combined)
- Best val MAE seen so far

---

### 4.6.3 — Checkpointing Strategy

On Colab, sessions die. Checkpointing is not optional — it is survival infrastructure.

**What to save in every checkpoint:**

```python
checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
    'scaler_state_dict': scaler.state_dict(),  # for mixed precision
    'best_val_mae': best_val_mae,
    'config': config,              # full config saved inside checkpoint
    'train_history': train_history # list of dicts, one per epoch
}
```

**Saving logic:**
- Save `checkpoint_best.pt` whenever val MAE improves. This overwrites the previous best.
- Save `checkpoint_epoch_{N}.pt` every 10 epochs regardless. This is your safety net.
- Save `checkpoint_latest.pt` at the end of every epoch (overwrites). This lets you resume from the last completed epoch after a Colab session dies.

All saves go directly to Drive, not to `/content/`. A save to `/content/` disappears with the session.

**Resuming from checkpoint:**
At the top of your training notebook, check if `checkpoint_latest.pt` exists. If it does, load it and resume. If it does not, start fresh. This logic should be automatic — you should never have to manually manage resume logic.

---

### 4.6.4 — Early Stopping

Track the best validation MAE. If it does not improve for `early_stopping_patience` epochs (from config, typically 15), stop training and restore the best checkpoint. Log which epoch achieved the best result.

**Important nuance:** Early stopping should monitor validation MAE, not validation loss. If you are using a segmentation loss (BCE + Dice), the loss scale is not directly comparable to MAE, and a model that minimizes segmentation loss might not minimize MAE. Always use the physically meaningful metric (MAE in ms) for your stopping criterion.

---

### 4.6.5 — Learning Rate Scheduling

Use cosine annealing as your default: the learning rate decays smoothly from `lr` to `eta_min` over `T_max` epochs following a cosine curve. This is more stable than step decay and often outperforms it.

**Warmup:** For the first 5 epochs, linearly ramp the learning rate from `lr/10` to `lr`. This prevents early instability from large gradient updates when the model weights are random. Use `torch.optim.lr_scheduler.LinearLR` for warmup, then chain it with cosine annealing using `SequentialLR`.

**When to use ReduceLROnPlateau instead:**
If you are unsure about the right number of epochs or the training loss curve is erratic, ReduceLROnPlateau (reduce LR by factor 0.5 if val MAE does not improve for 5 epochs) is more forgiving. Use it for early experiments, switch to cosine annealing once you have a stable training regime.

---

## 4.7 — Hyperparameter Search Strategy

You cannot manually tune every model. You need a systematic search strategy that fits within Colab's constraints.

---

### 4.7.1 — What to Search and What to Fix

**Fix across all experiments (do not search):**
- Optimizer: AdamW (consistently best for deep learning on this type of task)
- Seed: 42 (for reproducibility)
- Split ratios: 70/15/15
- Augmentation types (only toggle on/off)
- Evaluation metrics (always the same)

**Search per model:**
- Learning rate: log-uniform search in [1e-5, 1e-2]
- Batch size: [4, 8, 16] (constrained by VRAM)
- Dropout rate: [0.0, 0.1, 0.3, 0.5]
- Weight decay: log-uniform in [1e-5, 1e-2]
- Model-specific: base_channels [32, 64], encoder_depth [3, 4, 5]

**Loss function:** Treat as a discrete hyperparameter. Run MAE vs. Huber for your best architecture specifically.

---

### 4.7.2 — Optuna Integration

Use Optuna for hyperparameter search. It is lightweight, integrates cleanly with PyTorch, and supports **pruning** — stopping unpromising trials early, which is critical given Colab's time limits.

**How to structure an Optuna study in Colab:**

Define an objective function that:
1. Samples hyperparameters from the Optuna trial object
2. Builds the model and dataloaders with those parameters
3. Trains for a fixed short number of epochs (e.g., 20 — enough to distinguish good from bad configurations)
4. Returns the best validation MAE achieved

Use `MedianPruner` — if a trial's validation MAE after epoch 10 is worse than the median of all previous trials at epoch 10, prune it. This alone can reduce your total search time by 50–70%.

**Saving Optuna studies to Drive:**
Optuna stores its study in an SQLite database. Point the storage path to Drive so your search history survives Colab session restarts. Load and continue the study in subsequent sessions.

**Search budget:** Run 30–50 trials per model. This typically finds a near-optimal configuration for the search space defined above. Beyond 50 trials, diminishing returns set in.

---

### 4.7.3 — Practical Search Order

Do not search hyperparameters for every model. Be strategic:

1. Run classical baselines with no search (they have few parameters)
2. Run 1D ResNet with a light Optuna search (20 trials) — establishes a reasonable LR and batch size regime
3. Transfer those findings as starting points for U-Net search (the LR regime from simpler models is a reasonable prior)
4. Run full Optuna search (50 trials) only on your best-performing architecture from the initial benchmark
5. For pretrained backbone models, search only the LR — everything else follows from the unfreezing schedule

---

## 4.8 — Experiment Tracking

Every training run must produce a complete record. Without this you will lose track of what you tried.

---

### 4.8.1 — Results Logging Structure

For each completed training run, save a results dict to Drive as a JSON file named `{experiment_name}_results.json`:

```json
{
  "experiment_name": "unet_seg_all_assets_v3",
  "config_path": "configs/model_unet.yaml",
  "timestamp": "2024-01-15T14:32:00",
  "best_epoch": 67,
  "training_time_hours": 2.3,
  "final_metrics": {
    "test_mae_ms": 4.2,
    "test_rmse_ms": 7.1,
    "test_within_5ms_pct": 78.3,
    "test_within_10ms_pct": 91.2,
    "per_asset": {
      "brunswick": {"mae_ms": 3.8, "within_5ms_pct": 81.0},
      "halfmile":  {"mae_ms": 5.1, "within_5ms_pct": 74.5},
      "lalor":     {"mae_ms": 3.9, "within_5ms_pct": 79.8},
      "sudbury":   {"mae_ms": 4.0, "within_5ms_pct": 77.9}
    }
  },
  "val_history": [
    {"epoch": 1, "val_mae_ms": 28.4, "lr": 0.001},
    ...
  ]
}
```

Also save training curves (loss and MAE vs. epoch) as PNG files to Drive.

---

### 4.8.2 — Master Benchmark Table

Maintain a single `benchmark_results.csv` on Drive. Every time a model finishes training, append one row:

```
experiment_name, architecture, framing, pretrained, test_mae_ms, test_within_5ms, 
test_within_10ms, brunswick_mae, halfmile_mae, lalor_mae, sudbury_mae, 
n_parameters, training_time_hours, best_epoch, notes
```

This becomes your source of truth for the final comparison. Load it at the start of your benchmarking notebook to instantly see where every model stands.

---

## 4.9 — Notebook Architecture

Each training notebook follows an identical structure so you can navigate any of them quickly:

```
Cell 1:  Drive mount + sys.path setup + seed fixing
Cell 2:  Config loading (specify which yaml file)
Cell 3:  Dataset and DataLoader instantiation + sanity checks
Cell 4:  Model instantiation + parameter count printout
Cell 5:  Loss function and optimizer setup
Cell 6:  Resume logic (load checkpoint if exists)
Cell 7:  Training loop (calls trainer.train())
Cell 8:  Load best checkpoint, run test evaluation
Cell 9:  Append results to benchmark_results.csv
Cell 10: Visualization (training curves, sample predictions)
```

**Sanity check cell (Cell 3) must verify:**
- Dataset loads without error
- `__getitem__` returns correct shapes for your framing
- All four assets are represented in the training split
- Label distribution in train/val/test splits looks similar (stratification worked)
- At least one gather visualized with labels overlaid (visual sanity, not just shape check)

**Never skip the sanity check cell.** A shape bug that would have been caught in 10 seconds in Cell 3 will waste 2 hours of Colab training time if discovered at epoch 50.

---

## 4.10 — Memory Management for Colab

Colab's RAM and VRAM are shared and limited. Explicit management prevents OOM crashes mid-training.

**RAM management:**
- Never load all four full HDF5 files simultaneously. Load one at a time during preprocessing, save NPZs, then close the HDF5 handle.
- If your processed NPZ dataset fits in RAM (~4–8GB typical), set `cache_in_ram: true` in the config. Loading from RAM is 10–100× faster than loading from Drive per batch.
- If it does not fit, use PyTorch's `num_workers=2` in DataLoader with `prefetch_factor=2` to overlap Drive I/O with GPU computation.

**VRAM management:**
- After each training run in a notebook, explicitly call `del model`, `del optimizer`, `torch.cuda.empty_cache()`, and `gc.collect()` before instantiating the next model. Otherwise VRAM fragments and subsequent models will OOM even if theoretically they fit.
- Monitor VRAM usage during training with `torch.cuda.memory_allocated()` and log it. If it grows monotonically (memory leak), you have a tensor that is not being freed — commonly caused by storing loss values as tensors instead of Python floats in your history dict.

---



# Phase 4.5 — Experiment Tracking, Device Management & Rerun-Safe Notebook Architecture

This is a self-contained extension to Phase 4. Nothing in Phases 4.1–4.10 needs to be deleted — this adds infrastructure on top of it. Read it as mandatory additions to your pipeline, not alternatives.

---

## 4.5.1 — MLflow Integration

MLflow is the right tool for your experiment tracking needs. It is free, runs entirely within Colab, persists its database to Drive, and gives you a browser-based UI to compare runs without writing a single visualization script.

---

### Why MLflow Over Manual CSV Logging

The `benchmark_results.csv` approach from Phase 4.8 is still worth keeping as a human-readable summary. But MLflow gives you things the CSV cannot:

- Automatic parameter hashing so identical runs are never duplicated
- Artifact storage — model checkpoints, plots, and example outputs attached directly to the run that produced them
- A queryable backend — "show me all runs where test MAE < 5ms and architecture is unet" in two clicks
- Nested runs — a hyperparameter search trial is a child run of the parent experiment, keeping your run list clean
- Direct diff view between any two runs

---

### MLflow Setup in Colab

MLflow stores everything in a tracking URI. Point that URI at your Drive so it survives session restarts:

```
tracking_uri: "file:///content/drive/MyDrive/seismic_fbp/mlruns"
```

Add this to your `configs/datasets.yaml` as a top-level field so every notebook reads it from the same place. Never hardcode the path inside a notebook.

At the top of every training notebook, after mounting Drive, initialize MLflow:

```
import mlflow
mlflow.set_tracking_uri(config.tracking_uri)
mlflow.set_experiment(config.experiment.name)
```

The experiment name groups related runs together — use the architecture name as the experiment name (e.g., `"unet"`, `"resnet_unet"`, `"1d_resnet"`), not the full run name. Individual runs within that experiment are differentiated by their logged parameters.

---

### What to Log to MLflow Per Run

Structure your logging into three categories:

**At run start — log parameters:**
Everything from your config that affects the result. This includes architecture name, framing, all training hyperparameters, augmentation toggles, normalization scheme, which assets were used for this run, the device it ran on, and the random seed. Log these with `mlflow.log_params(flat_config_dict)`. MLflow expects a flat dict, so flatten your nested YAML before logging.

**During training — log metrics per epoch:**
Use `mlflow.log_metric(key, value, step=epoch)` for: train loss, val loss, val MAE, val within-5ms accuracy, current learning rate, GPU memory usage, and per-asset val MAE if training on all combined. The `step` parameter is critical — it makes MLflow plot these as time series automatically.

**At run end — log artifacts:**
This is where MLflow becomes genuinely powerful. Log the following as artifacts with `mlflow.log_artifact(filepath)`:

- The config YAML file used for this run
- The best checkpoint file (`checkpoint_best.pt`)
- Training curve plots (loss and MAE vs epoch as PNG)
- Example output images (covered in 4.5.2)
- The benchmark row CSV for this run
- A text file containing the full console output of the run

All artifacts are stored inside `mlruns/` on Drive, organized by experiment and run ID automatically.

---

### Accessing the MLflow UI in Colab

MLflow has a web UI. To access it from Colab, start the server in a background thread and use a tunnel:

```python
import subprocess
import threading

def run_mlflow_ui():
    subprocess.run([
        "mlflow", "ui",
        "--backend-store-uri", config.tracking_uri,
        "--port", "5000"
    ])

thread = threading.Thread(target=run_mlflow_ui, daemon=True)
thread.start()
```

Then use `ngrok` or Colab's built-in port forwarding to access it in your browser. Add this as a standalone utility notebook (`notebooks/00_mlflow_ui.ipynb`) that you open separately when you want to browse results — do not put it in the training notebooks.

---

## 4.5.2 — Example Output Artifacts Per Run

Every completed training run must save a set of visual example outputs so you can immediately see what the model is actually predicting without writing new code each time.

---

### What to Save and How

At the end of every training run, after loading the best checkpoint, run inference on a fixed set of examples from each asset's test split. These examples must be the same across all runs — fix them by shot ID at the start of your project and save those IDs to a file called `visualization_examples.json` on Drive. Every model evaluates on the same examples, making visual comparison across models straightforward.

For each example shot gather, save one composite PNG containing four panels arranged vertically:

**Panel 1 — Raw gather image:**
The 2D shot gather displayed as a grayscale heatmap. Traces on the x-axis, time on the y-axis. No annotations.

**Panel 2 — Ground truth overlay:**
Same gather with the ground truth first break picks overlaid as a red line. For unlabeled traces, show a gap in the line.

**Panel 3 — Prediction overlay:**
Same gather with the model's predicted first break picks overlaid as a blue line. Ground truth in red at lower opacity for reference.

**Panel 4 — Error profile:**
A line plot showing the per-trace error in milliseconds (predicted minus ground truth) as a function of trace index. Include a horizontal band at ±5ms to show the within-threshold zone. Color the line green where error is within 5ms and red where it exceeds it.

Save this composite as `{experiment_name}_{asset}_{shot_id}_example.png`. Log it as a MLflow artifact. This gives you immediate visual diagnostic capability — you can see at a glance whether your model has a systematic bias (always predicts too early), whether it fails on specific offset ranges, or whether it handles noise traces gracefully.

---

### Fixed Visualization Example Selection

During your preprocessing phase, before any training, select and save your fixed visualization examples. Choose:

- 3 examples per asset (12 total)
- For each asset: one easy example (high SNR, clean first break), one medium example (moderate noise), one hard example (low SNR or complex geometry)
- Classify difficulty using the per-trace SNR proxy computed during EDA

Save the shot IDs of these 12 examples to `visualization_examples.json`. Every training notebook loads this file and generates the same 12 composite plots. Never regenerate which examples to use — consistency across all models is the entire point.

---

## 4.5.3 — Device Detection and Management

---

### Detecting Available Hardware

At the top of every notebook, immediately after imports, run a device detection block that determines what hardware is available and configures the pipeline accordingly. Colab may give you a CPU, a T4 GPU, an A100 GPU, or a TPU — and your code must handle all four without manual intervention.

**GPU detection:**
Check `torch.cuda.is_available()`. If True, query `torch.cuda.get_device_name(0)` and `torch.cuda.get_device_properties(0).total_memory` to record the GPU model and VRAM in bytes. Log both to MLflow as run parameters.

**TPU detection:**
Colab's TPU v2 (and occasionally TPU v4 in Colab Pro+) requires a separate backend. Check for TPU availability using the `torch_xla` library. TPU detection logic:

```python
try:
    import torch_xla.core.xla_model as xm
    device = xm.xla_device()
    device_type = "tpu"
    device_name = str(device)
except ImportError:
    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_type)
    device_name = torch.cuda.get_device_name(0) if device_type == "cuda" else "cpu"
```

**VRAM-aware batch size selection:**
After detecting the device, automatically select the batch size based on available VRAM rather than relying on the config value blindly. Define a lookup table in your config:

```yaml
vram_batch_size_map:
  unet:
    "< 8GB":  4
    "8-12GB": 8
    "> 12GB": 16
  resnet_unet:
    "< 8GB":  2
    "8-12GB": 4
    "> 12GB": 8
```

At runtime, query available VRAM, look up the appropriate batch size, and override the config value. Log the actual batch size used (not the config default) to MLflow. This prevents silent OOM failures when your session gets a weaker GPU than expected.

**CPU fallback:**
If neither GPU nor TPU is available, the code must still run — just slower. Add a warning print that training will be slow, reduce batch size to 2, and disable mixed precision (AMP does not help on CPU). Do not crash or require manual intervention.

---

### TPU-Specific Adaptations

If a TPU is detected, several parts of your pipeline need adjustment:

**Mixed precision:** Do not use `torch.cuda.amp` — it is CUDA-specific. TPU handles precision internally. Remove the GradScaler when on TPU.

**DataLoader:** TPU training benefits from a larger number of workers. Use `num_workers=8` on TPU vs. `num_workers=2` on GPU.

**Optimizer step:** On TPU, replace `optimizer.step()` with `xm.optimizer_step(optimizer)` and add `xm.mark_step()` after each forward-backward pass. These are XLA-specific synchronization calls.

**Checkpoint saving:** On TPU, saving checkpoints requires `xm.save()` instead of `torch.save()`. Abstract your checkpoint saving into a `save_checkpoint(checkpoint_dict, path)` helper function that internally checks `device_type` and uses the appropriate save method.

---

### Device Information in Every Checkpoint

Every checkpoint saved to Drive must record the device it was trained on. Add these fields to the checkpoint dict from Phase 4.6.3:

```python
checkpoint['device_info'] = {
    'device_type': device_type,
    'device_name': device_name,
    'vram_bytes': vram_bytes if device_type == 'cuda' else None,
    'torch_version': torch.__version__,
    'colab_session_id': os.environ.get('COLAB_BACKEND_URL', 'unknown')
}
```

This becomes critical during debugging — if a model trained on T4 behaves differently than one trained on A100, you want to know immediately.

---

## 4.5.4 — Rerun-Safe Progressive Asset Training

This is the most architecturally significant addition in this phase. The core idea: each training notebook should behave correctly whether it is being run for the first time, resumed mid-training, or re-run after completion — with no manual code edits required between runs.

---

> ### ⚠ Default vs. Fallback — Read This First
>
> **The default training approach for this project is combined simultaneous
> training across all four assets.** The `BalancedAssetSampler` loads shot
> gathers from Brunswick, Halfmile, Lalor, and Sudbury interleaved within
> every batch from epoch 1. The model learns all four assets simultaneously.
> This is simpler, more robust, and eliminates catastrophic forgetting entirely.
>
> **The progressive training pattern described in this section is a fallback
> option only.** It exists for cases where a single combined training run
> exceeds Colab session time limits. It is enabled via a config flag:
> `training_mode: combined | progressive` — and defaults to `combined`.
>
> The end-of-epoch checkpointing system already described in this document
> (saving `checkpoint_latest.pt` after every epoch to Drive) means that a
> Colab session death during a long combined run is not catastrophic — the
> next run resumes automatically from the last completed epoch. This makes
> the progressive fallback necessary far less often than it might appear.
>
> **If you are implementing this for the first time: use combined mode.
> Only consider switching to progressive mode if combined training consistently
> exceeds your available Colab session time after profiling actual epoch duration.**

This first-run / resume / re-run behavior applies to notebooks operating in `training_mode: progressive`.

### The Progressive Training Pattern

**Note: catastrophic forgetting is only relevant when `training_mode: progressive` is explicitly enabled. Under the default combined training mode it does not occur and the mitigations below are not needed.**

The sequential training across four assets is a form of **curriculum learning combined with domain adaptation**. You train the model on one asset, save it, then fine-tune on the next, and so on. By the fourth run, the model has seen all four assets.

**Before implementing this, understand the critical risk:** Sequential fine-tuning across assets causes **catastrophic forgetting — the model's performance on earlier assets degrades as it trains on later ones.** You must address this explicitly.

Three strategies, from simple to powerful:

**Strategy 1 — Replay buffer (recommended for your constraints):**
When training on Asset N, include a small random sample (10–20%) of the training data from all previous Assets 1 through N-1 alongside the full Asset N data. This prevents forgetting without requiring any architectural changes. Implement this in your BalancedAssetSampler — after the first asset, the sampler mixes current asset data with replay samples from previous assets.

**Strategy 2 — Elastic Weight Consolidation (EWC):**
A regularization approach that adds a penalty term to the loss, discouraging large changes to weights that were important for previous tasks. More theoretically grounded than replay but requires computing the Fisher information matrix after each asset, which is computationally expensive. Use only if Strategy 1 proves insufficient.

**Strategy 3 — Train on all assets simultaneously from the start (simplest):**
If after EDA the assets are sufficiently compatible, simply combine all four training sets from day one and train once. This avoids catastrophic forgetting entirely. The progressive pattern only makes sense if assets are sufficiently different that a curriculum helps, or if you want to explicitly study per-asset generalization.

**Recommendation:** Default to Strategy 1 (replay buffer). It is simple, effective, and fits naturally into your existing BalancedAssetSampler.

---

### The State Machine at the Heart of Each Notebook

Every training notebook implements a simple state machine with five states (in `training_mode: progressive`). The notebook detects its current state automatically and executes the correct logic for that state with no user intervention.

```
State 0: NO_CHECKPOINT_EXISTS
  → Train from scratch on Asset 1
  → Save checkpoint with state metadata
  → Done for this run

State 1: ASSET_1_COMPLETE
  → Load checkpoint
  → Fine-tune on Asset 2 (+ replay from Asset 1)
  → Save checkpoint with updated state metadata
  → Done for this run

State 2: ASSET_2_COMPLETE
  → Load checkpoint
  → Fine-tune on Asset 3 (+ replay from Assets 1-2)
  → Save checkpoint with updated state metadata
  → Done for this run

State 3: ASSET_3_COMPLETE
  → Load checkpoint
  → Fine-tune on Asset 4 (+ replay from Assets 1-3)
  → Save checkpoint with updated state metadata
  → Done for this run

State 4: TRAINING_COMPLETE
  → Print "Model fully trained. Skipping to evaluation."
  → Load best checkpoint
  → Run full test evaluation on all four assets
  → Log results to MLflow
  → Generate all visualization examples
  → Done
```

---

### Implementing the State Machine

Add a `training_state` field to every checkpoint dict. This is the single source of truth for which state the notebook is in:

```python
checkpoint['training_state'] = {
    'completed_assets': ['brunswick'],      # list grows with each run
    'current_asset_index': 0,               # 0-indexed
    'is_fully_trained': False,
    'per_asset_best_val_mae': {
        'brunswick': 3.8                    # filled in as assets complete
    }
}
```

At the very start of your training notebook (before any model construction), run the state detection block:

```python
checkpoint_path = os.path.join(config.output.checkpoint_dir, 'checkpoint_latest.pt')

if not os.path.exists(checkpoint_path):
    state = 'NO_CHECKPOINT_EXISTS'
    start_asset_index = 0
    resume_epoch = 0
else:
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    training_state = checkpoint['training_state']
    
    if training_state['is_fully_trained']:
        state = 'TRAINING_COMPLETE'
    else:
        completed = training_state['completed_assets']
        start_asset_index = len(completed)
        state = f'ASSET_{len(completed)}_COMPLETE' if completed else 'NO_CHECKPOINT_EXISTS'
        resume_epoch = checkpoint['epoch']

print(f"Detected state: {state}")
print(f"Next asset to train: {ASSET_ORDER[start_asset_index] if state != 'TRAINING_COMPLETE' else 'N/A'}")
```

Define `ASSET_ORDER = ['brunswick', 'halfmile', 'lalor', 'sudbury']` as a constant at the top of the notebook. This order determines the training curriculum and must be consistent across all runs.

---

### Mid-Asset Resume Safety

The state machine above handles inter-asset transitions (between full asset training runs). But Colab sessions also die mid-training within a single asset. The `checkpoint_latest.pt` saved every epoch (from Phase 4.6.3) handles this case.

When loading a checkpoint, check both the `training_state` (which asset to train next) AND the `epoch` field (where to resume within the current asset's training). If the latest checkpoint was saved mid-training on Asset 2 (e.g., at epoch 34 of 100), the next run resumes at epoch 34 of Asset 2 — it does not restart Asset 2 from scratch and it does not skip to Asset 3.

This requires your training loop to accept a `start_epoch` parameter:

```python
for epoch in range(resume_epoch, config.training.epochs):
    # training logic
```

When an asset completes (all epochs done or early stopping triggered), update `training_state.completed_assets`, set `resume_epoch = 0` for the next asset, and save `checkpoint_latest.pt` before the notebook ends.

---

### Per-Asset Fine-Tuning Hyperparameters

When fine-tuning on Asset 2 through 4, you should not use the same learning rate as the initial training on Asset 1. The model has already converged on a good solution — large learning rates will destroy it.

Add per-asset fine-tuning configs to your YAML:

```yaml
progressive_training:
  asset_order: ['brunswick', 'halfmile', 'lalor', 'sudbury']
  asset_epochs: [100, 50, 50, 50]        # more epochs for first asset, fewer for fine-tuning
  asset_lr:     [1e-3, 2e-4, 2e-4, 2e-4] # lower LR for fine-tuning runs
  replay_fraction: 0.15                   # 15% of each fine-tuning batch from previous assets
  reset_optimizer: true                   # reset optimizer state between assets
  reset_scheduler: true                   # restart LR schedule for each asset
```

**`reset_optimizer: true` is important:** The Adam optimizer accumulates momentum and adaptive learning rate estimates. Carrying these over from Asset 1 training into Asset 2 fine-tuning can cause erratic behavior in early fine-tuning epochs because the accumulated statistics are calibrated to Asset 1's gradient landscape. Resetting gives you a clean start for each new asset.

---

### The Evaluation-Only Run (State 4)

When the notebook detects `TRAINING_COMPLETE`, it must execute a comprehensive evaluation without any training. This run should:

- Load `checkpoint_best.pt` (the best checkpoint across all four asset training phases, not just the last one)
- Run full test evaluation on each asset's test split independently
- Generate all 12 visualization examples
- Log everything to MLflow
- Print a clean summary table to the notebook output
- Save the benchmark row to `benchmark_results.csv`

**Which "best checkpoint" to use:**
Because you save `checkpoint_best.pt` based on validation MAE, and validation MAE is computed on the current asset during each training phase, the "best" checkpoint saved during Asset 4 training might actually be worse on Assets 1-2 than the checkpoint from Asset 2 training. 

To handle this properly, save a separate `checkpoint_best_overall.pt` that is updated only when the combined validation MAE (averaged across all assets seen so far) improves. This requires running a brief validation pass on all previously seen assets after each epoch — computationally more expensive but gives you a truly best overall model rather than the best model on the last asset.

---

## 4.5.5 — Constraints Summary and Experimentation Routine

Consolidating all constraints into one place for reference during every session:

---

### Hard Constraints

**Memory:** Never load more than one full HDF5 file at a time. Always use NPZ-per-gather storage. Use Drive-backed DataLoader with `num_workers=2` unless the processed dataset fits in RAM (check at the start of each session with `shutil.disk_usage` and an estimate of dataset size from the master index CSV).

**Checkpointing:** Save `checkpoint_latest.pt` at the end of every epoch, to Drive, unconditionally. No epoch should complete without a checkpoint save. If saving fails (Drive I/O error), catch the exception, log a warning, and retry once — never silently continue without saving.

**Reproducibility:** Fix all random seeds at the very top of every notebook — Python `random`, `numpy`, `torch`, and `torch.cuda` (for deterministic CUDA ops). Log the seed to MLflow. Never generate splits or augmentation sequences before the seed is fixed.

**No manual edits between runs:** The notebook must work correctly from a fresh session (Drive mounted, packages installed) by running all cells top to bottom. Every path, every hyperparameter, every decision comes from the config file or the checkpoint state machine (for `training_mode: progressive`).

---

### Session Start Checklist (Automate This as Cell 1)

Print this checklist automatically at the start of every notebook session:

```
✓ Drive mounted at /content/drive
✓ Device: T4 GPU (14.7 GB VRAM)
✓ Checkpoint detected: checkpoint_latest.pt (epoch 34, asset: halfmile)
✓ Training state: ASSET_1_COMPLETE → Next: halfmile (resuming epoch 34)
✓ MLflow tracking URI: file:///content/drive/MyDrive/seismic_fbp/mlruns
✓ Active experiment: unet
✓ Processed dataset: 4823 shot gathers across 4 assets
✓ Replay buffer: 483 samples from brunswick
✓ Estimated time to asset completion: ~2.1 hours
```

This takes 10 seconds to compute and prevents every category of "wait why is it retraining from scratch" confusion.

---

### Recommended Session Routine

A practical routine for each working session given Colab's time limits:

1. Open the notebook for the model you are currently training
2. Run Cell 1 (setup + checklist) — verify the state is what you expect
3. If state looks wrong, investigate before running further — never blindly proceed
4. Run all remaining cells — training resumes automatically
5. When Colab warns about session timeout, your checkpoint is already saved — let it expire
6. Next session: repeat from step 1

If a session dies mid-epoch before the end-of-epoch checkpoint save, you lose that partial epoch and resume from the previous epoch's checkpoint. This is acceptable — one epoch of repeated work is trivial compared to the alternative of losing the entire run.

