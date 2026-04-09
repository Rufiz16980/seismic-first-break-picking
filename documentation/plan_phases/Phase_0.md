# Phase 0 — Environment & Repository Architecture

## Purpose

This document defines the foundational architecture of the project before any
code is written or data is downloaded. It covers three things: the repository
structure, the Colab/Drive infrastructure, and the clarification of what kind
of machine learning task this actually is. Everything in Phases 1 through 5
assumes the decisions made here are already in place.

Data acquisition and environment verification are covered in Phase_0.5.md.
Operational workflow, agent routing, and session management are covered in
Meta_Plan_Agent.md and Meta_Plan_User.md. This document does not repeat those.

---

## 0.1 — Task Framing Clarification

This must be understood before any modeling decision is made.

### What the task is NOT

This is not binary classification. The label stored in SPARE1 is not a class
label. It is a continuous time value in milliseconds representing when the
first seismic wave arrives at a receiver. Framing this as classification would
mean discretizing a continuous physical quantity into bins — lossy, arbitrary,
and unnecessary.

### What the task IS

The task is **regression**. Specifically, per-trace regression where the target
is a single float value (first break arrival time in milliseconds) for each
seismic trace.

However the way model input is structured creates three distinct framings, and
the choice of framing is an architectural decision that affects every model:

**Framing A — Trace-level regression:**
- Input: one 1D seismic trace, shape `[1, n_samples]`
- Output: one float (first break time in ms)
- The model sees one trace in isolation with no spatial context

**Framing B — Shot-gather regression:**
- Input: one 2D shot gather image, shape `[1, n_samples, n_traces]`
- Output: vector of first break times, shape `[n_traces]`
- The model sees all traces in a shot simultaneously and exploits spatial
  coherence of the first break curve across neighboring traces

**Framing C — Segmentation:**
- Input: one 2D shot gather image, shape `[1, n_samples, n_traces]`
- Output: binary mask, shape `[1, n_samples, n_traces]`
  where 1 = below first break, 0 = above first break
- First break times extracted as the boundary row index per column
- Enables use of U-Net and segmentation loss functions

Framing B and C are strictly more powerful than Framing A because they
provide the model with spatial context. A noisy trace whose first break
is ambiguous in isolation becomes much more predictable when the model can
see the clean first break curve on neighboring traces.

**Default recommendation:** Use Framing B for regression models, Framing C
for segmentation models (U-Net). Framing A for classical ML baselines only.
All models implement a unified `predict()` method that always returns
first break times in ms regardless of internal framing.

---

## 0.2 — Repository Structure

The canonical repository structure for this project is defined below.
Every agent session, every notebook, and every config file assumes this
structure exists exactly as shown. It is created by running
`scripts/create_folder_structure.py` as described in Phase_0.5.md.

```
seismic-first-break-picking/
│
├── configs/
│   ├── datasets.yaml              ← Drive paths, MLflow URI, asset metadata
│   ├── preprocessing.yaml         ← normalization, windowing, augmentation flags
│   ├── model_unet.yaml
│   ├── model_resnet_unet.yaml
│   ├── model_1dcnn.yaml
│   └── model_classical.yaml
│
├── data/
│   ├── raw/                       ← compressed .xz source files, never modified
│   ├── extracted/                 ← decompressed .hdf5 files, never modified
│   ├── processed/                 ← NPZ shot gather files, one per shot
│   │   ├── brunswick/
│   │   ├── halfmile/
│   │   ├── lalor/
│   │   └── sudbury/
│   └── datasets/
│       ├── train/
│       ├── val/
│       └── test/
│
├── documentation/
│   ├── plan_phases/               ← original phase plans, READ ONLY
│   │   ├── Phase_0.md             ← this file
│   │   ├── Phase_0.5.md
│   │   ├── Phase_1.md
│   │   ├── Phase_2.md
│   │   ├── Phase_3.md
│   │   ├── Phase_4.md             ← includes Phase 4.5 content
│   │   └── Phase_5.md
│   └── implementation_phases/     ← agent output, one file per phase
│       ├── Impl_Phase_0.5.md
│       ├── Impl_Phase_1.md
│       └── ...
│
├── mlruns/                        ← MLflow tracking database, auto-generated
│
├── models/
│   ├── checkpoints/               ← per-model training checkpoints
│   │   ├── unet/
│   │   ├── resnet_unet/
│   │   ├── 1dcnn/
│   │   └── classical/
│   └── final/                     ← final selected model checkpoint
│
├── notebooks/
│   ├── 00_environment_setup.ipynb
│   ├── 01_eda_brunswick.ipynb
│   ├── 01_eda_halfmile.ipynb
│   ├── 01_eda_lalor.ipynb
│   ├── 01_eda_sudbury.ipynb
│   ├── 01_eda_combined.ipynb
│   ├── 02_preprocessing_pipeline.ipynb
│   ├── 03_train_classical.ipynb
│   ├── 03_train_1dcnn.ipynb
│   ├── 03_train_unet.ipynb
│   ├── 03_train_resnet_unet.ipynb
│   └── 04_benchmark_and_compare.ipynb
│
├── results/
│   ├── sanity_plots/
│   ├── eda_plots/
│   ├── visualization_examples/    ← 12 fixed example composites, all models
│   └── benchmark/
│
├── scripts/
│   └── create_folder_structure.py
│
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   ├── hdf5_reader.py         ← reads raw HDF5, applies scale factors
│   │   ├── shot_gather_builder.py ← groups traces into 2D shot gathers
│   │   ├── dataset.py             ← PyTorch Dataset + BalancedAssetSampler
│   │   └── transforms.py          ← augmentation transform classes
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py             ← MAE, RMSE, within-N-ms accuracy
│   │   └── visualizer.py          ← composite output plot generation
│   ├── models/
│   │   ├── __init__.py
│   │   ├── base.py                ← BaseModel interface all models inherit
│   │   ├── unet.py
│   │   ├── resnet_unet.py
│   │   ├── cnn_1d.py
│   │   └── classical.py
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py             ← training loop, checkpointing, early stopping
│   │   └── losses.py              ← MAE, Huber, BCE+Dice, Wing losses
│   └── utils/
│       ├── __init__.py
│       ├── config_loader.py       ← YAML loading, validation, flattening
│       └── logger.py              ← CSV logging, MLflow integration
│
├── CURRENT_STATUS.md              ← agent/user handoff document
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 0.3 — Colab and Drive Infrastructure

### Drive Layout

The project root on Drive is:
```
/content/drive/MyDrive/seismic-first-break-picking/
```

This is the only path that should ever appear in notebooks and configs.
All other paths are relative to this root or derived from it at runtime
via the `datasets.yaml` config. Nothing is hardcoded beyond this root.

### What Lives Where

| Location | Contents | Modified by |
|---|---|---|
| Drive: `data/raw/` | Compressed .xz source files | Downloaded once by 00_environment_setup.ipynb, never touched again |
| Drive: `data/extracted/` | Decompressed HDF5 files | Created once by 00_environment_setup.ipynb, never touched again |
| Drive: `data/processed/` | NPZ shot gather files | Created by 02_preprocessing_pipeline.ipynb |
| Drive: `models/checkpoints/` | Training checkpoints | Written by training notebooks every epoch |
| Drive: `mlruns/` | MLflow experiment database | Written by MLflow automatically during training |
| Drive: `results/` | Plots, metrics, benchmark CSV | Written by all notebooks |
| Local + Drive: everything else | Code, configs, docs, notebooks | Agent writes, Drive syncs to local |

Large binary files (HDF5, NPZ, PT checkpoints) live exclusively on Drive
and are never pushed to Git. Code, configs, documentation, and notebooks
are in Git and also sync to local via Drive desktop app.

### Colab Session Requirements

Every notebook begins with these three operations in Cell 1, in this order:

1. Mount Google Drive
2. Append `src/` to sys.path so all modules are importable
3. Run device detection and print the device report

No notebook assumes a clean Colab environment. Every notebook installs its
required packages in Cell 2 if they are not already present. This makes every
notebook runnable from a fresh Colab session without any manual setup.

### Training Mode

The default training mode is **combined** — all four assets are loaded into
one DataLoader and trained simultaneously from epoch 1. The BalancedAssetSampler
ensures each batch contains a proportional mix from all four assets.

Progressive training (one asset per run) is a fallback option available via
`training_mode: progressive` in the model config. It is not the default and
should only be used if combined training consistently exceeds Colab session
time limits. See Phase_4.md for full details on both modes.

---

## 0.4 — Config System Architecture

Every configurable value in this project lives in a YAML file under `configs/`.
No hyperparameter, path, threshold, or flag is hardcoded in any notebook or
module. This is enforced, not suggested.

### datasets.yaml

Global project config shared by all notebooks and all models:

```yaml
project_root: "/content/drive/MyDrive/seismic-first-break-picking"

paths:
  raw:        "data/raw"
  extracted:  "data/extracted"
  processed:  "data/processed"
  datasets:   "data/datasets"
  models:     "models/checkpoints"
  results:    "results"
  mlruns:     "mlruns"

mlflow:
  tracking_uri: "file:///content/drive/MyDrive/seismic-first-break-picking/mlruns"

assets:
  names: ["brunswick", "halfmile", "lalor", "sudbury"]

visualization:
  examples_json: "results/visualization_examples.json"
  n_examples_per_asset: 3
```

### Per-model config files

Each model has its own YAML covering data settings, augmentation, model
architecture, training hyperparameters, loss function, and evaluation settings.
The full schema is defined in Phase_4.md Section 4.1.1.

### Config loader behavior

`src/utils/config_loader.py` loads any YAML file, validates that all required
fields are present, resolves all relative paths against `project_root`, and
returns a nested namespace object. It also saves a copy of the active config
into the MLflow run and alongside every checkpoint file. A checkpoint without
its config is considered incomplete.

---

## 0.5 — Base Model Interface

Every model in `src/models/` inherits from `BaseModel` defined in
`src/models/base.py`. The interface is fixed and non-negotiable — it is what
allows the evaluation and benchmarking code to treat all models identically
without knowing their internal framing.

```python
class BaseModel(nn.Module):

    def forward(self, x, offsets=None):
        """
        x: input tensor, shape depends on framing
           Framing A: [batch, 1, n_samples]
           Framing B/C: [batch, 1, n_samples, n_traces]
        offsets: optional [batch, n_traces] offset distances in meters
        returns: raw model output (logits for segmentation,
                 ms values for regression)
        """
        raise NotImplementedError

    def predict(self, x, offsets=None):
        """
        Always returns first break times in milliseconds.
        Shape: [batch, n_traces] for gather models
               [batch] for trace-level models
        Handles segmentation mask → ms conversion internally.
        Never raises — returns NaN for positions it cannot predict.
        """
        raise NotImplementedError

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters()
                   if p.requires_grad)

    def framing(self):
        """
        Returns one of: 'trace_regression',
                        'gather_regression',
                        'segmentation'
        """
        raise NotImplementedError
```

---

## 0.6 — Relationship Between Phase Documents

Read this before starting any implementation session to understand
which documents inform which phases:

```
Phase_0.md        ← you are here. Foundational architecture.
                     Read first in every agent session.

Phase_0.5.md      ← data acquisition and environment verification.
                     One-time setup. Gate 1 for all subsequent phases.

Phase_1.md        ← EDA across all four assets.
                     Depends on: Phase_0.5 complete, HDF5 files available.

Phase_2.md        ← preprocessing pipeline.
                     Depends on: Phase_1 EDA results.
                     Many decisions here cannot be made before EDA.

Phase_3.md        ← model catalog.
                     Can be read and planned during Phase_1/2.
                     Implementation depends on Phase_2 complete.

Phase_4.md        ← training pipeline including all Phase_4.5 content.
                     Read Phase_4.md in full before implementing any
                     training code. The two sections form one system.

Phase_5.md        ← benchmarking and final selection.
                     Scaffolding can begin during Phase_3/4.
                     Full execution requires at least two trained models.
```

No phase document should be read in isolation. When implementing any phase,
always have Phase_0.md and the most recent implementation document open
alongside the target phase plan.
```
