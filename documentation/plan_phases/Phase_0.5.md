# Phase 0.5 — Data Acquisition & Environment Verification

---

## Overview

This phase is a hard prerequisite for everything that follows. Nothing in Phases 1 through 5 can proceed until every item in this document is confirmed complete. The phase has no ML content — it is purely infrastructure and data verification. It should take one focused session of 2–4 hours depending on your internet speed for the downloads.

This phase has one agent session and one mandatory user-run notebook. The agent writes the verification notebook. You run it. The outputs confirm readiness.

---

## 0.5.1 — Local Environment Setup

Complete these steps on your local machine before any agent session begins.

---

### Google Drive for Desktop

Download and install Google Drive for Desktop from `drive.google.com/drive/download`. Sign in with the Google account that has your 2TB storage. During setup, choose **Stream files** mode, not Mirror mode — Stream keeps large files in the cloud and only downloads them on access, which preserves your local disk space given the size of the HDF5 files. Your Drive will appear as a lettered drive (e.g., `G:\My Drive`).

Move your repository into the Drive folder:

```
G:\My Drive\seismic-first-break-picking\
```

Verify the move completed by checking that the Google Drive taskbar icon shows sync complete (no spinning indicator). Open one of your Phase plan MD files from the new location in your editor and confirm it opens correctly.

From this point forward, `G:\My Drive\seismic-first-break-picking\` is your one true repo location. Do not maintain a separate copy on F:. Having two copies will cause confusion about which is current.

---

### Git Initialization

Open a terminal in `G:\My Drive\seismic-first-break-picking\` and initialize Git if not already done. Create a `.gitignore` immediately before making any commits. The gitignore must include the following categories or individual files will accidentally be staged and cause enormous commits:

```gitignore
# Large data files — never commit these
*.hdf5
*.hdf5.xz
*.hdf
*.hdf.xz
*.npz
*.npy
*.pt
*.pth

# Colab checkpoint and output files
checkpoints/
mlruns/

# Python
__pycache__/
*.pyc
*.egg-info/
.ipynb_checkpoints/

# OS
.DS_Store
Thumbs.db
```

Make your first commit after adding the gitignore and all existing documentation files. Tag this commit `v0.0-plan-complete` so you always have a clean reference point before any implementation begins.

---

### Python Environment for Local Development

Even though training runs on Colab, you will want a local Python environment for reading agent-generated code, running small utility scripts, and verifying notebook structure without launching Colab. Create a virtual environment in the repo root:

```
python -m venv venv
```

Add `venv/` to your `.gitignore`. Install the minimal local dependencies — not the full training stack, just what you need to open notebooks and inspect HDF5 files:

```
pip install jupyter h5py numpy pandas matplotlib seaborn pyyaml
```

This local environment is never used for training. Its sole purpose is enabling you to inspect files and run lightweight utility scripts without Colab.

---

## 0.5.2 — Google Drive Folder Structure Creation

Before downloading any data, create the full folder structure defined in Phase 0 Section 0.1. Create it directly inside your repo's Drive location. The agent's first task in this phase is to generate a single Python script called `scripts/create_folder_structure.py` that creates all required directories programmatically. You run this script once locally and it builds the entire structure.

Verify the following directories exist after running the script:

```
G:\My Drive\seismic-first-break-picking\
├── data\
│   ├── raw\
│   ├── extracted\
│   ├── processed\
│   │   ├── brunswick\
│   │   ├── halfmile\
│   │   ├── lalor\
│   │   └── sudbury\
│   └── datasets\
│       ├── train\
│       ├── val\
│       └── test\
├── notebooks\
├── src\
│   ├── data\
│   ├── models\
│   ├── training\
│   ├── evaluation\
│   └── utils\
├── configs\
├── results\
├── models\
│   ├── checkpoints\
│   └── final\
├── documentation\
│   ├── plan_phases\
│   └── implementation_phases\
├── scripts\
└── mlruns\
```

Once created locally, Drive sync pushes this structure to the cloud automatically. When you later open Colab and mount Drive, these folders are already there waiting.

---

## 0.5.3 — Colab Environment Notebook

The agent creates `notebooks/00_environment_setup.ipynb`. This notebook is structured as a one-time setup and verification tool. It is not a training notebook and does not need to be rerun-safe in the same way as training notebooks — it is designed to be run once per project setup, not once per session.

---

### Cell Structure of 00_environment_setup.ipynb

**Cell 1 — Drive Mount and Path Verification:**

Mounts Google Drive. Verifies the repo folder exists at the expected path. Prints the full directory tree of the repo to confirm structure matches what was created locally. Fails loudly with a descriptive error if the repo path is not found — never silently continues.

**Cell 2 — Package Installation:**

Installs all required packages for the full project. Pin every version explicitly — do not use unpinned installs. The full package list includes:

```
h5py==3.9.0
numpy==1.24.3
pandas==2.0.3
matplotlib==3.7.2
seaborn==0.12.2
scipy==1.11.1
scikit-learn==1.3.0
torch==2.0.1+cu118          # CUDA 11.8 build for Colab T4
torchvision==0.15.2+cu118
tqdm==4.65.0
pyyaml==6.0
mlflow==2.5.0
optuna==3.2.0
lightgbm==4.0.0
xgboost==1.7.6
obspy==1.4.0                # for STA/LTA classical picker
segmentation-models-pytorch==0.3.3
albumentations==1.3.1
```

After installation, print the installed version of each critical package (torch, h5py, numpy, mlflow) to confirm no version conflicts occurred.

**Cell 3 — Device Detection:**

Runs the full device detection block from Phase 4.5.3. Prints a formatted device report:

```
============================================================
DEVICE REPORT
============================================================
Device type     : CUDA
Device name     : Tesla T4
VRAM total      : 15,109 MB
VRAM available  : 14,987 MB
CUDA version    : 11.8
PyTorch version : 2.0.1+cu118
TPU available   : No (torch_xla not found)
Recommended batch size (U-Net)     : 8
Recommended batch size (ResNet-UNet): 4
============================================================
```

If only CPU is available, print a prominent warning that training will be extremely slow and suggest switching to a GPU runtime before proceeding.

**Cell 4 — HDF5 Download:**

Downloads all four compressed HDF5 files from their CloudFront URLs directly to `/content/drive/MyDrive/seismic-first-break-picking/data/raw/`. Uses `wget` with the `--continue` flag so interrupted downloads resume rather than restart. Downloads sequentially, not in parallel — parallel downloads to Drive from Colab can cause I/O errors.

For each file, after download completes:
- Print the file size in GB
- Compute and print the MD5 hash of the downloaded file
- Compare against expected size (if known from prior download attempts recorded in this document)

The expected approximate compressed sizes based on the dataset description are large — budget at least 30–60 minutes total download time depending on Colab's network speed. Do not run this cell and immediately close the notebook. Stay present until all four files confirm downloaded.

**Cell 5 — Decompression:**

Decompresses each `.xz` file using Python's `lzma` module. Decompresses to the `data/extracted/` folder. Verifies the decompressed file is a valid HDF5 file by opening it with h5py and checking the root group exists. Prints decompressed file size.

Decompression is CPU-bound and will take 10–30 minutes per file. The notebook runs them sequentially. Do not interrupt — a partially decompressed file will not be a valid HDF5 and will cause confusing errors in Phase 1.

**Cell 6 — HDF5 Structural Verification:**

For each of the four decompressed HDF5 files, opens the file and runs a structural audit. Prints a verification table:

```
============================================================
HDF5 STRUCTURAL VERIFICATION — brunswick
============================================================
File path       : data/extracted/brunswick.hdf5
File size       : 12.3 GB
Root groups     : ['TRACE_DATA']
Target group    : TRACE_DATA/DEFAULT — FOUND ✓
Keys present:
  data_array    : shape (45231, 6000), dtype float32  ✓
  SHOTID        : shape (45231,), dtype int32          ✓
  SHOT_PEG      : shape (45231,), dtype int32          ✓
  SOURCE_X      : shape (45231,), dtype int32          ✓
  SOURCE_Y      : shape (45231,), dtype int32          ✓
  SOURCE_HT     : shape (45231,), dtype int32          ✓
  REC_PEG       : shape (45231,), dtype int32          ✓
  REC_X         : shape (45231,), dtype int32          ✓
  REC_Y         : shape (45231,), dtype int32          ✓
  REC_HT        : shape (45231,), dtype int32          ✓
  SAMP_RATE     : shape (45231,), dtype int32          ✓
  COORD_SCALE   : shape (45231,), dtype int32          ✓
  HT_SCALE      : shape (45231,), dtype int32          ✓
  SAMP_NUM      : shape (45231,), dtype int32          ✓
  SPARE1        : shape (45231,), dtype float32        ✓
Undocumented keys: none
SAMP_RATE constant: YES — value 250 µs
COORD_SCALE constant: YES — value -100
HT_SCALE constant: YES — value -100
SAMP_NUM constant: YES — value 6000
Labeled traces (SPARE1 ≠ 0 and ≠ -1): 38,441 / 45,231 (85.0%)
STATUS: PASS ✓
============================================================
```

If any documented key is missing, print `STATUS: FAIL` with the specific missing key. Do not proceed to Phase 1 if any asset fails this check — investigate the HDF5 file before continuing.

For any undocumented keys found (keys present in the file but not listed in the task description), print them explicitly. These are candidates for additional useful features and must be recorded in the Phase 1 implementation document.

**Cell 7 — Quick Sanity Visualization:**

For each asset, loads a single random trace and plots it as a line plot with the SPARE1 first break time marked as a vertical red line. This is the most basic possible sanity check — you are visually confirming that the data looks like a seismic trace and the label falls at a plausible position.

Saves all four plots as a single PNG to `results/sanity_plots/00_first_traces.png`. Does not display inline only — always saves to Drive so the output persists after the session ends.

**Cell 8 — Environment Report Save:**

Writes a JSON file to `results/00_environment_report.json` containing every piece of information printed in cells 3 and 6 in machine-readable form. This file is read by subsequent agent sessions as part of their context injection — it tells them the exact shapes, sample rates, scale factors, and label densities for all four assets without requiring them to re-run the HDF5 audit.

Ends by printing:

```
============================================================
PHASE 0.5 COMPLETE
All four assets verified. Environment ready for Phase 1.
Update CURRENT_STATUS.md and begin Phase 1 agent session.
============================================================
```

---

## 0.5.4 — Verification Checklist

After running `00_environment_setup.ipynb` to completion, confirm every item before starting the Phase 1 agent session:

```
□ Google Drive for Desktop installed and syncing
□ Repo moved to G:\My Drive\seismic-first-break-picking\
□ Git initialized with correct .gitignore
□ First commit made, tagged v0.0-plan-complete
□ Full folder structure created and visible on Drive
□ All four .xz files downloaded to data/raw/
□ All four .hdf5 files decompressed to data/extracted/
□ All four HDF5 files passed structural verification (STATUS: PASS)
□ Undocumented keys (if any) noted for Phase 1
□ Sanity visualization plots saved and visually inspected
□ 00_environment_report.json saved to results/
□ CURRENT_STATUS.md updated to reflect Phase 0.5 complete
□ Notebook saved in Colab (Ctrl+S) before closing
□ Drive sync confirmed complete on local machine
```

Do not proceed to Phase 1 until every box is checked. If any HDF5 file fails structural verification, investigate and resolve it before moving on — every downstream phase assumes all four files are valid and accessible.

---

## 0.5.5 — Known Risks and Mitigations

**Risk — Download interrupted mid-file:**
The `wget --continue` flag handles this. If a download fails, rerun Cell 4. It will resume from where it stopped rather than restarting. Verify the final file size matches expectations before decompressing.

**Risk — Colab session dies during decompression:**
Decompression writes the output file incrementally. A partial decompressed file will exist but will fail the h5py verification in Cell 6. Delete the partial file and rerun Cell 5 for the affected asset only. Add a per-asset completion flag check at the start of Cell 5 so already-decompressed files are skipped on rerun.

**Risk — Drive storage quota during extraction:**
Four compressed files plus four decompressed files simultaneously requires significant Drive space. At typical seismic data compression ratios, the decompressed files may be 3–5× larger than the compressed versions. Budget approximately 50–100GB total for raw and extracted data. With 2TB available this is not a constraint, but verify available space before starting decompression.

**Risk — SAMP_RATE or COORD_SCALE not constant:**
If Cell 6 reports these as non-constant, do not treat it as a minor issue. Non-constant SAMP_RATE means you cannot safely build a unified time axis for that asset. Flag it in `00_environment_report.json`, note it in `CURRENT_STATUS.md`, and raise it explicitly at the start of the Phase 1 agent session. The Phase 1 agent must address it before proceeding with any analysis.

**Risk — SPARE1 labeling convention differs across assets:**
The task description states that 0 and -1 mean unlabeled. If Cell 6 shows a third sentinel value appearing in any asset (e.g., -999 or 9999), record it. The Phase 1 agent must investigate whether it represents a third unlabeled category or something else before any label statistics are computed.