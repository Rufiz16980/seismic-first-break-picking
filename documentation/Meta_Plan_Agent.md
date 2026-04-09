# Seismic First Break Picking — Agent Meta Plan

## Who This Document Is For

This document is for the AI agent (Codex or equivalent) responsible for implementing
the seismic first break picking project. Read this document fully at the start of
every session before touching any code. It defines your role, your constraints, your
interaction protocol with the user, and the complete implementation order.

---

## Project Summary

The goal is to build the best possible automated seismic first break picking model.
Four real-world seismic datasets (Brunswick, Halfmile, Lalor, Sudbury) are provided
as HDF5 files. Each contains thousands of seismic traces manually labeled with first
break arrival times in milliseconds. The task is to train a deep learning model that
predicts these times automatically and generalizes across all four datasets.

The full technical plan is documented across six phase files in:
`documentation/plan_phases/`

You are responsible for implementing those plans in code, documenting every decision
in `documentation/implementation_phases/`, and communicating clearly with the user
about what you have done, what you need from them, and what comes next.

The ultimate goal is not to follow the plan — it is to build the best possible model.
The plan is the minimum. You are expected to exceed it wherever the data and results
justify doing so.

---

## Repository Structure

```
seismic-first-break-picking/
├── configs/                    ← YAML config files, one per model
├── data/
│   ├── raw/                    ← compressed .xz files, never modified
│   ├── extracted/              ← decompressed .hdf5 files
│   ├── processed/              ← per-asset NPZ shot gathers
│   │   ├── brunswick/
│   │   ├── halfmile/
│   │   ├── lalor/
│   │   └── sudbury/
│   └── datasets/               ← final train/val/test splits
│       ├── train/
│       ├── val/
│       └── test/
├── documentation/
│   ├── plan_phases/            ← original phase plans (READ ONLY, never edit)
│   │   ├── Phase_0.md
│   │   ├── Phase_0.5.md
│   │   ├── Phase_1.md
│   │   ├── Phase_2.md
│   │   ├── Phase_3.md
│   │   ├── Phase_4.md          ← includes Phase 4.5 content at the end
│   │   └── Phase_5.md
│   └── implementation_phases/  ← YOUR output, one file per phase
│       ├── Impl_Phase_0.5.md
│       ├── Impl_Phase_1.md
│       └── ...
├── mlruns/                     ← MLflow tracking database (auto-generated)
├── models/
│   ├── checkpoints/            ← per-model checkpoint files
│   └── final/                  ← final selected model
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
├── results/
│   ├── sanity_plots/
│   ├── eda_plots/
│   └── benchmark/
├── scripts/
│   └── create_folder_structure.py
├── src/
│   ├── data/
│   │   ├── hdf5_reader.py
│   │   ├── shot_gather_builder.py
│   │   ├── dataset.py
│   │   └── transforms.py
│   ├── evaluation/
│   │   ├── metrics.py
│   │   └── visualizer.py
│   ├── models/
│   │   ├── unet.py
│   │   ├── resnet_unet.py
│   │   ├── cnn_1d.py
│   │   └── classical.py
│   ├── training/
│   │   ├── trainer.py
│   │   └── losses.py
│   └── utils/
│       ├── config_loader.py
│       └── logger.py
├── CURRENT_STATUS.md           ← your primary handoff document
├── .gitignore
└── requirements.txt
```

---

## Your Core Responsibilities

### 1 — Implement the phase plans

Each phase plan in `documentation/plan_phases/` describes what must be built.
Your job is to implement it in code. The plans are intentionally detailed but
deliberately incomplete in places that depend on data you have not seen yet.
Where the plan says "decide based on EDA results," you make that decision after
seeing the actual outputs and document your reasoning.

### 2 — Document every decision in implementation files

For every phase you implement, create or update the corresponding file in
`documentation/implementation_phases/`. This is not optional and not a summary.
It is a complete technical record. See the Implementation Document Format section
below for the exact required structure.

### 3 — Never block silently

If you cannot proceed because you need the user to run a notebook, confirm an
output, or make a judgment call — say so immediately and update CURRENT_STATUS.md.
Do not write speculative code for things you cannot yet know. Write up to the
blocking point, document what you built, mark yourself BLOCKED, and stop.

### 4 — Extend the plan, never deviate silently

You are expected to extend the plan based on what the data shows. If EDA reveals
something the plan did not anticipate, if a model architecture suggests itself that
was not in Phase 3, if a preprocessing step turns out to be unnecessary — handle it.
But every extension and every deviation must be documented in the implementation file
with explicit reasoning. A deviation that is not documented is a bug in your process.

### 5 — Optimize relentlessly toward the best possible model

The plan defines the minimum acceptable work. Your goal is the best possible model.
If after benchmarking you see an opportunity to improve — better augmentation, a
different loss function, a preprocessing change, an ensemble approach — pursue it,
document it, and implement it. Never stop at "good enough."

---

## Implementation Document Format

Every implementation phase document must begin with this header block, filled in
completely before you write anything else:

```
# Implementation — Phase [N]: [Phase Name]

## Header

| Field | Value |
|---|---|
| STATUS | IN PROGRESS / COMPLETE / BLOCKED |
| BLOCKED_BY | (what is preventing progress, or "N/A") |
| PHASE_PLAN_REF | documentation/plan_phases/Phase_N.md |
| LAST_UPDATED | (date of last update) |
| IMPLEMENTING_AGENT | (note "Agent session [number]" for tracking) |

## Decisions Made
(bulleted list — every non-trivial implementation choice and why it was made)

## Deviations From Plan
(any departure from the phase plan document, with explicit justification)
(if none: write "None")

## Data Findings
(for EDA phases: everything discovered about the data that was not known before)
(for other phases: any data-related surprises encountered during implementation)

## Open Questions
(things flagged but not resolvable in this session — carry these forward)

## Next Agent Must Know
(critical context for whoever continues this phase or starts the next one)
(assume the next agent has zero memory of this session)

---

## Implementation Details

[Full technical content below this line]
```

The implementation document must be MORE detailed than the original plan, not less.
The plan describes intentions. The implementation document describes reality — exact
shapes, actual values, real decisions, actual results. If the plan says "check if
SAMP_RATE is constant" your implementation document says "SAMP_RATE is 250µs across
all 45,231 traces in Brunswick, confirmed constant, stored as scalar in config."

---

## Phase Implementation Order

The phases have dependencies. Some are strictly sequential. Others can be worked on
in parallel. Follow this order:

```
STRICTLY SEQUENTIAL (cannot parallelize):
────────────────────────────────────────
Phase 0     — Environment & Repository Architecture
    ↓
Phase 0.5   — Data Acquisition & Environment Verification
    ↓ [USER MUST RUN 00_environment_setup.ipynb]
Phase 1     — EDA (all four assets + combined)
    ↓ [USER MUST RUN all five EDA notebooks]
Phase 2     — Preprocessing Pipeline
    ↓ [USER MUST RUN 02_preprocessing_pipeline.ipynb]
Phase 4     — Training Infrastructure (DataLoader, trainer, MLflow, state machine)
    ↓
Phase 3/4   — Model training (one model family per agent session)
    ↓ [USER MUST RUN each training notebook 4 times per model]
Phase 5     — Benchmarking, Error Analysis, Final Selection


CAN BE DONE AHEAD OR IN PARALLEL:
──────────────────────────────────
Phase 3 documentation (reading prior art, cataloging models) → during Phase 1
Phase 4 infrastructure code (DataLoader, transforms, trainer) → during Phase 2
Phase 5 notebook scaffolding → during Phase 3/4 training
```

Phase 4 in this repo includes all content from the original Phase 4 AND Phase 4.5
documents. When implementing Phase 4, read BOTH sections fully before writing any
code. They describe one unified system, not two separate ones.

---

## Bottleneck Gates

These are hard stops where you cannot proceed without user action. When you reach
one, update CURRENT_STATUS.md, clearly state what the user must do, and stop.

```
GATE 1 — After Phase 0.5 implementation:
  User must run: 00_environment_setup.ipynb (full run, all cells)
  You need back: 00_environment_report.json contents

GATE 2 — After Phase 1 implementation:
  User must run: 01_eda_brunswick.ipynb
                 01_eda_halfmile.ipynb
                 01_eda_lalor.ipynb
                 01_eda_sudbury.ipynb
                 01_eda_combined.ipynb
  You need back: all EDA outputs + eda_summary.json

GATE 3 — After Phase 2 implementation:
  User must run: 02_preprocessing_pipeline.ipynb
  You need back: confirmation that NPZ files saved correctly,
                 master_index.csv row counts per asset,
                 any errors or warnings from the run

GATE 4 — After each training notebook implementation:
  User must run: the training notebook 4 times (once per asset)
  You need back: final val MAE per asset, any training errors,
                 confirmation that checkpoint_best_overall.pt saved

GATE 5 — After all training notebooks complete:
  User must run: 04_benchmark_and_compare.ipynb
  You need back: benchmark_results.csv contents
```

At every gate, be precise about what you need back from the user.
Do not ask for vague confirmation. Ask for specific file contents or
specific numbers. The user can paste these directly into the next session.

---

## Rerun-Safe Notebook Requirements

Every notebook you write must be safe to rerun from top to bottom with no
manual edits between runs. This is non-negotiable. The user will sometimes
rerun notebooks by accident, will rerun them intentionally to resume after
a crashed session, and must never have to edit a cell to do so.

For every notebook, verify these properties before considering it complete:

**Idempotency:** Running a cell twice produces the same result as running it
once. File creation checks for existence before creating. Downloads check
for existing files before downloading. Preprocessing checks for existing NPZ
files before reprocessing.

**State detection:** At the top of every training notebook, the state machine
detects the current training state from the checkpoint on Drive and prints
what it found. The user sees exactly what will happen before any computation
begins.

**Graceful resume:** If a session died mid-training, the next run resumes from
the last saved checkpoint automatically. No data is lost. No training is
needlessly repeated.

**No hardcoded paths:** Every path comes from the config file. The only
hardcoded string in any notebook is the path to the config file itself, which
is always relative to the Drive mount point.

**Explicit skip messages:** When a step is skipped because it already completed,
print a clear message: "Skipping download — brunswick.hdf5.xz already exists
(12.3 GB)." Never skip silently.

---

## MLflow Integration Requirements

Every training run must log to MLflow. The tracking URI always points to
`mlruns/` inside the Drive repo folder. This is set in `configs/datasets.yaml`
and read by every notebook — never hardcoded.

At minimum, every run logs:

**Parameters (at run start):**
All config fields flattened to a dict. Architecture name, framing, asset(s)
used, all hyperparameters, augmentation toggles, normalization scheme,
device name, VRAM available, PyTorch version, random seed.

**Metrics (per epoch):**
train_loss, val_loss, val_mae_ms, val_within_5ms_pct, learning_rate,
gradient_norm, gpu_memory_mb. Per-asset val MAE when training on combined.

**Artifacts (at run end):**
Config YAML, best checkpoint file, training curve plots, example output
composites (12 fixed examples, 3 per asset), benchmark row CSV.

---

## Device Management Requirements

Every notebook detects its device at startup and adapts automatically.
Never hardcode batch sizes, never assume CUDA is available, never crash
on CPU. The device detection block from Phase 4.5.3 runs in Cell 1 of
every training notebook and its output is logged to MLflow.

For TPU sessions: use torch_xla throughout. Checkpoint saving uses xm.save().
Optimizer steps use xm.optimizer_step(). Never mix CUDA and XLA calls.

---

## The CURRENT_STATUS.md Contract

This file is your primary communication channel with the user.
Update it at the end of every session without exception.
Its format is fixed:

```markdown
# Current Project Status

**Last Updated:** [date and time]
**Last Agent Session:** [brief description of what was worked on]

## Completed
[bulleted list of everything fully complete across all phases]

## Just Completed This Session
[bulleted list of what this specific session accomplished]

## Current Status
[one of: IN PROGRESS / BLOCKED / READY FOR NEXT PHASE]

## Blocked On (if applicable)
[exact description of what the user must do]
[exact notebook(s) to run]
[exact information to paste into the next agent session]

## Next Agent Session Should
[clear description of the immediate next task]
[which files the next agent must read before starting]

## Open Questions Requiring User Decision
[any architectural or data decisions that require human judgment]
```

This file must never be vague. "User should run the notebook" is not acceptable.
"User must run notebooks/01_eda_brunswick.ipynb on Colab, run all cells,
confirm the final cell prints PHASE 1 EDA COMPLETE, then paste the contents
of results/eda/eda_summary.json into the next agent session" is acceptable.

---

## Context Injection for New Sessions

When you are a new agent beginning a session, you need specific context to
work effectively. The user will provide this to you according to their routing
guide. When you receive it, read each piece in this order before responding:

1. This Meta_Plan_Agent.md file (you are reading it now)
2. CURRENT_STATUS.md (tells you where things stand)
3. 00_environment_report.json (if Phase 0.5 is complete — dataset facts)
4. The implementation document for the most recently completed phase
5. The phase plan document for the phase you are about to implement
6. The current src/ and notebooks/ directory tree

Do not begin implementing until you have confirmed you understand all six.
If any of these files were not provided, ask for them before proceeding.
State explicitly what you received and what you understood from each.

---

## Code Quality Standards

**No magic numbers:** Every threshold, every size, every rate comes from the
config file or is derived from the data at runtime. Nothing is hardcoded in
model or training code.

**Every function has a docstring:** State input shapes, output shapes, and
any non-obvious behavior. Shape comments in code are mandatory for tensor
operations: `# [batch, channels, n_samples, n_traces]`

**Fail loudly:** Use assertions for shape checks, data type checks, and
value range checks at boundaries between components. A bad tensor shape
caught at the DataLoader boundary saves hours of debugging a training crash.

**One responsibility per function:** Functions that download AND decompress
AND verify are three functions called in sequence, not one function that
does all three. This makes debugging and rerunning individual steps possible.

**Imports at the top:** No inline imports inside functions except for
optional heavy dependencies (torch_xla, obspy) that may not be installed.
Those get try/except import blocks at the top of the file.

---

## The Visualization Examples Contract

At the start of Phase 1, before any model is trained, select and save 12
fixed visualization examples to `results/visualization_examples.json`.
Three per asset: one easy (high SNR), one medium, one hard (low SNR).
These same 12 examples are used by every model for output visualization.
Never change them after they are set. Consistency across models is the point.

Every training notebook generates these 12 composite output plots at run end
and logs them as MLflow artifacts. The composite has four panels per example:
raw gather, ground truth overlay, prediction overlay, per-trace error profile.

---

## Prior Art Research Requirement

Before implementing any model in Phase 3/4, search for published work on
these exact four datasets. The Brunswick, Halfmile, Lalor, and Sudbury 3D
datasets form a known seismic first break picking benchmark. Search:

- Google Scholar: "seismic first break picking Brunswick Halfmile deep learning"
- arXiv: "first break picking HDF5 mining seismic"
- GitHub: "FBP benchmark seismic"
- SEG Annual Meeting proceedings 2019-2023

For each paper found, record in the implementation document:
architecture used, preprocessing choices, reported MAE per asset, whether
code was released. Published MAE scores are your performance targets.
If you cannot find any prior work, document that search was conducted and
note it as an open question for the user to investigate independently.

---

## Catastrophic Forgetting Mitigation (Progressive Training)

The progressive asset training pattern (train on Asset 1, fine-tune on
Asset 2, etc.) risks catastrophic forgetting. The default mitigation is
a replay buffer: when training on Asset N, include 15% of training data
from all previous assets in each batch. This is implemented in the
BalancedAssetSampler and controlled by `replay_fraction` in the config.

If after cross-asset evaluation (Phase 5.3.5) forgetting is still detected
(large MAE gap between in-distribution and leave-one-out performance),
escalate to Elastic Weight Consolidation. Document the decision and
implement it as an optional loss term controlled by a config flag.

---

## Final Reminder

The plan phases are the floor. Your job is to reach the ceiling.
Every decision you make that is not explicitly covered by the plan is an
opportunity to improve the final model. Use those opportunities.
Document every one of them.
```

-




