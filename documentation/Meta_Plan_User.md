# Seismic First Break Picking — User Guide

## What This Project Is

You are building an automated seismic first break picking system using deep
learning. The work is split between you and an AI agent (Codex). Your role
is to set up the environment, run notebooks on Google Colab, and manage the
handoff between sessions. The agent's role is to write all the code and
document all decisions.

This document tells you exactly how to do your part.

---

## One-Time Setup (Do This First, Do It Once)

### Step 1 — Install Google Drive for Desktop

Download from: drive.google.com/drive/download
Sign in with your Google account (the one with 2TB storage).
During setup, choose STREAM FILES mode (not Mirror).

After installation, your Drive appears as a drive letter on your PC
(e.g., G:\My Drive). Open it and confirm you can browse your Drive files.

### Step 2 — Move the Repository Into Drive

Move the entire project folder into your Drive:

```
FROM: F:\Projects\seismic-first-break-picking\
TO:   G:\My Drive\seismic-first-break-picking\
```

Wait for the Drive taskbar icon to show sync complete (no spinner).
This is now your one true repo location. Do not keep a copy on F:.

From this point: every file the agent creates, every notebook output
you save on Colab, every result — all of it lives here and syncs
automatically to your local machine.

### Step 3 — Initialize Git

Open a terminal inside `G:\My Drive\seismic-first-break-picking\` and run:

```bash
git init
git add .
git commit -m "initial commit - all phase plans complete"
git tag v0.0-plan-complete
```

Git gives you version history. If the agent makes a breaking change,
you can revert. Commit at the end of every meaningful session.

### Step 4 — Verify Sync Is Working

Open any file in the repo on your local machine and make a small edit.
Wait 30 seconds. Open Google Drive in your browser and confirm the edit
appears there. Then open the same file path on Colab (after mounting Drive
in the next step) and confirm it matches. If all three match, sync is working.

---

## How Colab Works With Your Drive

When you open a notebook in Colab:

1. Mount Drive by running the first cell (it will contain the mount command)
2. Your repo is accessible at: `/content/drive/MyDrive/seismic-first-break-picking/`
3. Run all cells
4. Save before closing: Ctrl+S or File → Save
5. The saved notebook (with all cell outputs) syncs back to your local machine
   automatically within 30–60 seconds

You never need to manually copy files between Colab and your local machine.
Drive sync handles everything. The only thing you must do is remember to
save the notebook in Colab before closing it.

---

## Your Workflow — Session by Session

### Before Starting Any Session

1. Open `CURRENT_STATUS.md` in the repo root
2. Read "Current Status" — is it BLOCKED, IN PROGRESS, or READY FOR NEXT PHASE?
3. Read "Blocked On" — is there a notebook you need to run first?
4. Read "Next Agent Session Should" — this tells you what to ask the agent to do

If status is BLOCKED and you have not yet run the required notebook, run it
on Colab before starting the agent session. The agent cannot proceed without
those outputs.

### Starting an Agent Session

Open a new Codex chat. Paste the following files in this exact order,
one after another, before asking the agent to do anything:

```
REQUIRED CONTEXT — paste all of these:

1. documentation/Meta_Plan_Agent.md          ← always, every session
2. CURRENT_STATUS.md                          ← always, every session
3. results/00_environment_report.json         ← once Phase 0.5 is done
4. documentation/implementation_phases/
   [most recently completed phase impl doc]   ← always, after Phase 0.5
5. documentation/plan_phases/
   [phase you want implemented next]          ← always
6. Current directory tree of src/ and
   notebooks/ (just the file names/paths)     ← always
```

To get the directory tree quickly, run this in your terminal:
```
tree G:\My Drive\seismic-first-break-picking\src /f
tree G:\My Drive\seismic-first-break-picking\notebooks /f
```

After pasting all context, tell the agent exactly what you want this session
to accomplish. Use the "Next Agent Session Should" field from CURRENT_STATUS.md
as your instruction. Be specific — a good instruction looks like:

  "Implement the SeismicDataset class and BalancedAssetSampler from
   Phase 4 section 4.2. The preprocessing NPZ files are confirmed saved
   to data/processed/. The master_index.csv has 4823 rows. Here are the
   EDA results: [paste eda_summary.json contents]."

A bad instruction looks like: "Continue with Phase 4."

### During a Session

Watch the agent's responses. If it asks you a clarifying question, answer it
precisely. If it asks you to confirm something about the data, check the
relevant output file and paste the relevant section back.

If the agent starts contradicting earlier decisions, losing track of the file
structure, or producing code that ignores constraints it established earlier
in the same session — this is context degradation. See "When to End a Session"
below.

### Ending a Session

Before closing any session where the task is not fully complete, ask the agent:

  "Summarize everything you implemented this session, all decisions made,
   and exactly what remains to complete this task."

Save that summary somewhere (a note, a text file). You will paste it into
the next session as additional context.

Also confirm the agent updated CURRENT_STATUS.md. If it did not, ask it to
do so before you close the chat.

---

## When to Start and End Agent Sessions

### Start a new session when:

- You are beginning a new phase
- A GATE has been cleared (you ran the required notebook and have outputs)
- The previous session ended with BLOCKED status and you have now unblocked it
- The previous session produced an incomplete task and you are resuming it
  (new session, not continuation — paste the summary from the old session)

### End the current session when:

- The agent marks CURRENT_STATUS.md as BLOCKED (it is waiting for you)
- The phase task is fully complete and documented
- You notice the agent contradicting itself or forgetting earlier decisions
  (context degradation — end gracefully, start fresh)
- The session has been very long (40+ exchanges on a complex task) even if
  not obviously degrading — quality drops gradually before it drops sharply

### Signs of context degradation to watch for:

- Agent re-asks questions it already answered earlier in the session
- Agent proposes implementing something it already implemented
- Agent's code ignores a constraint it explicitly stated 10 messages ago
- Agent forgets the file structure and starts creating files in wrong locations
- Agent gives vague answers where it previously gave specific ones

When you see these: ask for the summary, end the session, start fresh.

---

## The Notebook Running Routine

Some notebooks you run once. Some you run multiple times. Here is the full list:

### Run Once:
```
00_environment_setup.ipynb      ← Phase 0.5 gate. Downloads data, verifies HDF5.
02_preprocessing_pipeline.ipynb ← Phase 2 gate. Builds NPZ shot gather files.
04_benchmark_and_compare.ipynb  ← Phase 5. Run after all models are trained.
```

### Run Once Per Asset (4 times total):
```
01_eda_brunswick.ipynb    ← run once, outputs saved to Drive
01_eda_halfmile.ipynb     ← run once, outputs saved to Drive
01_eda_lalor.ipynb        ← run once, outputs saved to Drive
01_eda_sudbury.ipynb      ← run once, outputs saved to Drive
01_eda_combined.ipynb     ← run once AFTER all four individual EDAs complete
```

### Run 4 Times Per Model (progressive asset training):
```
03_train_[model_name].ipynb

Run 1: notebook detects no checkpoint → trains on Brunswick → saves checkpoint
Run 2: notebook detects Brunswick complete → trains on Halfmile → saves checkpoint
Run 3: notebook detects Halfmile complete → trains on Lalor → saves checkpoint
Run 4: notebook detects Lalor complete → trains on Sudbury → saves checkpoint
Run 5: notebook detects all assets complete → skips training → runs final evaluation
```

You do not need to edit anything between runs. The notebook detects its state
automatically from the checkpoint file. Just open it, connect to a GPU runtime,
and run all cells.

### What to report back to the agent after running a notebook:

After running any notebook that is a GATE, paste these into the next agent session:

**After 00_environment_setup.ipynb:**
Paste the full contents of `results/00_environment_report.json`

**After EDA notebooks:**
Paste the full contents of `results/eda/eda_summary.json`
Report any cells that produced errors or unexpected outputs

**After 02_preprocessing_pipeline.ipynb:**
Paste the first 10 and last 10 rows of `data/datasets/master_index.csv`
Report total shot gather counts per asset
Report any validation warnings printed during the run

**After training notebooks:**
Report the final val MAE per asset printed by the notebook
Report which epoch achieved the best result
Report any training errors or warnings
Confirm that `models/checkpoints/[model]/checkpoint_best_overall.pt` exists

---

## Navigating the Repository

### Where to find things:

| What you need | Where it is |
|---|---|
| Phase plans (read-only reference) | `documentation/plan_phases/` |
| Implementation records + decisions | `documentation/implementation_phases/` |
| Current project status + next steps | `CURRENT_STATUS.md` |
| All trained model checkpoints | `models/checkpoints/[model_name]/` |
| MLflow experiment UI data | `mlruns/` (browse via MLflow UI) |
| Benchmark comparison table | `results/benchmark/benchmark_results.csv` |
| EDA plots and outputs | `results/eda_plots/` |
| Example prediction visualizations | `results/visualization_examples/` |
| Config files for each model | `configs/model_[name].yaml` |

### Files you should never manually edit:

```
documentation/plan_phases/*     ← original plans, read-only
data/raw/*                      ← compressed source files, never touch
data/extracted/*                ← decompressed HDF5 files, never touch
mlruns/*                        ← managed by MLflow automatically
```

### Files you should read regularly:

```
CURRENT_STATUS.md               ← read this before every session
documentation/implementation_phases/Impl_Phase_[N].md
                                ← read after each phase completes to understand
                                   what was built and why
```

---

## Viewing MLflow Results

To browse experiment results in the MLflow UI:

1. Open any notebook on Colab
2. Run this code in a cell:

```python
import subprocess, threading, time
def run_mlflow():
    subprocess.run(["mlflow", "ui", "--backend-store-uri",
                    "/content/drive/MyDrive/seismic-first-break-picking/mlruns",
                    "--port", "5000"])
t = threading.Thread(target=run_mlflow, daemon=True)
t.start()
time.sleep(3)
print("MLflow UI starting...")
```

3. Then run this to get a public URL:
```python
from google.colab.output import eval_js
print(eval_js("google.colab.kernel.proxyPort(5000)"))
```

4. Open the printed URL in your browser

This gives you a full UI to compare all training runs, view metrics over time,
and download artifacts. You do not need to write any code to compare models —
it is all in the UI.

---

## Git Commit Routine

Commit at the end of every meaningful working session. Good commit points:

```
After Phase 0.5 completes:    git commit -m "Phase 0.5: environment verified"
After EDA completes:          git commit -m "Phase 1: EDA complete, all 4 assets"
After preprocessing:          git commit -m "Phase 2: preprocessing pipeline"
After each model trains:      git commit -m "Phase 4: unet training complete - MAE 4.2ms"
After benchmarking:           git commit -m "Phase 5: final benchmark complete"
```

Never commit large binary files. Your `.gitignore` handles this but verify
with `git status` before committing — if you see `.hdf5`, `.npz`, or `.pt` files
staged, something is wrong with your gitignore.

---

## Troubleshooting

### "The notebook is retraining from scratch even though I already ran it"

The state machine relies on finding `checkpoint_latest.pt` in the model's
checkpoint directory on Drive. If this file is missing, the notebook starts
over. Check `models/checkpoints/[model_name]/` on Drive. If the file is there
but the notebook does not find it, the Drive path in the config is wrong.
Ask the agent to verify the checkpoint path in `configs/model_[name].yaml`.

### "Drive is not syncing"

Check the Google Drive for Desktop taskbar icon. If it shows an error,
right-click → Quit, then relaunch the app. If a specific large file is stuck,
pause sync, wait 30 seconds, resume. HDF5 files are large and may take several
minutes to sync after first creation.

### "Colab ran out of RAM mid-notebook"

The preprocessing and training notebooks are designed to load data in batches
from Drive rather than all at once. If you hit OOM: check that `cache_in_ram`
is set to `false` in the relevant config YAML. If it is already false, reduce
`batch_size` in the config. Ask the agent to verify the DataLoader is not
accidentally loading the full dataset in a non-obvious place.

### "The agent wrote code but I cannot find it"

The agent always writes files to specific paths defined in the repository
structure above. Check `src/` for module code, `notebooks/` for notebooks,
`configs/` for config files, `scripts/` for utility scripts. If you still
cannot find something, search the implementation document for that phase —
the agent is required to document every file it creates and its location.

### "I want to rerun a phase from scratch"

For EDA: delete the relevant `results/eda/` output files and rerun the notebook.
For preprocessing: delete the `data/processed/[asset]/` NPZ files and rerun.
For training: delete `models/checkpoints/[model]/` and rerun.
For the benchmark: delete `results/benchmark/benchmark_results.csv` and rerun.
Never delete anything in `data/raw/` or `data/extracted/`.

---

## Quick Reference — What To Do Right Now

If you are reading this for the first time, here is your immediate sequence:

```
□ Install Google Drive for Desktop
□ Move repo to G:\My Drive\seismic-first-break-picking\
□ Verify sync is working (edit a file, confirm it appears on Drive)
□ Initialize Git and make first commit
□ Start a Codex session
□ Paste: Meta_Plan_Agent.md + CURRENT_STATUS.md
□ Ask: "Implement Phase 0.5 — create the folder structure script and
        the 00_environment_setup.ipynb notebook"
□ Run 00_environment_setup.ipynb on Colab
□ Verify all four HDF5 files pass structural verification
□ Update CURRENT_STATUS.md if agent did not
□ You are now ready for Phase 1
```
```

