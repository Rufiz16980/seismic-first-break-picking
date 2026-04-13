import os
import nbformat as nbf

MODELS = {
    "classical": {"config": "model_classical.yaml", "framing": "trace"},
    "tabular_lgbm": {"config": "model_tabular_lgbm.yaml", "framing": "trace"},
    "cnn1d": {"config": "model_cnn1d.yaml", "framing": "trace"},
    "resnet1d": {"config": "model_resnet1d.yaml", "framing": "trace"},
    "tcn": {"config": "model_tcn.yaml", "framing": "trace"},
    "bilstm": {"config": "model_bilstm.yaml", "framing": "trace"},
    "unet1d": {"config": "model_unet1d.yaml", "framing": "trace"},
    "transformer1d": {"config": "model_transformer1d.yaml", "framing": "trace"},
    "cnn2d": {"config": "model_cnn2d.yaml", "framing": "gather"},
    "resnet_unet": {"config": "model_resnet_unet.yaml", "framing": "gather"},
    "unet": {"config": "model_unet_softargmax.yaml", "framing": "gather"} 
}

def generate_notebook(model_key, spec):
    nb = nbf.v4.new_notebook()
    cells = []
    
    # 1. Environment
    cells.append(nbf.v4.new_markdown_cell(f"# Phase 4: Training Pipeline ({model_key})"))
    cells.append(nbf.v4.new_code_cell("""\
# Cell 1: Environment Setup & Hardware Detection
import os
import sys

try:
    from google.colab import drive
    drive.mount('/content/drive', force_remount=False)
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

def find_project_root(indicator_file="configs/datasets.yaml"):
    # Dynamically find the project root relative to the notebook location.
    curr = os.getcwd()
    # Check current and up to 4 parent levels (Notebooks are usually in notebooks/)
    for _ in range(5):
        if os.path.exists(os.path.join(curr, indicator_file)):
            return curr
        curr = os.path.dirname(curr)
    # Fallback for local dev or standard Colab location
    return '/content/drive/MyDrive/seismic-first-break-picking'

PROJECT_ROOT = find_project_root()

if IN_COLAB:
    # Ensure all required packages are installed in the fresh Colab runtime
    !pip install -q mlflow optuna lightgbm segmentation-models-pytorch pyyaml

if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

import torch
import numpy as np
import random
import matplotlib.pyplot as plt

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
device = torch.device(device_type)
device_name = torch.cuda.get_device_name(0) if device_type == 'cuda' else 'cpu'
vram_bytes = torch.cuda.get_device_properties(0).total_memory if device_type == 'cuda' else 0

print(f"Device: {device_name}")
print(f"VRAM: {vram_bytes / 1e9:.2f} GB" if vram_bytes > 0 else "")
"""))

    # 2. Config & Tracker Setup
    cells.append(nbf.v4.new_code_cell(f"""\
# Cell 2: Config & MLflow Setup
import mlflow
import types
from src.utils.config_loader import load_model_config
from src.training.mlflow_logger import MLFlowLogger

config_path = os.path.join(PROJECT_ROOT, 'configs', '{spec["config"]}')
config = load_model_config(config_path, as_namespace=True)

# Defensive fallback: guarantee config.output exists with all required keys
if not hasattr(config, 'output'):
    config.output = types.SimpleNamespace()
if not hasattr(config.output, 'tracking_uri'):
    config.output.tracking_uri = f'file:///{{PROJECT_ROOT}}/mlruns'
if not hasattr(config.output, 'checkpoint_dir'):
    config.output.checkpoint_dir = f'models/{model_key}'

# CRITICAL: Resolve checkpoint_dir to an absolute path inside Google Drive.
# If it is a relative path (e.g. 'models/cnn1d'), Colab resolves it against
# CWD=/content (ephemeral disk) — models would be LOST when the session ends.
# Prepend PROJECT_ROOT so all checkpoints land on Drive unconditionally.
if not os.path.isabs(config.output.checkpoint_dir):
    config.output.checkpoint_dir = os.path.join(PROJECT_ROOT, config.output.checkpoint_dir)
os.makedirs(config.output.checkpoint_dir, exist_ok=True)
print(f"Checkpoint dir (absolute): {{config.output.checkpoint_dir}}")

# Terminate any active MLflow run from a previous cell execution (rerun safety)
mlflow.end_run()

# --- Resume-safe MLflow init ---
# _save_checkpoint writes mlflow_run_id.txt on every epoch save.
# If the file exists, we rejoin that run so all epochs appear on one timeline.
# If the run was FINISHED (or the file is stale) resume_run() transparently
# falls back to a fresh run, so re-running a completed notebook is fully safe.
_run_id_file = os.path.join(config.output.checkpoint_dir, 'mlflow_run_id.txt')
_existing_run_id = None
if os.path.exists(_run_id_file):
    with open(_run_id_file) as _f:
        _existing_run_id = _f.read().strip() or None

logger = MLFlowLogger(config.output.tracking_uri, config.experiment.name)
if _existing_run_id:
    logger.resume_run(_existing_run_id)
    print(f"Resumed MLflow run: {{_existing_run_id}}")
else:
    logger.start_run()

logger.log_params(config)
print(f"Loaded config for: {{config.experiment.name}}")
print(f"Checkpoint dir: {{config.output.checkpoint_dir}}")
"""))

    # 3. Training State Detection
    cells.append(nbf.v4.new_code_cell("""\
# Cell 3: Checkpoint State Detection
import os
import torch

# Trainer saves as '{experiment_name}_latest.pt' — match that filename exactly.
checkpoint_path = os.path.join(config.output.checkpoint_dir, f'{config.experiment.name}_latest.pt')
is_progressive = hasattr(config, 'progressive_training') and getattr(config.training, 'training_mode', 'combined') == 'progressive'

if not os.path.exists(checkpoint_path):
    state = 'NO_CHECKPOINT_EXISTS'
    start_asset_index = 0
    resume_epoch = 0
    completed_assets = []
else:
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    training_state = checkpoint.get('training_state', {})
    resume_epoch = checkpoint.get('epoch', 0)
    
    if is_progressive:
        if training_state.get('is_fully_trained', False):
            state = 'TRAINING_COMPLETE'
            completed_assets = config.progressive_training.asset_order
            start_asset_index = len(completed_assets)
        else:
            completed_assets = training_state.get('completed_assets', [])
            start_asset_index = len(completed_assets)
            state = f"ASSET_{start_asset_index}_COMPLETE" if completed_assets else "NO_CHECKPOINT_EXISTS"
    else:
        # Combined Training (Default)
        completed_assets = []
        start_asset_index = 0
        state = 'RESUMING_COMBINED_TRAINING'

print(f"Detected state: {state}")
print(f"Resuming from epoch: {resume_epoch}")
"""))

    # 4. Collate function setup
    collate = "trace_collate_fn" if spec["framing"] == "trace" else "lambda x: variable_width_collate_fn(x, width_divisibility=16)"
    
    cells.append(nbf.v4.new_code_cell(f"""\
# Cell 4: Data & Sampler Setup
from torch.utils.data import DataLoader
from src.data.dataset import ShotGatherDataset, ProgressiveAssetSampler, variable_width_collate_fn, trace_collate_fn
from src.data.transforms import build_transforms
from src.utils.config_loader import load_yaml

csv_path = config.data.index_csv if os.path.isabs(config.data.index_csv) else os.path.join(PROJECT_ROOT, config.data.index_csv)

# Load augmentation config and build training transforms
preproc_path = os.path.join(PROJECT_ROOT, 'configs', 'preprocessing.yaml')
preproc_cfg = load_yaml(preproc_path)
train_transform = build_transforms(preproc_cfg.get('augmentation', {{}}), is_training=True)
print(f"Train augmentations: {{train_transform}}")

train_ds = ShotGatherDataset(csv_path, split='train', transform=train_transform)
val_ds = ShotGatherDataset(csv_path, split='val')  # No augmentation for validation

val_loader = DataLoader(
    val_ds, batch_size=config.training.batch_size, shuffle=False,
    collate_fn={collate}, num_workers=2, pin_memory=True
)

train_loader = DataLoader(
    train_ds, batch_size=config.training.batch_size, shuffle=True,
    collate_fn={collate}, num_workers=2, pin_memory=True
)
print(f"Train samples: {{len(train_ds)}} | Val samples: {{len(val_ds)}}")
"""))

    # 5. Model Injection
    model_impl = ""
    if model_key == "classical":
        model_impl = """from src.models.classical import STALTAPicker\nmodel = STALTAPicker()\nprint("Classical model Ready")"""
    elif model_key == "tabular_lgbm":
        model_impl = """from src.models.tabular import LightGBMWrapper\nmodel = LightGBMWrapper()\nprint("LightGBM Ready")"""
    elif model_key == "cnn1d":
        model_impl = """from src.models.cnn_1d import Conv1DRegressor\nmodel = Conv1DRegressor(base_channels=getattr(config.model, 'base_channels', 32)).to(device)\nprint(f"Parameters: {model.count_parameters()}")"""
    elif model_key == "resnet1d":
        model_impl = """from src.models.cnn_1d import ResNet1DRegressor\nmodel = ResNet1DRegressor(base_channels=getattr(config.model, 'base_channels', 32)).to(device)\nprint(f"Parameters: {model.count_parameters()}")"""
    elif model_key == "tcn":
        model_impl = """from src.models.cnn_1d import TCNRegressor\nmodel = TCNRegressor().to(device)\nprint(f"Parameters: {model.count_parameters()}")"""
    elif model_key == "bilstm":
        model_impl = """from src.models.rnn import BiLSTMRegressor\nmodel = BiLSTMRegressor(hidden_size=getattr(config.model, 'hidden_size', 64)).to(device)\nprint(f"Parameters: {model.count_parameters()}")"""
    elif model_key == "cnn2d":
        model_impl = """from src.models.cnn_2d import Standard2DCNN\nmodel = Standard2DCNN(base_channels=getattr(config.model, 'base_channels', 32)).to(device)\nprint(f"Parameters: {model.count_parameters()}")"""
    elif model_key == "resnet_unet":
        model_impl = """from src.models.unet import ResNetUNet\nmodel = ResNetUNet(encoder_name=getattr(config.model, 'encoder_name', 'resnet34'), encoder_weights=getattr(config.model, 'encoder_weights', 'imagenet')).to(device)\nprint(f"Parameters: {model.count_parameters()}")"""
    elif model_key == "unet":
        model_impl = """from src.models.unet import SoftArgmaxUNet\nmodel = SoftArgmaxUNet(base_channels=getattr(config.model, 'base_channels', 64)).to(device)\nprint(f"Parameters: {model.count_parameters()}")"""
    elif model_key == "unet1d":
        model_impl = """from src.models.unet_1d import SoftArgmax1DUNet\nmodel = SoftArgmax1DUNet(base_channels=getattr(config.model, 'base_channels', 32)).to(device)\nprint(f"Parameters: {model.count_parameters()}")"""
    elif model_key == "transformer1d":
        model_impl = """from src.models.transformer import TransformerRegressor\nmodel = TransformerRegressor(embed_dim=getattr(config.model, 'embed_dim', 128), num_heads=getattr(config.model, 'num_heads', 8), num_layers=getattr(config.model, 'num_layers', 4)).to(device)\nprint(f"Parameters: {model.count_parameters()}")"""

    cells.append(nbf.v4.new_code_cell(f"# Cell 5: Model Map\n{model_impl}"))

    # 6. Optuna Module for Hyperparameter Sweeping
    if "tabular" not in model_key and "classical" not in model_key:
        cells.append(nbf.v4.new_code_cell("""\
# Cell 6: Phase 4.7 Optuna Sweeper (Optional: Run ONLY on final best architecture)
try:
    import optuna
    from optuna.pruners import MedianPruner
except ImportError:
    optuna = None
    print("Install optuna for sweeps.")

def run_optuna_sweep(n_trials=30):
    if optuna is None: return
    
    print("WARNING: Do not run Optuna sweeps on all models. Compute limits apply.")
    print("Execute this function manually only on the singular best architecture.")
    
    def objective(trial):
        lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
        bs = trial.suggest_categorical("batch_size", [4, 8, 16] if '2d' in model_key else [32, 64, 128])
        # Insert discrete model instantiations + train epochs.
        # Report intermediate validation MAE to the pruner at each epoch:
        # trial.report(val_mae, epoch)
        # if trial.should_prune():
        #     raise optuna.TrialPruned()
        return 0.0 # Return absolute Val MAE here.
        
    study = optuna.create_study(direction="minimize", pruner=MedianPruner(n_warmup_steps=10))
    study.optimize(objective, n_trials=n_trials)
    print("Best params:", study.best_params)
"""))

    # 7. Training loop (DL) or tabular fitting (Tabular) — classical has nothing here
    if "tabular" in model_key:
        fit_code = """\
# Cell 7: LightGBM Fitting on Train Set
import numpy as np
from src.features.features import extract_features

print("Fitting LightGBM on train_loader features...")
all_features, all_labels = [], []

for batch in train_loader:
    traces = batch['traces'].squeeze(1).numpy()
    labels = batch['labels_ms'].numpy()
    mask = batch['valid_mask'].numpy()  # always present: trace_collate emits it
    offsets = batch.get('offsets', np.zeros(traces.shape[0]))
    if mask.any():
        f = extract_features(traces, offsets)
        valid_rows = mask.astype(bool) if mask.ndim == 1 else mask.any(axis=1)
        all_features.append(f[valid_rows])
        all_labels.extend(labels[valid_rows].tolist())

if all_features:
    X = np.vstack(all_features)
    y = np.array(all_labels)
    print(f"Fitting on {{X.shape[0]}} traces with {{X.shape[1]}} features...")
    model.model.fit(X, y)
    print("LightGBM fitting complete!")
    
    # Persist to Google Drive (survives session resets)
    import joblib
    lgbm_save_path = os.path.join(config.output.checkpoint_dir, 'lgbm_model.pkl')
    os.makedirs(config.output.checkpoint_dir, exist_ok=True)
    joblib.dump(model.model, lgbm_save_path)
    print(f"Model saved to Drive: {{lgbm_save_path}}")
    logger.log_artifact(lgbm_save_path)
else:
    print("WARNING: No valid training traces found.")
"""
        cells.append(nbf.v4.new_code_cell(fit_code))
    elif "classical" not in model_key:
        loop_code = f"""\
# Cell 7: Execute Progressive Loop
from src.models.loss import MaskedMAELoss
from src.training.trainer import Trainer

criterion = MaskedMAELoss()

if state == 'TRAINING_COMPLETE':
    print("Skipping to evaluation")
else:
    # We iterate over the sequence of assets dynamically
    # For Combined mode, asset_order just has 'all'
    is_progressive = hasattr(config, 'progressive_training') and getattr(config.training, 'training_mode', 'combined') == 'progressive'
    
    if not is_progressive:
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.training.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.training.epochs)
        trainer = Trainer(model, optimizer, criterion, config, device, scheduler, logger)
        # Resume from Drive checkpoint if one was detected in Cell 3
        if state == 'RESUMING_COMBINED_TRAINING' and os.path.isfile(checkpoint_path):
            trainer.load_checkpoint(checkpoint_path)
            print(f"Resumed from checkpoint at epoch {{trainer.start_epoch - 1}}")
        trainer.run(train_loader, val_loader)
    else:
        # Loop Progressive Assets
        for i in range(start_asset_index, len(config.progressive_training.asset_order)):
            current_asset = config.progressive_training.asset_order[i]
            print(f"--- Training Phase: {{current_asset}} ---")
            
            p_sampler = ProgressiveAssetSampler(
                train_ds, 
                current_asset=current_asset, 
                previous_assets=config.progressive_training.asset_order[:i],
                replay_fraction=config.progressive_training.replay_fraction
            )
            
            p_loader = DataLoader(
                train_ds, batch_size=config.training.batch_size, sampler=p_sampler,
                collate_fn={collate}, num_workers=2, pin_memory=True
            )
            
            optimizer = torch.optim.AdamW(model.parameters(), lr=config.progressive_training.asset_lr[i])
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.progressive_training.asset_epochs[i])
            
            trainer = Trainer(model, optimizer, criterion, config, device, scheduler, logger)
            
            # On the first resumable asset, reload model weights from the checkpoint.
            # (The optimizer is intentionally NOT restored — each asset starts a fresh
            # optimiser phase with its own LR.  Only the learned weights carry over.)
            if i == start_asset_index and resume_epoch > 0 and os.path.isfile(checkpoint_path):
                _ckpt = torch.load(checkpoint_path, map_location=device)
                model.load_state_dict(_ckpt['model_state_dict'])
                print(f"  Loaded model weights from checkpoint (epoch {{resume_epoch}})")
            
            trainer.run(p_loader, val_loader,
                        start_epoch=resume_epoch,
                        total_epochs=config.progressive_training.asset_epochs[i])
            
            # Mark this asset complete inside the trainer so _save_checkpoint
            # persists the completed list to disk — prevents progressive amnesia
            # if the session dies before all assets finish.
            trainer.training_state = {{
                'completed_assets': config.progressive_training.asset_order[:i + 1],
                'is_fully_trained': (i + 1) == len(config.progressive_training.asset_order),
            }}
            trainer._save_checkpoint(
                config.progressive_training.asset_epochs[i], False,
                f"{{config.experiment.name}}_latest.pt"
            )
            
            # Reset resume so next asset starts at epoch 1
            resume_epoch = 0
"""
        cells.append(nbf.v4.new_code_cell(loop_code))
        
    # 8. Eval & Plotting Check
    cells.append(nbf.v4.new_code_cell(f"""\
# Cell 8: Universal Evaluation (all model tiers)
# Force-purge ALL src.* modules from Colab's module cache so that any fixes
# pushed to Drive (loss.py, trainer.py, dataset.py, evaluator.py, etc.) are
# reloaded fresh. Without this, Colab reuses stale bytecode from the session's
# first import, silently ignoring edits on Drive.
import importlib, sys
_stale = [k for k in sys.modules if k.startswith('src.')]
for _mod_name in _stale:
    del sys.modules[_mod_name]
print(f"Cache cleared: {{len(_stale)}} src.* modules evicted.")

from src.training.evaluator import ModelEvaluator

_is_dl = '{model_key}' not in ('classical', 'tabular_lgbm')
_history = trainer.history if 'trainer' in dir() and hasattr(trainer, 'history') else {{}}

evaluator = ModelEvaluator(
    model=model,
    val_loader=val_loader,
    logger=logger,
    device=device,
    model_key='{model_key}',
    is_dl=_is_dl,
    history=_history,
)
final_metrics = evaluator.run()

# Close MLflow run cleanly
import mlflow
mlflow.end_run()
print("MLflow run closed.")
"""))

    nb['cells'] = cells
    
    out_dir = os.path.join(os.path.dirname(__file__), '..', 'notebooks')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f'03_train_{model_key}.ipynb')
    
    with open(out_path, 'w') as f:
        nbf.write(nb, f)
    print(f"Generated {out_path}")

if __name__ == "__main__":
    for key, spec in MODELS.items():
        generate_notebook(key, spec)
    print("All notebooks created successfully!")
