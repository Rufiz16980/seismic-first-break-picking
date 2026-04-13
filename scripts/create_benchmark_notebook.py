"""
Generator for the full Benchmark & Leaderboard Notebook (v5 - Val, Test, Latency & CSVs).

Produces 04_benchmark_and_compare.ipynb with:
  - Checkpoint metadata table
  - Training curves
  - Per-model inference on CPU measuring true Inference Latency
  - Test-set Scatter plots & Error histograms
  - Leaderboards exported as CSV
"""

import os
import nbformat as nbf

def cell(src): return nbf.v4.new_code_cell(src)
def md(src): return nbf.v4.new_markdown_cell(src)

CELL_SETUP = """\
# ── Cell 1: Environment Setup ──────────────────────────────────────────────
import os, sys, gc, time
import warnings
warnings.filterwarnings('ignore')

try:
    from google.colab import drive
    drive.mount('/content/drive', force_remount=False)
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

def find_project_root():
    curr = os.getcwd()
    for _ in range(5):
        if os.path.exists(os.path.join(curr, 'configs/datasets.yaml')):
            return curr
        curr = os.path.dirname(curr)
    return '/content/drive/MyDrive/seismic-first-break-picking'

PROJECT_ROOT = find_project_root()
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

if IN_COLAB:
    import subprocess
    # Only install packages NOT already bundled in Colab — reinstalling torch/numpy
    # causes a large temporary RAM spike that kills the session on free T4.
    subprocess.run(['pip', 'install', '-q', 'mlflow', 'lightgbm', 'joblib',
                    'segmentation-models-pytorch'], check=True)

import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

ARTIFACTS_DIR = os.path.join(PROJECT_ROOT, 'artifacts', 'plots')
os.makedirs(ARTIFACTS_DIR, exist_ok=True)
os.makedirs(os.path.join(ARTIFACTS_DIR, 'leaderboard'), exist_ok=True)

# Swap to 'cuda' here if running on a safe, dedicated instance
DEVICE = torch.device('cpu')
print(f'Project root : {PROJECT_ROOT}')
print(f'Artifacts dir: {ARTIFACTS_DIR}')
print('Environment ready.')
"""

CELL_METADATA = """\
# ── Cell 2: Checkpoint Metadata (no inference) ─────────────────────────────
MODEL_REGISTRY = {
    'CNN-1D':      ('models/cnn1d',           'cnn1d_standard'),
    'ResNet-1D':   ('models/resnet1d',        'resnet1d_deep'),
    'UNet-1D':     ('models/unet1d',          'unet1d_standard'),
    'UNet-2D':     ('models/unet_softargmax', 'unet_softargmax'),
    'ResNet-UNet': ('models/resnet_unet',     'resnet_unet'),
    'LightGBM':    ('models/tabular_lgbm',    'tabular_lgbm'),
    'Classical':   ('models/classical',       'classical_baselines'),
}

def find_checkpoint(ckpt_dir, suffix='_best.pt'):
    if not os.path.isdir(ckpt_dir): return None
    for fname in os.listdir(ckpt_dir):
        if fname.endswith(suffix): return os.path.join(ckpt_dir, fname)
    if suffix == '_best.pt': return find_checkpoint(ckpt_dir, '_latest.pt')
    return None

records = []
for name, (rel_dir, _) in MODEL_REGISTRY.items():
    ckpt_dir = os.path.join(PROJECT_ROOT, rel_dir)
    ckpt = find_checkpoint(ckpt_dir)
    row = {'Model': name, 'Status': 'Not started', 'Epoch': None, 'Best Val MAE (ms)': None}
    if ckpt:
        try:
            d = torch.load(ckpt, map_location='cpu', weights_only=False)
            row['Epoch'] = d.get('epoch')
            val = d.get('best_val_mae')
            row['Best Val MAE (ms)'] = round(float(val), 3) if val is not None else None
            row['Status'] = 'Finished' if d.get('is_fully_trained') else f"Checkpoint @ ep {d.get('epoch','?')}"
        except Exception as e:
            row['Status'] = f'Read error: {e}'
    else:
        pkl_dir = os.path.join(PROJECT_ROOT, rel_dir)
        if os.path.isdir(pkl_dir) and any(f.endswith('.pkl') for f in os.listdir(pkl_dir)):
            row['Status'] = 'Trained (pkl)'
    records.append(row)

meta_df = pd.DataFrame(records)
print(meta_df.to_string(index=False))
"""

CELL_TRAINING_CURVES = """\
# ── Cell 3: Training Curves ────────────────────────────────────────────────
def save_fig(fig, model_name, fname):
    out_dir = os.path.join(ARTIFACTS_DIR, model_name.replace(' ', '_').replace('-', '_'))
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, fname)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return path

for name, (rel_dir, _) in MODEL_REGISTRY.items():
    ckpt_dir = os.path.join(PROJECT_ROOT, rel_dir)
    ckpt = find_checkpoint(ckpt_dir)
    if not ckpt: continue
    try: d = torch.load(ckpt, map_location='cpu', weights_only=False)
    except: continue

    history = d.get('history', {})
    val_mae = history.get('val_mae', [])
    train_mae= history.get('train_mae', [])

    if not val_mae and not train_mae: continue

    fig, ax = plt.subplots(figsize=(8, 4))
    epochs = range(1, len(val_mae) + 1)
    if train_mae:
        ax.plot(range(1, len(train_mae)+1), train_mae, color='steelblue', lw=2, linestyle='--', label='Train MAE')
    if val_mae:
        ax.plot(epochs, val_mae, color='tomato', lw=2, marker='o', markersize=3, label='Val MAE')
    if d.get('best_val_mae'):
        ax.axhline(d['best_val_mae'], color='green', lw=1.2, linestyle=':', label=f"Best Val MAE = {d['best_val_mae']:.2f} ms")
    ax.set_title(f'{name} — Training Curves', fontsize=14, fontweight='bold')
    ax.legend(); ax.grid(True, linestyle='--', alpha=0.5)
    fig.tight_layout()
    save_fig(fig, name, 'training_curve.png')

print('Training curves saved.')
"""

CELL_DATALOADER = """\
# ── Cell 4: Validation & Test DataLoaders ──────────────────────────────────
from torch.utils.data import DataLoader
from src.data.dataset import ShotGatherDataset, trace_collate_fn, variable_width_collate_fn

csv_path = os.path.join(PROJECT_ROOT, 'data/processed/split_index.csv')

val_ds = ShotGatherDataset(csv_path, split='val')
# batch_size=8 for trace models: safe on both CPU and GPU runtimes.
# batch_size=2 for gather models: wide gathers (e.g. Brunswick W=4090) are large.
trace_val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, collate_fn=trace_collate_fn, num_workers=0)
gather_val_loader = DataLoader(val_ds, batch_size=2, shuffle=False, collate_fn=lambda b: variable_width_collate_fn(b, width_divisibility=16), num_workers=0)

test_ds = ShotGatherDataset(csv_path, split='test')
trace_test_loader = DataLoader(test_ds, batch_size=8, shuffle=False, collate_fn=trace_collate_fn, num_workers=0)
gather_test_loader = DataLoader(test_ds, batch_size=2, shuffle=False, collate_fn=lambda b: variable_width_collate_fn(b, width_divisibility=16), num_workers=0)

print(f'Val Samples:  {len(val_ds)} gathers')
print(f'Test Samples: {len(test_ds)} gathers')
print('DataLoaders ready.')
"""

CELL_INFERENCE = """\
# ── Cell 5: Dual Inference (Val + Test) + Latency Timing ───────────────────
import joblib
from src.models.cnn_1d import Conv1DRegressor, ResNet1DRegressor
from src.models.unet_1d import SoftArgmax1DUNet
from src.models.unet import SoftArgmaxUNet, ResNetUNet
from src.utils.config_loader import load_model_config

inference_results = {}  

def _load_state(model, ckpt_path):
    d = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    model.load_state_dict(d['model_state_dict'])
    model.to(DEVICE).eval()
    return model

def _cfg(yaml_rel):
    return load_model_config(os.path.join(PROJECT_ROOT, yaml_rel), as_namespace=True)

def infer_trace(model, loader):
    preds, labels = [] , []
    tot_time = 0.0
    tot_traces = 0
    with torch.no_grad():
        for batch in loader:
            x = batch['traces'].float().to(DEVICE)
            y = batch['labels_ms'].numpy()
            m = batch['valid_mask'].numpy().astype(bool)
            
            t0 = time.perf_counter()
            # --- VRAM ARMOR: Hardcap trace computation at 4096 per pass to prevent GPU explosion ---
            chunk_size = 4096
            out_chunks = []
            for i in range(0, x.shape[0], chunk_size):
                chunk_pred = model(x[i:i+chunk_size])
                if torch.is_tensor(chunk_pred): chunk_pred = chunk_pred.detach().cpu().numpy()
                out_chunks.append(chunk_pred)
            out = np.concatenate(out_chunks, axis=0) if out_chunks else np.array([])
            t1 = time.perf_counter()
            
            out = out.reshape(-1)
            traces_in_batch = int(m.sum())
            if traces_in_batch > 0:
                tot_time += (t1 - t0)
                tot_traces += traces_in_batch
            preds.extend(out[m].tolist()); labels.extend(y[m].tolist())
            
    latency_ms = (tot_time / max(1, tot_traces)) * 1000.0 if tot_traces > 0 else float('nan')
    return {'preds': np.array(preds), 'labels': np.array(labels), 'latency': latency_ms}

def infer_gather(model, loader):
    preds, labels = [], []
    tot_time = 0.0
    tot_traces = 0
    with torch.no_grad():
        for batch in loader:
            x = batch['traces'].float().to(DEVICE)
            y = batch['labels_ms'].numpy()
            m = batch.get('label_mask', batch['valid_mask']).numpy().astype(bool)
            
            t0 = time.perf_counter()
            # --- VRAM ARMOR: Strict 1-gather-at-a-time evaluation to prevent 2D OOM ---
            out_chunks = []
            for i in range(x.shape[0]):
                chunk_pred = model(x[i:i+1])
                if torch.is_tensor(chunk_pred): chunk_pred = chunk_pred.detach().cpu().numpy()
                out_chunks.append(chunk_pred)
            out = np.concatenate(out_chunks, axis=0) if out_chunks else np.array([])
            t1 = time.perf_counter()
            
            fp = out.reshape(-1); fl = y.reshape(-1); fm = m.reshape(-1)
            traces_in_batch = int(fm.sum())
            if traces_in_batch > 0:
                tot_time += (t1 - t0)
                tot_traces += traces_in_batch
            preds.extend(fp[fm].tolist()); labels.extend(fl[fm].tolist())
            
    latency_ms = (tot_time / max(1, tot_traces)) * 1000.0 if tot_traces > 0 else float('nan')
    return {'preds': np.array(preds), 'labels': np.array(labels), 'latency': latency_ms}

def run_both(model_name, model, infer_fn, val_loader, test_loader):
    cache_dir = os.path.join(ARTIFACTS_DIR, model_name.replace(' ', '_').replace('-', '_'))
    cache_path = os.path.join(cache_dir, 'inference_cache.pkl')
    
    if os.path.exists(cache_path):
        print(f'[CACHE HIT] Loaded {model_name} completely from Drive. Skipping inference.')
        inference_results[model_name] = joblib.load(cache_path)
        if model is not None: del model; gc.collect()
        return
        
    print(f'Inferring {model_name} on Val...')
    val_res = infer_fn(model, val_loader)
    print(f'Inferring {model_name} on Test...')
    test_res = infer_fn(model, test_loader)
    
    res_dict = {'val': val_res, 'test': test_res}
    inference_results[model_name] = res_dict
    
    os.makedirs(cache_dir, exist_ok=True)
    joblib.dump(res_dict, cache_path)
    print(f'  [SAVED CACHE] {model_name} results safely backed up to Drive.')
    
    if model is not None: del model; gc.collect()
    print(f'  {model_name} complete: {len(test_res["preds"])} test traces. Latency: {test_res["latency"]:.3f} ms/trace\\n')

# ── CNN-1D ──
name = 'CNN-1D'; ckpt = find_checkpoint(os.path.join(PROJECT_ROOT, 'models/cnn1d'))
if ckpt:
    model = _load_state(Conv1DRegressor(base_channels=getattr(_cfg('configs/model_cnn1d.yaml').model,'base_channels',32)), ckpt)
    run_both(name, model, infer_trace, trace_val_loader, trace_test_loader)

# ── ResNet-1D ──
name = 'ResNet-1D'; ckpt = find_checkpoint(os.path.join(PROJECT_ROOT, 'models/resnet1d'))
if ckpt:
    model = _load_state(ResNet1DRegressor(base_channels=getattr(_cfg('configs/model_resnet1d.yaml').model,'base_channels',32)), ckpt)
    run_both(name, model, infer_trace, trace_val_loader, trace_test_loader)

# ── UNet-1D ──
name = 'UNet-1D'; ckpt = find_checkpoint(os.path.join(PROJECT_ROOT, 'models/unet1d'))
if ckpt:
    model = _load_state(SoftArgmax1DUNet(base_channels=getattr(_cfg('configs/model_unet1d.yaml').model,'base_channels',32)), ckpt)
    run_both(name, model, infer_trace, trace_val_loader, trace_test_loader)

# ── UNet-2D ──
name = 'UNet-2D'; ckpt = find_checkpoint(os.path.join(PROJECT_ROOT, 'models/unet_softargmax'))
if ckpt:
    model = _load_state(SoftArgmaxUNet(base_channels=getattr(_cfg('configs/model_unet_softargmax.yaml').model,'base_channels',32)), ckpt)
    run_both(name, model, infer_gather, gather_val_loader, gather_test_loader)

# ── ResNet-UNet ──
name = 'ResNet-UNet'; ckpt = find_checkpoint(os.path.join(PROJECT_ROOT, 'models/resnet_unet'))
if ckpt:
    model = _load_state(ResNetUNet(encoder_name=getattr(_cfg('configs/model_resnet_unet.yaml').model,'encoder_name','resnet34'), encoder_weights=None), ckpt)
    run_both(name, model, infer_gather, gather_val_loader, gather_test_loader)

# ── LightGBM ──
name = 'LightGBM'
lgbm_dir = os.path.join(PROJECT_ROOT, 'models/tabular_lgbm')
pkl_files = [f for f in os.listdir(lgbm_dir) if f.endswith('.pkl')] if os.path.isdir(lgbm_dir) else []
if pkl_files:
    import joblib
    from src.features.features import extract_features
    lgbm = joblib.load(os.path.join(lgbm_dir, pkl_files[0]))
    
    def infer_lgbm(loader):
        preds, labels = [], []
        tot_time = 0.0; tot_traces = 0
        for batch in loader:
            x = batch['traces'].squeeze(1).numpy()
            y = batch['labels_ms'].numpy()
            m = batch['valid_mask'].numpy().astype(bool)
            offsets = getattr(batch.get('offsets', np.zeros(x.shape[0])), 'numpy', lambda: getattr(batch.get('offsets', np.zeros(x.shape[0])), 'values', np.zeros(x.shape[0])))() if hasattr(batch.get('offsets'), 'numpy') else np.zeros(x.shape[0])
            
            t0 = time.perf_counter()
            feats = extract_features(x, offsets)
            p = lgbm.predict(feats)
            t1 = time.perf_counter()
            
            traces_in_batch = int(m.sum())
            if traces_in_batch > 0:
                tot_time += (t1 - t0)
                tot_traces += traces_in_batch
            preds.extend(p[m].tolist()); labels.extend(y[m].tolist())
        latency_ms = (tot_time / max(1, tot_traces)) * 1000.0 if tot_traces > 0 else float('nan')
        return {'preds': np.array(preds), 'labels': np.array(labels), 'latency': latency_ms}
    
    run_both(name, None, lambda m, l: infer_lgbm(l), trace_val_loader, trace_test_loader)
"""

CELL_PLOTS = """\
# ── Cell 6: Test Set Scatter Plots & Error Histograms ──────────────────────
for name, res in inference_results.items():
    preds  = res['test']['preds']
    labels = res['test']['labels']
    errors = np.abs(preds - labels)

    mae = np.mean(errors); p90 = np.percentile(errors, 90)

    # ── Scatter Plot ──
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(labels, preds, s=6, alpha=0.3, color='steelblue', rasterized=True)
    lo, hi = min(labels.min(), preds.min()), max(labels.max(), preds.max())
    ax.plot([lo, hi], [lo, hi], 'r--', lw=1.5, label='Perfect')
    ax.set_title(f'{name} — TEST Predicted vs Ground Truth\\nTEST MAE = {mae:.2f} ms')
    ax.legend(); ax.grid(True, linestyle='--', alpha=0.4); fig.tight_layout()
    save_fig(fig, name, 'test_scatter.png')

    # ── Error Histogram ──
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(errors, bins=60, color='tomato', edgecolor='white', linewidth=0.5)
    ax.axvline(mae, color='navy', lw=2, linestyle='--', label=f'TEST MAE = {mae:.2f} ms')
    ax.axvline(p90, color='darkorange', lw=1.5, linestyle=':', label=f'TEST P90 = {p90:.2f} ms')
    ax.set_title(f'{name} — TEST Error Distribution')
    ax.legend(); ax.grid(True, linestyle='--', alpha=0.4); fig.tight_layout()
    save_fig(fig, name, 'test_error_hist.png')

print('\\nAll Test-Set plots generated and saved.')
"""

CELL_LEADERBOARD = """\
# ── Cell 7: Final Dual Leaderboard & Extraction ────────────────────────────

def build_lb(split_name):
    lb = []
    for name, res in inference_results.items():
        preds, labels = res[split_name]['preds'], res[split_name]['labels']
        latency = res[split_name]['latency']
        errors = np.abs(preds - labels)
        lb.append({
            'Model': name,
            f'{split_name.title()} MAE (ms)': round(float(np.mean(errors)), 3),
            'P90 Error (ms)': round(float(np.percentile(errors, 90)), 3),
            '<5ms (%)': round(float(np.mean(errors <= 5.0) * 100), 2),
            '<10ms (%)': round(float(np.mean(errors <= 10.0) * 100), 2),
            f'{split_name.title()} CPU Latency (ms/trace)': round(float(latency), 4)
        })
    df = pd.DataFrame(lb).sort_values(f'{split_name.title()} MAE (ms)').reset_index(drop=True)
    df.index += 1
    return df

val_df = build_lb('val')
test_df = build_lb('test')

print('\\n' + '='*75)
print(' VALIDATION LEADERBOARD (Tuning)')
print('='*75)
print(val_df.to_string())

print('\\n\\n' + '='*75)
print(' OFFICIAL TEST LEADERBOARD (Tournament Result)')
print('='*75)
print(test_df.to_string())

# Export tables to CSV!
val_df.to_csv(os.path.join(ARTIFACTS_DIR, 'leaderboard', 'val_leaderboard.csv'), index=False)
test_df.to_csv(os.path.join(ARTIFACTS_DIR, 'leaderboard', 'test_leaderboard.csv'), index=False)
print('\\nSaved Leaderboards to artifacts/plots/leaderboard/ as CSV files.')

# ── Plots ──
combined = val_df.merge(test_df, on='Model')
plot_df = combined.melt(id_vars='Model', value_vars=['Val MAE (ms)', 'Test MAE (ms)'], 
                        var_name='Split', value_name='MAE (ms)')

fig, ax = plt.subplots(figsize=(10, max(5, len(inference_results) * 1.0)))
sns.barplot(data=plot_df, y='Model', x='MAE (ms)', hue='Split', palette=['deepskyblue', 'crimson'], ax=ax)
ax.set_title('Seismic First-Break Picking — Val vs Test Leaderboard', fontsize=14, fontweight='bold')
ax.set_xlabel('MAE (ms) — lower is better')
ax.grid(True, axis='x', linestyle='--', alpha=0.5)

for p in ax.patches:
    w = p.get_width()
    if w > 0: ax.text(w + 0.2, p.get_y() + p.get_height()/2, f'{w:.2f}', va='center', fontsize=9)

fig.tight_layout()
save_fig(fig, 'leaderboard', 'val_vs_test_leaderboard.png')

# ── Latency vs Accuracy Plot ──
fig, ax = plt.subplots(figsize=(8, 6))
sns.scatterplot(data=test_df, x='Test CPU Latency (ms/trace)', y='Test MAE (ms)', 
                hue='Model', s=250, palette='tab10', ax=ax, zorder=5)
for _, row in test_df.iterrows():
    ax.annotate(row['Model'], xy=(row['Test CPU Latency (ms/trace)'], row['Test MAE (ms)']),
                xytext=(8, 4), textcoords='offset points', fontsize=9)
ax.set_xlabel('CPU Inference Latency per Trace (ms)')
ax.set_ylabel('Official Test MAE (ms)')
ax.set_title('Deployment Pareto Frontier (Accuracy vs Speed)')
ax.grid(True, linestyle='--', alpha=0.5)
fig.tight_layout()
save_fig(fig, 'leaderboard', 'deployment_pareto.png')

print('\\nLeaderboard charts saved.')
test_df
"""

def create_notebook(output_path="notebooks/04_benchmark_and_compare.ipynb"):
    nb = nbf.v4.new_notebook()
    nb['cells'] = [
        md("# Phase 5: Ultimate Tournament Benchmark \nComputes Official **Test** set metrics vs Validation set metrics seamlessly."),
        cell(CELL_SETUP), cell(CELL_METADATA), cell(CELL_TRAINING_CURVES),
        cell(CELL_DATALOADER), cell(CELL_INFERENCE), cell(CELL_PLOTS), cell(CELL_LEADERBOARD)
    ]
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f: nbf.write(nb, f)
    print(f"Created: {output_path}")

if __name__ == "__main__": create_notebook()
