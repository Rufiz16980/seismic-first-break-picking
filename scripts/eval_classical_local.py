import os
import sys

PROJECT_ROOT = r"g:\My Drive\seismic-first-break-picking"
sys.path.insert(0, PROJECT_ROOT)

import torch
import numpy as np
from torch.utils.data import DataLoader
from src.data.dataset import ShotGatherDataset, trace_collate_fn
from src.models.classical import STALTAPicker, MERPicker, AICPicker

import warnings
warnings.filterwarnings('ignore')

val_ds = ShotGatherDataset(os.path.join(PROJECT_ROOT, "data/processed/split_index.csv"), split='val')
val_loader = DataLoader(
    val_ds, batch_size=64, shuffle=False,
    collate_fn=trace_collate_fn, num_workers=0
)

models = {
    "STA/LTA": STALTAPicker(),
    "MER": MERPicker(),
    "AIC": AICPicker()
}

results = {name: {"maes": [], "within_5ms": []} for name in models}

print(f"Evaluating {len(val_ds)} validation gathers...")

for batch_idx, batch in enumerate(val_loader):
    traces = batch['traces'].squeeze(1).numpy()
    labels_ms = batch['labels_ms'].numpy()
    valid_mask = batch['valid_mask'].numpy()
    
    if not valid_mask.any():
        continue

    valid_labels = labels_ms[valid_mask]

    for name, model in models.items():
        preds_ms = model.predict(traces)
        valid_preds = preds_ms[valid_mask]
        
        mae = np.abs(valid_preds - valid_labels)
        results[name]["maes"].extend(mae.tolist())
        results[name]["within_5ms"].extend((mae <= 5.0).astype(float).tolist())

print("\n--- Validation Results ---")
for name in models:
    if len(results[name]["maes"]) > 0:
        mean_mae = np.mean(results[name]["maes"])
        pct_5ms = np.mean(results[name]["within_5ms"]) * 100.0
        print(f"{name:8s} | MAE: {mean_mae:6.2f} ms | <= 5ms: {pct_5ms:5.1f}%")
    else:
        print(f"{name:8s} | No valid traces")
