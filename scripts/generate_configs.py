import os
import yaml

configs = {
    "model_classical": {
        "experiment": {"name": "classical_baselines", "seed": 42, "asset": "all", "framing": "trace"},
        "data": {
            "index_csv": "data/processed/split_index.csv",
            "processed_dir": "processed/",
            "target_n_samples": 1500,
            "target_n_traces": 120,
            "normalization": "per_trace",
            "cache_in_ram": False
        },
        "model": {"architecture": "sta_lta"},
        "training": {
            "batch_size": 128,  # very fast
            "epochs": 1 # no actual training
        },
        "output": {"tracking_uri": "file:///content/drive/MyDrive/seismic-first-break-picking/mlruns", "checkpoint_dir": "models/classical"}
    },
    
    "model_tabular_lgbm": {
        "experiment": {"name": "tabular_lgbm", "seed": 42, "asset": "all", "framing": "trace"},
        "data": {
            "index_csv": "data/processed/split_index.csv", "processed_dir": "processed/",
            "target_n_samples": 1500, "target_n_traces": 120,
            "normalization": "per_trace", "cache_in_ram": False
        },
        "model": {"architecture": "lightgbm", "n_estimators": 500, "learning_rate": 0.05, "max_depth": 15},
        "training": {
            "batch_size": 256,
            "epochs": 1
        },
        "output": {"tracking_uri": "file:///content/drive/MyDrive/seismic-first-break-picking/mlruns", "checkpoint_dir": "models/tabular_lgbm"}
    },

    "model_cnn1d": {
        "experiment": {"name": "cnn1d_standard", "seed": 42, "asset": "all", "framing": "trace"},
        "data": {"index_csv": "datasets/master_index.csv", "processed_dir": "processed/", 
                 "target_n_samples": 1500, "target_n_traces": 120, "normalization": "per_trace", "cache_in_ram": False},
        "model": {"architecture": "cnn1d", "base_channels": 32},
        "training": {"batch_size": 64, "epochs": 50, "optimizer": "adamw", "lr": 1e-3, "weight_decay": 1e-4, "scheduler_params": {"T_max": 50, "eta_min": 1e-6}, "early_stopping_patience": 15, "gradient_clip_norm": 1.0},
        "loss": {"primary": "masked_mae", "huber_delta": 5.0},
        "output": {"tracking_uri": "file:///content/drive/MyDrive/seismic-first-break-picking/mlruns", "checkpoint_dir": "models/cnn1d"},
        "progressive_training": {
            "asset_order": ["brunswick", "halfmile", "lalor", "sudbury"],
            "asset_epochs": [50, 25, 25, 25],
            "asset_lr": [5e-4, 1e-4, 1e-4, 1e-4],
            "replay_fraction": 0.15,
            "reset_optimizer": True,
            "reset_scheduler": True
        }
    },

    "model_resnet1d": {
        "experiment": {"name": "resnet1d_deep", "seed": 42, "asset": "all", "framing": "trace"},
        "data": {"index_csv": "datasets/master_index.csv", "processed_dir": "processed/", "target_n_samples": 1500, "target_n_traces": 120, "normalization": "per_trace", "cache_in_ram": False},
        "model": {"architecture": "resnet1d", "base_channels": 32},
        "training": {"batch_size": 64, "epochs": 50, "optimizer": "adamw", "lr": 1e-3, "weight_decay": 1e-4, "scheduler_params": {"T_max": 50, "eta_min": 1e-6}, "early_stopping_patience": 15, "gradient_clip_norm": 1.0},
        "loss": {"primary": "masked_mae", "huber_delta": 5.0},
        "output": {"tracking_uri": "file:///content/drive/MyDrive/seismic-first-break-picking/mlruns", "checkpoint_dir": "models/resnet1d"}
    },

    "model_tcn": {
        "experiment": {"name": "tcn_dilated", "seed": 42, "asset": "all", "framing": "trace"},
        "data": {"index_csv": "datasets/master_index.csv", "processed_dir": "processed/", "target_n_samples": 1500, "target_n_traces": 120, "normalization": "per_trace", "cache_in_ram": False},
        "model": {"architecture": "tcn", "kernel_size": 3},
        "training": {"batch_size": 64, "epochs": 50, "optimizer": "adamw", "lr": 1e-3, "weight_decay": 1e-4, "scheduler_params": {"T_max": 50, "eta_min": 1e-6}, "early_stopping_patience": 15, "gradient_clip_norm": 1.0},
        "loss": {"primary": "masked_mae", "huber_delta": 5.0},
        "output": {"tracking_uri": "file:///content/drive/MyDrive/seismic-first-break-picking/mlruns", "checkpoint_dir": "models/tcn"},
        "progressive_training": {
            "asset_order": ["brunswick", "halfmile", "lalor", "sudbury"],
            "asset_epochs": [50, 25, 25, 25],
            "asset_lr": [5e-4, 1e-4, 1e-4, 1e-4],
            "replay_fraction": 0.15,
            "reset_optimizer": True,
            "reset_scheduler": True
        }
    },

    "model_bilstm": {
        "experiment": {"name": "bilstm_sequential", "seed": 42, "asset": "all", "framing": "trace"},
        "data": {"index_csv": "datasets/master_index.csv", "processed_dir": "processed/", "target_n_samples": 1500, "target_n_traces": 120, "normalization": "per_trace", "cache_in_ram": False},
        "model": {"architecture": "bilstm", "hidden_size": 64, "num_layers": 2},
        "training": {"batch_size": 32, "epochs": 50, "optimizer": "adamw", "lr": 5e-4, "weight_decay": 1e-4, "scheduler_params": {"T_max": 50, "eta_min": 1e-6}, "early_stopping_patience": 15, "gradient_clip_norm": 1.0},
        "loss": {"primary": "masked_mae", "huber_delta": 5.0},
        "output": {"tracking_uri": "file:///content/drive/MyDrive/seismic-first-break-picking/mlruns", "checkpoint_dir": "models/bilstm"},
        "progressive_training": {
            "asset_order": ["brunswick", "halfmile", "lalor", "sudbury"],
            "asset_epochs": [50, 25, 25, 25],
            "asset_lr": [5e-4, 1e-4, 1e-4, 1e-4],
            "replay_fraction": 0.15,
            "reset_optimizer": True,
            "reset_scheduler": True
        }
    },

    "model_cnn2d": {
        "experiment": {"name": "cnn2d_standard", "seed": 42, "asset": "all", "framing": "gather"},
        "data": {"index_csv": "datasets/master_index.csv", "processed_dir": "processed/", "target_n_samples": 1500, "target_n_traces": 120, "normalization": "per_trace", "cache_in_ram": False},
        "model": {"architecture": "cnn2d", "base_channels": 32},
        "training": {"batch_size": 8, "epochs": 75, "optimizer": "adamw", "lr": 1e-3, "weight_decay": 1e-4, "scheduler_params": {"T_max": 75, "eta_min": 1e-6}, "early_stopping_patience": 15, "gradient_clip_norm": 1.0},
        "loss": {"primary": "masked_mae", "huber_delta": 5.0},
        "output": {"tracking_uri": "file:///content/drive/MyDrive/seismic-first-break-picking/mlruns", "checkpoint_dir": "models/cnn2d"},
        "progressive_training": {
            "asset_order": ["brunswick", "halfmile", "lalor", "sudbury"],
            "asset_epochs": [100, 50, 50, 50],
            "asset_lr": [1e-3, 2e-4, 2e-4, 2e-4],
            "replay_fraction": 0.15,
            "reset_optimizer": True,
            "reset_scheduler": True
        }
    },

    "model_resnet_unet": {
        "experiment": {"name": "resnet_unet_pretrained", "seed": 42, "asset": "all", "framing": "gather"},
        "data": {"index_csv": "datasets/master_index.csv", "processed_dir": "processed/", "target_n_samples": 1500, "target_n_traces": 120, "normalization": "per_trace", "cache_in_ram": False},
        "model": {"architecture": "resnet_unet", "encoder_name": "resnet34", "encoder_weights": "imagenet"},
        "training": {"training_mode": "combined", "batch_size": 8, "epochs": 75, "optimizer": "adamw", "lr": 5e-4, "weight_decay": 1e-4, "scheduler_params": {"T_max": 75, "eta_min": 1e-6}, "early_stopping_patience": 15, "gradient_clip_norm": 1.0},
        "loss": {"primary": "masked_mae", "huber_delta": 5.0},
        "output": {"tracking_uri": "file:///content/drive/MyDrive/seismic-first-break-picking/mlruns", "checkpoint_dir": "models/resnet_unet"},
        "progressive_training": {
            "asset_order": ["brunswick", "halfmile", "lalor", "sudbury"],
            "asset_epochs": [100, 50, 50, 50],
            "asset_lr": [5e-4, 1e-4, 1e-4, 1e-4],
            "replay_fraction": 0.15,
            "reset_optimizer": True,
            "reset_scheduler": True
        }
    }
}

os.makedirs('configs', exist_ok=True)
for name, conf in configs.items():
    with open(f"configs/{name}.yaml", "w") as f:
        yaml.dump(conf, f, sort_keys=False)

print("Generated 8 Configs")
