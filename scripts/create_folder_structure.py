from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DIRS = [
    "configs",
    "data/raw",
    "data/extracted",
    "data/processed/brunswick",
    "data/processed/halfmile",
    "data/processed/lalor",
    "data/processed/sudbury",
    "data/datasets/train",
    "data/datasets/val",
    "data/datasets/test",
    "documentation/implementation_phases",
    "mlruns",
    "models/checkpoints",
    "models/checkpoints/unet",
    "models/checkpoints/resnet_unet",
    "models/checkpoints/1dcnn",
    "models/checkpoints/classical",
    "models/final",
    "notebooks",
    "results",
    "results/sanity_plots",
    "results/eda_plots",
    "results/benchmark",
    "results/visualization_examples",
    "src/data",
    "src/models",
    "src/training",
    "src/evaluation",
    "src/utils",
]

for rel in DIRS:
    path = ROOT / rel
    path.mkdir(parents=True, exist_ok=True)
    keep = path / ".gitkeep"
    if not any(path.iterdir()) and not keep.exists():
        keep.touch()

print(f"Created/verified {len(DIRS)} directories under {ROOT}")
