"""Package initialization for seismic first break picking.

Exposes core pipeline components:
  - hdf5_reader: HDF5 loading with coordinate scaling
  - shot_gather_builder: preprocessing pipeline (gather → NPZ)
  - transforms: data augmentation classes
  - dataset: PyTorch Dataset and DataLoader utilities
"""

from src.data.hdf5_reader import open_hdf5, apply_segy_scale, load_all_metadata
from src.data.shot_gather_builder import (
    process_asset,
    write_master_index,
    stratified_split,
    write_split_index,
    load_shot_npz,
    save_shot_npz,
    normalize_traces,
    harmonize_trace,
    harmonize_gather,
)
from src.data.transforms import (
    Compose,
    AmplitudeScale,
    GaussianNoise,
    TraceDropout,
    TimeShift,
    PolarityReversal,
    build_transforms,
)

__all__ = [
    "open_hdf5", "apply_segy_scale", "load_all_metadata",
    "process_asset", "write_master_index", "stratified_split",
    "write_split_index", "load_shot_npz", "save_shot_npz",
    "normalize_traces", "harmonize_trace", "harmonize_gather",
    "Compose", "AmplitudeScale", "GaussianNoise", "TraceDropout",
    "TimeShift", "PolarityReversal", "build_transforms",
]
