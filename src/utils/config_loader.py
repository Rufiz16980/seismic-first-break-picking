"""Configuration loading and path resolution helpers.

Loads YAML configs and resolves all paths relative to the project root.
Works both locally and on Google Colab with Drive mount.
"""

import os
from pathlib import Path
from typing import Any, Dict
import types

import yaml

def _dict_to_namespace(d: Dict[str, Any]) -> types.SimpleNamespace:
    """Recursively convert a dictionary to a SimpleNamespace."""
    namespace = types.SimpleNamespace()
    for k, v in d.items():
        if isinstance(v, dict):
            setattr(namespace, k, _dict_to_namespace(v))
        else:
            setattr(namespace, k, v)
    return namespace

def load_yaml(path: str) -> Dict[str, Any]:
    """Load a YAML file and return as a dictionary.

    Args:
        path: absolute or relative path to a YAML file.

    Returns:
        dict with parsed YAML contents.

    Raises:
        FileNotFoundError: if the YAML file does not exist.
    """
    path = str(path)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def resolve_paths(cfg: Dict[str, Any], project_root: str) -> Dict[str, Any]:
    """Resolve all relative paths in config against the project root.

    Processes both the top-level ``paths`` dict and the ``output`` dict,
    converting every relative path to an absolute path rooted at *project_root*.

    Args:
        cfg: parsed config dictionary (from load_yaml).
        project_root: absolute path to the project root directory.

    Returns:
        The same dict with path values resolved to absolute paths.
    """
    root = Path(project_root)

    # Resolve paths dict
    if "paths" in cfg:
        for key, rel_path in cfg["paths"].items():
            cfg["paths"][key] = str(root / rel_path)

    # Resolve output dict
    if "output" in cfg:
        for key, rel_path in cfg["output"].items():
            if isinstance(rel_path, str) and not os.path.isabs(rel_path):
                resolved = str(root / rel_path)
                # Special handle for MLflow tracking URI: it MUST be a URI (file://...)
                if key == "tracking_uri":
                    # Ensure properly formatted file URI for Windows/Linux
                    prefix = "file:///" if os.name == 'nt' else "file://"
                    cfg["output"][key] = prefix + resolved.replace('\\', '/')
                else:
                    cfg["output"][key] = resolved

    return cfg


def load_project_config(project_root: str) -> Dict[str, Any]:
    """Load datasets.yaml and resolve all paths.

    Args:
        project_root: absolute path to the project root directory.

    Returns:
        Fully-resolved configuration dictionary.
    """
    cfg_path = os.path.join(project_root, "configs", "datasets.yaml")
    cfg = load_yaml(cfg_path)
    cfg["project_root"] = project_root
    return resolve_paths(cfg, project_root)


def load_preprocessing_config(project_root: str) -> Dict[str, Any]:
    """Load preprocessing.yaml and resolve output paths.

    Args:
        project_root: absolute path to the project root directory.

    Returns:
        Fully-resolved preprocessing configuration dictionary.
    """
    cfg_path = os.path.join(project_root, "configs", "preprocessing.yaml")
    cfg = load_yaml(cfg_path)
    return resolve_paths(cfg, project_root)


def get_asset_hdf5_path(project_cfg: Dict[str, Any],
                        asset_name: str) -> str:
    """Get the absolute path to an asset's extracted HDF5 file.

    Args:
        project_cfg: loaded project config (from load_project_config).
        asset_name: one of 'brunswick', 'halfmile', 'lalor', 'sudbury'.

    Returns:
        Absolute path to the HDF5 file.

    Raises:
        KeyError: if asset_name is not in the config.
    """
    root = project_cfg["project_root"]
    meta = project_cfg["asset_meta"][asset_name]
    extracted_dir = os.path.join(root, project_cfg["paths"]["extracted"])
    return os.path.join(extracted_dir, meta["extracted_file"])


def load_model_config(cfg_path: str, as_namespace: bool = True) -> Any:
    """Load model training config (Phase 4).
    
    Args:
        cfg_path: Path to the model specific yaml config.
        as_namespace: If True, returns a SimpleNamespace for dot-notation access. 
                      If False, returns raw Dict.
                      
    Returns:
        Configuration object.
    """
    cfg = load_yaml(cfg_path)
    if as_namespace:
        return _dict_to_namespace(cfg)
    return cfg

