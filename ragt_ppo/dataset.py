"""
dataset.py
----------
Loads a collection of AAG parts from your REAL aag_generator output and
splits them into train/validation/test sets.

Your actual file schema (confirmed from a real exported part) is a single
JSON object per part with (relevant fields only):

    {
      "num_nodes": <int>,
      "catalogs": {"msc_labels": [...14 strings...], "tool_labels": [...36 strings...]},
      "static_attributes": {
          "feasible_solution_matrix_P": {"0": [[...n_c...], ...n_t rows...], "1": ..., ...},
          "distance_matrix_M": [[...n...], ...n rows...]
      },
      "edge_attributes": {
          "edge_list": [{"src_index": i, "dst_index": j, "attr": [dep,adj,datum]}, ...]
      },
      "feature_reference": {"0": {"feature_type": "...", ...}, ...}   # optional, for naming
    }

Every part in the directory must share the SAME catalog (same msc_labels /
tool_labels) -- that's what fixes the RAGT-PPO network's input/output
dimensions for the whole training run. `load_aag_dataset` checks this and
raises a clear error if a file disagrees.

A legacy/simplified schema (top-level "edges" list, flat "feasibility" /
"distance" arrays) is also supported as a fallback, for hand-built test
fixtures -- see `_load_one_legacy`.
"""
from __future__ import annotations
import json
import pickle
import random
from pathlib import Path
from typing import List, Tuple

import numpy as np

from .aag import AAGPart, AAGConfig, build_synthetic_part, DEFAULT_CONFIG


def _load_one_real_schema(d: dict) -> Tuple[AAGPart, AAGConfig]:
    n = d["num_nodes"]
    msc_labels = d["catalogs"]["msc_labels"]
    tool_labels = d["catalogs"]["tool_labels"]
    cfg = AAGConfig.from_labels(msc_labels, tool_labels)

    P_dict = d["static_attributes"]["feasible_solution_matrix_P"]
    feasibility = np.stack([np.array(P_dict[str(i)], dtype=np.float32) for i in range(n)])
    assert feasibility.shape == (n, cfg.n_t, cfg.n_c), (
        f"P shape {feasibility.shape} doesn't match catalog sizes "
        f"(n_t={cfg.n_t}, n_c={cfg.n_c}) -- check catalogs.tool_labels/msc_labels"
    )

    distance = np.array(d["static_attributes"]["distance_matrix_M"], dtype=np.float32)

    edges = [(e["src_index"], e["dst_index"], tuple(int(b) for b in e["attr"]))
              for e in d["edge_attributes"]["edge_list"]]

    feature_names = [f"F{i+1}" for i in range(n)]
    fref = d.get("feature_reference")
    if fref:
        # optional, human-readable: "F3 (circular_end_pocket)"
        feature_names = [
            f"F{i+1} ({fref.get(str(i), {}).get('feature_type', '?')})" for i in range(n)
        ]

    part = AAGPart(n_features=n, feasibility=feasibility, distance=distance,
                    edges=edges, feature_names=feature_names)
    return part, cfg


def _load_one_legacy(d: dict) -> Tuple[AAGPart, AAGConfig]:
    """Fallback for a flat/simplified schema: {"n_features", "feasibility",
    "distance", "edges", "feature_names"[, "msc_labels", "tool_labels"]}."""
    cfg = (AAGConfig.from_labels(d["msc_labels"], d["tool_labels"])
           if "msc_labels" in d and "tool_labels" in d else DEFAULT_CONFIG)
    edges = [(e[0], e[1], tuple(e[2])) for e in d["edges"]]
    part = AAGPart(
        n_features=d["n_features"],
        feasibility=np.array(d["feasibility"], dtype=np.float32),
        distance=np.array(d["distance"], dtype=np.float32),
        edges=edges,
        feature_names=d.get("feature_names", [f"F{i+1}" for i in range(d["n_features"])]),
    )
    return part, cfg


def _load_one(path: Path) -> Tuple[AAGPart, AAGConfig]:
    if path.suffix == ".json":
        with open(path) as f:
            d = json.load(f)
        if "static_attributes" in d and "edge_attributes" in d:
            return _load_one_real_schema(d)
        elif "edges" in d:
            return _load_one_legacy(d)
        else:
            raise ValueError(
                f"{path}: unrecognized JSON schema (expected either the real "
                f"aag_generator export with 'static_attributes'/'edge_attributes', "
                f"or a legacy dict with a top-level 'edges' key)"
            )
    elif path.suffix in (".pkl", ".pickle"):
        with open(path, "rb") as f:
            obj = pickle.load(f)
        if isinstance(obj, tuple) and len(obj) == 2:
            return obj  # (AAGPart, AAGConfig) pickled directly
        if isinstance(obj, AAGPart):
            return obj, DEFAULT_CONFIG
        return _load_one_legacy(obj)
    else:
        raise ValueError(f"Unrecognized AAG file format: {path}")


def load_aag_dataset(directory: str) -> Tuple[List[AAGPart], AAGConfig]:
    """Loads every part file in `directory` (sorted for reproducibility).
    Returns (parts, cfg) -- cfg is taken from the first file and every
    other file's catalog is checked to match it."""
    directory = Path(directory)
    files = sorted([p for p in directory.iterdir()
                     if p.suffix in (".json", ".pkl", ".pickle")
                     and p.name != "manifest.json"])
    if not files:
        raise FileNotFoundError(f"No AAG part files found in {directory}")

    parts: List[AAGPart] = []
    cfg: AAGConfig | None = None
    for fpath in files:
        part, file_cfg = _load_one(fpath)
        if cfg is None:
            cfg = file_cfg
        elif file_cfg.msc_labels != cfg.msc_labels or file_cfg.tool_labels != cfg.tool_labels:
            raise ValueError(
                f"{fpath} uses a different catalog than earlier files in this "
                f"directory (msc_labels/tool_labels mismatch). All parts fed to "
                f"one RAGT-PPO model must share the same catalog."
            )
        parts.append(part)
    return parts, cfg


def split_dataset(parts: List[AAGPart], n_train: int = 60, n_val: int = 20,
                   n_test: int = 20, seed: int = 42
                   ) -> Tuple[List[AAGPart], List[AAGPart], List[AAGPart]]:
    """Shuffles and splits `parts` into (train, val, test)."""
    assert n_train + n_val + n_test <= len(parts), (
        f"Requested split {n_train}+{n_val}+{n_test}={n_train+n_val+n_test} "
        f"exceeds dataset size {len(parts)}"
    )
    idx = list(range(len(parts)))
    random.Random(seed).shuffle(idx)
    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train + n_val]
    test_idx = idx[n_train + n_val: n_train + n_val + n_test]
    train = [parts[i] for i in train_idx]
    val = [parts[i] for i in val_idx]
    test = [parts[i] for i in test_idx]
    return train, val, test


def build_synthetic_dataset(n_parts: int = 100, seed: int = 0
                             ) -> Tuple[List[AAGPart], AAGConfig]:
    """For dry-running the whole pipeline: builds `n_parts` synthetic AAGs
    (paper's original 10-MSC/23-tool catalog) of varying complexity."""
    rng = np.random.default_rng(seed)
    parts = []
    for i in range(n_parts):
        n_feat = int(rng.integers(4, 15))
        parts.append(build_synthetic_part(DEFAULT_CONFIG, n_features=n_feat,
                                           seed=seed * 10_000 + i))
    return parts, DEFAULT_CONFIG
