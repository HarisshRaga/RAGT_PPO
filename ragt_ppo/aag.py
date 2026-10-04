"""
aag.py
------
Data structures for the Attributed Adjacency Graph (AAG) state.

IMPORTANT: unlike the first version of this file, MSC/tool catalog sizes
are NOT hardcoded module constants anymore. Your real aag_generator uses an
EXTENDED catalog (14 MSCs including drilling/reaming/tapping/chamfering,
36 tools -- see Feature_Operation_Mapping.xlsx / Tools_Available.xlsx),
which is larger than the paper's original 10 MSCs / 23 tools. Every part
you export carries its own `catalogs.msc_labels` / `catalogs.tool_labels`
list, so this module builds an `AAGConfig` from those labels at load time,
and every other module (rules/model/env/reward/schedule_output) takes that
config as an explicit argument rather than importing fixed constants.

Two independent precedence CHAINS are recognized from the MSC labels:
  - "milling" chain : rough -> secondary-rough -> semi-finish -> finish
                       (Table 1, rules 1-3)
  - "hole" chain     : drilling -> reaming -> tapping -> chamfering
                       (your guaranteed-precision-hole workflow; not in the
                       original paper's Table 1, but the same kind of
                       stage-precedence rule, applied to its own MSC subset)
Any MSC label that doesn't match a known pattern falls back to chain="other",
stage=0 (no precedence requirement), so future catalog extensions degrade
gracefully instead of crashing.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple, Dict
import numpy as np

EDGE_TYPES = ["dependence", "adjacency", "datum"]   # matches relation_vocabulary in your JSON
N_EDGE_TYPES = len(EDGE_TYPES)

# ---------------------------------------------------------------------------
# MSC label -> (chain, stage-within-chain, method, region, stage_label)
# ---------------------------------------------------------------------------
_MSC_RULES = [
    # (match-key after lowercase+underscore normalization, chain, stage, method, region, stage_label)
    ("rough_milling",                    "milling", 0, "3axis", "-",       "Roughing"),
    ("secondary_rough_milling",          "milling", 1, "3axis", "-",       "Secondary roughing"),
    ("semi_finish_milling_sides",        "milling", 2, "3axis", "Sides",   "Semi-finishing"),
    ("semi_finish_milling_sides_5axis",  "milling", 2, "5axis", "Sides",   "Semi-finishing"),
    ("semi_finish_milling_bottom",       "milling", 2, "3axis", "Bottom",  "Semi-finishing"),
    ("finish_milling_sides",             "milling", 3, "3axis", "Sides",   "Finishing"),
    ("finish_milling_sides_5axis",       "milling", 3, "5axis", "Sides",   "Finishing"),
    ("finish_milling_bottom",            "milling", 3, "3axis", "Bottom",  "Finishing"),
    ("finish_milling_corners",           "milling", 3, "3axis", "Corners", "Finishing"),
    ("finish_milling_corners_5axis",     "milling", 3, "5axis", "Corners", "Finishing"),
    ("drilling",                         "hole",    0, "3axis", "-",       "Drilling"),
    ("reaming",                          "hole",    1, "3axis", "-",       "Reaming"),
    ("tapping",                          "hole",    2, "3axis", "-",       "Tapping"),
    ("chamfering",                       "hole",    3, "3axis", "-",       "Chamfering"),
]
_MSC_LOOKUP = {r[0]: r[1:] for r in _MSC_RULES}


def _normalize_label(label: str) -> str:
    return label.strip().lower().replace(" ", "_").replace("-", "_")


def _infer_msc_metadata(label: str):
    key = _normalize_label(label)
    if key in _MSC_LOOKUP:
        return _MSC_LOOKUP[key]
    # graceful fallback for any future/unrecognized MSC label
    return ("other", 0, "3axis", "-", label)


@dataclass
class AAGConfig:
    """Everything downstream (rules/model/env/reward/schedule_output) needs
    to know about the catalog sizes and semantics for ONE dataset. Build
    once via `AAGConfig.from_labels(...)` and pass it everywhere -- do not
    hardcode n_t / n_c anywhere else in the codebase."""
    msc_labels: List[str]
    tool_labels: List[str]
    chain_of: List[str] = field(default_factory=list)
    stage_of: np.ndarray = None
    method_of: List[str] = field(default_factory=list)
    region_of: List[str] = field(default_factory=list)
    stage_label_of: List[str] = field(default_factory=list)

    @property
    def n_c(self) -> int:
        return len(self.msc_labels)

    @property
    def n_t(self) -> int:
        return len(self.tool_labels)

    @property
    def n_edge_types(self) -> int:
        return N_EDGE_TYPES

    @classmethod
    def from_labels(cls, msc_labels: List[str], tool_labels: List[str]) -> "AAGConfig":
        chain_of, stage_of, method_of, region_of, stage_label_of = [], [], [], [], []
        for lbl in msc_labels:
            chain, stage, method, region, stage_label = _infer_msc_metadata(lbl)
            chain_of.append(chain)
            stage_of.append(stage)
            method_of.append(method)
            region_of.append(region)
            stage_label_of.append(stage_label)
        return cls(
            msc_labels=list(msc_labels), tool_labels=list(tool_labels),
            chain_of=chain_of, stage_of=np.array(stage_of, dtype=np.int64),
            method_of=method_of, region_of=region_of, stage_label_of=stage_label_of,
        )


# A default config matching the paper's ORIGINAL 10-MSC / 23-tool catalog,
# used only by the synthetic demo builders below so run_demo.py keeps
# working without any real data attached.
_DEFAULT_MSC_LABELS = [
    "rough_milling", "secondary_rough_milling",
    "semi_finish_milling_sides", "semi_finish_milling_sides_5axis",
    "semi_finish_milling_bottom",
    "finish_milling_sides", "finish_milling_sides_5axis",
    "finish_milling_bottom", "finish_milling_corners", "finish_milling_corners_5axis",
]
_DEFAULT_TOOL_LABELS = [
    "Face Mill D50 R1", "End Mill D30 R5", "End Mill D30 R3", "End Mill D30 R1",
    "End Mill D25 R5", "End Mill D25 R3", "End Mill D25 R1",
    "End Mill D20 R5", "End Mill D20 R3", "End Mill D20 R1",
    "End Mill D18 R5", "End Mill D18 R3", "End Mill D18 R1",
    "End Mill D16 R5", "End Mill D16 R3", "End Mill D16 R1",
    "End Mill D12 R5", "End Mill D12 R3", "End Mill D12 R1",
    "End Mill D10 R3", "End Mill D10 R1", "End Mill D8 R3", "End Mill D8 R1",
]
DEFAULT_CONFIG = AAGConfig.from_labels(_DEFAULT_MSC_LABELS, _DEFAULT_TOOL_LABELS)


@dataclass
class AAGPart:
    """One machining part's Attributed Adjacency Graph (static content only;
    dynamic attributes are tracked by the environment, not stored here)."""
    n_features: int
    feasibility: np.ndarray          # (n, n_t, n_c) -- P_i stacked; 0 = infeasible,
                                      # >0 = normalized cutting time
    distance: np.ndarray             # (n, n) normalized Euclidean distance matrix M
    edges: List[Tuple[int, int, Tuple[int, int, int]]]  # (src, dst, (dep,adj,datum) bits)
    feature_names: List[str] = field(default_factory=list)

    def datum_prereqs(self) -> List[List[int]]:
        """For each feature j, list of feature indices i that are its datum
        (datum bit set on edge i -> j) and therefore must be FULLY machined
        before j can receive any operation (Rule 4, Table 1)."""
        prereqs = [[] for _ in range(self.n_features)]
        for (i, j, bits) in self.edges:
            if bits[2] == 1:
                prereqs[j].append(i)
        return prereqs


def build_synthetic_part(cfg: AAGConfig = DEFAULT_CONFIG, n_features: int = 7,
                          seed: int = 0) -> AAGPart:
    """Builds a small synthetic AAG part with the same shapes/semantics your
    real aag_generator would output, for end-to-end testing of RAGT-PPO."""
    rng = np.random.default_rng(seed)
    n_t, n_c = cfg.n_t, cfg.n_c
    milling_ids = [i for i, c in enumerate(cfg.chain_of) if c == "milling"]
    rough_id = next((i for i in milling_ids if cfg.stage_of[i] == 0), None)
    secrough_id = next((i for i in milling_ids if cfg.stage_of[i] == 1), None)
    semi_ids = [i for i in milling_ids if cfg.stage_of[i] == 2]
    finish_ids = [i for i in milling_ids if cfg.stage_of[i] == 3]

    feas = np.zeros((n_features, n_t, n_c), dtype=np.float32)
    for f in range(n_features):
        if rough_id is not None:
            for t in rng.choice(n_t, size=min(3, n_t), replace=False):
                feas[f, t, rough_id] = rng.uniform(0.2, 1.0)
        if secrough_id is not None and rng.random() < 0.5:
            feas[f, rng.choice(n_t), secrough_id] = rng.uniform(0.2, 1.0)
        for m in semi_ids[:2] + finish_ids[:2]:
            feas[f, rng.choice(n_t), m] = rng.uniform(0.2, 1.0)

    coords = rng.uniform(0, 1, size=(n_features, 2))
    d = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
    d = d / (d.max() + 1e-8)

    edges: List[Tuple[int, int, Tuple[int, int, int]]] = []
    for j in range(1, n_features):
        edges.append((0, j, (0, 0, 1)))
    for i in range(n_features - 1):
        edges.append((i, i + 1, (0, 1, 0)))
        edges.append((i + 1, i, (0, 1, 0)))
    for i in range(1, n_features - 1):
        edges.append((i, i + 1, (1, 0, 0)))

    names = [f"F{i+1}" for i in range(n_features)]
    return AAGPart(n_features, feas, d.astype(np.float32), edges, names)
