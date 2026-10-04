"""
rules.py
--------
Implements the eight hard rules from Table 1 of Zhang et al. (2025) / the
design doc's Section 1.1, as an ACTION-MASKING function (never as an
environment crash): a rule violation costs a zero reward and leaves the
state unchanged, exactly as in the paper.

Rule -> mechanism used here
 1-3. Stage order within a chain (milling: rough -> secondary-rough ->
      semi-finish -> finish; hole: drilling -> reaming -> tapping ->
      chamfering)                                   -> _stage_cleared_for_msc
 4. Datum features machined first                     -> _datum_ready
 5. Roughing tool != semi-finish/finish tool           -> tool-reuse gate
    (milling chain only, tool wear)
 6. No further machining of an already finish-machined
    region                                             -> subsumed by (8):
    each (feature, MSC) pair is unique, and MSC already encodes the region
 7. Fully machined features need no further ops        -> fully-machined gate
 8. Each feature can receive a given MSC only once      -> applied-MSC gate

This module is CATALOG-SIZE-AGNOSTIC: it reads n_t/n_c/chain/stage
metadata from the `AAGConfig` you pass in, so it works unmodified whether
your dataset has the paper's original 10 MSCs / 23 tools or your real,
extended 14 MSC / 36 tool catalog.
"""
from __future__ import annotations
import numpy as np
from .aag import AAGPart, AAGConfig


class FeatureDynamicState:
    """Dynamic attributes d_i^(t) for one feature (Zhang et al. Sec 3.2.1),
    sized from AAGConfig rather than a hardcoded constant."""

    def __init__(self, cfg: AAGConfig):
        self.cfg = cfg
        self.msc_applied = np.zeros(cfg.n_c, dtype=np.float32)     # rule 8 memory
        self.msc_prev = np.zeros(cfg.n_c, dtype=np.float32)
        self.tool_prev = np.zeros(cfg.n_t, dtype=np.float32)
        self.machined_prev = 0.0
        self.fully_machined = False
        self.rough_tool = -1                                        # rule 5 memory

    def as_vector(self) -> np.ndarray:
        return np.concatenate([
            self.msc_applied, self.msc_prev, self.tool_prev,
            np.array([self.machined_prev, float(self.fully_machined)], dtype=np.float32)
        ])


class RuleEngine:
    """Stateful mask builder: tracks dynamic attributes for every feature in
    a part and produces the feasibility mask over the full n*n_t*n_c action
    space at every step, enforcing rules 1-8."""

    def __init__(self, part: AAGPart, cfg: AAGConfig):
        self.part = part
        self.cfg = cfg
        self.n = part.n_features
        self.dyn = [FeatureDynamicState(cfg) for _ in range(self.n)]
        self.datum_prereqs = part.datum_prereqs()
        self._rough_msc_ids = {m for m in range(cfg.n_c)
                                if cfg.chain_of[m] == "milling" and cfg.stage_of[m] == 0}

    # ---- helpers -----------------------------------------------------
    def _stage_cleared_for_msc(self, f: int, m: int) -> bool:
        """Rules 1-3 (and their hole-chain analogue): True if every
        feasible (P_i>0) MSC that (a) belongs to the SAME chain as m and
        (b) has a strictly earlier stage than m, has already been applied
        to feature f at least once."""
        chain, stage = self.cfg.chain_of[m], self.cfg.stage_of[m]
        if stage == 0:
            return True
        feas = self.part.feasibility[f]                # (n_t, n_c)
        for m2 in range(self.cfg.n_c):
            if self.cfg.chain_of[m2] != chain or self.cfg.stage_of[m2] >= stage:
                continue
            if feas[:, m2].max() > 0 and self.dyn[f].msc_applied[m2] == 0:
                return False
        return True

    def _datum_ready(self, f: int) -> bool:
        """Rule 4: every datum-prerequisite feature must be FULLY machined
        before feature f may receive any operation."""
        return all(self.dyn[p].fully_machined for p in self.datum_prereqs[f])

    def _recompute_fully_machined(self, f: int):
        feas = self.part.feasibility[f]
        feasible_mscs = np.where(feas.max(axis=0) > 0)[0]
        self.dyn[f].fully_machined = bool(
            all(self.dyn[f].msc_applied[m] == 1 for m in feasible_mscs)
        ) if len(feasible_mscs) > 0 else True

    # ---- public API ----------------------------------------------------
    def build_mask(self) -> np.ndarray:
        """Returns boolean mask of shape (n, n_t, n_c); True = legal action."""
        n_t, n_c = self.cfg.n_t, self.cfg.n_c
        mask = np.zeros((self.n, n_t, n_c), dtype=bool)
        for f in range(self.n):
            d = self.dyn[f]
            if d.fully_machined:                        # rule 7
                continue
            if not self._datum_ready(f):                 # rule 4
                continue
            feas = self.part.feasibility[f] > 0           # (n_t, n_c) feasibility gate
            for m in range(n_c):
                if d.msc_applied[m] == 1:                  # rule 8
                    continue
                if not self._stage_cleared_for_msc(f, m):  # rules 1-3 (+ hole chain)
                    continue
                for t in range(n_t):
                    if not feas[t, m]:
                        continue
                    if (self.cfg.chain_of[m] == "milling" and self.cfg.stage_of[m] >= 2
                            and t == d.rough_tool):          # rule 5
                        continue
                    mask[f, t, m] = True
        return mask

    def apply(self, f: int, t: int, m: int):
        """Executes a legal (feature, tool, msc) action: updates dynamic
        attributes exactly as Fig. 4 of the paper describes."""
        cfg = self.cfg
        d = self.dyn[f]
        d.msc_applied[m] = 1
        d.msc_prev = np.zeros(cfg.n_c, dtype=np.float32)
        d.msc_prev[m] = 1
        d.tool_prev = np.zeros(cfg.n_t, dtype=np.float32)
        d.tool_prev[t] = 1
        d.machined_prev = 1.0
        if m in self._rough_msc_ids and d.rough_tool == -1:
            d.rough_tool = t
        self._recompute_fully_machined(f)
        for other in range(self.n):
            if other != f:
                self.dyn[other].machined_prev = 0.0

    def all_fully_machined(self) -> bool:
        return all(d.fully_machined for d in self.dyn)
