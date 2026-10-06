"""
env.py
------
The environment: state s_t = current AAG (static P_i/distance graph +
per-feature dynamic attributes), action a_t = machining operation
(feature, tool, MSC), transition dynamics and termination conditions
follow Zhang et al. (2025) Sec. 3.2.3 exactly:

  * invalid action (fails Table-1 rules)          -> s_t+1 = s_t, r_t = 0
  * valid action                                  -> dynamic attrs updated,
                                                      r_t from lead-time formula
  * episode ends when: (1) every feature fully machined, cumulative reward
                            = TERMINAL_BONUS, or
                        (2) N_s consecutive invalid actions, cumulative
                            reward = 0

Catalog sizes (n_t, n_c) and MSC method (3axis/5axis) come from the
`AAGConfig` you pass in -- NOT from hardcoded constants -- so this works
unmodified against your real, extended 14-MSC/36-tool catalog.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import torch

from .aag import AAGPart, AAGConfig
from .rules import RuleEngine
from .reward import non_cutting_time, immediate_reward, TERMINAL_BONUS

N_S_MAX_INVALID = 50           # matches Table 3 of the paper


@dataclass
class StepResult:
    reward: float
    done: bool
    info: dict


class MachiningSequencingEnv:
    """One-episode-per-part environment (Application phase style: initial
    state has all dynamic attributes zeroed, per Zhang et al. Sec. 3.2.6)."""

    def __init__(self, part: AAGPart, cfg: AAGConfig, device: str = "cpu"):
        self.part = part
        self.cfg = cfg
        self.n = part.n_features
        self.device = device
        self.reset()

    # ------------------------------------------------------------------
    def reset(self):
        self.rules = RuleEngine(self.part, self.cfg)
        self.prev_feature = None
        self.prev_tool = None
        self.prev_method = None
        self.consecutive_invalid = 0
        self.log = []             # list of executed operations, for the schedule output
        return self.get_state()

    # ------------------------------------------------------------------
    def clone(self) -> "MachiningSequencingEnv":
        """Independent copy of the current episode state (for MCTS look-ahead);
        the part and config are shared, not copied."""
        c = MachiningSequencingEnv.__new__(MachiningSequencingEnv)
        c.part, c.cfg, c.n, c.device = self.part, self.cfg, self.n, self.device
        c.rules = self.rules.clone()
        c.prev_feature, c.prev_tool, c.prev_method = self.prev_feature, self.prev_tool, self.prev_method
        c.consecutive_invalid = self.consecutive_invalid
        c.log = list(self.log)
        return c

    # ------------------------------------------------------------------
    def get_state(self) -> dict:
        """Packs everything the RAGT-PPO network needs, as torch tensors."""
        n_t, n_c = self.cfg.n_t, self.cfg.n_c
        dev = self.device
        static = torch.tensor(
            self.part.feasibility.reshape(self.n, n_t * n_c), dtype=torch.float32, device=dev
        )
        dyn = torch.tensor(
            np.stack([d.as_vector() for d in self.rules.dyn]), dtype=torch.float32, device=dev
        )
        distance = torch.tensor(self.part.distance, dtype=torch.float32, device=dev)
        if len(self.part.edges) > 0:
            src = [e[0] for e in self.part.edges]
            dst = [e[1] for e in self.part.edges]
            attrs = [e[2] for e in self.part.edges]
            edge_index = torch.tensor([src, dst], dtype=torch.long, device=dev)
            edge_attr = torch.tensor(attrs, dtype=torch.float32, device=dev)
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long, device=dev)
            edge_attr = torch.zeros((0, self.cfg.n_edge_types), dtype=torch.float32, device=dev)
        mask = torch.tensor(self.rules.build_mask(), dtype=torch.bool, device=dev)
        return dict(static=static, dyn=dyn, distance=distance,
                    edge_index=edge_index, edge_attr=edge_attr,
                    mask=mask.reshape(self.n * n_t * n_c))

    # ------------------------------------------------------------------
    def step(self, action_index: int) -> StepResult:
        n_t, n_c = self.cfg.n_t, self.cfg.n_c
        f, t, m = np.unravel_index(action_index, (self.n, n_t, n_c))
        legal_mask = self.rules.build_mask()

        if not legal_mask[f, t, m]:
            self.consecutive_invalid += 1
            done = self.consecutive_invalid >= N_S_MAX_INVALID
            return StepResult(reward=0.0, done=done,
                               info={"cumulative_reward": 0.0 if done else None,
                                     "cutting_time": 0.0, "non_cutting_time": 0.0})

        self.consecutive_invalid = 0
        cutting_time_hat = float(self.part.feasibility[f, t, m])
        curr_method = self.cfg.method_of[m]
        nct = non_cutting_time(
            self.part, self.prev_feature, f, self.prev_tool, t,
            self.prev_method, curr_method,
        )
        r = immediate_reward(cutting_time_hat, nct)

        self.rules.apply(f, t, m)
        self.log.append((f, t, m))
        self.prev_feature, self.prev_tool, self.prev_method = f, t, curr_method

        done = self.rules.all_fully_machined()
        info = {"cumulative_reward": TERMINAL_BONUS if done else None,
                "cutting_time": cutting_time_hat, "non_cutting_time": nct}
        return StepResult(reward=r, done=done, info=info)

    # ------------------------------------------------------------------
    def action_dim(self) -> int:
        return self.n * self.cfg.n_t * self.cfg.n_c
