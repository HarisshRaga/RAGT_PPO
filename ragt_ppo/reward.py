"""
reward.py
---------
Goal: minimize LEAD TIME = cutting time + non-cutting time, where

    cutting time      = normalized T_c from P_i (exactly as produced by
                         aag_generator / Eq. 1-2 of Zhang et al.)
    non-cutting time  = tool travel time + tool change time + orientation
                         change time

This mirrors Eq. 3 of Zhang et al. (2025) structurally (immediate reward
built from a weighted, normalized cutting/non-cutting split, zero on rule
violation, terminal bonus of 1/0.367 on full completion) but extends the
non-cutting term with an explicit orientation-change component, since your
tool catalog spans 3-axis and 5-axis machining methods and the original
paper's T_n only modeled travel + tool-change time.

It also charges a small fixed cost whenever consecutive operations are on
features with different tool approach directions (TAD, from
static_attributes.tool_approach_direction in the AAG file). Unlike the AAG
generator's noncutting.py (0.6 min for a 90-degree change, 2.0 min for a
flip), this is deliberately ONE flat, small cost for any change: a nudge to
group same-direction features, not a full re-fixturing model.
"""
from __future__ import annotations
import numpy as np
from .aag import AAGPart

TAU = 0.8              # same weighting hyperparameter as Zhang et al. Eq. 3
TOOL_CHANGE_COST = 0.15        # normalized fixed cost per tool change
ORIENTATION_CHANGE_COST = 0.35  # normalized fixed cost per 3-axis <-> 5-axis switch
# Fixed cost per change of tool approach direction between consecutive
# operations, kept to a few seconds of machine time. Converted to the same
# normalized scale as TOOL_CHANGE_COST by treating that constant as one real
# tool change (0.5 min = 30 s in noncutting.py): 5 s -> 0.15 * 5/30 = 0.025.
TAD_CHANGE_SECONDS = 5.0
TOOL_CHANGE_SECONDS = 30.0
TAD_CHANGE_COST = TOOL_CHANGE_COST * TAD_CHANGE_SECONDS / TOOL_CHANGE_SECONDS
TERMINAL_BONUS = 1.0 / 0.367   # identical constant to Zhang et al. / Schulman 2016


def non_cutting_time(part: AAGPart, prev_feature: int, curr_feature: int,
                      prev_tool: int, curr_tool: int,
                      prev_method: str, curr_method: str) -> float:
    """Normalized non-cutting time for the transition into this operation."""
    travel = 0.0 if prev_feature is None else float(part.distance[prev_feature, curr_feature])
    tool_change = 0.0 if (prev_tool is None or prev_tool == curr_tool) else TOOL_CHANGE_COST
    orientation_change = 0.0 if (prev_method is None or prev_method == curr_method) else ORIENTATION_CHANGE_COST
    return travel + tool_change + orientation_change + tad_change_cost(part, prev_feature, curr_feature)


def tad_change_cost(part: AAGPart, prev_feature: int, curr_feature: int) -> float:
    """TAD_CHANGE_COST if the two features' tool approach directions differ
    (any angle), else 0. Zero for the first operation and for parts loaded
    without TAD data (e.g. synthetic parts)."""
    tad = part.tool_approach_direction
    if prev_feature is None or tad is None:
        return 0.0
    same = float(np.dot(tad[prev_feature], tad[curr_feature])) > 1.0 - 1e-6
    return 0.0 if same else TAD_CHANGE_COST


def immediate_reward(cutting_time_hat: float, non_cutting_time_hat: float) -> float:
    """Eq. 3-style immediate reward, r_t in [~ -tau/(1+tau), 1]."""
    return 1.0 - (cutting_time_hat + TAU * non_cutting_time_hat) / (1.0 + TAU)
