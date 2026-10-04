"""
RAGT-PPO: Relation-Aware Graph Transformer + PPO for machining process sequencing.

Replaces the ConvGNN(PNA) backbone of Zhang et al. (2025, J. Manuf. Syst. 78:370-393)
with a relation-aware graph transformer, while keeping the paper's environment design
(state = AAG, action = machining operation, transition dynamics, termination
conditions) and Table 1 process rules intact.

Policy network : Option A  -> self-attention output feeds action logits directly
                               (no separate SENet / PMA gating module).
Value network  : PMA (Pooling by Multihead Attention, Lee et al. 2019) collapses
                               the variable-size node set into one fixed-size vector.
"""
