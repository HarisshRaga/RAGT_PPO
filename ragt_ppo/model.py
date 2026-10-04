"""
model.py
--------
The RAGT-PPO network described in Step 2/3 of the design conversation:

  1. Node embedding  : static (P_i, flattened) + dynamic attrs -> d_model
  2. Distance branch : distance row M_i,* projected independently and
                        concatenated after the first encoder layer (keeps
                        Zhang et al.'s fix for the "distance-vector-size
                        scales with n" imbalance problem, backbone-agnostic)
  3. L x Relation-aware Graph Transformer encoder layers:
        - multi-head attention restricted to graph neighborhood (self + any
          edge, either direction)
        - edge attribute vector (dependence/adjacency/datum bits) -> small
          MLP -> per-head additive attention bias (generalizes "separate
          projection matrix per edge type" to arbitrary edge-type
          combinations, since a single edge can carry multiple bits, e.g.
          the paper's "101" = dependence+datum)
        - residual + LayerNorm, position-wise FFN, residual + LayerNorm
  4. POLICY HEAD (Option A): final node embeddings -> shared linear layer
     -> per-node (n_t*n_c) logits -> flatten -> feasibility+rule mask
     (Fig. 7 style) -> softmax. No SENet / no PMA in the policy head.
  5. VALUE HEAD (PMA): one learnable seed query cross-attends over the
     final node embeddings (keys/values via learned projections) -> single
     pooled vector, regardless of n -> MLP -> scalar v_t.

This file processes ONE graph (one part / one state) per forward call, which
is the natural granularity for this on-policy PPO setup (parts vary in n,
so naive batching needs padding+masking -- left as an extension point).
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .aag import N_EDGE_TYPES, AAGConfig


class RelationAwareEncoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model, self.n_heads = d_model, n_heads
        self.d_head = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        # edge-attribute -> per-head additive attention bias (relation-aware term)
        self.edge_bias_mlp = nn.Sequential(
            nn.Linear(N_EDGE_TYPES, d_model // 2), nn.ReLU(),
            nn.Linear(d_model // 2, n_heads),
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.ReLU(), nn.Linear(d_ff, d_model)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: torch.Tensor, neighbor_mask: torch.Tensor) -> torch.Tensor:
        """
        h              : (n, d_model)
        edge_index     : (2, m) long
        edge_attr      : (m, N_EDGE_TYPES) float, bits possibly co-occurring
        neighbor_mask  : (n, n) bool, True where attention is ALLOWED
                          (self-loops + any edge in either direction)
        """
        n = h.size(0)
        q = self.q_proj(h).view(n, self.n_heads, self.d_head)
        k = self.k_proj(h).view(n, self.n_heads, self.d_head)
        v = self.v_proj(h).view(n, self.n_heads, self.d_head)

        # scores: (heads, n, n)
        scores = torch.einsum("ihd,jhd->hij", q, k) / math.sqrt(self.d_head)

        # relation-aware additive bias, aggregated over (possibly several)
        # edges between the same ordered pair -- sum of per-edge biases
        if edge_index.numel() > 0:
            eb = self.edge_bias_mlp(edge_attr)                     # (m, heads)
            bias = torch.zeros(self.n_heads, n, n, device=h.device)
            src, dst = edge_index[0], edge_index[1]
            bias[:, dst, src] += eb.t()   # node dst attends TO src -> bias at (dst,src)
            scores = scores + bias

        neg_inf = torch.finfo(scores.dtype).min
        allow = neighbor_mask.unsqueeze(0)                          # (1, n, n)
        scores = scores.masked_fill(~allow, neg_inf)

        attn = torch.softmax(scores, dim=-1)                        # (heads, n, n)
        out = torch.einsum("hij,jhd->ihd", attn, v).reshape(n, self.d_model)
        out = self.out_proj(out)

        h = self.norm1(h + self.dropout(out))
        h = self.norm2(h + self.dropout(self.ffn(h)))
        return h


class PMAValueHead(nn.Module):
    """Pooling by Multihead Attention (Lee et al., 2019, Set Transformer):
    a single learnable seed query attends over all node embeddings,
    producing ONE fixed-size vector regardless of n."""

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.d_model, self.n_heads = d_model, n_heads
        self.d_head = d_model // n_heads
        self.seed = nn.Parameter(torch.randn(1, d_model) * 0.02)     # learnable query S
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.q_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, 1)
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        n = h.size(0)
        q = self.q_proj(self.seed).view(1, self.n_heads, self.d_head)         # (1,H,dh)
        k = self.k_proj(h).view(n, self.n_heads, self.d_head)                 # (n,H,dh)
        v = self.v_proj(h).view(n, self.n_heads, self.d_head)
        scores = torch.einsum("qhd,nhd->hqn", q, k) / math.sqrt(self.d_head)  # (H,1,n)
        attn = torch.softmax(scores, dim=-1)
        pooled = torch.einsum("hqn,nhd->qhd", attn, v).reshape(1, self.d_model)
        pooled = self.out_proj(pooled)                                        # (1, d_model)
        value = self.mlp(pooled).squeeze(-1)                                  # (1,)
        return value.squeeze(0)                                               # scalar tensor


class RAGT_PPO(nn.Module):
    def __init__(self, cfg: AAGConfig, d_model: int = 128, n_heads: int = 4,
                 n_layers: int = 3, d_ff: int = 256, dist_dim: int = 32):
        super().__init__()
        self.cfg = cfg
        static_dim = cfg.n_t * cfg.n_c
        dyn_dim = 2 * cfg.n_c + cfg.n_t + 2

        self.node_in = nn.Sequential(
            nn.Linear(static_dim + dyn_dim, d_model), nn.ReLU(),
        )
        # distance branch: variable-length row (size n) -> fixed dist_dim,
        # via a size-agnostic linear-on-each-entry + mean/max pool, exactly
        # mirroring Zhang et al.'s "extract distance matrix independently,
        # project into a fixed-dimensional space" fix (Sec 3.2.5)
        self.dist_embed = nn.Sequential(nn.Linear(1, dist_dim), nn.ReLU())
        self.dist_merge = nn.Linear(dist_dim * 2, d_model)
        self.post_dist = nn.Sequential(nn.Linear(d_model * 2, d_model), nn.ReLU())

        self.layers = nn.ModuleList([
            RelationAwareEncoderLayer(d_model, n_heads, d_ff) for _ in range(n_layers)
        ])

        # ---- Policy head (Option A): shared linear per node, no gating module
        self.policy_proj = nn.Linear(d_model, cfg.n_t * cfg.n_c)

        # ---- Value head: PMA
        self.value_head = PMAValueHead(d_model, n_heads)

        self.d_model = d_model

    # ------------------------------------------------------------------
    def _neighbor_mask(self, n: int, edge_index: torch.Tensor, device) -> torch.Tensor:
        mask = torch.eye(n, dtype=torch.bool, device=device)
        if edge_index.numel() > 0:
            src, dst = edge_index[0], edge_index[1]
            mask[dst, src] = True
            mask[src, dst] = True   # attention allowed both directions along any edge
        return mask

    # ------------------------------------------------------------------
    def encode(self, static, dyn, distance, edge_index, edge_attr) -> torch.Tensor:
        n = static.size(0)
        device = static.device
        h = self.node_in(torch.cat([static, dyn], dim=-1))            # (n, d_model)

        # distance branch: per-row mean+max pool of the per-entry embedding
        d_embed = self.dist_embed(distance.unsqueeze(-1))              # (n, n, dist_dim)
        d_pool = torch.cat([d_embed.mean(dim=1), d_embed.max(dim=1).values], dim=-1)
        d_feat = self.dist_merge(d_pool)                               # (n, d_model)
        h = self.post_dist(torch.cat([h, d_feat], dim=-1))             # (n, d_model)

        neighbor_mask = self._neighbor_mask(n, edge_index, device)
        for layer in self.layers:
            h = layer(h, edge_index, edge_attr, neighbor_mask)
        return h                                                       # H = {h_1..h_n}

    # ------------------------------------------------------------------
    def forward(self, state: dict):
        h = self.encode(state["static"], state["dyn"], state["distance"],
                         state["edge_index"], state["edge_attr"])

        # ---- Option A policy head: no pooling/gating module, straight from H
        logits = self.policy_proj(h).reshape(-1)                       # (n*n_t*n_c,)
        mask = state["mask"]
        logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
        dist = torch.distributions.Categorical(logits=logits)

        # ---- PMA value head
        value = self.value_head(h)
        return dist, value
