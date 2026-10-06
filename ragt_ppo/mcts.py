"""
mcts.py
-------
Application-phase Monte Carlo Tree Search on top of a TRAINED RAGT-PPO
network, following Zhang et al. (2025) Sec. 3.2.6 / Fig. 9-10. Training is
untouched -- MCTS only replaces the greedy "take the argmax action" loop
when generating the final machining process scheme for a part.

Per time step t, a fresh tree is rooted at s_t and grown for `n_iterations`
iterations of:

  Selection     : from the root, repeatedly pick the child maximizing Eq. 7.
                  Two variants (`selection=`):
                  "paper" -- literal Eq. 7
                      Q(s,a)/Q(s) + C * P(s,a) / (1 + N(s,a))
                    with Q the ACCUMULATED (summed) benefit, so the first
                    term is the child's share of its parent's benefit in
                    [0, 1]. In practice this never explores: the first
                    visited child takes share ~1, unvisited siblings score
                    only C*P(s,a), and the exploration term only shrinks,
                    so the search reproduces the greedy policy.
                  "puct" (default) -- same structure, but Q(s,a) is the
                    MEAN benefit min-max normalized over visited siblings
                    (still in [0, 1]) and the exploration term carries
                    Silver et al.'s sqrt(N(s)) factor:
                      Qnorm(s,a) + C * P(s,a) * sqrt(N(s)) / (1 + N(s,a))
  Expansion     : at a leaf, list the legal actions whose prior is >= p_e =
                  (p_max + p_min) / 2 (p_max/p_min taken over the legal
                  actions), then pick one of them with Eq. 7 and step into it.
  Simulation    : from the new node, `n_rollouts` policy rollouts of at most
                  `sim_depth` steps (`rollout="greedy"` = argmax actions,
                  default; `"sample"` = sampled from the policy, which is
                  much noisier unless the policy is sharply peaked; with
                  greedy rollouts n_rollouts > 1 adds nothing); each gives
                  return R and
                  Score = (V(s) + gamma*R) / (1 + gamma), V from the value
                  head. The outcome is the mean Score.
  Backprop      : visit counts +1 and the outcome added to the benefit value
                  of every node on the path.

After the iterations, the root child with the most visits is executed, and
the search restarts from s_{t+1}.

One deliberate addition to the paper's description: the benefit added to a
node also includes the immediate rewards on the tree edges between that node
and the evaluated leaf. Without it, sibling actions that differ only in
their own cutting / non-cutting time (exactly the lead-time signal) would be
indistinguishable to the search.
"""
from __future__ import annotations
import torch

from .env import MachiningSequencingEnv
from .aag import AAGPart, AAGConfig
from .reward import TERMINAL_BONUS
from .ppo import RolloutBuffer


class _Node:
    __slots__ = ("env", "edge_reward", "edge_info", "terminal", "completed",
                 "priors", "value", "children", "expanded", "visits", "value_sum")

    def __init__(self, env: MachiningSequencingEnv, edge_reward: float = 0.0,
                 edge_info: dict | None = None, terminal: bool = False):
        self.env = env
        self.edge_reward = edge_reward     # reward of the action that led here
        self.edge_info = edge_info or {}
        self.terminal = terminal
        self.completed = env.rules.all_fully_machined()
        self.priors = None                 # {action: P(s,a)} over legal actions
        self.value = None                  # V(s) from the value head
        self.children: dict[int, "_Node | None"] = {}   # None = listed but not created
        self.expanded = False
        self.visits = 0
        self.value_sum = 0.0


class MCTSPlanner:
    def __init__(self, model, n_iterations: int = 50, c_puct: float = 1.0,
                 sim_depth: int = 10, n_rollouts: int = 1, score_gamma: float = 1.0,
                 selection: str = "puct", rollout: str = "greedy"):
        assert selection in ("puct", "paper")
        assert rollout in ("greedy", "sample")
        self.selection = selection
        self.rollout = rollout
        self.model = model
        self.n_iterations = n_iterations
        self.c = c_puct
        self.sim_depth = sim_depth
        self.n_rollouts = n_rollouts
        self.score_gamma = score_gamma

    # ---- network evaluation ---------------------------------------------
    @torch.no_grad()
    def _evaluate(self, node: _Node):
        """Fills node.priors (legal actions only) and node.value."""
        if node.priors is not None:
            return
        state = node.env.get_state()
        mask = state["mask"]
        if not bool(mask.any()):
            node.priors, node.value = {}, 0.0
            return
        dist, value = self.model(state)
        legal = mask.nonzero(as_tuple=True)[0]
        probs = dist.probs[legal]
        node.priors = {int(a): float(p) for a, p in zip(legal.tolist(), probs.tolist())}
        node.value = float(value.item())

    # ---- Eq. 7 ------------------------------------------------------------
    def _select_action(self, node: _Node) -> int:
        if self.selection == "paper":
            # literal Eq. 7: summed child benefit / summed parent benefit
            parent_q = node.value_sum
            def q_term(child):
                return child.value_sum / parent_q if parent_q > 1e-8 else 0.0
            explore = 1.0
        else:
            # "puct": mean benefit min-max normalized over visited siblings
            # (keeps the term in [0, 1] as the paper intends) + Silver et
            # al.'s sqrt(N(s)) so unvisited siblings eventually get tried
            means = [ch.value_sum / ch.visits for ch in node.children.values()
                     if ch is not None and ch.visits > 0]
            lo, hi = (min(means), max(means)) if means else (0.0, 0.0)
            def q_term(child):
                return (child.value_sum / child.visits - lo) / (hi - lo) if hi - lo > 1e-8 else 0.0
            explore = max(1.0, node.visits) ** 0.5

        best_a, best_score = None, -float("inf")
        for a, child in node.children.items():
            visited = child is not None and child.visits > 0
            n_sa = child.visits if visited else 0
            score = (q_term(child) if visited else 0.0) + self.c * node.priors[a] * explore / (1 + n_sa)
            if score > best_score:
                best_a, best_score = a, score
        return best_a

    def _child(self, node: _Node, a: int) -> _Node:
        child = node.children[a]
        if child is None:
            env = node.env.clone()
            res = env.step(a)
            terminal = res.done or not env.rules.build_mask().any()
            child = _Node(env, res.reward, res.info, terminal)
            node.children[a] = child
        return child

    # ---- expansion with policy-based pruning --------------------------------
    def _expand(self, node: _Node):
        self._evaluate(node)
        if node.priors:
            p_max, p_min = max(node.priors.values()), min(node.priors.values())
            p_e = (p_max + p_min) / 2.0
            node.children = {a: None for a, p in node.priors.items() if p >= p_e}
        node.expanded = True

    # ---- simulation ---------------------------------------------------------
    @torch.no_grad()
    def _simulate(self, node: _Node) -> float:
        if node.terminal:
            return TERMINAL_BONUS if node.completed else 0.0
        self._evaluate(node)
        scores = []
        for _ in range(self.n_rollouts):
            env = node.env.clone()
            ret = 0.0
            for _ in range(self.sim_depth):
                state = env.get_state()
                if not bool(state["mask"].any()):
                    break
                dist, _ = self.model(state)
                a = dist.probs.argmax() if self.rollout == "greedy" else dist.sample()
                res = env.step(int(a.item()))
                ret += res.reward
                if res.done:
                    if env.rules.all_fully_machined():
                        ret += TERMINAL_BONUS
                    break
            scores.append((node.value + self.score_gamma * ret) / (1.0 + self.score_gamma))
        return sum(scores) / len(scores)

    # ---- one search from the current state ---------------------------------
    def search(self, env: MachiningSequencingEnv) -> int:
        root = _Node(env.clone())
        self._expand(root)
        if len(root.children) == 1:          # nothing to search over
            return next(iter(root.children))

        for _ in range(self.n_iterations):
            node, path = root, [root]
            # selection
            while node.expanded and not node.terminal and node.children:
                node = self._child(node, self._select_action(node))
                path.append(node)
            # expansion (a visited, unexpanded leaf -> grow one child below it)
            if not node.terminal and not node.expanded and node.visits > 0:
                self._expand(node)
                if node.children:
                    node = self._child(node, self._select_action(node))
                    path.append(node)
            # simulation
            outcome = self._simulate(node)
            # backpropagation: each node's benefit = reward of its own incoming
            # action + rewards on the path below it + the simulation outcome
            acc = outcome
            for n in reversed(path):
                acc += n.edge_reward
                n.visits += 1
                n.value_sum += acc

        def rank(a):
            ch = root.children[a]
            return ((ch.visits if ch is not None else 0), root.priors[a])
        return max(root.children, key=rank)


def plan_with_mcts(model, part: AAGPart, cfg: AAGConfig, device="cpu",
                   max_steps: int | None = None, **mcts_kwargs
                   ) -> tuple[RolloutBuffer, MachiningSequencingEnv]:
    """Drop-in counterpart of ppo.collect_episode(..., greedy=True) that picks
    each action by MCTS. Returns a RolloutBuffer with only rewards / dones /
    cutting / non-cutting times filled, so episode_lead_time() works on it."""
    was_training = model.training
    model.eval()
    planner = MCTSPlanner(model, **mcts_kwargs)
    env = MachiningSequencingEnv(part, cfg, device=device)
    buf = RolloutBuffer()
    steps = 0
    try:
        while True:
            if not env.rules.build_mask().any():
                break                       # dead end: nothing legal left
            action = planner.search(env)
            res = env.step(action)
            buf.actions.append(action)
            buf.rewards.append(res.reward)
            buf.dones.append(res.done)
            buf.cutting_times.append(res.info.get("cutting_time", 0.0))
            buf.non_cutting_times.append(res.info.get("non_cutting_time", 0.0))
            steps += 1
            if res.done or (max_steps is not None and steps >= max_steps):
                break
    finally:
        model.train(was_training)
    return buf, env
