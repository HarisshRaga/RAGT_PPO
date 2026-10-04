"""
ppo.py
------
Standard clipped-surrogate PPO + GAE (Schulman et al. 2017 / 2015), trained
on-policy against MachiningSequencingEnv. Kept deliberately close to Zhang
et al.'s own training recipe (Table 3): Adam, gamma=0.99, GAE lambda=0.95,
multiple epochs per collected batch, clipped objective.

For clarity this reference trainer collects ONE full episode (one complete
machining-process rollout for one part) per update; swap in multiple
parallel envs / parts for the "200 parallel environments" style speed-up
the paper uses (Sec 4.3.1) once you're running on real parts.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List
import torch
import torch.nn.functional as F

from .env import MachiningSequencingEnv
from .aag import AAGPart, AAGConfig


@dataclass
class RolloutBuffer:
    states: List[dict] = field(default_factory=list)
    actions: List[int] = field(default_factory=list)
    logprobs: List[torch.Tensor] = field(default_factory=list)
    values: List[torch.Tensor] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    dones: List[bool] = field(default_factory=list)
    cutting_times: List[float] = field(default_factory=list)     # for reporting
    non_cutting_times: List[float] = field(default_factory=list)  # (not for the loss)


def collect_episode(model, part: AAGPart, cfg: AAGConfig, device="cpu", greedy: bool = False,
                     max_steps: int | None = None) -> tuple[RolloutBuffer, MachiningSequencingEnv]:
    env = MachiningSequencingEnv(part, cfg, device=device)
    state = env.get_state()
    buf = RolloutBuffer()

    done = False
    steps = 0
    while not done:
        dist, value = model(state)
        action = dist.probs.argmax() if greedy else dist.sample()
        logprob = dist.log_prob(action)

        result = env.step(int(action.item()))

        buf.states.append(state)
        buf.actions.append(int(action.item()))
        buf.logprobs.append(logprob.detach())
        buf.values.append(value.detach())
        buf.rewards.append(result.reward)
        buf.dones.append(result.done)
        buf.cutting_times.append(result.info.get("cutting_time", 0.0))
        buf.non_cutting_times.append(result.info.get("non_cutting_time", 0.0))

        done = result.done
        state = env.get_state()
        steps += 1
        if max_steps is not None and steps >= max_steps:
            break

    return buf, env


def episode_lead_time(buf: RolloutBuffer) -> float:
    """Total normalized lead time actually incurred this episode
    (cutting + non-cutting), the metric the design brief asks to minimize --
    distinct from the shaped PPO reward used for the gradient."""
    return float(sum(buf.cutting_times) + sum(buf.non_cutting_times))


def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    values = values + [torch.tensor(0.0)]     # bootstrap terminal value = 0
    advantages, gae = [], 0.0
    for t in reversed(range(len(rewards))):
        mask = 0.0 if dones[t] else 1.0
        delta = rewards[t] + gamma * values[t + 1].item() * mask - values[t].item()
        gae = delta + gamma * lam * mask * gae
        advantages.insert(0, gae)
    returns = [a + v.item() for a, v in zip(advantages, values[:-1])]
    return torch.tensor(advantages, dtype=torch.float32), torch.tensor(returns, dtype=torch.float32)


def ppo_update(model, optimizer, buf: RolloutBuffer, clip_eps=0.2,
                n_epochs=20, value_coef=0.5, entropy_coef=0.01):
    advantages, returns = compute_gae(buf.rewards, buf.values, buf.dones)
    if advantages.std() > 1e-6:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    batch = dict(states=buf.states, actions=buf.actions, logprobs=buf.logprobs,
                 advantages=advantages, returns=returns)
    stats = ppo_update_from_batch(model, optimizer, batch, clip_eps, n_epochs,
                                   value_coef, entropy_coef)
    stats["episode_return"] = sum(buf.rewards)
    return stats


# ---------------------------------------------------------------------------
# Multi-part batching: collect one episode per part in a mini-batch, compute
# GAE PER EPISODE (trajectories from different parts must not be mixed), then
# concatenate everything into one flat batch for a single PPO gradient step.
# ---------------------------------------------------------------------------
def collect_batch(model, parts: List[AAGPart], cfg: AAGConfig, device="cpu", max_steps=None):
    all_states, all_actions, all_logprobs = [], [], []
    all_advantages, all_returns = [], []
    episode_stats = []

    for part in parts:
        buf, env = collect_episode(model, part, cfg, device=device, greedy=False,
                                    max_steps=max_steps)
        adv, ret = compute_gae(buf.rewards, buf.values, buf.dones)
        if adv.std() > 1e-6:
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        all_states.extend(buf.states)
        all_actions.extend(buf.actions)
        all_logprobs.extend(buf.logprobs)
        all_advantages.append(adv)
        all_returns.append(ret)

        episode_stats.append(dict(
            n_features=part.n_features, steps=len(buf.rewards),
            episode_return=sum(buf.rewards), lead_time=episode_lead_time(buf),
            fully_machined=env.rules.all_fully_machined(),
        ))

    batch = dict(states=all_states, actions=all_actions, logprobs=all_logprobs,
                 advantages=torch.cat(all_advantages), returns=torch.cat(all_returns))
    return batch, episode_stats


def ppo_update_from_batch(model, optimizer, batch: dict, clip_eps=0.2,
                           n_epochs=10, value_coef=0.5, entropy_coef=0.01):
    device = next(model.parameters()).device
    old_logprobs = torch.stack(batch["logprobs"]).to(device)
    actions = torch.tensor(batch["actions"], dtype=torch.long, device=device)
    advantages = batch["advantages"].to(device)
    returns = batch["returns"].to(device)
    states = batch["states"]

    for _ in range(n_epochs):
        new_logprobs, new_values, entropies = [], [], []
        for s, a in zip(states, actions):
            dist, v = model(s)
            new_logprobs.append(dist.log_prob(a))
            new_values.append(v)
            entropies.append(dist.entropy())
        new_logprobs = torch.stack(new_logprobs)
        new_values = torch.stack(new_values)
        entropies = torch.stack(entropies)

        ratio = torch.exp(new_logprobs - old_logprobs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        value_loss = F.mse_loss(new_values, returns)
        entropy_bonus = entropies.mean()

        loss = policy_loss + value_coef * value_loss - entropy_coef * entropy_bonus

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

    return dict(policy_loss=policy_loss.item(), value_loss=value_loss.item(),
                entropy=entropy_bonus.item())


def evaluate(model, parts: List[AAGPart], cfg: AAGConfig, device="cpu", max_steps=None) -> dict:
    """Greedy rollout on every part in `parts` (val or test set). Returns
    per-part stats plus dataset-level aggregates."""
    per_part = []
    for part in parts:
        buf, env = collect_episode(model, part, cfg, device=device, greedy=True,
                                    max_steps=max_steps)
        per_part.append(dict(
            n_features=part.n_features, steps=len(buf.rewards),
            episode_return=sum(buf.rewards), lead_time=episode_lead_time(buf),
            fully_machined=env.rules.all_fully_machined(),
        ))
    completion_rate = sum(p["fully_machined"] for p in per_part) / len(per_part)
    avg_lead_time = sum(p["lead_time"] for p in per_part) / len(per_part)
    avg_return = sum(p["episode_return"] for p in per_part) / len(per_part)
    return dict(per_part=per_part, completion_rate=completion_rate,
                avg_lead_time=avg_lead_time, avg_return=avg_return)
