"""
train.py
--------
Full pipeline for your 100 AAG parts: 60/20/20 train/val/test split,
multi-part PPO training with periodic validation, best-checkpoint
selection, and a final test-set report + saved schedules.

    python3 train.py --data_dir /path/to/your/aag_parts        # real data
    python3 train.py --synthetic                                # dry run

Outputs (under --out_dir, default ./ragt_ppo_run relative to your working directory):
    best_model.pt                 -- checkpoint with best validation lead time
    training_log.csv              -- per-update train/val metrics
    test_report.csv               -- per-test-part lead time / completion
    schedules/test_part_XX.csv    -- Table-A.1-style schedule for each test part

With --use_mcts, every test part is ALSO planned with Monte Carlo Tree Search
on top of the trained policy/value networks (Zhang et al. 2025, Sec. 3.2.6).
Training and validation are unchanged; test_report.csv gains mcts_* columns
next to the greedy ones, and schedules/test_part_XX_mcts.csv is written too.
--eval_checkpoint skips training and only runs the test phase on an existing
best_model.pt (use the same --split_seed / --n_* so the test set matches).
"""
from __future__ import annotations
import argparse
import csv
import os
import random
import time

import numpy as np
import torch

from ragt_ppo.dataset import load_aag_dataset, split_dataset, build_synthetic_dataset
from ragt_ppo.model import RAGT_PPO
from ragt_ppo.ppo import (collect_batch, ppo_update_from_batch, evaluate, collect_episode,
                          episode_lead_time)
from ragt_ppo.mcts import plan_with_mcts
from ragt_ppo.schedule_output import build_schedule_table, save_schedule_csv


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train(args, model, optimizer, cfg, train_parts, val_parts, device) -> float:
    """Step 3: PPO training with periodic validation + best-checkpoint tracking.
    Returns the best validation lead time."""
    log_path = os.path.join(args.out_dir, "training_log.csv")
    log_rows = []
    best_val_lead_time = float("inf")
    best_ckpt_path = os.path.join(args.out_dir, "best_model.pt")

    for update in range(1, args.n_updates + 1):
        batch_parts = random.sample(train_parts, k=min(args.batch_size, len(train_parts)))
        batch, ep_stats = collect_batch(model, batch_parts, cfg, device=device)
        stats = ppo_update_from_batch(model, optimizer, batch,
                                       n_epochs=args.epochs_per_update,
                                       entropy_coef=args.entropy_coef)

        train_return = sum(s["episode_return"] for s in ep_stats) / len(ep_stats)
        train_lead_time = sum(s["lead_time"] for s in ep_stats) / len(ep_stats)
        train_completion = sum(s["fully_machined"] for s in ep_stats) / len(ep_stats)

        row = dict(update=update, split="train",
                   avg_return=train_return, avg_lead_time=train_lead_time,
                   completion_rate=train_completion,
                   policy_loss=stats["policy_loss"], value_loss=stats["value_loss"])
        log_rows.append(row)

        msg = (f"[update {update:4d}] train return={train_return:+.3f} "
               f"lead_time={train_lead_time:.3f} completion={train_completion:.0%}")

        if update % args.val_every == 0 or update == args.n_updates:
            val_metrics = evaluate(model, val_parts, cfg, device=device)
            log_rows.append(dict(update=update, split="val",
                                  avg_return=val_metrics["avg_return"],
                                  avg_lead_time=val_metrics["avg_lead_time"],
                                  completion_rate=val_metrics["completion_rate"],
                                  policy_loss=None, value_loss=None))
            msg += (f"  || val lead_time={val_metrics['avg_lead_time']:.3f} "
                    f"completion={val_metrics['completion_rate']:.0%}")

            # model-selection criterion: lowest validation lead time AMONG
            # checkpoints that at least match the best completion rate seen
            # so far (never trade completion for a marginal lead-time gain)
            if val_metrics["avg_lead_time"] < best_val_lead_time and val_metrics["completion_rate"] >= 0.5:
                best_val_lead_time = val_metrics["avg_lead_time"]
                torch.save(model.state_dict(), best_ckpt_path)
                msg += "  [saved best checkpoint]"

        print(msg)

    with open(log_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(log_rows[0].keys()))
        writer.writeheader()
        writer.writerows(log_rows)
    print(f"\nTraining log saved to {log_path}")
    return best_val_lead_time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default=None,
                     help="Directory of real AAG part files (aag_generator output)")
    ap.add_argument("--synthetic", action="store_true",
                     help="Use a 100-part synthetic dataset instead of --data_dir")
    ap.add_argument("--n_train", type=int, default=60)
    ap.add_argument("--n_val", type=int, default=20)
    ap.add_argument("--n_test", type=int, default=20)
    ap.add_argument("--split_seed", type=int, default=42)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_updates", type=int, default=300,
                     help="Number of PPO updates (each update = one mini-batch of parts)")
    ap.add_argument("--batch_size", type=int, default=8,
                     help="Number of training parts per PPO update")
    ap.add_argument("--epochs_per_update", type=int, default=8)
    ap.add_argument("--entropy_coef", type=float, default=0.03,
                     help="Higher values help the agent explore past 'almost "
                          "done' states to actually reach full completion")
    ap.add_argument("--val_every", type=int, default=10,
                     help="Run validation every N updates")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--d_model", type=int, default=128)
    ap.add_argument("--n_heads", type=int, default=4)
    ap.add_argument("--n_layers", type=int, default=3)
    ap.add_argument("--d_ff", type=int, default=256)
    ap.add_argument("--device", type=str, default="cpu",
                     help="'cpu' or 'cuda'. If 'cuda' is requested but unavailable the "
                          "script ABORTS instead of silently falling back to CPU.")
    ap.add_argument("--out_dir", type=str, default="./ragt_ppo_run",
                     help="Directory to write checkpoints/logs/schedules to "
                          "(created relative to wherever you run this script)")
    # ---- MCTS (application phase only; off unless --use_mcts is given) ----
    ap.add_argument("--use_mcts", action="store_true",
                     help="Also plan every test part with MCTS on top of the trained "
                          "networks (Zhang et al. 2025 Sec. 3.2.6) and report it "
                          "alongside the plain greedy policy")
    ap.add_argument("--mcts_iterations", type=int, default=50,
                     help="Search iterations per decision (per machining operation)")
    ap.add_argument("--mcts_c", type=float, default=1.0,
                     help="Exploration constant C in Eq. 7 (paper: 1)")
    ap.add_argument("--mcts_sim_depth", type=int, default=10,
                     help="Max policy-rollout steps per simulation")
    ap.add_argument("--mcts_rollouts", type=int, default=1,
                     help="Simulations averaged per expanded node")
    ap.add_argument("--mcts_score_gamma", type=float, default=1.0,
                     help="gamma in Score = (V(s) + gamma*R) / (1 + gamma) (paper: 1)")
    ap.add_argument("--mcts_selection", choices=["puct", "paper"], default="puct",
                     help="'paper' = literal Eq. 7 (degenerates to the greedy policy in "
                          "practice); 'puct' = normalized mean Q + sqrt(N(s)) exploration")
    ap.add_argument("--mcts_rollout", choices=["greedy", "sample"], default="greedy",
                     help="Simulation policy: argmax actions (default) or sampled from the "
                          "policy distribution (noisier; combine with --mcts_rollouts > 1)")
    ap.add_argument("--mcts_max_parts", type=int, default=0,
                     help="Plan only the first N test parts with MCTS (0 = all). "
                          "MCTS is far slower than greedy, so cap this on big test sets")
    ap.add_argument("--eval_checkpoint", type=str, default=None,
                     help="Skip training; load this checkpoint and run only the test phase")
    args = ap.parse_args()

    set_seed(args.seed)
    device = args.device
    if device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise SystemExit("error: --device cuda requested but torch.cuda.is_available() is False "
                             "(no GPU allocated, or torch build/driver mismatch)")
        print(f"Device: {device} ({torch.cuda.get_device_name(0)})")
    else:
        print(f"Device: {device}")
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, "schedules"), exist_ok=True)

    # ---- 1. Load + split data --------------------------------------------
    if args.synthetic:
        parts, cfg = build_synthetic_dataset(n_parts=100, seed=args.seed)
    else:
        assert args.data_dir is not None, "Pass --data_dir or --synthetic"
        parts, cfg = load_aag_dataset(args.data_dir)
    print(f"Catalog: n_c={cfg.n_c} MSCs, n_t={cfg.n_t} tools")
    assert len(parts) >= args.n_train + args.n_val + args.n_test, (
        f"Need >= {args.n_train + args.n_val + args.n_test} parts, found {len(parts)}"
    )
    train_parts, val_parts, test_parts = split_dataset(
        parts, n_train=args.n_train, n_val=args.n_val, n_test=args.n_test,
        seed=args.split_seed,
    )
    print(f"Dataset: {len(parts)} parts -> train={len(train_parts)} "
          f"val={len(val_parts)} test={len(test_parts)}")

    # ---- 2. Build model + optimizer ---------------------------------------
    model = RAGT_PPO(cfg, d_model=args.d_model, n_heads=args.n_heads,
                      n_layers=args.n_layers, d_ff=args.d_ff).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_ckpt_path = os.path.join(args.out_dir, "best_model.pt")
    if args.eval_checkpoint is None:
        best_val_lead_time = train(args, model, optimizer, cfg, train_parts, val_parts, device)

    # ---- 4. Final test-set evaluation with the BEST checkpoint -----------
    if args.eval_checkpoint is not None:
        model.load_state_dict(torch.load(args.eval_checkpoint, map_location=device))
        print(f"\nLoaded {args.eval_checkpoint} for testing (training skipped).")
    elif os.path.exists(best_ckpt_path):
        model.load_state_dict(torch.load(best_ckpt_path, map_location=device))
        print(f"\nLoaded best checkpoint (val lead_time={best_val_lead_time:.3f}) for testing.")
    else:
        print("\nNo checkpoint passed the completion-rate bar; testing with final weights.")

    n_mcts = 0
    if args.use_mcts:
        n_mcts = len(test_parts) if args.mcts_max_parts <= 0 else min(args.mcts_max_parts, len(test_parts))
        print(f"MCTS enabled for {n_mcts}/{len(test_parts)} test parts "
              f"(iterations={args.mcts_iterations}, C={args.mcts_c}, "
              f"sim_depth={args.mcts_sim_depth}, rollouts={args.mcts_rollouts}, "
              f"selection={args.mcts_selection}, rollout={args.mcts_rollout})")
    mcts_kwargs = dict(n_iterations=args.mcts_iterations, c_puct=args.mcts_c,
                       sim_depth=args.mcts_sim_depth, n_rollouts=args.mcts_rollouts,
                       score_gamma=args.mcts_score_gamma, selection=args.mcts_selection,
                       rollout=args.mcts_rollout)

    test_rows = []
    for i, part in enumerate(test_parts):
        t0 = time.perf_counter()
        buf, env = collect_episode(model, part, cfg, device=device, greedy=True)
        row = dict(test_part=i, n_features=part.n_features,
                   steps=len(buf.rewards), lead_time=episode_lead_time(buf),
                   fully_machined=env.rules.all_fully_machined(),
                   compute_time_s=round(time.perf_counter() - t0, 3))
        rows = build_schedule_table(part, env.log, cfg)
        save_schedule_csv(rows, os.path.join(args.out_dir, "schedules", f"test_part_{i:02d}.csv"))

        if args.use_mcts:
            if i < n_mcts:
                t0 = time.perf_counter()
                mbuf, menv = plan_with_mcts(model, part, cfg, device=device, **mcts_kwargs)
                row.update(mcts_steps=len(mbuf.rewards), mcts_lead_time=episode_lead_time(mbuf),
                           mcts_fully_machined=menv.rules.all_fully_machined(),
                           mcts_compute_time_s=round(time.perf_counter() - t0, 3))
                rows = build_schedule_table(part, menv.log, cfg)
                save_schedule_csv(rows, os.path.join(args.out_dir, "schedules",
                                                     f"test_part_{i:02d}_mcts.csv"))
                print(f"  test part {i:3d}: greedy lead_time={row['lead_time']:.3f} "
                      f"| mcts lead_time={row['mcts_lead_time']:.3f} "
                      f"({row['mcts_compute_time_s']:.1f}s)")
            else:
                row.update(mcts_steps=None, mcts_lead_time=None,
                           mcts_fully_machined=None, mcts_compute_time_s=None)
        test_rows.append(row)

    test_report_path = os.path.join(args.out_dir, "test_report.csv")
    with open(test_report_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(test_rows[0].keys()))
        writer.writeheader()
        writer.writerows(test_rows)

    completion_rate = sum(r["fully_machined"] for r in test_rows) / len(test_rows)
    avg_lead_time = sum(r["lead_time"] for r in test_rows) / len(test_rows)
    print(f"\n=== TEST SET RESULTS ({len(test_parts)} parts) ===")
    print(f"Completion rate : {completion_rate:.1%}")
    print(f"Avg lead time   : {avg_lead_time:.3f}")
    if n_mcts > 0:
        m_rows = test_rows[:n_mcts]
        print(f"--- greedy vs MCTS on the {n_mcts} MCTS-planned parts ---")
        print(f"Greedy : completion {sum(r['fully_machined'] for r in m_rows) / n_mcts:.1%}, "
              f"avg lead time {sum(r['lead_time'] for r in m_rows) / n_mcts:.3f}, "
              f"avg compute {sum(r['compute_time_s'] for r in m_rows) / n_mcts:.2f}s")
        print(f"MCTS   : completion {sum(r['mcts_fully_machined'] for r in m_rows) / n_mcts:.1%}, "
              f"avg lead time {sum(r['mcts_lead_time'] for r in m_rows) / n_mcts:.3f}, "
              f"avg compute {sum(r['mcts_compute_time_s'] for r in m_rows) / n_mcts:.2f}s")
    print(f"Full report     : {test_report_path}")
    print(f"Per-part schedules under: {os.path.join(args.out_dir, 'schedules')}")

if __name__ == "__main__":
    main()
