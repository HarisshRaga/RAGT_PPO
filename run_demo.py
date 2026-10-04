"""
run_demo.py
-----------
End-to-end demonstration of RAGT-PPO on a synthetic AAG part built from the
paper's original 10-MSC/23-tool catalog. For your real, extended
14-MSC/36-tool catalog and 100 real parts, use train.py instead.

    python3 run_demo.py
"""
import os
import sys
sys.path.insert(0, "ragt_ppo")

import torch
from ragt_ppo.aag import build_synthetic_part, DEFAULT_CONFIG
from ragt_ppo.model import RAGT_PPO
from ragt_ppo.ppo import collect_episode, ppo_update
from ragt_ppo.schedule_output import build_schedule_table, print_schedule_table, save_schedule_csv


def main():
    torch.manual_seed(0)
    cfg = DEFAULT_CONFIG
    part = build_synthetic_part(cfg, n_features=7, seed=1)

    model = RAGT_PPO(cfg, d_model=64, n_heads=4, n_layers=3, d_ff=128, dist_dim=16)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    n_updates = 60
    for update in range(1, n_updates + 1):
        buf, env = collect_episode(model, part, cfg, greedy=False)
        stats = ppo_update(model, optimizer, buf, n_epochs=4)
        if update % 10 == 0 or update == 1:
            print(f"[update {update:3d}] steps={len(buf.rewards):3d} "
                  f"return={stats['episode_return']:+7.3f} "
                  f"policy_loss={stats['policy_loss']:+.4f} "
                  f"value_loss={stats['value_loss']:.4f} "
                  f"fully_machined={env.rules.all_fully_machined()}")

    # ---- inference: greedy rollout with the trained policy (Fig. 8 style) ----
    print("\n=== Final greedy machining process scheme ===")
    buf, env = collect_episode(model, part, cfg, greedy=True)
    rows = build_schedule_table(part, env.log, cfg)
    print_schedule_table(rows)
    os.makedirs("./demo_output", exist_ok=True)
    save_schedule_csv(rows, "./demo_output/machining_process_scheme.csv")
    print(f"\nfully_machined = {env.rules.all_fully_machined()}  "
          f"total steps = {len(env.log)}  episode_return = {sum(buf.rewards):+.3f}")


if __name__ == "__main__":
    main()
