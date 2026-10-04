# RAGT-PPO — Machining Process Sequencing

A working PyTorch implementation of the **Relation-Aware Graph Transformer +
PPO** architecture, replacing the ConvGNN(PNA) backbone of Zhang et al.
(2025, *J. Manuf. Syst.* 78:370–393) while keeping their environment design
(state = AAG, action = machining operation, transition dynamics,
termination conditions) intact.

- **Policy network**: **Option A** — the graph transformer's own attention
  layers absorb SENet's job; final node embeddings feed action logits
  directly (`model.py :: RAGT_PPO.forward`, no separate gating module).
- **Value network**: **PMA** (Pooling by Multihead Attention, Lee et al.
  2019) — a single learnable seed query attends over all node embeddings to
  produce one fixed-size vector regardless of feature count `n`
  (`model.py :: PMAValueHead`).
- **Goal**: minimize **lead time = cutting time + non-cutting time**, where
  non-cutting time = tool travel time + tool change time + orientation
  change time (`reward.py`).
- **Output**: same table structure as the paper's Table A.1 / Fig. 16
  (`schedule_output.py`).

## Catalog-driven design (`AAGConfig`)

Unlike the paper's original setup (10 MSCs / 23 tools, fixed), **every
catalog size in this codebase is read from your data at load time**, via
an `AAGConfig` object (`aag.py`). Your real `aag_generator` catalog has
**14 MSCs (including drilling / reaming / tapping / chamfering) and 36
tools**, and the code adapts to that automatically — nothing is hardcoded.

Two independent stage-precedence chains are recognized from MSC label text:
- **milling** chain: rough -> secondary-rough -> semi-finish -> finish (Table 1, rules 1-3)
- **hole** chain: drilling -> reaming -> tapping -> chamfering (your guaranteed-precision-hole workflow)

Any MSC label the code doesn't recognize falls back to "no precedence
requirement" instead of crashing, so future catalog extensions degrade
gracefully.

`AAGConfig` is built once (from your dataset's `catalogs.msc_labels` /
`catalogs.tool_labels`) and passed explicitly into every constructor
(`RuleEngine`, `MachiningSequencingEnv`, `RAGT_PPO`) -- there's no hidden
global state.

## Files

| File | Contents |
|---|---|
| `ragt_ppo/aag.py` | `AAGPart`, `AAGConfig` (catalog-driven dims/stage/chain inference), synthetic part builder |
| `ragt_ppo/dataset.py` | Loader for your **real** aag_generator JSON schema (`static_attributes.feasible_solution_matrix_P`, `edge_attributes.edge_list`, `catalogs.msc_labels/tool_labels`) + 60/20/20 split |
| `ragt_ppo/rules.py` | Section 1.1's eight hard rules as dynamic feasibility masking, generalized to any `AAGConfig` (both precedence chains) |
| `ragt_ppo/reward.py` | Lead-time reward: cutting time (from your `P_i`) + non-cutting time (travel + tool-change + orientation-change) |
| `ragt_ppo/env.py` | Gym-like environment: state/action/transition dynamics/termination |
| `ragt_ppo/model.py` | RAGT-PPO network: relation-aware graph transformer encoder, Option-A policy head, PMA value head |
| `ragt_ppo/ppo.py` | Clipped-surrogate PPO + GAE, single-episode and multi-part-batch variants |
| `ragt_ppo/schedule_output.py` | Converts a rollout into the paper-style schedule table (print + CSV) |
| `run_demo.py` | Single-part sanity check on synthetic data (paper's original 10-MSC/23-tool catalog) |
| `train.py` | Full 60/20/20 train/val/test pipeline for your real dataset |

## Your real AAG file schema (what `dataset.py` expects)

One JSON file per part, with (relevant fields):

```json
{
  "num_nodes": 11,
  "catalogs": {"msc_labels": ["...14 strings..."], "tool_labels": ["...36 strings..."]},
  "static_attributes": {
      "feasible_solution_matrix_P": {"0": [["...14 cols..."], "...36 rows..."], "1": "..."},
      "distance_matrix_M": [["...11..."], "...11 rows..."]
  },
  "edge_attributes": {
      "edge_list": [{"src_index": 0, "dst_index": 1, "attr": [1,1,1]}]
  },
  "feature_reference": {"0": {"feature_type": "rectangular_through_step"}}
}
```

This is exactly your `aag_generator`'s output format -- confirmed against a
real exported part. All 100 of your files should follow this same layout
and **share the same catalog** (same `msc_labels`/`tool_labels` lists);
`load_aag_dataset` checks this and raises a clear error if one file
disagrees.

## Running the demo (synthetic, paper's original catalog)

```bash
pip install torch
python3 run_demo.py
```

## Train / validation / test workflow (60 / 20 / 20) on your real 100 parts

```bash
python3 train.py --data_dir /path/to/your/aag_parts \
                  --n_train 60 --n_val 20 --n_test 20 \
                  --n_updates 300 --batch_size 8 --epochs_per_update 8 \
                  --val_every 10 \
                  --out_dir ./runs/ragt_ppo_run
```

This: (1) loads and randomly splits your parts 60/20/20 (`dataset.py`),
building `AAGConfig` from the first file and checking every other file
matches it; (2) trains RAGT-PPO with mini-batches of *training* parts per
PPO update (GAE computed per episode, then pooled for one gradient step);
(3) evaluates on the *validation* set every `--val_every` updates and
checkpoints whenever validation lead time improves (subject to a minimum
completion-rate bar); (4) after training, loads the best checkpoint and
reports final metrics on the held-out *test* set, saving a Table-A.1-style
schedule CSV for every test part.

`--out_dir` defaults to `./ragt_ppo_run`, created relative to wherever you
run the script -- override it if you want output somewhere specific.

Outputs: `best_model.pt`, `training_log.csv`, `test_report.csv`,
`schedules/test_part_XX.csv`.

### A note on `--entropy_coef`

In testing against a real part, the agent regularly got to within 1-2
operations of full completion, then got stuck oscillating among the same
few masked-out actions rather than exploring the remaining valid ones. A
higher entropy bonus fixes this (default here is `0.03`, vs. the more
conservative `0.01` typical for PPO) -- if your own runs show completion
rate plateauing well below 100% while lead time still looks reasonable,
try raising `--entropy_coef` further before assuming something else is
wrong.

## Mapping to Section 1.1 rules (design doc)

| Rule | Where enforced |
|---|---|
| 1-3 (stage order, both chains) | `RuleEngine._stage_cleared_for_msc` -- gates an MSC's stage until all *feasible* earlier-stage MSCs **within the same chain** for that feature have been applied |
| 4 (datum first) | `RuleEngine._datum_ready` |
| 5 (no roughing tool reuse in semi-finish/finish, milling chain only) | `RuleEngine.build_mask`, tool == `rough_tool` check |
| 6 (no re-machining a finished region) | subsumed by rule 8 |
| 7 (fully machined -> no further ops) | `RuleEngine.build_mask`, `fully_machined` early-continue |
| 8 (each MSC applied to a feature at most once) | `RuleEngine.build_mask`, `msc_applied[m] == 1` check |

Rules 1-8 are enforced purely through **masking + zero reward on
violation** (never a crash). The Section 1.1 *soft* rules (setup/TAD
grouping, explicit tool-change-count minimization, distortion-risk
shaping) are not yet implemented as reward-shaping terms -- natural next
additions to `reward.py` once you have the relevant geometric proxies.

## Extending to real training scale

`train.py`'s default trainer collects one episode per training part in
each mini-batch (`--batch_size`). To match Table 3 of the paper (200
parallel environments), vectorize `collect_batch` across multiple
`MachiningSequencingEnv` instances running concurrently rather than
sequentially.
