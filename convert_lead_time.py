#!/usr/bin/env python3
"""
convert_lead_time.py
--------------------
Converts the normalized, dimensionless lead times in RAGT-PPO results
(test_report.csv `lead_time` / `mcts_lead_time`) back into real time, in
seconds.

Why this can't be a single scale factor
---------------------------------------
Every term of the normalized lead time is scaled differently, per part and
per feature:

  - cutting time  P[f, t, m] = (Tc - Tc_min) / (Tc_max - Tc_min) + 0.1,
                  min-max normalized separately for every (feature, MSC)
                  over its candidate tools (aag_generator/msc_engine.py);
  - travel        M[i, j] = distance / (largest centroid distance in that
                  part) (aag_generator/generate_aag_states.py);
  - tool / orientation / TAD change: flat constants of reward.py.

So the conversion REPLAYS each saved schedule (schedules/test_part_XX.csv)
against its own part and prices every operation with the AAG generator's
real time model, the same one its process_sequences/ and ground_truth/
files use:

  - cutting time   : real Tc (Eq. 1) of the exact (feature, MSC, tool) cell
                     the agent chose, rebuilt with aag_generator's msc_engine
                     from the part's dimensions/ and graphs/ files;
  - non-cutting    : aag_generator/noncutting.py -- rapid travel between
                     feature centroids + 0.5 min per tool change + 0.6 / 2.0
                     min per 90 / 180 degree tool-approach-direction change.

reward.py's 3-axis <-> 5-axis method-switch cost has no counterpart in that
model; it is priced at --method_change_seconds (default 0, i.e. ignored).

Which AAG file is test part i is recovered by redoing train.py's split
(same sorted file list, same split_dataset, same seed). Two checks guard
this: the rebuilt P matrix must equal the one stored in the state file, and
replaying the schedule with reward.py must give back the reported normalized
lead time (column `lead_time_replay_ok`).

Outputs, per run folder:
    test_report_seconds.csv         -- test_report.csv + real-time columns
    schedules_seconds/*.csv         -- each schedule with per-operation times
and, when given a folder of runs, lead_time_seconds_summary.csv next to them.

Usage:
    python3 convert_lead_time.py                                 # every run in ragt_ppo_results/
    python3 convert_lead_time.py --results_dir ragt_ppo_results/run_batch_16
"""
from __future__ import annotations
import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path

import numpy as np

from ragt_ppo.aag import AAGConfig
from ragt_ppo.dataset import _load_one, split_dataset
from ragt_ppo.reward import non_cutting_time

HERE = Path(__file__).resolve().parent
DEFAULT_RESULTS_DIR = HERE / "ragt_ppo_results"
DEFAULT_DATA_ROOT = Path.home() / "mfpp_results" / "run_23975074"
DEFAULT_GENERATOR_DIR = HERE.parent / "mfpp_dataset" / "aag_generator" / "aag_generator"

# train.py split used by every run in ragt_ppo_results (see ragt_ppo_train.sh);
# overridden per run by its hparams.txt when that file exists
DEFAULT_SPLIT = dict(n_train=3500, n_val=500, n_test=1000, split_seed=42)

REPLAY_TOL = 1e-4


# ---------------------------------------------------------------------------
# Real time model of one part (aag_generator side)
# ---------------------------------------------------------------------------
class PartTimer:
    """Real cutting / non-cutting times, in minutes, for one AAG state file."""

    def __init__(self, state_path: Path, data_root: Path, gen):
        self.gen = gen
        with open(state_path) as f:
            state = json.load(f)
        self.model_id = state["model_id"]
        self.node_order = state["node_order"]
        self.msc_labels = state["catalogs"]["msc_labels"]
        fref = state["feature_reference"]
        tad = state["static_attributes"]["tool_approach_direction"]["values"]
        self.centroid = {fid: fref[str(fid)]["centroid"] for fid in self.node_order}
        self.tad = {fid: tad[str(fid)] for fid in self.node_order}

        with open(data_root / "dimensions" / f"{self.model_id}_dims.json") as f:
            dims = json.load(f)
        with open(data_root / "graphs" / f"{self.model_id}_aag.json") as f:
            aag = json.load(f)

        # same calls as generate_aag_states.build_state_and_sequence
        nc, nt = len(gen.catalogs.MSC_LABELS), len(gen.tools)
        tool_index = {t.id: i for i, t in enumerate(gen.tools)}
        P_stored = state["static_attributes"]["feasible_solution_matrix_P"]
        self.cells = {}  # (fid, msc, tool_id) -> cutting_time_min
        for fid in self.node_order:
            rec = dims["features"][str(fid)]
            a = gen.msc_engine.build_feature_assignment(
                fid, rec["feature_type"], rec["dimensions"], rec["depth_kind"], aag,
                gen.mapping, gen.tools, tool_index, nc=nc, nt=nt)
            if not np.allclose(a.P, np.array(P_stored[str(fid)]), atol=1e-5):
                raise RuntimeError(
                    f"{state_path.name}: rebuilt P for feature {fid} differs from the "
                    f"stored one -- --data_root / --generator_dir don't match the "
                    f"catalogs this dataset was generated with")
            for msc, cells in a.feasible_cells.items():
                for c in cells:
                    self.cells[(fid, msc, c.tool.id)] = c.cutting_time_min

    def operation_times(self, ops, tools, method_change_min: float) -> list[dict]:
        """`ops` is [(node_index, tool_index, msc_index, method)]. Returns one dict
        of real times (minutes) per operation, same model as noncutting.py."""
        out, prev = [], None
        for f, t, m, method in ops:
            fid, tool = self.node_order[f], tools[t]
            key = (fid, self.msc_labels[m], tool.id)
            if key not in self.cells:
                raise KeyError(f"{self.model_id}: no feasible cell for feature {fid}, "
                               f"MSC {self.msc_labels[m]}, tool {tool.name}")
            curr = {"centroid": self.centroid[fid], "tool_id": tool.id,
                    "tool_approach_direction": self.tad[fid]}
            nct = self.gen.noncutting.transition_time_min(
                None if prev is None else prev[0], curr)
            method_change = 0.0 if prev is None or prev[1] == method else method_change_min
            out.append(dict(
                cutting=self.cells[key],
                travel=nct["travel_time_min"],
                tool_change=nct["tool_change_time_min"],
                orientation_change=nct["orientation_change_time_min"],
                method_change=method_change,
            ))
            prev = (curr, method)
        return out


class Generator:
    """The aag_generator modules + catalogs, imported from --generator_dir."""

    def __init__(self, generator_dir: Path):
        sys.path.insert(0, str(generator_dir))
        import catalogs
        import msc_engine
        import noncutting
        self.catalogs, self.msc_engine, self.noncutting = catalogs, msc_engine, noncutting
        self.tools = catalogs.load_tools()
        self.mapping = catalogs.load_feature_operation_mapping()


# ---------------------------------------------------------------------------
# Schedule CSV -> (node, tool, msc) actions
# ---------------------------------------------------------------------------
def schedule_key_to_msc(cfg: AAGConfig) -> dict:
    """Inverse of schedule_output.build_schedule_table's (stage, region, method)."""
    lookup = {}
    for m in range(cfg.n_c):
        method = cfg.method_of[m].replace("3axis", "3-axis milling").replace("5axis", "5-axis milling")
        key = (cfg.stage_label_of[m], cfg.region_of[m], method)
        assert key not in lookup, f"schedule columns don't identify the MSC uniquely: {key}"
        lookup[key] = m
    return lookup


def read_schedule(path: Path, cfg: AAGConfig, msc_lookup: dict):
    rows, ops = [], []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            f_idx = int(re.match(r"F(\d+)", r["Associate machining feature"]).group(1)) - 1
            t_idx = cfg.tool_labels.index(r["Cutting tool"])
            m_idx = msc_lookup[(r["Machining stage"], r["Machining region"], r["Machining method"])]
            rows.append(r)
            ops.append((f_idx, t_idx, m_idx, cfg.method_of[m_idx]))
    return rows, ops


def replay_normalized(part, ops) -> float:
    """reward.py's normalized lead time for these operations (what train.py reports)."""
    total, prev_f, prev_t, prev_method = 0.0, None, None, None
    for f, t, m, method in ops:
        total += float(part.feasibility[f, t, m])
        total += non_cutting_time(part, prev_f, f, prev_t, t, prev_method, method)
        prev_f, prev_t, prev_method = f, t, method
    return total


# ---------------------------------------------------------------------------
# One run folder
# ---------------------------------------------------------------------------
def read_split(run_dir: Path) -> dict:
    split = dict(DEFAULT_SPLIT)
    hp = run_dir / "hparams.txt"
    if hp.exists():
        for line in hp.read_text().splitlines():
            k, _, v = line.partition("=")
            if k in split and v:
                split[k] = int(v)
    return split


def test_files(data_dir: Path, split: dict) -> list[Path]:
    # same file list as dataset.load_aag_dataset, same split as train.py
    files = sorted(p for p in data_dir.iterdir()
                   if p.suffix in (".json", ".pkl", ".pickle") and p.name != "manifest.json")
    _, _, test = split_dataset(files, n_train=split["n_train"], n_val=split["n_val"],
                               n_test=split["n_test"], seed=split["split_seed"])
    return test


def load_gt(data_root: Path, model_id: str) -> dict:
    path = data_root / "ground_truth" / f"{model_id}_optimal.json"
    if not path.exists():
        return {}
    with open(path) as f:
        d = json.load(f)
    return dict(gt_lead_time_s=round(60.0 * d["total_time_min"], 3),
                baseline_lead_time_s=round(60.0 * d["baseline_total_time_min"], 3))


def convert_run(run_dir: Path, data_root: Path, gen: Generator, cache: dict,
                method_change_s: float) -> dict:
    split = read_split(run_dir)
    files = test_files(data_root / "aag_states", split)
    out_sched = run_dir / "schedules_seconds"
    out_sched.mkdir(exist_ok=True)

    with open(run_dir / "test_report.csv", newline="") as f:
        report = list(csv.DictReader(f))

    out_rows, n_mismatch = [], 0
    for row in report:
        i = int(row["test_part"])
        path = files[i]
        if path not in cache:
            part, cfg = _load_one(path)
            cache[path] = (part, cfg, PartTimer(path, data_root, gen))
        part, cfg, timer = cache[path]
        if int(row["n_features"]) != part.n_features:
            raise RuntimeError(f"{run_dir.name} test part {i}: n_features "
                               f"{row['n_features']} != {part.n_features} in {path.name} "
                               f"-- wrong split settings or --data_root")
        msc_lookup = schedule_key_to_msc(cfg)

        out = dict(row)
        out["model_id"] = timer.model_id
        for prefix, suffix in (("", ""), ("mcts_", "_mcts")):
            if not row.get(f"{prefix}lead_time"):
                continue  # no MCTS column, or MCTS not run on this part
            sched_rows, ops = read_schedule(
                run_dir / "schedules" / f"test_part_{i:02d}{suffix}.csv", cfg, msc_lookup)
            times = timer.operation_times(ops, gen.tools, method_change_s / 60.0)

            replay = replay_normalized(part, ops)
            ok = abs(replay - float(row[f"{prefix}lead_time"])) < REPLAY_TOL
            n_mismatch += not ok

            totals = {k: 60.0 * sum(t[k] for t in times) for k in times[0]} if times else {}
            cutting_s = totals.get("cutting", 0.0)
            non_cutting_s = sum(v for k, v in totals.items() if k != "cutting")
            out.update({
                f"{prefix}lead_time_replay_ok": ok,
                f"{prefix}cutting_time_s": round(cutting_s, 3),
                f"{prefix}travel_time_s": round(totals.get("travel", 0.0), 3),
                f"{prefix}tool_change_time_s": round(totals.get("tool_change", 0.0), 3),
                f"{prefix}orientation_change_time_s": round(totals.get("orientation_change", 0.0), 3),
                f"{prefix}method_change_time_s": round(totals.get("method_change", 0.0), 3),
                f"{prefix}non_cutting_time_s": round(non_cutting_s, 3),
                f"{prefix}lead_time_s": round(cutting_s + non_cutting_s, 3),
            })

            with open(out_sched / f"test_part_{i:02d}{suffix}.csv", "w", newline="") as f:
                fields = list(sched_rows[0].keys()) + [
                    "Cutting time (s)", "Travel time (s)", "Tool change time (s)",
                    "Orientation change time (s)", "Method change time (s)",
                    "Non-cutting time (s)", "Total time (s)", "Cumulative time (s)"]
                w = csv.DictWriter(f, fieldnames=fields)
                w.writeheader()
                cum = 0.0
                for r, t in zip(sched_rows, times):
                    nc_s = 60.0 * (t["travel"] + t["tool_change"]
                                   + t["orientation_change"] + t["method_change"])
                    total = 60.0 * t["cutting"] + nc_s
                    cum += total
                    w.writerow({**r,
                                "Cutting time (s)": round(60.0 * t["cutting"], 3),
                                "Travel time (s)": round(60.0 * t["travel"], 3),
                                "Tool change time (s)": round(60.0 * t["tool_change"], 3),
                                "Orientation change time (s)": round(60.0 * t["orientation_change"], 3),
                                "Method change time (s)": round(60.0 * t["method_change"], 3),
                                "Non-cutting time (s)": round(nc_s, 3),
                                "Total time (s)": round(total, 3),
                                "Cumulative time (s)": round(cum, 3)})
        out.update(load_gt(data_root, timer.model_id))
        out_rows.append(out)

    fields = list(dict.fromkeys(k for r in out_rows for k in r))
    with open(run_dir / "test_report_seconds.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(out_rows)

    return summarize(run_dir.name, out_rows, n_mismatch)


def summarize(name: str, rows: list[dict], n_mismatch: int) -> dict:
    def mean(key, only_complete=None):
        vals = [float(r[key]) for r in rows if r.get(key) not in (None, "")
                and (only_complete is None or str(r.get(only_complete)) == "True")]
        return (round(sum(vals) / len(vals), 3) if vals else None), len(vals)

    s = {"run": name, "n_test": len(rows), "replay_mismatches": n_mismatch}
    s["completion_rate"] = round(sum(str(r["fully_machined"]) == "True" for r in rows) / len(rows), 4)
    s["avg_lead_time_norm"], _ = mean("lead_time")
    # averages over fully machined parts only: a cut-short episode's time is not a lead time
    s["avg_lead_time_s"], _ = mean("lead_time_s", "fully_machined")
    s["avg_cutting_time_s"], _ = mean("cutting_time_s", "fully_machined")
    s["avg_non_cutting_time_s"], _ = mean("non_cutting_time_s", "fully_machined")
    s["avg_gt_lead_time_s"], _ = mean("gt_lead_time_s", "fully_machined")
    s["avg_baseline_lead_time_s"], _ = mean("baseline_lead_time_s", "fully_machined")
    if any(r.get("mcts_lead_time_s") not in (None, "") for r in rows):
        s["avg_mcts_lead_time_s"], s["n_mcts"] = mean("mcts_lead_time_s", "mcts_fully_machined")
    return s


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results_dir", type=Path, default=DEFAULT_RESULTS_DIR,
                    help="one run folder (with test_report.csv), or a folder of them")
    ap.add_argument("--data_root", type=Path, default=DEFAULT_DATA_ROOT,
                    help="mfpp dataset root the runs were trained on (aag_states/, "
                         "dimensions/, graphs/, ground_truth/)")
    ap.add_argument("--generator_dir", type=Path, default=DEFAULT_GENERATOR_DIR,
                    help="aag_generator source folder (catalogs.py, msc_engine.py, noncutting.py)")
    ap.add_argument("--method_change_seconds", type=float, default=0.0,
                    help="real time charged per 3-axis <-> 5-axis switch (reward.py's "
                         "ORIENTATION_CHANGE_COST); aag_generator has no such constant")
    args = ap.parse_args()

    if (args.results_dir / "test_report.csv").exists():
        run_dirs = [args.results_dir]
    else:
        run_dirs = sorted(d for d in args.results_dir.iterdir() if (d / "test_report.csv").exists())
    if not run_dirs:
        sys.exit(f"no test_report.csv found in {args.results_dir} or its subfolders")

    gen = Generator(args.generator_dir)
    cache, summaries = {}, []
    for run_dir in run_dirs:
        print(f"converting {run_dir.name} ...", flush=True)
        s = convert_run(run_dir, args.data_root, gen, cache, args.method_change_seconds)
        summaries.append(s)
        msg = (f"  avg lead time {s['avg_lead_time_s']} s (greedy, fully machined parts) | "
               f"ground truth {s['avg_gt_lead_time_s']} s | baseline {s['avg_baseline_lead_time_s']} s")
        if "avg_mcts_lead_time_s" in s:
            msg += f" | MCTS {s['avg_mcts_lead_time_s']} s over {s['n_mcts']} parts"
        print(msg)
        if s["replay_mismatches"]:
            print(f"  note: {s['replay_mismatches']} schedule(s) don't reproduce the reported "
                  f"normalized lead time with the current reward.py (run predates a reward "
                  f"change?) -- the seconds are still priced from the schedule itself")

    if len(run_dirs) > 1:
        path = args.results_dir / "lead_time_seconds_summary.csv"
        fields = list(dict.fromkeys(k for s in summaries for k in s))
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(summaries)
        print(f"summary: {path}")


if __name__ == "__main__":
    main()
