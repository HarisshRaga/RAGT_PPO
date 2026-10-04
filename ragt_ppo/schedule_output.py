"""
schedule_output.py
-------------------
Produces the machining process scheme in EXACTLY the structure of Table A.1
/ Fig. 16 of Zhang et al. (2025):

  No. | Associate machining feature | Cutting tool | Machining stage |
      Machining region | Machining method
"""
from __future__ import annotations
import csv
from .aag import AAGPart, AAGConfig


def build_schedule_table(part: AAGPart, log, cfg: AAGConfig) -> list[dict]:
    rows = []
    for i, (f, t, m) in enumerate(log, start=1):
        method = cfg.method_of[m].replace("3axis", "3-axis milling").replace("5axis", "5-axis milling")
        rows.append({
            "No.": i,
            "Associate machining feature": part.feature_names[f],
            "Cutting tool": cfg.tool_labels[t],
            "Machining stage": cfg.stage_label_of[m],
            "Machining region": cfg.region_of[m],
            "Machining method": method,
        })
    return rows


def print_schedule_table(rows: list[dict]):
    headers = ["No.", "Associate machining feature", "Cutting tool",
               "Machining stage", "Machining region", "Machining method"]
    widths = {h: max(len(h), max((len(str(r[h])) for r in rows), default=0)) for h in headers}
    line = " | ".join(h.ljust(widths[h]) for h in headers)
    print(line)
    print("-" * len(line))
    for r in rows:
        print(" | ".join(str(r[h]).ljust(widths[h]) for h in headers))


def save_schedule_csv(rows: list[dict], path: str):
    headers = ["No.", "Associate machining feature", "Cutting tool",
               "Machining stage", "Machining region", "Machining method"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)
