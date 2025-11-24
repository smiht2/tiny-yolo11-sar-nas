#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
pick_knee.py — pick a knee-point from NSGA-II Pareto set by nearest-to-ideal (normalized) distance.
Minimize objectives: [-mAP, latency_ms, size_mb] → ideal = [min, min, min] = [-1, 0, 0] approximately.
We normalize each column, flip mAP to (1 - mAP_norm), then pick the minimal L2 distance.
"""

import csv
import json
import math
import argparse
from pathlib import Path

def load_pareto(csv_path):
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            for k in ["mAP","lat_ms","on_disk_mb","sd_size_mb"]:
                try:
                    row[k] = float(row[k])
                except Exception:
                    row[k] = -1.0
            rows.append(row)
    return rows

def pick_knee(rows):
    rows = [r for r in rows if r["mAP"]>=0 and r["lat_ms"]>=0]
    if not rows: return None

    def norm(arr):
        if not arr: return []
        vmin, vmax = min(arr), max(arr)
        if vmax == vmin: return [0.0]*len(arr)
        return [(v - vmin)/(vmax - vmin) for v in arr]

    m = [r["mAP"] for r in rows]
    l = [r["lat_ms"] for r in rows]
    s = [ (r["on_disk_mb"] if r["on_disk_mb"]>0 else r["sd_size_mb"]) for r in rows]

    m_n = norm(m)   # higher is better
    l_n = norm(l)   # lower is better
    s_n = norm(s)   # lower is better

    m_min = [1.0 - x for x in m_n]  # to minimize

    d = [math.sqrt(m_min[i]**2 + l_n[i]**2 + s_n[i]**2) for i in range(len(rows))]
    best_i = min(range(len(rows)), key=lambda i: d[i])
    rows[best_i]["knee_distance"] = d[best_i]
    return rows[best_i]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pareto_csv", type=str, required=True)
    ap.add_argument("--out_json", type=str, default="best_nsga.json")
    args = ap.parse_args()

    rows = load_pareto(args.pareto_csv)
    best = pick_knee(rows)
    if best is None:
        print("No valid rows in Pareto set")
        return
    outp = Path(args.out_json)
    outp.write_text(json.dumps(best, indent=2), encoding="utf-8")
    print("Saved knee-point to", outp)
    print("YAML:", best.get("yaml",""))
    print("Weights:", best.get("weights",""))

if __name__ == "__main__":
    main()