#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
nas_yolo11_nsga2.py — Multi-objective NAS for YOLO11-style tiny models using NSGA-II
Objectives: minimize [-mAP, latency_ms, size_MB]  (i.e., maximize mAP, minimize latency & size)
- Uses the same YOLOv11-style YAML generator (with proper P5/32) as the EA script you ran.
- Supports optional hard feasibility constraints vs. your baseline (accuracy floor, latency cap, size cap).

Example:
  python nas_yolo11_nsga2.py \
    --data "C:/.../SSDD.v1i.yolov11/data.yaml" \
    --baseline_weights "C:/.../runs_ssdd/y11s_1024/weights/best.pt" \
    --device 0 --imgsz 640 \
    --outdir "C:/.../runs/nas_yolo11_ssdd_nsga2" \
    --pop_size 12 --generations 10 \
    --epochs 10 --batch 16 --workers 0 \
    --hard_constraints True --min_map_drop 0.2 --latency_slack 0.03 --size_cap_mb 10.0
"""

import argparse
import csv
import json
import math
import random
import time
from copy import deepcopy
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Any, List, Tuple

import torch
from ultralytics import YOLO

try:
    from thop import profile as thop_profile  # optional
    THOP_OK = True
except Exception:
    THOP_OK = False

# -----------------------------
# Search space
# -----------------------------
WIDTH_MULTS = [0.25, 0.33, 0.50, 0.67, 0.75]
REPEATS = [1, 2, 3]  # for C2f repeats at 128/256/512
DOWNSAMPLE_OPS = ["Conv", "DWConv", "GhostConv"]
DOWNSAMPLE_KS = [3, 5]
NECK_REPEATS = [1, 2]

def rand_choice(seq):
    return random.choice(seq)

@dataclass
class Candidate:
    rep_128: int
    rep_256: int
    rep_512: int
    width_mult: float
    down_op: str
    down_k: int
    pan_repeats: int

    def mutate(self, prob=0.2) -> 'Candidate':
        c = deepcopy(self)
        import random as _r
        if _r.random() < prob: c.rep_128 = rand_choice(REPEATS)
        if _r.random() < prob: c.rep_256 = rand_choice(REPEATS)
        if _r.random() < prob: c.rep_512 = rand_choice(REPEATS)
        if _r.random() < prob: c.width_mult = rand_choice(WIDTH_MULTS)
        if _r.random() < prob: c.down_op = rand_choice(DOWNSAMPLE_OPS)
        if _r.random() < prob: c.down_k = rand_choice(DOWNSAMPLE_KS)
        if _r.random() < prob: c.pan_repeats = rand_choice(NECK_REPEATS)
        return c

    @staticmethod
    def crossover(a: 'Candidate', b: 'Candidate') -> 'Candidate':
        import random as _r
        return Candidate(
            rep_128=_r.choice([a.rep_128, b.rep_128]),
            rep_256=_r.choice([a.rep_256, b.rep_256]),
            rep_512=_r.choice([a.rep_512, b.rep_512]),
            width_mult=_r.choice([a.width_mult, b.width_mult]),
            down_op=_r.choice([a.down_op, b.down_op]),
            down_k=_r.choice([a.down_k, b.down_k]),
            pan_repeats=_r.choice([a.pan_repeats, b.pan_repeats])
        )

    @staticmethod
    def random() -> 'Candidate':
        return Candidate(
            rep_128=rand_choice(REPEATS),
            rep_256=rand_choice(REPEATS),
            rep_512=rand_choice(REPEATS),
            width_mult=rand_choice(WIDTH_MULTS),
            down_op=rand_choice(DOWNSAMPLE_OPS),
            down_k=rand_choice(DOWNSAMPLE_KS),
            pan_repeats=rand_choice(NECK_REPEATS),
        )


# -----------------------------
# YAML template (with proper P5/32 backbone level)
# P3=/8 -> layer 4, P4=/16 -> layer 6, P5=/32 -> layer 9
# -----------------------------
YAML_TEMPLATE = """# Auto-generated YOLO11-style tiny model
nc: {nc}
depth_multiple: 1.0
width_multiple: {width_mult}

backbone:
  # [from, number, module, args]
  - [-1, 1, {down_op}, [64, {ks}, 2]]          # 0-P1/2
  - [-1, 1, Conv, [128, {ks}, 2]]              # 1-P2/4
  - [-1, {rep128}, C2f, [128, True]]           # 2
  - [-1, 1, {down_op}, [256, {ks}, 2]]         # 3-P3/8
  - [-1, {rep256}, C2f, [256, True]]           # 4 (P3)
  - [-1, 1, {down_op}, [512, {ks}, 2]]         # 5-P4/16
  - [-1, {rep512}, C2f, [512, True]]           # 6 (P4)
  - [-1, 1, {down_op}, [1024, {ks}, 2]]        # 7-P5/32
  - [-1, 1, C2f, [1024, True]]                 # 8
  - [-1, 1, SPPF, [1024, 5]]                   # 9 (P5)

head:
  - [-1, 1, Conv, [512, 1, 1]]                  # 10
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]  # 11  /32->/16
  - [[-1, 6], 1, Concat, [1]]                   # 12  (up + P4)
  - [-1, {pan_r}, C2f, [512]]                   # 13  N4

  - [-1, 1, Conv, [256, 1, 1]]                  # 14
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]  # 15  /16->/8
  - [[-1, 4], 1, Concat, [1]]                   # 16  (up + P3)
  - [-1, {pan_r}, C2f, [256]]                   # 17  N3 (small)

  - [-1, 1, {down_op}, [512, {ks}, 2]]          # 18  /8->/16
  - [[-1, 13], 1, Concat, [1]]                  # 19  (down + N4)
  - [-1, {pan_r}, C2f, [512]]                   # 20  N4 (mid)

  - [-1, 1, {down_op}, [1024, {ks}, 2]]         # 21  /16->/32
  - [[-1, 9], 1, Concat, [1]]                   # 22  (down + P5)
  - [-1, {pan_r}, C2f, [1024]]                  # 23  N5 (large)

  - [[17, 20, 23], 1, Detect, [{nc}]]           # 24 Detect(P3,P4,P5)
"""

# -----------------------------
# Utils: latency, size, THOP
# -----------------------------
def try_thop(model_module: torch.nn.Module, imgsz: int, device: str):
    if not THOP_OK:
        return -1.0, -1.0
    try:
        model_module.eval().to(device)
        dummy = torch.randn(1, 3, imgsz, imgsz, device=device)
        macs, params = thop_profile(model_module, inputs=(dummy,), verbose=False)
        return float(macs), float(params)
    except Exception:
        return -1.0, -1.0

def measure_latency(model: YOLO, imgsz: int, device: str, iters=50, warmup=10) -> float:
    mdl = model.model  # nn.Module
    mdl.eval().to(device)
    x = torch.randn(1, 3, imgsz, imgsz, device=device)
    torch.cuda.empty_cache()
    with torch.no_grad():
        for _ in range(max(1, warmup)):
            _ = mdl(x)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(max(1, iters)):
            _ = mdl(x)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        dt = (time.time() - t0) / max(1, iters)
    return dt * 1000.0

def state_dict_size_mb(model_module: torch.nn.Module) -> float:
    size = 0
    for p in model_module.state_dict().values():
        size += p.numel() * p.element_size()
    return size / (1024.0 ** 2)

def on_disk_size_mb(path: Path) -> float:
    if path and path.is_file():
        return path.stat().st_size / (1024.0 ** 2)
    return -1.0

# -----------------------------
# Evaluation
# -----------------------------
def evaluate_candidate(cand: Candidate, args, run_dir: Path, nc: int):
    cand_dir = run_dir / f"cand_{int(time.time()*1000)}_{random.randint(0,9999)}"
    cand_dir.mkdir(parents=True, exist_ok=True)

    # Build YAML
    yaml_text = YAML_TEMPLATE.format(
        nc=nc, width_mult=cand.width_mult, rep128=cand.rep_128,
        rep256=cand.rep_256, rep512=cand.rep_512, down_op=cand.down_op,
        ks=cand.down_k, pan_r=cand.pan_repeats
    )
    yaml_path = cand_dir / "model.yaml"
    yaml_path.write_text(yaml_text, encoding="utf-8")

    model = YOLO(str(yaml_path))

    # Train short
    try:
        model.train(
            data=args.data,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            workers=args.workers,
            project=str(cand_dir),
            name="train",
            verbose=False,
            optimizer="auto",
            patience=0,
            seed=args.seed,
        )
    except Exception as e:
        return {"error": f"train_fail: {e}", **asdict(cand)}

    weights_path = cand_dir / "train" / "weights" / "best.pt"
    if not weights_path.exists():
        w_last = cand_dir / "train" / "weights" / "last.pt"
        weights_path = w_last if w_last.exists() else None

    # Validate
    mAP = -1.0
    try:
        val_res = model.val(
            data=args.data, imgsz=args.imgsz, device=args.device, split="val", workers=args.workers
        )
        mAP = float(getattr(getattr(val_res, "box", None), "map", -1.0))
        if mAP < 0 and hasattr(val_res, "results_dict"):
            mAP = float(val_res.results_dict.get("metrics/mAP50-95(B)", -1.0))
    except Exception as e:
        return {"error": f"val_fail: {e}", **asdict(cand)}

    # Latency & THOP
    if weights_path and weights_path.exists():
        model = YOLO(str(weights_path))
    device_str = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
    try:
        lat_ms = measure_latency(model, args.imgsz, device_str, iters=args.lat_iters, warmup=max(5, args.lat_warmup))
    except Exception:
        lat_ms = -1.0
    macs, params = try_thop(model.model, args.imgsz, device_str) if THOP_OK else (-1.0, -1.0)

    # Sizes
    sd_size_mb = state_dict_size_mb(model.model)
    on_disk_mb = on_disk_size_mb(weights_path) if weights_path else -1.0

    out = {
        **asdict(cand),
        "mAP": mAP,
        "lat_ms": lat_ms,
        "sd_size_mb": sd_size_mb,
        "on_disk_mb": on_disk_mb,
        "macs": macs,
        "params": params,
        "weights": str(weights_path) if weights_path else "",
        "yaml": str(yaml_path),
        "run_dir": str(cand_dir),
    }
    return out

# -----------------------------
# Baseline (for optional hard constraints)
# -----------------------------
def measure_baseline(args, nc: int):
    device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
    if args.baseline_weights and Path(args.baseline_weights).exists():
        base_model = YOLO(args.baseline_weights)
        try:
            val_res = base_model.val(data=args.data, imgsz=args.imgsz, device=args.device, split="val", workers=args.workers)
            baseline_map = float(getattr(getattr(val_res, "box", None), "map", -1.0))
            if baseline_map < 0 and hasattr(val_res, "results_dict"):
                baseline_map = float(val_res.results_dict.get("metrics/mAP50-95(B)", -1.0))
        except Exception:
            baseline_map = -1.0
        try:
            baseline_lat = measure_latency(base_model, args.imgsz, device,
                                           iters=args.lat_iters, warmup=max(5, args.lat_warmup))
        except Exception:
            baseline_lat = -1.0
        return baseline_map, baseline_lat
    return -1.0, -1.0

# -----------------------------
# NSGA-II core
# -----------------------------
def objectives(rec):
    """Return objective vector to minimize: (-mAP, latency_ms, size_mb)"""
    m = rec.get("mAP", -1.0)
    lat = rec.get("lat_ms", 1e9)
    size = rec.get("on_disk_mb", rec.get("sd_size_mb", 1e9))
    if size <= 0: size = rec.get("sd_size_mb", 1e9)
    return (-m if m > 0 else 1e3, lat if lat > 0 else 1e9, size if size > 0 else 1e9)

def dominates(a, b):
    """Return True if a dominates b (all objs <= and at least one <)."""
    return all(x <= y for x, y in zip(a, b)) and any(x < y for x, y in zip(a, b))

def fast_non_dominated_sort(pop_objs):
    S = [set() for _ in pop_objs]
    n = [0 for _ in pop_objs]
    fronts = [[]]
    for p in range(len(pop_objs)):
        for q in range(len(pop_objs)):
            if p == q: continue
            if dominates(pop_objs[p], pop_objs[q]):
                S[p].add(q)
            elif dominates(pop_objs[q], pop_objs[p]):
                n[p] += 1
        if n[p] == 0:
            fronts[0].append(p)
    i = 0
    while fronts[i]:
        next_front = []
        for p in fronts[i]:
            for q in S[p]:
                n[q] -= 1
                if n[q] == 0:
                    next_front.append(q)
        i += 1
        fronts.append(next_front)
    if not fronts[-1]:
        fronts.pop()
    return fronts

def crowding_distance(front, pop_objs):
    distance = {i: 0.0 for i in front}
    if len(front) == 0:
        return distance
    num_obj = 3
    for m in range(num_obj):
        front_sorted = sorted(front, key=lambda idx: pop_objs[idx][m])
        fmin = pop_objs[front_sorted[0]][m]
        fmax = pop_objs[front_sorted[-1]][m]
        distance[front_sorted[0]] = float('inf')
        distance[front_sorted[-1]] = float('inf')
        if fmax == fmin:
            continue
        for i in range(1, len(front_sorted)-1):
            prev = pop_objs[front_sorted[i-1]][m]
            nextv = pop_objs[front_sorted[i+1]][m]
            distance[front_sorted[i]] += (nextv - prev) / (fmax - fmin)
    return distance

def tournament_select(pop, ranks, crowd):
    a, b = random.sample(range(len(pop)), 2)
    ra, rb = ranks[a], ranks[b]
    if ra < rb: return deepcopy(pop[a])
    if rb < ra: return deepcopy(pop[b])
    ca, cb = crowd.get(a, 0.0), crowd.get(b, 0.0)
    return deepcopy(pop[a] if ca >= cb else pop[b])

def apply_hard_constraints(rec, baseline_map, baseline_lat, min_map_drop, latency_slack, size_cap_mb):
    if baseline_map > 0 and rec.get("mAP", -1.0) < baseline_map - min_map_drop:
        return False
    if baseline_lat > 0 and rec.get("lat_ms", 1e9) > baseline_lat * (1.0 + latency_slack):
        return False
    if size_cap_mb > 0:
        size = rec.get("on_disk_mb", rec.get("sd_size_mb", 1e9))
        if size <= 0: size = rec.get("sd_size_mb", 1e9)
        if size > size_cap_mb:
            return False
    return True

def nsga2_search(args):
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Load nc
    import yaml
    with open(args.data, "r", encoding="utf-8") as f:
        data_yaml = yaml.safe_load(f)
    names = data_yaml.get("names", [])
    if isinstance(names, dict):
        nc = len(names)
    elif isinstance(names, list):
        nc = len(names)
    else:
        nc = int(data_yaml.get("nc", 1))

    baseline_map, baseline_lat = measure_baseline(args, nc)
    if baseline_map > 0 and baseline_lat > 0:
        print(f"[baseline] mAP={baseline_map:.4f}, lat={baseline_lat:.2f} ms")
    else:
        print("[baseline] not measured (no baseline provided).")

    # CSV log
    csv_path = outdir / "nsga2_log.csv"
    cand_keys = list(Candidate.random().__dict__.keys())
    header = ["gen", "idx"] + cand_keys + ["mAP","lat_ms","sd_size_mb","on_disk_mb","macs","params","rank","crowd","weights","yaml","run_dir","feasible"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(header)

    # Initialize population
    population = []
    for i in range(args.pop_size):
        cand = Candidate.random()
        rec = evaluate_candidate(cand, args, outdir, nc)
        rec["gen"] = 0; rec["idx"] = i
        rec["feasible"] = apply_hard_constraints(rec, baseline_map, baseline_lat, args.min_map_drop, args.latency_slack, args.size_cap_mb) if args.hard_constraints else True
        population.append(rec)
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([rec.get(k, "") for k in header])

    # NSGA-II evolve
    for gen in range(1, args.generations+1):
        pop_objs = [objectives(r) for r in population]
        fronts = fast_non_dominated_sort(pop_objs)
        rank_map = {}
        for rnk, front in enumerate(fronts):
            for idx in front: rank_map[idx] = rnk
        crowd_map = {}
        for front in fronts:
            crowd = crowding_distance(front, pop_objs)
            crowd_map.update(crowd)

        # Make offspring
        offspring = []
        for i in range(args.pop_size):
            p1 = tournament_select(population, ranks=rank_map, crowd=crowd_map)
            child_cand = Candidate(**{k:p1[k] for k in cand_keys})
            # crossover
            if random.random() < args.crossover_prob:
                p2 = tournament_select(population, ranks=rank_map, crowd=crowd_map)
                p2_cand = Candidate(**{k:p2[k] for k in cand_keys})
                child_cand = Candidate.crossover(child_cand, p2_cand)
            # mutate
            child_cand = child_cand.mutate(prob=args.mutation_prob)
            rec = evaluate_candidate(child_cand, args, outdir, nc)
            rec["gen"] = gen; rec["idx"] = i
            rec["feasible"] = apply_hard_constraints(rec, baseline_map, baseline_lat, args.min_map_drop, args.latency_slack, args.size_cap_mb) if args.hard_constraints else True
            offspring.append(rec)
            with open(csv_path, "a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow([rec.get(k, "") for k in header])

        # Combine and select next generation
        combined = population + offspring
        objs = [objectives(r) for r in combined]
        fronts = fast_non_dominated_sort(objs)
        next_pop = []
        for front in fronts:
            if len(next_pop) + len(front) <= args.pop_size:
                next_pop.extend([combined[i] for i in front])
            else:
                crowd = crowding_distance(front, objs)
                front_sorted = sorted(front, key=lambda i: crowd[i], reverse=True)
                slots = args.pop_size - len(next_pop)
                next_pop.extend([combined[i] for i in front_sorted[:slots]])
                break
        population = next_pop

        # Report best front summary
        pop_objs = [objectives(r) for r in population]
        fronts = fast_non_dominated_sort(pop_objs)
        f0 = fronts[0]
        best_acc = max((population[i]["mAP"] for i in f0 if population[i]["mAP"] > 0), default=-1)
        best_lat = min((population[i]["lat_ms"] for i in f0 if population[i]["lat_ms"] > 0), default=1e9)
        best_size = min((population[i].get("on_disk_mb", population[i]["sd_size_mb"]) for i in f0), default=1e9)
        print(f"[gen {gen}] Pareto front: |F0|={len(f0)} | max mAP={best_acc:.4f} | min lat={best_lat:.2f} ms | min size={best_size:.2f} MB")

    # Save final Pareto front
    pop_objs = [objectives(r) for r in population]
    fronts = fast_non_dominated_sort(pop_objs)
    f0 = fronts[0]
    pareto = [population[i] for i in f0]

    # Save JSON & CSV
    with open(Path(args.outdir) / "pareto.json", "w", encoding="utf-8") as f:
        json.dump(pareto, f, indent=2)
    with open(Path(args.outdir) / "pareto.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(cand_keys + ["mAP","lat_ms","on_disk_mb","sd_size_mb","weights","yaml","run_dir"])
        for r in pareto:
            writer.writerow([r.get(k, "") for k in cand_keys] + [r.get("mAP",""), r.get("lat_ms",""),
                                                                 r.get("on_disk_mb",""), r.get("sd_size_mb",""),
                                                                 r.get("weights",""), r.get("yaml",""), r.get("run_dir","")])
    print("[done] Pareto front saved to pareto.json / pareto.csv")

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, required=True, help="Path to data.yaml")
    p.add_argument("--baseline_weights", type=str, default="", help="Baseline .pt weights")
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--outdir", type=str, default="runs/nas_y11_nsga2")
    p.add_argument("--pop_size", type=int, default=12)
    p.add_argument("--generations", type=int, default=10)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)

    # Latency eval params
    p.add_argument("--lat_iters", type=int, default=50)
    p.add_argument("--lat_warmup", type=int, default=10)

    # Variation rates
    p.add_argument("--mutation_prob", type=float, default=0.2)
    p.add_argument("--crossover_prob", type=float, default=0.3)

    # Optional hard constraints
    p.add_argument("--hard_constraints", type=lambda x: str(x).lower() in ["1","true","t","yes","y"], default=False)
    p.add_argument("--min_map_drop", type=float, default=0.2)
    p.add_argument("--latency_slack", type=float, default=0.03)
    p.add_argument("--size_cap_mb", type=float, default=10.0)

    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    nsga2_search(args)