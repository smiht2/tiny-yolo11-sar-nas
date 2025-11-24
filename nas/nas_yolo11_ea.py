#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
nas_yolo11_ea.py  (v2) — YAML fixed to include proper P5/32 backbone level.
- Adds final downsample to 1024 before SPPF (so P5 is /32)
- Head indices updated to match P3(/8), P4(/16), P5(/32)
"""

import argparse
import csv
import json
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
# Search space definition
# -----------------------------
WIDTH_MULTS = [0.25, 0.33, 0.50, 0.67, 0.75]
REPEATS = [1, 2, 3]  # for C2f repeats at 3 backbone stages (128/256/512)
DOWNSAMPLE_OPS = ["Conv", "DWConv", "GhostConv"]
DOWNSAMPLE_KS = [3, 5]
NECK_REPEATS = [1, 2]


def rand_choice(seq):
    return random.choice(seq)


@dataclass
class Candidate:
    rep_128: int   # C2f repeats at 128-ch stage
    rep_256: int   # C2f repeats at 256-ch stage
    rep_512: int   # C2f repeats at 512-ch stage
    width_mult: float
    down_op: str   # op for stride-2 downsample convs
    down_k: int    # kernel for down_op
    pan_repeats: int  # C2f repeats in PAN head blocks

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
# YAML template (YOLOv11-style). Only backbone + head.
# P3=/8 is layer 4, P4=/16 is layer 6, P5=/32 is layer 9.
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
  - [-1, {rep256}, C2f, [256, True]]           # 4  (P3)
  - [-1, 1, {down_op}, [512, {ks}, 2]]         # 5-P4/16
  - [-1, {rep512}, C2f, [512, True]]           # 6  (P4)
  - [-1, 1, {down_op}, [1024, {ks}, 2]]        # 7-P5/32
  - [-1, 1, C2f, [1024, True]]                 # 8
  - [-1, 1, SPPF, [1024, 5]]                   # 9  (P5)

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


def try_thop(model_module: torch.nn.Module, imgsz: int, device: str) -> Tuple[float, float]:
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


def score_candidate(mAP: float, lat_ms: float, size_mb: float,
                    baseline_map: float, baseline_lat: float,
                    min_map_drop: float, latency_slack: float, size_cap_mb: float) -> float:
    score = mAP  # primary
    lat_cap = baseline_lat * (1.0 + latency_slack) if baseline_lat > 0 else float('inf')
    if 0 < lat_ms <= lat_cap:
        score += 0.02 * (lat_cap / max(lat_ms, 1e-3))
    if 0 < size_mb <= size_cap_mb:
        score += 0.01 * (size_cap_mb / max(size_mb, 1e-3))
    if baseline_map > 0 and mAP < baseline_map - min_map_drop:
        score -= 2.0 * (baseline_map - min_map_drop - mAP)
    if baseline_lat > 0 and lat_ms > lat_cap:
        score -= 1.0 * ((lat_ms - lat_cap) / max(lat_cap, 1e-3))
    if size_cap_mb > 0 and size_mb > size_cap_mb:
        score -= 1.0 * ((size_mb - size_cap_mb) / max(size_cap_mb, 1e-3))
    return score


def evaluate_candidate(cand: Candidate, args, run_dir: Path, nc: int,
                       baseline_map: float, baseline_lat: float) -> Dict[str, Any]:
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

    # Train
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

    # Score
    sc = score_candidate(
        mAP=mAP, lat_ms=lat_ms, size_mb=on_disk_mb if on_disk_mb > 0 else sd_size_mb,
        baseline_map=baseline_map, baseline_lat=baseline_lat,
        min_map_drop=args.min_map_drop, latency_slack=args.latency_slack, size_cap_mb=args.size_cap_mb
    )

    out = {
        **asdict(cand),
        "mAP": mAP,
        "lat_ms": lat_ms,
        "macs": macs,
        "params": params,
        "sd_size_mb": sd_size_mb,
        "on_disk_mb": on_disk_mb,
        "score": sc,
        "weights": str(weights_path) if weights_path else "",
        "yaml": str(yaml_path),
        "run_dir": str(cand_dir),
    }
    return out


def measure_baseline(args, nc: int) -> Tuple[float, float]:
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

    print("[info] No --baseline_weights provided. Training a quick baseline from yolo11n for reference...")
    tmp_dir = Path(args.outdir) / "baseline_quick"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO("yolo11n.pt")
    model.train(data=args.data, epochs=max(5, args.epochs//2), imgsz=args.imgsz,
                batch=args.batch, device=args.device, workers=args.workers,
                project=str(tmp_dir), name="train", patience=0, seed=args.seed, verbose=False)
    w = tmp_dir / "train" / "weights" / "best.pt"
    if not w.exists():
        w = tmp_dir / "train" / "weights" / "last.pt"
    base_model = YOLO(str(w)) if w.exists() else model
    val_res = base_model.val(data=args.data, imgsz=args.imgsz, device=args.device, split="val", workers=args.workers)
    baseline_map = float(getattr(getattr(val_res, "box", None), "map", -1.0))
    if baseline_map < 0 and hasattr(val_res, "results_dict"):
        baseline_map = float(val_res.results_dict.get("metrics/mAP50-95(B)", -1.0))
    baseline_lat = measure_latency(base_model, args.imgsz, device,
                                   iters=args.lat_iters, warmup=max(5, args.lat_warmup))
    return baseline_map, baseline_lat


def evolutionary_search(args):
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

    # Baseline
    baseline_map, baseline_lat = measure_baseline(args, nc=nc)
    print(f"[baseline] mAP: {baseline_map:.4f} | latency: {baseline_lat:.2f} ms")

    # CSV log
    csv_path = outdir / "ea_log.csv"
    header_cand = list(Candidate.random().__dict__.keys())
    header = ["gen", "idx"] + header_cand + \
             ["mAP", "lat_ms", "macs", "params", "sd_size_mb", "on_disk_mb", "score", "weights", "yaml", "run_dir"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(header)

    # Init population
    population: List[Dict[str, Any]] = []
    for i in range(args.pop_size):
        cand = Candidate.random()
        res = evaluate_candidate(cand, args, outdir, nc, baseline_map, baseline_lat)
        res["gen"] = 0
        res["idx"] = i
        population.append(res)
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([res.get(k, "") for k in header])

    # Keep top_k
    population.sort(key=lambda x: x.get("score", -1e9), reverse=True)
    population = population[:args.top_k]
    if population:
        best0 = population[0]
        print(f"[gen 0] best score={best0['score']:.4f}, mAP={best0['mAP']:.4f}, lat={best0['lat_ms']:.2f}ms, size={best0['on_disk_mb']:.2f}MB")

    # Evolve
    for gen in range(1, args.generations + 1):
        new_pop: List[Dict[str, Any]] = []
        for i in range(args.pop_size):
            if len(population) >= 2:
                a, b = random.sample(population, k=2)
                parent = a if a.get("score", -1e9) > b.get("score", -1e9) else b
            elif population:
                parent = population[0]
            else:
                parent = {"rep_128": 2, "rep_256": 2, "rep_512": 2, "width_mult": 0.5,
                          "down_op": "Conv", "down_k": 3, "pan_repeats": 1, "score": -1e9}

            pkeys = list(Candidate.random().__dict__.keys())
            parent_cand = Candidate(**{k: parent[k] for k in pkeys})
            cand = parent_cand.mutate(prob=args.mutation_prob)
            if random.random() < args.crossover_prob and len(population) >= 2:
                other = random.choice(population)
                other_cand = Candidate(**{k: other[k] for k in pkeys})
                cand = Candidate.crossover(cand, other_cand)

            res = evaluate_candidate(cand, args, outdir, nc, baseline_map, baseline_lat)
            res["gen"] = gen
            res["idx"] = i
            new_pop.append(res)

            with open(csv_path, "a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow([res.get(k, "") for k in header])

        population = (population + new_pop)
        population.sort(key=lambda x: x.get("score", -1e9), reverse=True)
        population = population[:args.top_k]

        best = population[0]
        print(f"[gen {gen}] best score={best['score']:.4f}, mAP={best['mAP']:.4f}, lat={best['lat_ms']:.2f}ms, size={best['on_disk_mb']:.2f}MB")

    best = population[0]
    with open(outdir / "best.json", "w", encoding="utf-8") as f:
        json.dump(best, f, indent=2)
    print("[done] Best candidate saved to", outdir / "best.json")
    print("YAML:", best.get("yaml", ""))
    print("Weights:", best.get("weights", ""))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, required=True, help="Path to data.yaml")
    p.add_argument("--baseline_weights", type=str, default="", help="Baseline .pt weights for mAP/latency reference")
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--outdir", type=str, default="runs/nas_y11")
    p.add_argument("--pop_size", type=int, default=12)
    p.add_argument("--generations", type=int, default=10)
    p.add_argument("--top_k", type=int, default=6)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--min_map_drop", type=float, default=0.5)
    p.add_argument("--latency_slack", type=float, default=0.05)
    p.add_argument("--size_cap_mb", type=float, default=10.0)

    p.add_argument("--lat_iters", type=int, default=50)
    p.add_argument("--lat_warmup", type=int, default=10)

    p.add_argument("--mutation_prob", type=float, default=0.2)
    p.add_argument("--crossover_prob", type=float, default=0.3)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evolutionary_search(args)
