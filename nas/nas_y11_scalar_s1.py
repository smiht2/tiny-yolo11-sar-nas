
"""
This script reuses your existing evaluator & search space from nas_yolo11_ea.py:
- Uses base.Candidate.random()/mutate()/crossover()
- Calls base.evaluate_candidate(Candidate, args, Path(args.outdir), nc, baseline_map, baseline_lat)
- Measures baseline with base.measure_baseline(args, nc)
Place this file next to nas_yolo11_ea.py or add that folder to PYTHONPATH.
"""

import argparse, json, os, random, sys
from pathlib import Path
from typing import Any, Dict, List

def load_nc_from_data_yaml(data_path: str) -> int:
    import yaml
    with open(data_path, "r", encoding="utf-8") as f:
        d = yaml.safe_load(f)
    names = d.get("names", [])
    if isinstance(names, dict):
        return len(names)
    elif isinstance(names, list):
        return len(names)
    return int(d.get("nc", 1))

def build_parser(default_out: str):
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, required=True)
    p.add_argument("--baseline_weights", type=str, default="")
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--outdir", type=str, default=default_out)

    p.add_argument("--pop_size", type=int, default=12)
    p.add_argument("--generations", type=int, default=10)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)

    # latency evaluator knobs (needed by base.measure_baseline / evaluate)
    p.add_argument("--lat_iters", type=int, default=50)
    p.add_argument("--lat_warmup", type=int, default=10)

    # old additive scoring knobs are parsed but unused here (we keep for compatibility)
    p.add_argument("--min_map_drop", type=float, default=0.5)
    p.add_argument("--latency_slack", type=float, default=0.05)
    p.add_argument("--size_cap_mb", type=float, default=10.0)

    # normalization targets
    p.add_argument("--mb", type=float, default=0.0, help="Baseline mAP if you want to set manually")
    p.add_argument("--Lb", type=float, default=0.0, help="Baseline latency(ms) if you want to set manually")
    p.add_argument("--Lstar", type=float, default=7.5, help="Latency target (ms)")
    p.add_argument("--Sstar", type=float, default=10.0, help="Size cap (MB)")
    return p

def run_search(args, score_fn, score_name: str):
    # import base
    try:
        import importlib
        base = importlib.import_module("nas_yolo11_ea")
    except Exception as e:
        print("ERROR: Could not import nas_yolo11_ea. Put this script next to it or add its folder to PYTHONPATH.")
        print("Exception:", e)
        sys.exit(2)

    if not hasattr(base, "Candidate") or not hasattr(base, "evaluate_candidate") or not hasattr(base, "measure_baseline"):
        print("ERROR: nas_yolo11_ea.py must expose Candidate, evaluate_candidate, and measure_baseline.")
        sys.exit(2)

    random.seed(args.seed)
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    # load number of classes
    nc = load_nc_from_data_yaml(args.data)

    # ensure args carry what base.measure_baseline expects
    # (it uses args.baseline_weights, device, imgsz, data, workers, lat_iters, lat_warmup, epochs, batch, seed, outdir)
    # so we add missing attributes dynamically if needed.
    if not hasattr(args, "outdir"): setattr(args, "outdir", str(outdir))

    # measure baseline (or take supplied)
    mb, Lb = float(args.mb), float(args.Lb)
    if (mb <= 0 or Lb <= 0):
        # base.measure_baseline returns (mAP, latency)
        mb2, Lb2 = base.measure_baseline(args, nc)
        if mb <= 0: mb = float(mb2)
        if Lb <= 0: Lb = float(Lb2)
    print(f"[baseline] mAP={mb:.4f}, latency={Lb:.2f} ms (normalization)")

    # helper to eval + scalar
    def eval_and_score(cand) -> Dict[str, Any]:
        rec = base.evaluate_candidate(cand, args, outdir, nc, baseline_map=mb, baseline_lat=Lb)
        rec["scalar"] = score_fn(rec, mb=mb, Lb=Lb, Lstar=args.Lstar, Sstar=args.Sstar)
        rec["__cand__"] = cand.__dict__.copy()
        return rec

    # init population
    population: List[Dict[str, Any]] = []
    for _ in range(args.pop_size):
        c = base.Candidate.random()
        population.append(eval_and_score(c))

    # generations
    for g in range(1, args.generations + 1):
        offspring: List[Dict[str, Any]] = []
        while len(offspring) < args.pop_size:
            a, b = random.sample(population, 2)
            pa = a if a["scalar"] >= b["scalar"] else b
            c2, d2 = random.sample(population, 2)
            pb = c2 if c2["scalar"] >= d2["scalar"] else d2
            # crossover/mutate using Candidate methods
            candA = base.Candidate(**pa["__cand__"])
            candB = base.Candidate(**pb["__cand__"])
            child = base.Candidate.crossover(candA, candB) if random.random() < 0.3 else candA
            child = child.mutate(prob=args.mutation_prob)
            offspring.append(eval_and_score(child))

        combined = population + offspring
        combined.sort(key=lambda r: r["scalar"], reverse=True)
        population = combined[:args.pop_size]

        best = population[0]
        size_mb = best.get("on_disk_mb", best.get("sd_size_mb", -1))
        print(f"[gen {g}] best {score_name}={best['scalar']:.4f} | mAP={best.get('mAP',-1):.4f} "
              f"| lat={best.get('lat_ms',-1):.2f}ms | size={size_mb:.2f}MB")

    best = max(population, key=lambda r: r["scalar"])
    with open(outdir / f"best_{score_name}.json", "w", encoding="utf-8") as f:
        json.dump(best, f, indent=2)
    print(f"[done] Saved best to {outdir / f'best_{score_name}.json'}")
    print("YAML:", best.get("yaml"))
    print("Weights:", best.get("weights"))

import math
def _softplus(x: float) -> float:
    return math.log1p(math.exp(x)) if x < 20 else x

def score_s1(rec, mb, Lb, Lstar, Sstar, rho=0.5, beta=0.5, gamma=0.3, kappa=10.0, mu=10.0):
    m = float(rec.get("mAP", -1)); L = float(rec.get("lat_ms", -1)); S = float(rec.get("on_disk_mb", rec.get("sd_size_mb", -1)))
    if m <= 0 or L <= 0 or S <= 0: return -1e9
    mt = max(1e-9, m/mb); Lt = max(1e-9, Lb/L); St = max(1e-9, Sstar/S)
    U = math.log(mt) if abs(rho-1.0) < 1e-9 else ((mt**(1.0-rho))-1.0)/(1.0-rho)
    barrier = math.exp(-kappa*_softplus(L/Lstar - 1.0)) * math.exp(-mu*_softplus(S/Sstar - 1.0))
    return U * (Lt**beta) * (St**gamma) * barrier

def main():
    p = build_parser("runs/nas_scalar_s1")
    p.add_argument("--rho", type=float, default=0.5)
    p.add_argument("--beta", type=float, default=0.5)
    p.add_argument("--gamma", type=float, default=0.3)
    p.add_argument("--kappa", type=float, default=10.0)
    p.add_argument("--mu", type=float, default=10.0)
    p.add_argument("--mutation_prob", type=float, default=0.2)
    args = p.parse_args()

    def score_fn(rec, mb, Lb, Lstar, Sstar):
        return score_s1(rec, mb, Lb, Lstar, Sstar, rho=args.rho, beta=args.beta, gamma=args.gamma, kappa=args.kappa, mu=args.mu)

    run_search(args, score_fn, "s1")

if __name__ == "__main__":
    main()
