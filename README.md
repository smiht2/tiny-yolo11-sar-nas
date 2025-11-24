# Tiny YOLO11 NAS for SAR Ship Detection

This project discovers and trains **tiny YOLO11-style** models for **SAR ship detection** under SWaP constraints. It includes two search modes you can run locally on an RTX 4080:

* **Scalar EA** — single score with soft penalties (size/latency/mAP constraints).
* **NSGA-II** — true Pareto multi-objective search with **knee-point** selection.

## Key Results (SSDD @ 640)

| Model               |    Params |   GFLOPs |  mAP50-95 | Inference @640 |
| ------------------- | --------: | -------: | --------: | -------------: |
| YOLO11x (baseline)  |     56.8M |    194.4 | **0.737** |   ~**15.5 ms** |
| EA tiny (final)     | **2.78M** |  **7.2** | **0.722** |    ~**1.3 ms** |
| NSGA-II knee (warm) | **5.21M** | **14.3** | **0.706** |    ~**2.0 ms** |

> Notes:
> • Baseline measured from trained YOLO11x checkpoint.
> • NSGA-knee “warm” = 50-epoch retrain; accuracy typically improves with 120–200 epochs (early stop).
> • Latency is Ultralytics’ GPU forward timing; confirm on device (TensorRT) for deployment.

---

## Repo Layout (what’s included)

```
cfg/                 # dataset configs (user-edited)
nas/                 # search scripts (EA + NSGA-II) and knee picker
models/              # discovered architectures (YAML)
weights/             # LFS-tracked model weights (.pt)
artifacts/           # logs, Pareto, best.json records
```

---

## Environment (tested on Windows, CUDA 12.1 / RTX 4080)

```bash
# Conda (example)
conda create -n sar-nas python=3.12 -y
conda activate sar-nas

# PyTorch (CUDA 12.1 wheels for Ada/4080)
pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio

# Project deps
pip install ultralytics==8.3.203 thop pyyaml numpy pillow opencv-python matplotlib
```

> If you see OpenMP DLL clashes on Windows, set:
>
> * PowerShell:
>
>   ```powershell
>   $env:KMP_DUPLICATE_LIB_OK="TRUE"
>   $env:OMP_NUM_THREADS="8"
>   ```
> * CMD: `set KMP_DUPLICATE_LIB_OK=TRUE`

---

## Dataset config

Edit `cfg/data_ssdd.yaml` to point to your dataset path.

```yaml
# cfg/data_ssdd.yaml
path: C:/Users/pretha/Desktop/optimization-paper-code/SSDD.v1i.yolov11
train: ${path}/train/images
val:   ${path}/val/images
test:  ${path}/test/images   # optional
names: [ship]
```

---

## Baseline checkpoint (used for constraints)

Place your trained baseline YOLO11x weights here (tracked by Git LFS):

```
weights/baseline_y11x_best.pt
```

---

## Run — Scalar EA (single-score with penalties)

```powershell
python nas/nas_yolo11_ea.py `
  --data cfg/data_ssdd.yaml `
  --baseline_weights weights/baseline_y11x_best.pt `
  --device 0 --imgsz 640 `
  --outdir runs/nas_yolo11_ssdd `
  --pop_size 12 --generations 10 `
  --epochs 10 --batch 16 --workers 0 `
  --min_map_drop 0.5 --latency_slack 0.05 --size_cap_mb 10.0
```

**What it does:**

* Builds candidate YAMLs, short-trains (5–10 ep), validates (mAP50-95), times latency, computes size.
* **Score(x)** = mAP − penalties if size/latency/mAP constraints are violated.
* Tournament selection + uniform crossover + per-gene mutation; (μ+λ) replacement.

---

## Run — NSGA-II (Pareto search + knee selection)

```powershell
python nas/nas_yolo11_nsga2.py `
  --data cfg/data_ssdd.yaml `
  --baseline_weights weights/baseline_y11x_best.pt `
  --device 0 --imgsz 640 `
  --outdir runs/nas_yolo11_ssdd_nsga2 `
  --pop_size 12 --generations 10 `
  --epochs 10 --batch 16 --workers 0 `
  --hard_constraints True --min_map_drop 0.2 --latency_slack 0.03 --size_cap_mb 10.0
```

**What it does:**

* Minimizes **(−mAP, latency, size)** via fast non-dominated sorting + crowding distance.
* Produces a Pareto set; picks the **knee** (closest to ideal after normalization).
* Logs: `pareto.csv`, `nsga2_log.csv`, and `best_nsga.json`.

---

## Retrain finalists (long)

EA tiny:

```powershell
yolo detect train `
  model=models/y11_tiny_nas.yaml `
  data=cfg/data_ssdd.yaml imgsz=640 device=0 batch=16 `
  epochs=200 optimizer=AdamW lr0=0.002 cos_lr=True warmup_epochs=3 `
  amp=True patience=40
```

NSGA knee:

```powershell
yolo detect train `
  model=models/nsga_knee.yaml `
  data=cfg/data_ssdd.yaml imgsz=640 device=0 batch=16 `
  epochs=200 optimizer=AdamW lr0=0.002 cos_lr=True warmup_epochs=3 `
  amp=True patience=40
```

---

## Evaluate & compare

```powershell
yolo detect val model=weights/baseline_y11x_best.pt data=cfg/data_ssdd.yaml imgsz=640
yolo detect val model=weights/ea_tiny_best.pt        data=cfg/data_ssdd.yaml imgsz=640
yolo detect val model=weights/nsga_knee_best.pt      data=cfg/data_ssdd.yaml imgsz=640
```

---

## Latency & deployment (Jetson/edge)

* Export ONNX/TensorRT and benchmark **FP16 / INT8** on target hardware.
* Prefer **PTQ** first; if drop is high, use **QAT** (few epochs).
* Rank final candidates by **post-quantization mAP** at the **device latency** budget.

---

## Compression extras

* **Pruning:** channel-wise (L1/L2) after sensitivity check; re-train briefly.
* **Quantization:** PTQ → QAT; maintain calibration set from SSDD val images.
* **Knowledge Distillation:** teacher YOLO11x → student (EA tiny / NSGA-knee).

---

## Reproducibility

* `--seed 42` for retrains; logs saved under `runs/…` and `artifacts/logs/`.
* NAS runs short-train for speed; **always** long-retrain finalists.

---

## Troubleshooting (Windows)

* **OpenMP conflict:** set `KMP_DUPLICATE_LIB_OK=TRUE`.
* **Pillow/NumPy DLL errors:** pin compatible versions (as in `requirements.txt`).
* **Long paths on Windows:** enable long paths or avoid deep directories.

---

## Third-party software

This repo uses the **official Ultralytics YOLO v11** library (`ultralytics==8.3.203`) for model build/train/val.
Refer to Ultralytics’ license and documentation for usage terms.

---

## License

-
---

## Acknowledgements

* Ultralytics YOLO v11
* SSDD dataset maintainers
* Inspiration from efficient ML literature (ICLR/ICML/NeurIPS; Han Lab for system-aware training, quantization, and compression)
