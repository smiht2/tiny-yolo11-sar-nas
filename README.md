# Tiny YOLO11 NAS for SAR Ship Detection (SSDD)

- Baseline: YOLO11x — mAP50-95 ~0.737, ~15.5 ms @640
- EA tiny:   2.78M params, 7.2 GFLOPs — mAP50-95 ~0.722, ~1.3 ms
- NSGA knee (warm): 5.21M params, 14.3 GFLOPs — mAP50-95 ~0.706, ~2.0 ms

## How to run
- EA:    python nas/nas_yolo11_ea.py --data cfg/data_ssdd.yaml --baseline_weights weights/baseline_y11x_best.pt ...
- NSGA2: python nas/nas_yolo11_nsga2.py --data cfg/data_ssdd.yaml --baseline_weights weights/baseline_y11x_best.pt ...
