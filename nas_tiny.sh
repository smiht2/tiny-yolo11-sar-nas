# python nas/nas_yolo11_ea.py \
#   --data cfg/data_new_scene.yaml \
#   --baseline_weights weights/baseline_y11x_best.pt \
#   --device 0 --imgsz 640 \
#   --outdir runs/nas_yolo11_xview3test \
#   --pop_size 12 --generations 10 \
#   --epochs 10 --batch 16 --workers 0 \
#   --min_map_drop 0.5 --latency_slack 0.05 --size_cap_mb 10.0


## training
# yolo detect train \
#   model=models/y11_tiny_nas.yaml \
#   data=cfg/data_new_scene.yaml imgsz=640 device=0 batch=16 \
#   epochs=10 optimizer=AdamW lr0=0.002 cos_lr=True warmup_epochs=1 \
#   amp=True patience=40

## Validation with CSV output
python -c "
import csv
from ultralytics import YOLO
import pandas as pd

model = YOLO('weights/nsga_knee_best.pt')
results = model.val(
    data='cfg/data_ssdd.yaml',
    imgsz=640,
    device=0,
    project='runs/detect',
    name='val_ssdd_on_val_set',
    save_json=True
)

# Save results to CSV
metrics = results.results_dict
csv_data = {
    'model': ['nsga_knee_best'],
    'mAP50': [metrics.get('metrics/mAP50(B)', 0)],
    'mAP50-95': [metrics.get('metrics/mAP50-95(B)', 0)],
    'precision': [metrics.get('metrics/precision(B)', 0)],
    'recall': [metrics.get('metrics/recall(B)', 0)],
    'fitness': [results.fitness]
}
df = pd.DataFrame(csv_data)
df.to_csv('runs/detect/val_ssdd_on_test_set/results.csv', index=False)
print(f'Results saved to runs/detect/val_ssdd_on_test_set/results.csv')
"