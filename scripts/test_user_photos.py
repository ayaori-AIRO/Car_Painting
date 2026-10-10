from pathlib import Path
import json
import torch
from ultralytics import YOLO

torch.set_num_threads(4)
source = (Path(__file__).resolve().parents[1]/'dataset/test')
output = Path(__file__).resolve().parents[1] / 'runs/user_photo_results_20261009'
output.mkdir(exist_ok=True)
model = YOLO(str(Path(__file__).resolve().parents[1]/'runs/carparts_fender_refresh_20261009/weights/best.pt'))
summary = []
for n, path in enumerate(sorted(source.iterdir()), 1):
    if path.suffix.lower() not in {'.png', '.jpg', '.jpeg', '.webp', '.bmp'}:
        continue
    result = model.predict(str(path), imgsz=640, conf=0.25, device=0, retina_masks=True, verbose=False)[0]
    target = output / f'result_{n:02d}.jpg'
    result.save(filename=str(target), labels=True, conf=True, boxes=True)
    result.save(filename=str(output / f'masks_{n:02d}.jpg'), labels=False, conf=False, boxes=False)
    detections = []
    for box in result.boxes:
        detections.append({'class': result.names[int(box.cls.item())], 'confidence': round(float(box.conf.item()), 4), 'xyxy': box.xyxy[0].tolist()})
    summary.append({'source': str(path), 'result': str(target), 'detections': detections})
    print(json.dumps(summary[-1], ensure_ascii=True), flush=True)
(output / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
