from pathlib import Path
import json
import torch
from ultralytics import YOLO

torch.set_num_threads(4)
source = Path(r'C:\Projects\Car_Painting\dataset\test\스크린샷 2026-10-09 184337 - 복사본.png')
output = Path(__file__).resolve().parents[1] / 'runs/single_photo_184337_copy'
output.mkdir(exist_ok=True)
model = YOLO(str(Path(__file__).resolve().parents[1]/'runs/carparts_fender_refresh_20261009/weights/best.pt'))
result = model.predict(str(source), imgsz=640, conf=0.25, device=0, retina_masks=True, verbose=False)[0]
result.save(filename=str(output / 'predictions.jpg'), labels=True, conf=True, boxes=True, line_width=1, font_size=10)
result.save(filename=str(output / 'masks.jpg'), labels=False, conf=False, boxes=False)
summary = [{'class': result.names[int(b.cls.item())], 'confidence': round(float(b.conf.item()), 4)} for b in result.boxes]
(output / 'detections.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(summary))
