from pathlib import Path
import argparse
from ultralytics import YOLO

if __name__ == '__main__':
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True)
    parser.add_argument('--model', default=str(root/'runs/carparts_12classes_20261012/weights/best.pt'))
    parser.add_argument('--conf', type=float, default=0.25)
    parser.add_argument('--device', default='cpu', help='Use 0 for the first CUDA GPU')
    args = parser.parse_args()
    YOLO(args.model).predict(source=args.source, imgsz=640, conf=args.conf, device=args.device,
                             save=True, project=str(root/'runs'), name='predict_latest')
