from pathlib import Path
import json
import shutil
import argparse
import torch
import yaml
from ultralytics import YOLO
from concurrent.futures import ThreadPoolExecutor

# Avoid Windows named pipes in multiprocessing.pool during label scanning.
# The dataset loader remains single-process (workers=0).
class ScanThreadPool:
    def __init__(self, workers):
        self.executor = ThreadPoolExecutor(max_workers=workers)
    def __enter__(self):
        return self
    def __exit__(self, *args):
        self.executor.shutdown(wait=True)
    def imap(self, func, iterable):
        return self.executor.map(func, iterable)

import ultralytics.data.dataset as dataset_module
import ultralytics.data.base as base_module
dataset_module.ThreadPool = ScanThreadPool
base_module.ThreadPool = ScanThreadPool

PROJECT = Path(__file__).resolve().parents[1]
DATA = PROJECT / 'dataset/carparts-seg'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--cache', default='false', choices=['false', 'ram'])
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    assert torch.cuda.is_available(), 'CUDA GPU unavailable'
    cfg = yaml.safe_load((DATA / 'data.yaml').read_text())
    cfg['path'] = DATA.as_posix()
    prepared = PROJECT / 'data.train.yaml'
    prepared.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding='utf-8')
    checkpoint = PROJECT / 'runs/carparts_yolo11n_seg/weights/last.pt'
    model = YOLO(str(checkpoint if args.resume else PROJECT / 'yolo11n-seg.pt'))
    options = dict(batch=args.batch, cache='ram' if args.cache=='ram' else False, device=0,
                   workers=0, plots=True)
    if args.resume:
        model.train(resume=True, **options)
    else:
        model.train(data=str(prepared), epochs=50, patience=12, imgsz=640, seed=42,
                    project=str(PROJECT / 'runs'), name='carparts_yolo11n_seg', exist_ok=True, **options)
    best = Path(model.trainer.best)
    trained = YOLO(str(best))
    metrics = trained.val(data=str(prepared), split='test', device=0, batch=8,
                          workers=0, project=str(PROJECT / 'runs'), name='test_evaluation', plots=True)
    (PROJECT / 'test_metrics.json').write_text(json.dumps(metrics.results_dict, indent=2), encoding='utf-8')
    selected = []
    for im in sorted((DATA / 'images/test').glob('*')):
        lab = DATA / 'labels/test' / (im.stem + '.txt')
        if any(line.split() and line.split()[0]=='23' for line in lab.read_text().splitlines()):
            selected.append(str(im))
    selected += [str(p) for p in sorted((DATA / 'images/test').glob('*')) if str(p) not in selected][:12]
    trained.predict(source=selected, device=0, imgsz=640, conf=0.25, save=True,
                    project=str(PROJECT / 'runs'), name='test_predictions', exist_ok=True)
    deliver = Path(__file__).resolve().parent
    shutil.copy2(best, deliver / 'carparts_yolo11n_seg_best.pt')
    shutil.copy2(PROJECT / 'test_metrics.json', deliver / 'test_metrics.json')
    print(f'TRAINING_AND_TEST_COMPLETE: {best}', flush=True)

if __name__ == '__main__':
    main()
