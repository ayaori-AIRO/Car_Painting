import os, sys, json, time, threading, gc
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
os.environ['ALBUMENTATIONS_NO_TELEMETRY']='1'
os.environ['NO_ALBUMENTATIONS_UPDATE']='1'
sys.path.insert(0,str(Path(__file__).parent/'augmentation_packages'))
import albumentations as A
import cv2, numpy as np, torch, yaml
from train_and_test import YOLO

ROOT=Path(__file__).resolve().parents[1]/'runs/augmentation_training_20261009'
DATA=(Path(__file__).resolve().parents[1]/'dataset/carparts-seg')
CHECKPOINT=(Path(__file__).resolve().parents[1]/'runs/carparts_fender_refresh_20261009/weights/best.pt')
ROOT.mkdir(exist_ok=True)
START=time.time(); lock=threading.Lock()
state={'state':'Preparing','experiment':'lighting','epochs':8,'output':str(ROOT)}
def update(**kw):
 with lock:
  state.update(kw); state['elapsed_seconds']=round(time.time()-START)
  (ROOT/'status.json').write_text(json.dumps(state,ensure_ascii=False,indent=2),encoding='utf-8')
class Handler(BaseHTTPRequestHandler):
 def do_GET(self):
  if self.path=='/status':
   with lock: payload=json.dumps(state,ensure_ascii=False).encode('utf-8')
   content='application/json; charset=utf-8'
  else:
   payload=Path(__file__).with_name('augmentation_live.html').read_bytes(); content='text/html; charset=utf-8'
  self.send_response(200); self.send_header('Content-Type',content); self.send_header('Cache-Control','no-store'); self.end_headers(); self.wfile.write(payload)
 def log_message(self,*a): pass

def metrics_report(metrics):
 rows=[]
 for idx,cls in enumerate(metrics.seg.ap_class_index):
  rows.append({'id':int(cls),'class':metrics.names[int(cls)],'precision':float(metrics.seg.p[idx]),'recall':float(metrics.seg.r[idx]),'AP50':float(metrics.seg.ap50[idx]),'AP50_95':float(metrics.seg.ap[idx])})
 return {'overall':metrics.results_dict,'classes':rows}

def main():
 torch.set_num_threads(4)
 threading.Thread(target=ThreadingHTTPServer(('127.0.0.1',8791),Handler).serve_forever,daemon=True).start()
 configs={
  'lighting':A.Compose([A.RandomBrightnessContrast(brightness_limit=0.18,contrast_limit=0.18,p=0.7),A.RandomGamma(gamma_limit=(80,120),p=0.5)]),
  'quality':A.Compose([A.OneOf([A.GaussianBlur(blur_limit=(3,3),sigma_limit=(0.3,0.8),p=1),A.ImageCompression(quality_range=(70,90),p=1),A.Downscale(scale_range=(0.7,0.85),p=1)],p=1)])}
 images=sorted(p for p in (DATA/'images/train').iterdir() if p.suffix.lower() in {'.jpg','.jpeg','.png','.webp'})
 cfg=yaml.safe_load((DATA/'data.yaml').read_text(encoding='utf-8'))
 manifest={'library_version':A.__version__,'checkpoint':str(CHECKPOINT),'seed':42,'epochs':8,'original_train_images':len(images),'experiments':{}}
 prepared_manifest=ROOT/'manifest.json'
 for name,pipeline in configs.items():
  if prepared_manifest.exists() and (ROOT/name/'data.yaml').exists():
   continue
  update(state='Preparing',experiment=name,generated=0,total=len(images),epoch=0,batch=0)
  folder=ROOT/name; imdir=folder/'images/train'; labdir=folder/'labels/train'
  imdir.mkdir(parents=True,exist_ok=True); labdir.mkdir(parents=True,exist_ok=True)
  train=[]
  for i,path in enumerate(images,1):
   image=cv2.imdecode(np.fromfile(path,dtype=np.uint8),cv2.IMREAD_COLOR)
   if image is None: raise RuntimeError(f'Image decode failed: {path}')
   rgb=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
   pipeline.set_random_seed(42+i)
   transformed=pipeline(image=rgb)['image']
   assert transformed.shape==rgb.shape
   target=imdir/f'aug_{path.stem}.png'; label=DATA/'labels/train'/f'{path.stem}.txt'
   if not label.exists(): raise RuntimeError(f'Missing label: {label}')
   label_target=labdir/f'{target.stem}.txt'
   cv2.imencode('.png',cv2.cvtColor(transformed,cv2.COLOR_RGB2BGR))[1].tofile(target)
   label_target.write_bytes(label.read_bytes())
   assert label_target.read_bytes()==label.read_bytes()
   train.extend([path.as_posix(),target.as_posix()])
   if i%100==0: update(generated=i)
  train_file=folder/'train.txt'; train_file.write_text('\n'.join(train),encoding='utf-8')
  config={**cfg,'path':folder.as_posix(),'train':train_file.as_posix(),'val':(DATA/'images/val').as_posix(),'test':(DATA/'images/test').as_posix()}
  yaml_path=folder/'data.yaml'; yaml_path.write_text(yaml.safe_dump(config,allow_unicode=True),encoding='utf-8')
  manifest['experiments'][name]={'train_images':len(train),'labels_verified':len(images),'data':str(yaml_path)}
 if not prepared_manifest.exists():
  prepared_manifest.write_text(json.dumps(manifest,indent=2),encoding='utf-8')
 # Evaluate the starting checkpoint on the same untouched test set as both experiments.
 comparison=ROOT/'comparison.json'
 if comparison.exists():
  reports=json.loads(comparison.read_text())
 else:
  update(state='Testing',experiment='baseline')
  base=YOLO(str(CHECKPOINT))
  metrics=base.val(data=str(ROOT/'lighting/data.yaml'),split='test',imgsz=640,device=0,batch=16,workers=2,project=str(ROOT/'runs'),name='baseline_test',plots=True)
  reports={'baseline':metrics_report(metrics)}
  comparison.write_text(json.dumps(reports,indent=2),encoding='utf-8')
  del base,metrics; gc.collect(); torch.cuda.empty_cache()
 for name in configs:
  if name in reports: continue
  update(state='Training',experiment=name,epoch=0,batch=0)
  resume_path=ROOT/'runs'/name/'weights/last.pt'
  model=YOLO(str(resume_path if resume_path.exists() else CHECKPOINT)); batches=[0]; last=[0]
  def epoch_start(trainer):
   batches[0]=0; update(state='Training',epoch=trainer.epoch+1,epochs=trainer.epochs,batch=0,batches=len(trainer.train_loader))
  def batch_done(trainer):
   batches[0]+=1
   if time.time()-last[0]>2:
    last[0]=time.time(); update(batch=batches[0],gpu_memory_gb=round(torch.cuda.memory_reserved()/1e9,2))
  def validation(trainer): update(state='Validating')
  def fit_done(trainer): update(metrics={k:float(v) for k,v in trainer.metrics.items()})
  model.add_callback('on_train_epoch_start',epoch_start); model.add_callback('on_train_batch_end',batch_done)
  model.add_callback('on_train_epoch_end',validation); model.add_callback('on_fit_epoch_end',fit_done)
  model.train(resume=resume_path.exists(),data=str(ROOT/name/'data.yaml'),epochs=8,patience=8,imgsz=640,batch=32,cache=False,device=0,workers=2,amp=True,optimizer='AdamW',lr0=0.0003,warmup_epochs=1,seed=42,
   hsv_h=0,hsv_s=0,hsv_v=0,fliplr=0,flipud=0,degrees=0,translate=0.1,scale=0.3,mosaic=0.5,close_mosaic=2,mixup=0,copy_paste=0,
   project=str(ROOT/'runs'),name=name,exist_ok=True,plots=True)
  best=str(model.trainer.best); del model; gc.collect(); torch.cuda.empty_cache()
  update(state='Testing',experiment=name,best=best)
  trained=YOLO(best)
  metrics=trained.val(data=str(ROOT/name/'data.yaml'),split='test',imgsz=640,device=0,batch=16,workers=2,project=str(ROOT/'runs'),name=f'{name}_test',plots=True)
  reports[name]=metrics_report(metrics)
  (ROOT/'comparison.json').write_text(json.dumps(reports,indent=2),encoding='utf-8')
  photos=sorted(str(p) for p in (Path(__file__).resolve().parents[1]/'dataset/test').glob('*.png'))
  trained.predict(source=photos,imgsz=640,conf=0.25,device=0,save=True,project=str(ROOT/'runs'),name=f'{name}_photo_predictions')
  del trained,metrics; gc.collect(); torch.cuda.empty_cache()
 update(state='Complete',experiment='all',comparison=str(ROOT/'comparison.json'))
 print('BOTH_EXPERIMENTS_COMPLETE',flush=True)
 # Keep the dashboard available, without retaining the trained GPU models.
 while True: time.sleep(5)
if __name__=='__main__':
 try: main()
 except Exception as exc:
  update(state='Failed',error=str(exc)); raise
