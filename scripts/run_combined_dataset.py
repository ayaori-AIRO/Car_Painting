import os,sys,time,json,threading,gc,shutil
from pathlib import Path
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
os.environ['ALBUMENTATIONS_NO_TELEMETRY']='1'
os.environ['NO_ALBUMENTATIONS_UPDATE']='1'
sys.path.insert(0,str(Path(__file__).parent/'augmentation_packages'))
from PIL import Image,JpegImagePlugin
import numpy as np,torch,yaml
from train_and_test import YOLO

PROJECT=Path(__file__).resolve().parents[1]
SOURCE=PROJECT/'dataset/carparts-seg'
DATA=PROJECT/'dataset/carparts-seg-augmented-all'
RUN=PROJECT/'runs/carparts_combined_augmentation_20261011'
INITIAL=PROJECT/'runs/carparts_fender_refresh_20261009/weights/best.pt'
WORK=Path(__file__).resolve().parent
GROUPS={
 'lighting':WORK/'augmentation_training_20261009/lighting',
 'quality':WORK/'augmentation_training_20261009/quality',
 'noise':WORK/'noise_occlusion_training_20261010/noise',
 'occlusion':WORK/'noise_occlusion_training_20261010/occlusion'}
RUN.mkdir(parents=True,exist_ok=True)
lock=threading.Lock(); started=time.monotonic()
state={'state':'Preparing','stage':'Starting','epochs':8,'dataset':str(DATA),'run':str(RUN)}
def update(**kwargs):
 with lock:
  state.update(kwargs);state['elapsed_seconds']=round(time.monotonic()-started)
  (RUN/'status.json').write_text(json.dumps(state,ensure_ascii=False,indent=2),encoding='utf-8')
class Handler(BaseHTTPRequestHandler):
 def do_GET(self):
  if self.path=='/status':
   with lock: payload=json.dumps(state,ensure_ascii=False).encode('utf-8')
   content='application/json; charset=utf-8'
  else:
   payload=Path(__file__).with_name('combined_training_live.html').read_bytes();content='text/html; charset=utf-8'
  self.send_response(200);self.send_header('Content-Type',content);self.send_header('Cache-Control','no-store');self.end_headers();self.wfile.write(payload)
 def log_message(self,*args):pass
def metrics_report(m):
 return {'overall':m.results_dict,'classes':[{'id':int(c),'class':m.names[int(c)],'precision':float(m.seg.p[i]),'recall':float(m.seg.r[i]),'AP50':float(m.seg.ap50[i]),'AP50_95':float(m.seg.ap[i])} for i,c in enumerate(m.seg.ap_class_index)]}
def image_paths(folder):return sorted(p for p in folder.iterdir() if p.suffix.lower() in {'.jpg','.jpeg','.png','.webp'})
def verify_label(path):
 for line in path.read_text().splitlines():
  parts=line.split()
  if not parts:continue
  cls=int(parts[0]); coords=np.array(list(map(float,parts[1:])))
  if not 0<=cls<24 or len(coords)<6 or len(coords)%2 or not np.isfinite(coords).all():raise ValueError(f'Invalid polygon: {path}')
def build_dataset():
 manifest_path=DATA/'manifest.json'
 if manifest_path.exists():
  return json.loads(manifest_path.read_text(encoding='utf-8'))
 DATA.mkdir(parents=True,exist_ok=True)
 source_train=image_paths(SOURCE/'images/train'); counts={}; ledger=[]
 total=sum(len(image_paths(SOURCE/'images'/s)) for s in ['train','val','test'])+4*len(source_train)
 done=0
 for split in ['train','val','test']:
  imdir=DATA/'images'/split;labdir=DATA/'labels'/split
  imdir.mkdir(parents=True,exist_ok=True);labdir.mkdir(parents=True,exist_ok=True)
  paths=image_paths(SOURCE/'images'/split)
  for p in paths:
   label=SOURCE/'labels'/split/f'{p.stem}.txt';verify_label(label)
   shutil.copyfile(p,imdir/p.name);shutil.copyfile(label,labdir/label.name)
   assert (imdir/p.name).stat().st_size==p.stat().st_size
   assert (labdir/label.name).read_bytes()==label.read_bytes()
   done+=1
   if done%20==0:update(stage=f'Copy original {split}',generated=done,total=total)
  counts[split]=len(paths)
 for name,folder in GROUPS.items():
  for index,p in enumerate(source_train,1):
   png=folder/'images/train'/f'aug_{p.stem}.png'
   label=folder/'labels/train'/f'aug_{p.stem}.txt'
   if not png.exists() or not label.exists():raise FileNotFoundError(str(png))
   verify_label(label)
   if label.read_bytes()!=(SOURCE/'labels/train'/f'{p.stem}.txt').read_bytes():raise ValueError(f'Changed label: {label}')
   target=DATA/'images/train'/f'{name}_{p.stem}.jpg'
   target_label=DATA/'labels/train'/f'{target.stem}.txt'
   with Image.open(p) as original:
    size=original.size
    tables=original.quantization if original.format=='JPEG' else None
    sampling=JpegImagePlugin.get_sampling(original) if original.format=='JPEG' else 2
   if target.exists() and target_label.exists() and target_label.read_bytes()==label.read_bytes():
    try:
     with Image.open(target) as existing:
      valid=existing.size==size and existing.format=='JPEG';existing.verify()
     if valid:
      ledger.append({'group':name,'source':str(png),'image':target.name,'bytes':target.stat().st_size})
      done+=1
      if done%20==0:update(stage=f'Verify existing {name}',generated=done,total=total,group=name,group_generated=index,group_total=len(source_train))
      continue
    except (OSError,ValueError):pass
   with Image.open(png) as image:
    if image.size!=size:raise ValueError(f'Changed image dimensions: {png}')
    options={'qtables':tables} if tables is not None else {'quality':95}
    image.convert('RGB').save(target,format='JPEG',subsampling=sampling,optimize=True,**options)
   with Image.open(target) as check:
    if check.size!=size:raise ValueError(f'Bad output dimensions: {target}')
    check.verify()
   shutil.copyfile(label,target_label)
   assert target_label.read_bytes()==label.read_bytes()
   ledger.append({'group':name,'source':str(png),'image':target.name,'bytes':target.stat().st_size})
   done+=1
   if done%20==0:update(stage=f'Convert {name}',generated=done,total=total,group=name,group_generated=index,group_total=len(source_train))
 cfg=yaml.safe_load((SOURCE/'data.yaml').read_text(encoding='utf-8'))
 cfg.update(path=DATA.as_posix(),train='images/train',val='images/val',test='images/test')
 (DATA/'data.yaml').write_text(yaml.safe_dump(cfg,allow_unicode=True),encoding='utf-8')
 for split in ['train','val','test']:
  ims=image_paths(DATA/'images'/split); labs=list((DATA/'labels'/split).glob('*.txt'))
  expected=counts[split]*(5 if split=='train' else 1)
  assert len(ims)==len(labs)==expected,(split,len(ims),len(labs),expected)
  assert {p.stem for p in ims}=={p.stem for p in labs}
 counts['train']*=5
 manifest={'counts':counts,'total_images':sum(counts.values()),'encoding':'Original JPEG quantization tables and chroma subsampling, optimize=True; non-JPEG originals use quality=95 and subsampling=2; original dimensions unchanged','source':str(SOURCE),'groups':list(GROUPS),'initial_model':str(INITIAL),'bytes':sum(p.stat().st_size for p in DATA.rglob('*') if p.is_file()),'labels_verified':True,'physical_files':True}
 (DATA/'augmentation_files.json').write_text(json.dumps(ledger,ensure_ascii=False),encoding='utf-8')
 manifest_path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
 return manifest
def main():
 torch.set_num_threads(4)
 threading.Thread(target=ThreadingHTTPServer(('127.0.0.1',8794),Handler).serve_forever,daemon=True).start()
 update(state='Preparing',stage='Copying and converting')
 manifest=build_dataset();update(dataset_GB=round(manifest['bytes']/1e9,2),counts=manifest['counts'])
 yaml_path=DATA/'data.yaml'; comparison=RUN/'comparison.json'
 reports=json.loads(comparison.read_text()) if comparison.exists() else {}
 if 'original_model' not in reports:
  update(state='Testing',stage='Original model evaluation')
  original=YOLO(str(INITIAL));m=original.val(data=str(yaml_path),split='test',imgsz=640,batch=16,device=0,workers=2,project=str(RUN),name='original_test',plots=True)
  reports['original_model']=metrics_report(m);comparison.write_text(json.dumps(reports,indent=2))
  del original,m;gc.collect();torch.cuda.empty_cache()
 if 'combined_model' not in reports:
  last=RUN/'weights/last.pt';model=YOLO(str(last if last.exists() else INITIAL))
  batches=[0];last_update=[0]
  def epoch_start(t):
   batches[0]=0;update(state='Training',stage='Combined dataset',epoch=t.epoch+1,epochs=t.epochs,batch=0,batches=len(t.train_loader))
  def batch_end(t):
   batches[0]+=1
   if time.monotonic()-last_update[0]>2:
    last_update[0]=time.monotonic();update(batch=batches[0],gpu_memory_gb=round(torch.cuda.memory_reserved()/1e9,2))
  model.add_callback('on_train_epoch_start',epoch_start);model.add_callback('on_train_batch_end',batch_end)
  model.add_callback('on_train_epoch_end',lambda t:update(state='Validating'))
  model.add_callback('on_fit_epoch_end',lambda t:update(metrics={k:float(v) for k,v in t.metrics.items()}))
  model.train(resume=last.exists(),data=str(yaml_path),epochs=8,patience=8,imgsz=640,batch=32,cache=False,device=0,workers=2,amp=True,optimizer='AdamW',lr0=0.0003,warmup_epochs=1,seed=42,
   hsv_h=0,hsv_s=0,hsv_v=0,fliplr=0,flipud=0,degrees=0,translate=0.1,scale=0.3,mosaic=0.5,close_mosaic=2,mixup=0,copy_paste=0,project=str(PROJECT/'runs'),name=RUN.name,exist_ok=True,plots=True)
  best=str(model.trainer.best);del model;gc.collect();torch.cuda.empty_cache()
  update(state='Testing',stage='Combined model evaluation',best=best)
  trained=YOLO(best);m=trained.val(data=str(yaml_path),split='test',imgsz=640,batch=16,device=0,workers=2,project=str(RUN),name='combined_test',plots=True)
  reports['combined_model']=metrics_report(m);comparison.write_text(json.dumps(reports,indent=2))
  photos=image_paths(PROJECT/'dataset/test')
  if photos:trained.predict(source=[str(p) for p in photos],imgsz=640,conf=0.25,device=0,save=True,project=str(RUN),name='photo_predictions')
  del trained,m;gc.collect();torch.cuda.empty_cache()
 update(state='Complete',stage='Training and comparison complete',comparison=str(comparison))
 print('COMBINED_EXPERIMENT_COMPLETE',flush=True)
 while True:time.sleep(5)
if __name__=='__main__':
 try:main()
 except Exception as e:
  update(state='Failed',error=str(e));raise
