import os,sys,json,time,threading,shutil,gc
from pathlib import Path
os.environ['ALBUMENTATIONS_NO_TELEMETRY']='1'
os.environ['NO_ALBUMENTATIONS_UPDATE']='1'
sys.path.insert(0,str(Path(__file__).parent/'augmentation_packages'))
import numpy as np,torch,yaml
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from train_and_test import YOLO
from run_combined_dataset import metrics_report

PROJECT=Path(__file__).resolve().parents[1]
DATA=PROJECT/'dataset/carparts-seg-augmented-all'
RUN=PROJECT/'runs/carparts_12classes_20261012'
BACKUP=PROJECT/'dataset/class_backups/carparts-seg-augmented-all_24classes_20261012'
INITIAL=PROJECT/'runs/carparts_combined_augmentation_20261011/weights/best.pt'
NAMES=['back_bumper','back_door','glass','back_light','front_bumper','front_door','front_light','hood','side_mirror','trunk','wheel','fender']
MAPPING={0:0,1:1,2:2,3:1,4:3,5:3,6:1,7:3,8:4,9:5,10:2,11:5,12:6,13:6,14:5,15:6,16:7,17:8,18:None,19:8,20:None,21:9,22:10,23:11}
RUN.mkdir(parents=True,exist_ok=True);lock=threading.Lock();started=time.monotonic()
state={'state':'Preparing','stage':'Class conversion','epochs':8,'dataset':str(DATA),'run':str(RUN),'classes':NAMES}
def update(**kwargs):
 with lock:
  state.update(kwargs);state['elapsed_seconds']=round(time.monotonic()-started)
  (RUN/'status.json').write_text(json.dumps(state,ensure_ascii=False,indent=2),encoding='utf-8')
class Handler(BaseHTTPRequestHandler):
 def do_GET(self):
  if self.path=='/status':
   with lock:payload=json.dumps(state,ensure_ascii=False).encode('utf-8')
   kind='application/json; charset=utf-8'
  else:
   payload=Path(__file__).with_name('classes12_training_live.html').read_bytes();kind='text/html; charset=utf-8'
  self.send_response(200);self.send_header('Content-Type',kind);self.send_header('Cache-Control','no-store');self.end_headers();self.wfile.write(payload)
 def log_message(self,*args):pass
def prepare():
 cfg=yaml.safe_load((DATA/'data.yaml').read_text(encoding='utf-8'))
 if list(cfg['names'].values())==NAMES:
  if not (RUN/'class_conversion.json').exists():raise RuntimeError('12-class config exists without completed conversion report')
  return
 if len(cfg['names'])!=24:raise RuntimeError('Unexpected input classes')
 BACKUP.mkdir(parents=True,exist_ok=True)
 if not (BACKUP/'backup_complete.json').exists():
  shutil.copyfile(DATA/'data.yaml',BACKUP/'data.yaml')
  files=sorted((DATA/'labels').rglob('*.txt'))
  for i,p in enumerate(files,1):
   dst=BACKUP/'labels'/p.relative_to(DATA/'labels');dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dst)
   if dst.read_bytes()!=p.read_bytes():raise RuntimeError('Backup mismatch')
   if i%100==0:update(stage='Backup 24-class labels',generated=i,total=len(files))
  (BACKUP/'backup_complete.json').write_text(json.dumps({'labels':len(files)}))
 removed={18:0,20:0};counts={};total=0
 for split in ['train','val','test']:
  files=sorted((BACKUP/'labels'/split).glob('*.txt'));instances=[0]*12;empty=0
  for i,p in enumerate(files,1):
   lines=[]
   for line in p.read_text().splitlines():
    parts=line.split()
    if not parts:continue
    old=int(parts[0]);new=MAPPING[old]
    if new is None:removed[old]+=1;continue
    coords=np.array(list(map(float,parts[1:])))
    if len(coords)<6 or len(coords)%2 or not np.isfinite(coords).all():raise ValueError(str(p))
    # Retain coordinate strings exactly; only the class ID changes.
    lines.append(str(new)+' '+' '.join(parts[1:]));instances[new]+=1
   target=DATA/'labels'/split/p.name;temporary=target.with_suffix('.tmp')
   temporary.write_text('\n'.join(lines)+('\n' if lines else ''),encoding='utf-8');temporary.replace(target)
   if not lines:empty+=1
   total+=1
   if i%100==0:update(stage=f'Remap {split}',generated=i,total=len(files))
  images=[p for p in (DATA/'images'/split).iterdir() if p.suffix.lower() in {'.jpg','.jpeg','.png','.webp'}]
  assert len(images)==len(files)
  assert {p.stem for p in images}=={p.stem for p in files}
  counts[split]={'images':len(images),'empty_labels':empty,'instances':dict(zip(NAMES,instances))}
 for cache in (DATA/'labels').glob('*.cache'):cache.unlink()
 cfg['names']=dict(enumerate(NAMES));cfg['path']=DATA.as_posix()
 report={'mapping':MAPPING,'classes':NAMES,'backup':str(BACKUP),'splits':counts,'removed_annotations':removed,'images_deleted':0,'polygons_preserved':True,'initial_model':str(INITIAL)}
 (RUN/'class_conversion.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
 (DATA/'data.yaml').write_text(yaml.safe_dump(cfg,allow_unicode=True),encoding='utf-8')
 manifest=DATA/'manifest.json';m=json.loads(manifest.read_text());m['classes']=NAMES;m['class_backup']=str(BACKUP);manifest.write_text(json.dumps(m,ensure_ascii=False,indent=2),encoding='utf-8')
def main():
 torch.set_num_threads(4)
 threading.Thread(target=ThreadingHTTPServer(('127.0.0.1',8795),Handler).serve_forever,daemon=True).start()
 update();prepare()
 last=RUN/'weights/last.pt';model=YOLO(str(last if last.exists() else INITIAL));counter=[0];tick=[0]
 def epoch_start(t):
  counter[0]=0;update(state='Training',stage='12 classes',epoch=t.epoch+1,epochs=t.epochs,batch=0,batches=len(t.train_loader))
 def batch(t):
  counter[0]+=1
  if time.monotonic()-tick[0]>2:
   tick[0]=time.monotonic();update(batch=counter[0],gpu_memory_gb=round(torch.cuda.memory_reserved()/1e9,2))
 model.add_callback('on_train_epoch_start',epoch_start);model.add_callback('on_train_batch_end',batch)
 model.add_callback('on_train_epoch_end',lambda t:update(state='Validating'))
 model.add_callback('on_fit_epoch_end',lambda t:update(metrics={k:float(v) for k,v in t.metrics.items()}))
 model.train(resume=last.exists(),data=str(DATA/'data.yaml'),epochs=8,patience=8,imgsz=640,batch=32,cache=False,device=0,workers=2,amp=True,optimizer='AdamW',lr0=0.0003,warmup_epochs=1,seed=42,
  hsv_h=0,hsv_s=0,hsv_v=0,fliplr=0,flipud=0,degrees=0,translate=0.1,scale=0.3,mosaic=0.5,close_mosaic=2,mixup=0,copy_paste=0,project=str(PROJECT/'runs'),name=RUN.name,exist_ok=True,plots=True)
 best=str(model.trainer.best);del model;gc.collect();torch.cuda.empty_cache()
 update(state='Testing',stage='12-class test evaluation',best=best)
 trained=YOLO(best);metrics=trained.val(data=str(DATA/'data.yaml'),split='test',device=0,batch=16,workers=2,project=str(RUN),name='test_evaluation',plots=True)
 (RUN/'test_metrics.json').write_text(json.dumps(metrics_report(metrics),indent=2),encoding='utf-8')
 photos=sorted(str(p) for p in (PROJECT/'dataset/test').iterdir() if p.suffix.lower() in {'.jpg','.jpeg','.png','.webp'})
 if photos:trained.predict(source=photos,imgsz=640,device=0,conf=0.25,save=True,project=str(RUN),name='photo_predictions')
 del trained,metrics;gc.collect();torch.cuda.empty_cache()
 update(state='Complete',stage='Training and test complete')
 while True:time.sleep(5)
if __name__=='__main__':
 try:main()
 except Exception as e:update(state='Failed',error=str(e));raise
