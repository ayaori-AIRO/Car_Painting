import json, time, threading, shutil
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import torch, yaml
from train_and_test import YOLO

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'dataset/carparts-seg'
RUN=ROOT/'runs/carparts_fender_refresh_20261009'
STATUS=RUN/'live_status.json'
started=time.time()
state={'state':'Preparing','elapsed_seconds':0,'run':str(RUN)}
lock=threading.Lock()
def update(**kw):
    with lock:
        state.update(kw);state['elapsed_seconds']=round(time.time()-started)
        STATUS.write_text(json.dumps(state),encoding='utf-8')
class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path=='/status':
            with lock: data=json.dumps({**state,'elapsed_seconds':round(time.time()-started)}).encode()
            kind='application/json'
        else:
            data=Path(__file__).with_name('fender_training_live.html').read_bytes();kind='text/html; charset=utf-8'
        self.send_response(200);self.send_header('Content-Type',kind);self.send_header('Cache-Control','no-store');self.end_headers();self.wfile.write(data)
    def log_message(self,*args):pass
def main():
    RUN.mkdir(parents=True,exist_ok=False)
    threading.Thread(target=ThreadingHTTPServer(('127.0.0.1',8790),Handler).serve_forever,daemon=True).start()
    torch.set_num_threads(4)
    assert torch.cuda.is_available()
    update(state='Preparing',gpu=torch.cuda.get_device_name(0))
    cfg=yaml.safe_load((DATA/'data.yaml').read_text());cfg['path']=DATA.as_posix()
    prepared=RUN/'data.yaml';prepared.write_text(yaml.safe_dump(cfg),encoding='utf-8')
    model=YOLO(str(ROOT/'runs/carparts_yolo11n_seg/weights/best.pt'))
    last=[0]
    counter=[0]
    def start_epoch(t):counter[0]=0
    def batch(t):
        counter[0]+=1
        if time.time()-last[0]<2:return
        last[0]=time.time()
        update(state='Training',epoch=t.epoch+1,epochs=t.epochs,batch=counter[0],batches=len(t.train_loader),gpu_memory_gb=round(torch.cuda.memory_reserved()/1e9,2))
    def epoch(t):
        update(state='Validating',epoch=t.epoch+1,epochs=t.epochs)
    def done_epoch(t):
        update(metrics={k:float(v) for k,v in t.metrics.items()},completed_epoch=t.epoch+1)
    model.add_callback('on_train_batch_end',batch)
    model.add_callback('on_train_epoch_start',start_epoch)
    model.add_callback('on_train_epoch_end',epoch)
    model.add_callback('on_fit_epoch_end',done_epoch)
    try:
        model.train(data=str(prepared),epochs=20,time=0.75,patience=8,imgsz=640,batch=32,cache='ram',device=0,workers=0,amp=True,optimizer='AdamW',lr0=0.0003,warmup_epochs=1,close_mosaic=3,seed=42,project=str(ROOT/'runs'),name=RUN.name,exist_ok=True,plots=True)
        best=Path(model.trainer.best)
        update(state='Testing',best=str(best))
        trained=YOLO(str(best))
        metrics=trained.val(data=str(prepared),split='test',device=0,batch=16,workers=0,project=str(RUN),name='test_evaluation',plots=True)
        (RUN/'test_metrics.json').write_text(json.dumps(metrics.results_dict,indent=2),encoding='utf-8')
        selected=[]
        for lab in sorted((DATA/'labels/test').glob('*.txt')):
            if any(l.split() and l.split()[0]=='23' for l in lab.read_text().splitlines()):
                selected.append(str(DATA/'images/test'/(lab.stem+'.jpg')))
            if len(selected)==24:break
        trained.predict(source=selected,imgsz=640,device=0,save=True,project=str(RUN),name='test_predictions')
        update(state='Complete',best=str(best),test_metrics=metrics.results_dict)
        print('TRAINING_AND_TEST_COMPLETE',best,flush=True)
    except Exception as e:
        update(state='Failed',error=str(e));raise
    while True:time.sleep(10)
if __name__=='__main__':main()
