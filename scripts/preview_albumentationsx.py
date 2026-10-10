import os, sys
from pathlib import Path
os.environ['ALBUMENTATIONS_NO_TELEMETRY']='1'
os.environ['NO_ALBUMENTATIONS_UPDATE']='1'
sys.path.insert(0, str(Path(__file__).parent/'augmentation_packages'))
import albumentations as A
import cv2, numpy as np, json, random
from PIL import Image, ImageDraw

BASE=(Path(__file__).resolve().parents[1]/'dataset/carparts-seg')
OUT=Path(__file__).resolve().parents[1]/'runs/albumentationsx_previews'
OUT.mkdir(exist_ok=True)
# Each variant is separate so its effects can be inspected individually.
variants={
 'brightness_contrast': A.RandomBrightnessContrast(brightness_limit=(-0.18,0.18),contrast_limit=(-0.18,0.18),p=1),
 'gamma': A.RandomGamma(gamma_limit=(80,120),p=1),
 'blur': A.GaussianBlur(blur_limit=(3,3),sigma_limit=(0.3,0.8),p=1),
 'jpeg': A.ImageCompression(quality_range=(70,90),p=1),
 'low_resolution': A.Downscale(scale_range=(0.7,0.85),p=1),
}
sources=sorted((BASE/'images/train').glob('*.jpg'))
rng=random.Random(42)
sources=rng.sample(sources,8)
sources+=sorted((Path(__file__).resolve().parents[1]/'dataset/test').glob('*.png'))
report=[]
def read(p):
 return cv2.cvtColor(cv2.imdecode(np.fromfile(p,dtype=np.uint8),cv2.IMREAD_COLOR),cv2.COLOR_BGR2RGB)
def overlay(img,label):
 if not label.exists(): return img
 painted=img.copy(); h,w=img.shape[:2]
 for line in label.read_text().splitlines():
  values=line.split()
  if len(values)<7: continue
  cls=int(values[0]); pts=np.array(list(map(float,values[1:])),dtype=np.float32).reshape(-1,2)
  pts=(pts*np.array([w,h])).round().astype(np.int32)
  color=((cls*71+30)%255,(cls*113+80)%255,(cls*43+160)%255)
  cv2.fillPoly(painted,[pts],color)
 img=cv2.addWeighted(img,0.65,painted,0.35,0)
 return img
for index,path in enumerate(sources,1):
 image=read(path); label=BASE/'labels/train'/f'{path.stem}.txt'
 folder=OUT/f'sample_{index:02d}'; folder.mkdir(exist_ok=True)
 samples={'original':image}; replay={}
 for name,transform in variants.items():
  pipeline=A.ReplayCompose([transform]); pipeline.set_random_seed(42+index)
  result=pipeline(image=image)
  assert result['image'].shape==image.shape
  samples[name]=result['image']; replay[name]=result['replay']
 sheet=Image.new('RGB',(960,600),'white'); draw=ImageDraw.Draw(sheet)
 for k,(name,img) in enumerate(samples.items()):
  Image.fromarray(img).save(folder/f'{name}.png')
  if label.exists(): (folder/f'{name}.txt').write_bytes(label.read_bytes())
  panel=Image.fromarray(overlay(img,label)); panel.thumbnail((310,265))
  x=(k%3)*320; y=(k//3)*300
  sheet.paste(panel,(x+(320-panel.width)//2,y+30))
  draw.text((x+8,y+8),name,fill='black')
 sheet.save(folder/'comparison.jpg')
 (folder/'replay.json').write_text(json.dumps(replay,default=lambda v:v.tolist() if hasattr(v,'tolist') else str(v),indent=2))
 report.append({'sample':index,'source':str(path),'has_ground_truth':label.exists(),'shape_preserved':True,'labels_byte_identical':all((folder/f'{name}.txt').read_bytes()==label.read_bytes() for name in samples) if label.exists() else None})
(OUT/'report.json').write_text(json.dumps({'albumentations_version':A.__version__,'seed':42,'samples':report},ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'version':A.__version__,'samples':len(report),'output':str(OUT)}))
