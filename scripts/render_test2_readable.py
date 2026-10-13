from pathlib import Path
import torch,json
from ultralytics import YOLO
from ultralytics.utils.plotting import colors
from PIL import Image,ImageDraw,ImageFont

torch.set_num_threads(4)
model=YOLO(r'C:\Projects\Car_Painting\runs\carparts_12classes_20261012\weights\best.pt')
source=Path(r'C:\Projects\Car_Painting\dataset\test2')
out=Path(__file__).parent/'test2_12classes_20261013/readable';out.mkdir(exist_ok=True)
names={'back_bumper':'뒤 범퍼','back_door':'뒷문','glass':'유리','back_light':'뒤 라이트','front_bumper':'앞 범퍼','front_door':'앞문','front_light':'앞 라이트','hood':'보닛','side_mirror':'사이드미러','trunk':'트렁크','wheel':'바퀴','fender':'앞휀더'}
font=ImageFont.truetype(r'C:\Windows\Fonts\malgunbd.ttf',30)
rows=[]
for index,p in enumerate(sorted(p for p in source.rglob('*') if p.suffix.lower() in {'.jpg','.jpeg','.png','.bmp','.webp'}),1):
 result=model.predict(str(p),imgsz=640,conf=0.25,device=0,retina_masks=True,verbose=False)[0]
 base=result.plot(labels=False,conf=False,line_width=4)
 image=Image.fromarray(base[:,:,::-1]);w,h=image.size;image.thumbnail((1600,1600));sx=image.width/w;sy=image.height/h
 draw=ImageDraw.Draw(image);occupied=[]
 for box in result.boxes:
  cls=int(box.cls.item());confidence=float(box.conf.item())
  text=f'{result.names[cls]} {confidence*100:.1f}%'
  bounds=draw.textbbox((0,0),text,font=font);tw=bounds[2]-bounds[0]+18;th=48
  coords=box.xyxy[0].tolist();x=max(0,min(int(coords[0]*sx),image.width-tw));y=max(0,int(coords[1]*sy)-th)
  # Shift label positions when boxes share a similar top edge.
  while any(x<ox+ow and x+tw>ox and y<oy+oh and y+th>oy for ox,oy,ow,oh in occupied) and y+2*th<image.height:y+=th
  occupied.append((x,y,tw,th));color=colors(cls)
  draw.rectangle((x,y,x+tw,y+th),fill=color)
  draw.text((x+9,y+3),text,font=font,fill='white',stroke_width=1,stroke_fill='black')
 target=out/f'result_{index:02d}.jpg';image.save(target,quality=95)
 rows.append({'result':str(target),'source':str(p)})
(out/'sources.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
print(f'Saved {len(rows)} readable images')
