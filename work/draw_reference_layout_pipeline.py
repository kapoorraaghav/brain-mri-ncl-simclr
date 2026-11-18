"""Render the requested three-panel design as editable SVG and vector PDF."""
from pathlib import Path
import json,hashlib,base64,io
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle,FancyArrowPatch

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'output/figures';OUT.mkdir(exist_ok=True,parents=True)
plt.rcParams.update({'font.family':'DejaVu Sans','pdf.fonttype':42,'svg.fonttype':'none'})
metadata_path=ROOT/'data/processed_flair_brats2023_expansion_v1/metadata.csv'
if metadata_path.is_file():
    meta=pd.read_csv(metadata_path)
    val=meta[meta.split=='val'];patient=sorted(val.patient_id.unique())[0]
    group=val[val.patient_id==patient].sort_values('slice_idx')
    record=group.iloc[len(group)//2]
    source=Path(record.file_path)
    image=np.asarray(np.load(source,allow_pickle=True).item()['image']).squeeze()
    image_hash=hashlib.sha256(source.read_bytes()).hexdigest()
    image_origin='original stored illustrative slice'
else:
    # Local datasets were removed after archiving. Reuse the same illustrative
    # tile already embedded in this vector asset, without fetching patient data.
    svg=OUT/'pipeline_reference_design.svg'
    tree=ET.fromstring(svg.read_bytes())
    tile=tree.find('.//{http://www.w3.org/2000/svg}image')
    uri=tile.attrib['{http://www.w3.org/1999/xlink}href']
    assert uri.startswith('data:image/png;base64,')
    raw=base64.b64decode(uri.split(',',1)[1])
    image=plt.imread(io.BytesIO(raw),format='png')
    image_hash=hashlib.sha256(raw).hexdigest()
    image_origin='same illustrative tile recovered from previously generated SVG'
fig,axes=plt.subplots(3,1,figsize=(10.6,7.0))
fig.subplots_adjust(left=.018,right=.992,top=.99,bottom=.065,hspace=.10)
labels=[];boxes=[]
red='#A72B30';gray='#777777';blue='#DCEAF6';green='#E5F0DA';ink='#27323B'

def box(ax,x,y,w,h,text,fill='white',edge='#5B636B',fontsize=10.4):
    patch=Rectangle((x,y-h/2),w,h,facecolor=fill,edgecolor=edge,lw=.85)
    ax.add_patch(patch)
    label=ax.text(x+w/2,y,text,ha='center',va='center',fontsize=fontsize,
        linespacing=1.10,color=ink)
    labels.append((patch,label));boxes.append((ax,x,y-h/2,w,h))

def arrow(ax,points,color=gray):
    if len(points)>2:ax.plot(*zip(*points[:-1]),color=color,lw=1)
    ax.add_patch(FancyArrowPatch(points[-2],points[-1],arrowstyle='-|>',
        mutation_scale=9,lw=1,color=color,shrinkA=0,shrinkB=0))

for i,(ax,method) in enumerate(zip(axes,['SUPERVISED','SIMCLR','NCL'])):
    ax.set(xlim=(0,16.5),ylim=(0,3.75));ax.axis('off')
    ax.text(.15,3.48,method,ha='left',va='center',fontsize=11,fontweight='bold',
        color=ink,bbox={'facecolor':'#DCEAF6','edgecolor':'none','pad':3})
    ax.imshow(image,cmap='gray',vmin=0,vmax=1,extent=(.15,1.35,1.55,2.75),aspect='auto')
    ax.text(.75,1.30,'MRI FLAIR\nslice',ha='center',va='top',fontsize=9.3,color=ink)
    top,low=2.72,1.15
    if i==0:
        box(ax,2.85,2.0,2.05,.88,'ViT Tiny\nbackbone: 192',blue)
        arrow(ax,[(1.35,2.0),(2.85,2.0)])
        arrow(ax,[(4.90,2.0),(5.25,2.0),(5.25,top),(5.8,top)])
        arrow(ax,[(4.90,2.0),(5.25,2.0),(5.25,low),(5.8,low)],red)
        head='Classification head\nCross entropy'
    else:
        box(ax,1.95,top,1.15,.68,'View 1','#F0F1F2',fontsize=9.5)
        box(ax,1.95,low,1.15,.68,'View 2','#F0F1F2',fontsize=9.5)
        arrow(ax,[(1.35,2.15),(1.65,2.15),(1.65,top),(1.95,top)])
        arrow(ax,[(1.35,2.15),(1.65,2.15),(1.65,low),(1.95,low)])
        for yy in [top,low]:
            box(ax,3.55,yy,1.95,.88,'ViT Tiny\nbackbone: 192',blue,fontsize=9.7)
            arrow(ax,[(3.10,yy),(3.55,yy)])
        ax.plot([4.525,4.525],[low+.44,top-.44],color='#8C969F',lw=.8)
        ax.text(4.525,(top+low)/2,'shared\nweights',ha='center',va='center',
            fontsize=8.8,color='#57616A',bbox={'facecolor':'white','edgecolor':'none','pad':1})
        arrow(ax,[(5.50,top),(5.8,top)])
        arrow(ax,[(5.50,low),(5.8,low)],red)
        head='Projection head: 128\n'+('Unconstrained' if i==1 else 'Terminal ReLU')
    box(ax,5.8,top,2.85,.88,head,green if i==2 else '#ECEEEF',
        edge='#537A43' if i==2 else '#5B636B')
    ax.text(8.94,top,'Discarded after training',ha='left',va='center',fontsize=9,color='#59626A')
    box(ax,5.8,low,2.25,.88,'Frozen CLS\n192 dimensions',blue)
    box(ax,8.53,low,1.75,.88,'TopK\nSAE','#ECEEEF')
    box(ax,10.77,low,2.15,.88,'Top 10 activating\nMRI slices',green,fontsize=9.5)
    box(ax,13.40,low,2.60,.88,'Label selectivity\nPurity / entropy',green,fontsize=9.8)
    for a,b in [(8.05,8.53),(10.28,10.77),(12.92,13.40)]:arrow(ax,[(a,low),(b,low)],red)
    if i<2:ax.plot([.10,16.10],[.08,.08],color='#D8DEE3',lw=.65)

fig.text(.02,.020,'Training path',fontsize=9.2,color=gray)
fig.text(.23,.020,'Frozen backbone to SAE analysis',fontsize=9.2,color=red)
fig.text(.992,.020,'Only the NCL projector output is constrained nonnegative.',
    ha='right',fontsize=9,color='#59626A')
fig.canvas.draw();renderer=fig.canvas.get_renderer()
for patch,label in labels:
    for _ in range(10):
        b=patch.get_window_extent(renderer);t=label.get_window_extent(renderer)
        if t.width<=b.width-6 and t.height<=b.height-6:break
        label.set_fontsize(label.get_fontsize()-.3);fig.canvas.draw()
    assert label.get_fontsize()>=8.3,label.get_text()
    assert t.width<=b.width-6 and t.height<=b.height-6,label.get_text()
for index,(ax,x,y,w,h) in enumerate(boxes):
    for aa,xx,yy,ww,hh in boxes[index+1:]:
        if aa is ax:assert not(min(x+w,xx+ww)>max(x,xx) and min(y+h,yy+hh)>max(y,yy))
base=OUT/'pipeline_reference_design'
for ext in ['svg','pdf','png']:
    kwargs={'dpi':240} if ext=='png' else {}
    fig.savefig(base.with_suffix('.'+ext),bbox_inches='tight',pad_inches=.04,**kwargs)
plt.close(fig)
(OUT/'pipeline_reference_design_checks.json').write_text(json.dumps({
    'text_boxes_checked':len(labels),'box_overlap_check':'passed','text_fit_check':'passed',
    'layout':'user supplied three-panel, paired-view design',
    'MRI_example':'illustrative mid retained slice from first validation patient; reused across methods',
    'image_source_sha256':image_hash,'image_source':image_origin,
    'output_labels':'label selectivity, not monosemanticity',
},indent=2),encoding='utf-8')
print('Saved:',base)
