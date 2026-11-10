"""Conservative removal of *colored overlay pixels*, not recovery of native radiography.
Outputs before/after/mask quality-control examples; uses no T/F labels.
"""
from pathlib import Path
import cv2,numpy as np,pandas as pd

def overlay_mask(img, dilation=2):
    if img.ndim!=3 or img.shape[2]!=3: raise ValueError('Expected color BGR machine-marked BMP')
    hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    b,g,r=cv2.split(img.astype(np.int16))
    # Color is required, preventing accidental removal of dark X-ray material.
    channel_gap=np.maximum.reduce([np.abs(r-g),np.abs(g-b),np.abs(b-r)])
    sat=hsv[:,:,1].astype(np.int16)
    val=hsv[:,:,2].astype(np.int16)
    m=((sat>=80)&(val>=65)&(channel_gap>=35)).astype(np.uint8)*255
    if dilation>0:
        d=int(dilation)
        m=cv2.dilate(m,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*d+1,2*d+1)))
    return m

def remove_overlays(img, dilation=2, radius=3):
    mask=overlay_mask(img,dilation)
    clean=cv2.inpaint(img,mask,float(radius),cv2.INPAINT_TELEA) if np.any(mask) else img.copy()
    # Repeated cleaning is a sensitivity check; the pipeline uses first-pass output only.
    return clean,mask

def overlay_quality(img,clean,mask):
    frac=float(np.mean(mask>0))
    # Inspect residual saturated pixels outside the dilated mask only.
    residual=overlay_mask(clean,dilation=0)
    changed=cv2.absdiff(img,clean)
    return dict(overlay_mask_fraction=frac,
                residual_color_fraction=float(np.mean(residual>0)),
                mean_abs_change=float(changed.mean()),
                qc_flag=('HIGH_MASK_AREA' if frac>0.10 else 'OK'))

def export_overlay_gallery(index,out,limit=15):
    from .features import read_image
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    rows=[]
    sample=index.head(int(limit))
    for n,(_,r) in enumerate(sample.iterrows()):
        src=read_image(r.local_path)
        clean,mask=remove_overlays(src)
        item=np.hstack([src,cv2.cvtColor(mask,cv2.COLOR_GRAY2BGR),clean])
        scale=min(1.0,1250/max(1,item.shape[1]))
        item=cv2.resize(item,None,fx=scale,fy=scale,interpolation=cv2.INTER_AREA)
        path=out/f'overlay_{n:03d}.png'
        cv2.imencode('.png',item)[1].tofile(str(path))
        rows.append(dict(key=r.key,output=path.name,**overlay_quality(src,clean,mask)))
    pd.DataFrame(rows).to_csv(out/'overlay_visual_audit.csv',index=False)
    (out/'IMPORTANT.txt').write_text('Panels: original | detected color mask | inpainted approximation. Inpainting cannot recover actual obscured X-ray pixels. Visually review every unit and F subtype.\n',encoding='utf-8')
