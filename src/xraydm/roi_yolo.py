"""Optional multi-scale frozen YOLO feature maps pooled at annotated ROIs.

Boxes represent supporting annotations, not exhaustive foreign-body truth.
The exported ROI vectors are descriptors, never an inferred Ishida-covered flag.
"""
from pathlib import Path
import numpy as np,pandas as pd,cv2
from tqdm.auto import tqdm


def extract_yolo_roi(index,out,side=640,layers=(4,6,9)):
    import torch
    from ultralytics import YOLO
    from .features import read_image,masks_and_gray
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    dest=out/'yolo_roi_multiscale.parquet'
    if dest.exists():return pd.read_parquet(dest)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    net=YOLO('yolo11n.pt').model.to(device).eval()
    taps={};hooks=[]
    for layer in layers:
        def make_hook(k):
            def capture(module,inputs,output):
                if isinstance(output,torch.Tensor):taps[k]=output
            return capture
        hooks.append(net.model[layer].register_forward_hook(make_hook(layer)))
    rows=[];errors=[]
    try:
        for _,row in tqdm(index.iterrows(),total=len(index),desc='YOLO candidate ROIs'):
            boxes=row.boxes if isinstance(row.boxes,list) else []
            if not boxes:continue
            try:
                im=read_image(row.local_path);gray,_=masks_and_gray(im)
                h,w=gray.shape
                rgb=cv2.cvtColor(cv2.resize(gray,(side,side)),cv2.COLOR_GRAY2RGB)
                t=torch.from_numpy(rgb).permute(2,0,1).unsqueeze(0).to(device).float()/255
                taps.clear()
                with torch.inference_mode():
                    _=net(t)
                    for j,b in enumerate(boxes):
                        x,y,bw,bh=[float(b[k]) for k in ('x','y','w','h')]
                        if not np.isfinite([x,y,bw,bh]).all():continue
                        feat=dict(key=row.key,candidate_index=j)
                        for k in layers:
                            if k not in taps:raise RuntimeError(f'Missing YOLO hook {k}')
                            f=taps[k];hh,ww=f.shape[-2:]
                            x1=int(np.floor((x-bw/2)*ww));x2=int(np.ceil((x+bw/2)*ww))
                            y1=int(np.floor((y-bh/2)*hh));y2=int(np.ceil((y+bh/2)*hh))
                            x1=max(0,min(x1,ww-1));x2=max(x1+1,min(x2,ww))
                            y1=max(0,min(y1,hh-1));y2=max(y1+1,min(y2,hh))
                            # Feature-map crop approximates ROIAlign; no external pixel-exact ROIAlign claim.
                            v=f[0,:,y1:y2,x1:x2].mean((-1,-2)).cpu().numpy()
                            feat.update({f'roi_yolo_l{k}_{ix:03d}':float(value) for ix,value in enumerate(v)})
                        rows.append(feat)
            except Exception as e:errors.append(dict(key=row.key,error=repr(e)))
    finally:
        for hook in hooks:hook.remove()
    pd.DataFrame(errors).to_csv(out/'roi_yolo_errors.csv',index=False)
    result=pd.DataFrame(rows)
    if not result.empty:result.to_parquet(dest,index=False)
    return result
