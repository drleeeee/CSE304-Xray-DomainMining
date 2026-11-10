from pathlib import Path
import numpy as np,pandas as pd
import cv2

def extract_deep(index,out,models=('resnet','yolo','dinov2'),batch_size=12):
    """Frozen multi-scale pretrained representations. Not validated defect detectors."""
    import torch
    from torchvision import models as tv
    from torchvision.transforms import Normalize
    from tqdm.auto import tqdm
    from .features import read_image,masks_and_gray
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    n=len(index)
    # Dense backbone on grayscale replicated across RGB, intentionally frozen.
    resnet=None;yolo=None;dino=None
    if 'resnet' in models:
        from torchvision.models import resnet18,ResNet18_Weights
        resnet=resnet18(weights=ResNet18_Weights.DEFAULT);resnet.fc=torch.nn.Identity();resnet.to(device).eval()
    if 'yolo' in models:
        from ultralytics import YOLO
        yolo=YOLO('yolo11n.pt').model.to(device).eval()
        # Feature tap chosen from actual sequential YOLO11n backbone stage.
        yolo_layer=yolo.model[9]
        captured=[]
        def collect(module,inputs,output):
            if isinstance(output,torch.Tensor):captured.append(output.detach())
            elif isinstance(output,(tuple,list)):
                tensor=next((x for x in output if isinstance(x,torch.Tensor)),None)
                if tensor is not None:captured.append(tensor.detach())
        hook=yolo_layer.register_forward_hook(collect)
    if 'dinov2' in models:
        from transformers import AutoModel
        dino=AutoModel.from_pretrained('facebook/dinov2-small').to(device).eval()
    names=[];vectors={k:[] for k in models}; failures=[]
    mean=torch.tensor([.485,.456,.406],device=device)[None,:,None,None]
    std=torch.tensor([.229,.224,.225],device=device)[None,:,None,None]
    for start in tqdm(range(0,n,batch_size),desc='Frozen embeddings'):
        entries=index.iloc[start:start+batch_size]
        arr=[];valid=[]
        for _,r in entries.iterrows():
            try:
                im=read_image(r.local_path);gray,_=masks_and_gray(im)
                color=cv2.cvtColor(cv2.resize(gray,(224,224)),cv2.COLOR_GRAY2RGB)
                arr.append(color);valid.append(r.key)
            except Exception as e:failures.append(dict(key=r.key,error=repr(e)))
        if not arr:continue
        t=torch.from_numpy(np.stack(arr)).permute(0,3,1,2).float().to(device)/255.
        with torch.inference_mode():
            if resnet is not None:
                x=resnet((t-mean)/std).cpu().numpy()
                vectors['resnet'].extend(x)
            if yolo is not None:
                captured.clear()
                # Input size is divisible by 32, avoid unnecessary resize.
                ty=torch.nn.functional.interpolate(t,size=(320,320),mode='bilinear',align_corners=False)
                _=yolo(ty)
                if not captured:raise RuntimeError('YOLO backbone hook did not capture feature map')
                fmap=captured[-1];vec=fmap.mean(dim=(-1,-2)).cpu().numpy()
                vectors['yolo'].extend(vec)
            if dino is not None:
                # DINOv2 HF expects ImageNet normalized pixel values.
                result=dino(pixel_values=(t-mean)/std)
                vec=result.last_hidden_state[:,0].cpu().numpy()
                vectors['dinov2'].extend(vec)
        names.extend(valid)
    if yolo is not None:hook.remove()
    pd.DataFrame(failures).to_csv(out/'deep_errors.csv',index=False)
    dfs=[]
    for name,values in vectors.items():
        a=np.stack(values)
        df=pd.DataFrame(a,columns=[f'{name}_{i:03d}' for i in range(a.shape[1])])
        df.insert(0,'key',names);df.to_parquet(out/f'{name}_embeddings.parquet',index=False)
        dfs.append(df.set_index('key'))
    return pd.concat(dfs,axis=1).reset_index() if dfs else pd.DataFrame({'key':names})
