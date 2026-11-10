import math
from pathlib import Path
import cv2,numpy as np,pandas as pd
from scipy.ndimage import gaussian_laplace
from skimage.feature import graycomatrix,graycoprops,local_binary_pattern
from skimage.measure import label,regionprops
from tqdm.auto import tqdm
from .domain import image_domain,candidate_domain
from .tableio import write_table,read_table,table_exists

def read_image(path):
    # Supports Unicode and mixed Korean filesystem paths.
    a=np.fromfile(str(path),dtype=np.uint8)
    img=cv2.imdecode(a,cv2.IMREAD_COLOR)
    if img is None:raise ValueError(f'Image cannot be decoded: {path}')
    return img

def masks_and_gray(im):
    from .overlay import remove_overlays
    approximate,mask=remove_overlays(im)
    return cv2.cvtColor(approximate,cv2.COLOR_BGR2GRAY),mask

def calc_global(gray,mask):
    g=gray.astype(np.float32);h,w=gray.shape
    lap=cv2.Laplacian(g,cv2.CV_32F)
    gl=cv2.resize(gray,(128,128),interpolation=cv2.INTER_AREA)//16
    matrix=graycomatrix(gl,[1,3],[0,np.pi/4,np.pi/2],levels=16,symmetric=True,normed=True)
    lbp=local_binary_pattern(gray,8,1,method='uniform')
    blackhat=cv2.morphologyEx(gray,cv2.MORPH_BLACKHAT,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(15,15)))
    rp=np.median(g,axis=1); cp=np.median(g,axis=0)
    d=dict(img_h=h,img_w=w,mean=float(g.mean()),std=float(g.std()),p05=float(np.percentile(g,5)),p95=float(np.percentile(g,95)),
        lap_var=float(lap.var()),high_freq=float(np.mean(lap**2)),overlay_fraction=float((mask>0).mean()),
        shortcut_overlay_component_count_proxy=float(max(0,cv2.connectedComponents((mask>0).astype('uint8'))[0]-1)),
        blackhat_mean=float(blackhat.mean()),blackhat_max=float(blackhat.max()),row_variation=float(rp.std()),
        col_variation=float(cp.std()),lbp_uniform_fraction=float((lbp<=8).mean()),
        edge_density=float((cv2.Canny(gray,50,120)>0).mean()))
    for prop in ['contrast','homogeneity','energy','correlation']:
        d['glcm_'+prop]=float(graycoprops(matrix,prop).mean())
    for s in [1.0,2.0,4.0]:
        r=np.abs(gaussian_laplace(g/255.0,sigma=s))
        d[f'log_max_s{s:g}']=float(r.max());d[f'log_mean_s{s:g}']=float(r.mean())
    # Dark-band analysis is a *pixel profile hypothesis*, not a physical band identity.
    profile=cv2.GaussianBlur(np.median(g,axis=1).astype(np.float32).reshape(-1,1),(1,9),0).ravel()
    band_threshold=float(np.percentile(profile,25))
    band_rows=profile<band_threshold
    d['band_row_fraction_proxy']=float(np.mean(band_rows))
    transitions=np.diff(np.r_[False,band_rows,False].astype(int))
    d['band_segments_proxy']=int(np.sum(transitions==1))
    # Blob candidate features without equating detected blobs with true foreign bodies.
    blob=cv2.morphologyEx(gray,cv2.MORPH_BLACKHAT,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(13,13)))
    threshold=np.percentile(blob,99.2)
    fg=(blob>threshold).astype('uint8')
    num,_,stats,_=cv2.connectedComponentsWithStats(fg,8)
    sizes=[int(stats[i,cv2.CC_STAT_AREA]) for i in range(1,num) if 3<=stats[i,cv2.CC_STAT_AREA]<=250]
    d['dark_blob_candidate_count_proxy']=len(sizes)
    d['dark_blob_mean_area_proxy']=float(np.mean(sizes)) if sizes else 0.0
    return d

def crop_roi(gray,box):
    h,w=gray.shape
    x,y,bw,bh=[float(box[k]) for k in ('x','y','w','h')]
    x1=max(0,int((x-bw/2)*w));x2=min(w,int((x+bw/2)*w))
    y1=max(0,int((y-bh/2)*h));y2=min(h,int((y+bh/2)*h))
    return gray[y1:y2,x1:x2],(x1,y1,x2,y2)

def candidate_props(gray,box):
    roi,rect=crop_roi(gray,box)
    if roi.size<9:return {}
    g=roi.astype(np.float32);dark=np.percentile(g,25)
    seg=(g<=dark).astype('uint8')
    objs=regionprops(label(seg)); best=max(objs,key=lambda x:x.area) if objs else None
    h,w=gray.shape;x1,y1,x2,y2=rect
    pad=max(5,int(max(x2-x1,y2-y1)))
    xa=max(0,x1-pad);xb=min(w,x2+pad);ya=max(0,y1-pad);yb=min(h,y2+pad)
    around=gray[ya:yb,xa:xb].astype(float)
    outer=np.ones(around.shape,dtype=bool)
    outer[y1-ya:y2-ya,x1-xa:x2-xa]=False
    b=around[outer]; mu=float(g.mean());bm=float(b.mean()) if b.size else float('nan')
    bs=float(b.std()) if b.size else float('nan')
    profile=cv2.GaussianBlur(np.median(gray.astype(np.float32),axis=1).reshape(-1,1),(1,9),0).ravel()
    is_dark=profile<float(np.percentile(profile,25))
    band_indices=np.where(is_dark)[0]
    band_distance_px=float(np.min(np.abs(band_indices-int(box['y']*h)))) if len(band_indices) else float('nan')
    return dict(cx=float(box['x']),cy=float(box['y']),band_distance_px_proxy=band_distance_px,
        band_inside_proxy=bool(is_dark[min(h-1,max(0,int(box['y']*h)))]),annotation_width=float(box['w']),annotation_height=float(box['h']),
        roi_mean=mu,roi_std=float(g.std()),background_mean=bm,background_std=bs,
        local_cnr=abs(mu-bm)/(bs+1e-6),bbox_w_pixels=x2-x1,bbox_h_pixels=y2-y1,
        apparent_component_area=float(best.area) if best else np.nan,
        apparent_eccentricity=float(best.eccentricity) if best else np.nan,
        apparent_solidity=float(best.solidity) if best else np.nan)

def extract_basic(index,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    existing=out/'basic_features_v3.parquet'; candidatesfile=out/'candidate_features_v3.parquet'
    if table_exists(existing) and table_exists(candidatesfile):
        return read_table(existing),read_table(candidatesfile)
    rows=[];candidates=[];failed=[]
    for _,r in tqdm(index.iterrows(),total=len(index),desc='Image / candidate features'):
        try:
            img=read_image(r.local_path);gray,overlay=masks_and_gray(img)
            feature=calc_global(gray,overlay)
            feature.update(image_domain(gray))
            feature.update(key=r.key,serial=r.serial,preset=r.preset,sha256=r.sha256,
                           review_label=r.review_label,review_subgroup=r.review_subgroup,manual_box_count=r.manual_box_count,
                           source_file=r.local_path,capture_group=r.capture_group)
            rows.append(feature)
            domain_items=candidate_domain(gray,r.boxes)
            by_index={x["candidate_index"]:x for x in domain_items}
            for j,box in enumerate(r.boxes):
                p=candidate_props(gray,box)
                p.update(by_index.get(j,{}))
                if p:p.update(key=r.key,serial=r.serial,review_label=r.review_label,review_subgroup=r.review_subgroup,candidate_index=j,annotation_source='manual_supporting_box');candidates.append(p)
        except Exception as e:failed.append(dict(key=r.key,error=repr(e)))
    pd.DataFrame(failed).to_csv(out/'feature_errors.csv',index=False)
    table=pd.DataFrame(rows);cts=pd.DataFrame(candidates)
    write_table(table,existing);write_table(cts,candidatesfile)
    return table,cts
