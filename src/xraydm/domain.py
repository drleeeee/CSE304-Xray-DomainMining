"""Domain-informed image/candidate descriptors.

All outputs are observational pixel proxies, not recovered attenuation, material
identity, Ishida sensitivity settings, or verified foreign-body segmentation.
"""
import numpy as np
import cv2
from scipy.ndimage import gaussian_filter, gaussian_laplace
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from skimage.measure import label, regionprops


def infer_bands(gray):
    """Robust horizontal dark-row candidates; does NOT establish physical identity."""
    g=gray.astype(np.float32)
    h,w=g.shape
    profile=np.median(g,axis=1)
    smooth=gaussian_filter(profile,sigma=max(2,h/100))
    deviation=profile-smooth
    # A local-threshold and a global darkness condition, explicitly exploratory.
    darkness=profile < np.quantile(profile,.30)
    threshold=min(float(np.quantile(deviation,.20)),0.)
    mask=(deviation < threshold)&darkness
    # Eliminate singleton rows, retain band candidates with >= 2 rows.
    if len(mask)>=3:
        kernel=np.ones((3,1),np.uint8)
        mask=cv2.morphologyEx(mask.astype('uint8')[:,None],cv2.MORPH_OPEN,kernel).ravel().astype(bool)
    bands=[]
    edge=np.diff(np.r_[False,mask,False].astype(int))
    for start,end in zip(np.where(edge==1)[0],np.where(edge==-1)[0]):
        bands.append((int(start),int(end)))
    return mask,bands,profile,deviation


def _rect(box,shape):
    h,w=shape
    try:
        x,y,bw,bh=(float(box[k]) for k in ('x','y','w','h'))
    except (KeyError,TypeError,ValueError):return None
    if not np.isfinite([x,y,bw,bh]).all() or bw<=0 or bh<=0:return None
    x1=max(0,int(np.floor((x-bw/2)*w)))
    y1=max(0,int(np.floor((y-bh/2)*h)))
    x2=min(w,int(np.ceil((x+bw/2)*w)))
    y2=min(h,int(np.ceil((y+bh/2)*h)))
    return (x1,y1,x2,y2) if x2>x1 and y2>y1 else None


def _roimask(gray,rect):
    """Heuristic dark-component segmentation; does not assert complete ground truth."""
    x1,y1,x2,y2=rect
    roi=gray[y1:y2,x1:x2]
    if roi.size<16:return None,None
    blur=cv2.GaussianBlur(roi,(3,3),0)
    # Morphological black-hat emphasizes small dark structures.
    d=cv2.morphologyEx(blur,cv2.MORPH_BLACKHAT,
                       cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7)))
    thr=float(np.quantile(d,.80))
    fg=(d>max(3,thr)).astype('uint8')
    props=regionprops(label(fg))
    if not props:return None,roi
    center=np.array([(roi.shape[0]-1)/2,(roi.shape[1]-1)/2])
    # Prefer sizable connected structure near annotation center.
    best=max(props,key=lambda r: r.area/(1+np.linalg.norm(np.array(r.centroid)-center)))
    return best,roi


def _texture(roi):
    if min(roi.shape)<5:return {}
    q=(cv2.resize(roi,(48,48),interpolation=cv2.INTER_AREA)//16).astype('uint8')
    glcm=graycomatrix(q,[1,2],[0,np.pi/2],levels=16,symmetric=True,normed=True)
    lbp=local_binary_pattern(roi,8,1,method='uniform')
    # LBP entropy, excluding absolute brightness.
    hist=np.bincount(np.clip(lbp.astype(int),0,9).ravel(),minlength=10).astype(float)
    hist=hist/max(1,hist.sum());nz=hist[hist>0]
    return dict(domain_roi_glcm_contrast=float(graycoprops(glcm,'contrast').mean()),
                domain_roi_glcm_homogeneity=float(graycoprops(glcm,'homogeneity').mean()),
                domain_roi_lbp_entropy=float(-(nz*np.log2(nz)).sum()))


def image_domain(gray):
    band,bands,profile,deviation=infer_bands(gray)
    # Camera/sensor-vs-product diagnosis needs alignment across multiple images;
    # these per-image measures alone do not establish a sensor artefact.
    residual=gray.astype(np.float32)-cv2.GaussianBlur(gray.astype(np.float32),(0,0),sigmaX=12)
    return dict(domain_band_row_fraction=float(band.mean()),
                domain_band_count_proxy=len(bands),
                domain_band_width_mean_px=float(np.mean([b-a for a,b in bands])) if bands else 0.,
                domain_row_darkness_variation=float(profile.std()),
                domain_row_residual_variation=float(deviation.std()),
                domain_background_residual_mad=float(np.median(np.abs(residual-np.median(residual)))),
                domain_background_residual_p95=float(np.quantile(np.abs(residual),.95)))


def candidate_domain(gray,boxes):
    """Features per MANUALLY ANNOTATED location; coverage unknown until verified."""
    h,w=gray.shape
    band,bands,profile,_=infer_bands(gray)
    band_boundaries=np.array([r for a,b in bands for r in (a,b)],dtype=float)
    centers=np.array([[float(b['x'])*w,float(b['y'])*h] for b in boxes if all(k in b for k in ('x','y'))])
    records=[]
    for i,box in enumerate(boxes):
        rect=_rect(box,gray.shape)
        if rect is None:continue
        x1,y1,x2,y2=rect;cy=(y1+y2)/2;cx=(x1+x2)/2
        r=gray[y1:y2,x1:x2]
        if r.size<16:continue
        pad=max(5,int(max(x2-x1,y2-y1)))
        ya,yb=max(0,y1-pad),min(h,y2+pad)
        xa,xb=max(0,x1-pad),min(w,x2+pad)
        surrounding=gray[ya:yb,xa:xb].astype(np.float32)
        outside=np.ones(surrounding.shape,bool)
        outside[y1-ya:y2-ya,x1-xa:x2-xa]=False
        bg=surrounding[outside]
        mean_object=float(r.mean());mean_bg=float(bg.mean()) if bg.size else np.nan
        sd_bg=float(bg.std()) if bg.size else np.nan
        local_residual=r.astype(np.float32)-cv2.GaussianBlur(surrounding,(0,0),max(2,pad/2))[y1-ya:y2-ya,x1-xa:x2-xa]
        # Multi-scale LoG persistence around region, not a verified detection score.
        persistence=[]
        for sigma in (1.,2.,4.):
            resp=np.abs(gaussian_laplace(r.astype(np.float32)/255.,sigma=sigma))
            persistence.append(float(resp.max()))
        nearest=np.nan
        if len(centers)>1 and i<len(centers):
            distances=np.linalg.norm(centers-centers[i],axis=1)
            distances[i]=np.inf
            nearest=float(np.min(distances))
        best,roi=_roimask(gray,rect)
        ecc=float(best.eccentricity) if best else np.nan
        angle=float(best.orientation) if best else np.nan
        rec=dict(candidate_index=i,
                 domain_roi_dark_residual_mean=float(np.mean(local_residual)),
                 domain_roi_dark_residual_min=float(np.min(local_residual)),
                 domain_roi_residual_std=float(np.std(local_residual)),
                 domain_local_cnr=abs(mean_object-mean_bg)/(sd_bg+1e-6),
                 domain_band_boundary_dist_px=float(np.min(np.abs(band_boundaries-cy))) if len(band_boundaries) else np.nan,
                 domain_band_inside_proxy=float(band[int(np.clip(cy,0,h-1))]),
                 domain_min_object_to_frame_px=float(min(cx,w-cx,cy,h-cy)),
                 domain_nearest_annotated_center_px=nearest,
                 domain_annotation_density=len(boxes),
                 domain_component_area_px=float(best.area) if best else np.nan,
                 domain_component_eccentricity=ecc,
                 domain_component_solidity=float(best.solidity) if best else np.nan,
                 domain_component_orientation_rad=angle,
                 domain_orientation_to_horizontal_band_rad=float(abs(angle)) if best else np.nan,
                 domain_log_response_s1=persistence[0],
                 domain_log_response_s2=persistence[1],
                 domain_log_response_s4=persistence[2],
                 domain_log_persistence_minmax=float(min(persistence)/(max(persistence)+1e-9)))
        rec.update(_texture(r))
        records.append(rec)
    return records


def aggregate_candidates(candidates):
    """Describe all annotated ROIs at IMAGE level without pretending each ROI has T/F ground truth."""
    import pandas as pd
    if candidates.empty:return pd.DataFrame(columns=['key'])
    cols=[c for c in candidates.columns if c.startswith(('domain_','roi_yolo_')) and pd.api.types.is_numeric_dtype(candidates[c])]
    if not cols:return pd.DataFrame({'key':candidates.key.unique()})
    agg=candidates.groupby('key',sort=False)[cols].agg(['mean','std'] )
    agg.columns=[f'agg_{name}_{stat}' for name,stat in agg.columns]
    return agg.reset_index()
