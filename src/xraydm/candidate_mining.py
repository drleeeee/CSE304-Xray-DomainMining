"""Exploratory ROI mining; candidate review label is IMAGE level only.

There is no independently verified candidate coverage GT in the current review.
Never report these candidate groups as covered-vs-missed sensitivity estimates.
"""
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors


def mine_candidates(candidates,out,seed=42):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    if candidates.empty:
        (out/'NO_CANDIDATE_DATA.txt').write_text('No annotated candidate ROIs.\n')
        return
    frame=candidates.copy()
    cols=[c for c in frame.columns if (c.startswith('domain_') or c.startswith('roi_yolo_'))
          and pd.api.types.is_numeric_dtype(frame[c]) and frame[c].nunique(dropna=True)>1]
    # Prevent row-wise label leakage: image-level T/F is metadata only.
    if not cols or len(frame)<8:return
    data=frame[cols].replace([np.inf,-np.inf],np.nan)
    data=data.loc[:,data.notna().sum()>3]
    if data.empty:return
    transformed=StandardScaler().fit_transform(SimpleImputer(strategy='median').fit_transform(data))
    pc=PCA(n_components=2,random_state=seed).fit_transform(transformed)
    clusters=KMeans(n_clusters=min(4,len(frame)),n_init=10,random_state=seed).fit_predict(transformed)
    result=frame[['key','candidate_index','review_label','serial']].copy()
    result['pc1']=pc[:,0];result['pc2']=pc[:,1];result['cluster']=clusters
    result.to_csv(out/'candidate_pca_clusters.csv',index=False)
    # Nearest annotated ROI examples from reviewed T images for each F-image ROI.
    # F image does NOT imply every annotation box is an unmarked object.
    f_idx=np.where(frame.review_label.to_numpy()=='F')[0]
    t_idx=np.where(frame.review_label.to_numpy()=='T')[0]
    pairrows=[]
    if len(f_idx) and len(t_idx):
        for i in f_idx:
            same=t_idx[(frame.iloc[t_idx].serial.values==frame.iloc[i].serial) &
                       (frame.iloc[t_idx].key.values!=frame.iloc[i].key)]
            if not len(same):continue
            dist=np.linalg.norm(transformed[same]-transformed[i],axis=1)
            j=same[np.argmin(dist)]
            pairrows.append(dict(f_image_key=frame.iloc[i].key,
                                 f_image_candidate_index=int(frame.iloc[i].candidate_index),
                                 closest_T_image_key=frame.iloc[j].key,
                                 closest_T_image_candidate_index=int(frame.iloc[j].candidate_index),
                                 standardized_distance=float(np.min(dist)),
                                 caveat='image-level F vs T ONLY; per-candidate marking status unknown'))
    pd.DataFrame(pairrows,columns=['f_image_key','f_image_candidate_index','closest_T_image_key',
            'closest_T_image_candidate_index','standardized_distance','caveat']).to_csv(out/'candidate_nearest_T_examples.csv',index=False)
    (out/'CANDIDATE_MINING_LIMITS.txt').write_text(
      'ROI labels are supporting box locations; F is the whole-image review label.\n'
      'F-image ROI candidates must NOT be individually called missed or uncovered.\n'
      'Clustering and nearest matches are descriptive hypotheses requiring visual review.\n',encoding='utf-8')
