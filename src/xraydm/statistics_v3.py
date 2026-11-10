"""E4: descriptive T/F differences, BH-FDR and acquisition-group bootstrap CIs."""
from pathlib import Path
import numpy as np,pandas as pd
from scipy.stats import mannwhitneyu

def e4_stats(df,out,seed=42,n_boot=250):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    x=df.loc[df.review_label.isin(['T','F'])].drop_duplicates('key').copy()
    if x.empty or x.review_label.nunique()<2:return pd.DataFrame()
    y=(x.review_label=='F').to_numpy()
    forbidden=('overlay_','annotation_','manual_box','bbox_','roi_yolo_','yolo_','resnet_','dinov2_')
    cols=[c for c in x.select_dtypes(include='number') if not c.startswith(forbidden)
          and c not in ('capture_epoch','capture_group_size') and x[c].nunique(dropna=True)>1]
    # For descriptive examination, numeric image / aggregate features only.
    groups=x.capture_group.to_numpy() if 'capture_group' in x else x.sha256.to_numpy()
    group_ids=np.unique(groups)
    group_to_idx={g:np.where(groups==g)[0] for g in group_ids}
    rng=np.random.default_rng(seed)
    rows=[]
    for col in cols:
        v=pd.to_numeric(x[col],errors='coerce').to_numpy(dtype=float)
        t=v[(~y)&np.isfinite(v)];f=v[y&np.isfinite(v)]
        if len(t)<5 or len(f)<5:continue
        U,p=mannwhitneyu(f,t,alternative='two-sided')
        delta=2*U/(len(f)*len(t))-1
        boot=[]
        for _ in range(n_boot):
            chosen=rng.choice(group_ids,size=len(group_ids),replace=True)
            idx=np.concatenate([group_to_idx[g] for g in chosen])
            vv=v[idx];yy=y[idx]
            ft=vv[yy&np.isfinite(vv)];tt=vv[(~yy)&np.isfinite(vv)]
            if len(ft)<2 or len(tt)<2:continue
            ub=mannwhitneyu(ft,tt,alternative='two-sided').statistic
            boot.append(float(2*ub/(len(ft)*len(tt))-1))
        lo,hi=np.percentile(boot,[2.5,97.5]) if len(boot)>=30 else (np.nan,np.nan)
        rows.append(dict(feature=col,n_T=len(t),n_F=len(f),median_T=float(np.median(t)),median_F=float(np.median(f)),cliff_delta=float(delta),cliff_ci_low=lo,cliff_ci_high=hi,p_value=float(p),bootstrap_valid=len(boot)))
    r=pd.DataFrame(rows)
    if not r.empty:
        r=r.sort_values('p_value').reset_index(drop=True)
        q=np.minimum.accumulate((r.p_value.to_numpy()*len(r)/np.arange(1,len(r)+1))[::-1])[::-1]
        r['bh_fdr_q']=np.clip(q,0,1)
    r.to_csv(out/'e4_group_bootstrap_statistics.csv',index=False)
    (out/'IMPORTANT.txt').write_text('Mann-Whitney p-values assume independent observations and are descriptive only; acquisition-group bootstrap CIs account for cluster dependence approximately. Prefer a permutation test at acquisition-group level for confirmatory significance.\n',encoding='utf-8')
    return r
