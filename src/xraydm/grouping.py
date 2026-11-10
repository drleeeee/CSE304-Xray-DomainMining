"""Leakage-oriented acquisition groups based on byte hash, image pHash and proximity.
Conservative candidate grouping is an approximation requiring manual audits.
"""
from pathlib import Path
import cv2,numpy as np,pandas as pd
from collections import defaultdict

def phash(gray):
    small=cv2.resize(gray,(32,32),interpolation=cv2.INTER_AREA).astype(np.float32)
    dct=cv2.dct(small)[:8,:8]
    med=np.median(dct.flat[1:])
    return int(sum((int(v>med)<<i) for i,v in enumerate(dct.flat)))

def hamming(a,b):return (int(a)^int(b)).bit_count()

def seconds_from_metadata(r):
    s=str(r.get('date',''))+str(r.get('time',''))
    try:
        return pd.Timestamp.strptime(s,'%Y%m%d%H%M%S').timestamp()+max(0,int(r.get('tenths',0)))/10
    except Exception:
        try:return pd.to_datetime(s,format='%Y%m%d%H%M%S').timestamp()+max(0,int(r.get('tenths',0)))/10
        except Exception:return float('nan')

class UnionFind:
    def __init__(self,n):self.p=list(range(n))
    def find(self,x):
        while x!=self.p[x]:self.p[x]=self.p[self.p[x]];x=self.p[x]
        return x
    def join(self,a,b):self.p[self.find(a)]=self.find(b)

def assign_groups(index,output_dir,phash_limit=7,near_seconds=12,near_phash_limit=14):
    from .features import read_image
    from .overlay import remove_overlays
    out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
    x=index.reset_index(drop=True).copy(); n=len(x)
    fingerprints=[];errors=[]
    for i,r in x.iterrows():
        try:
            im=read_image(r.local_path)
            clean,_=remove_overlays(im)
            g=cv2.cvtColor(clean,cv2.COLOR_BGR2GRAY)
            v=phash(g)
            # independent check of histogram, not used as sole criterion
            thumb=cv2.resize(g,(32,32)).astype(np.float32)
            fingerprints.append((v,thumb))
        except Exception as e:
            fingerprints.append((None,None));errors.append(dict(key=r.key,error=repr(e)))
    x['phash_hex']=[format(t[0],'016x') if t[0] is not None else '' for t in fingerprints]
    x['capture_epoch']=x.apply(seconds_from_metadata,axis=1)
    uf=UnionFind(n);by_sha=defaultdict(list)
    for i,s in enumerate(x.sha256):by_sha[s].append(i)
    exact_edges=0;near_edges=0
    for ids in by_sha.values():
        for j in ids[1:]:uf.join(ids[0],j);exact_edges+=1
    # Only compare within the same physical machine; avoid global O(n^2).
    for serial,part in x.groupby('serial'):
        ids=part.index.tolist()
        # pairs with equal pHash regardless of timestamp require strong pixel likeness.
        by_ph=defaultdict(list)
        for i in ids:
            if fingerprints[i][0] is not None:by_ph[fingerprints[i][0]].append(i)
        for same in by_ph.values():
            for j in same[1:]:uf.join(same[0],j);near_edges+=1
        ordered=sorted((i for i in ids if np.isfinite(x.loc[i,'capture_epoch'])),key=lambda i:x.loc[i,'capture_epoch'])
        for ii,i in enumerate(ordered):
            for j in ordered[ii+1:]:
                gap=x.loc[j,'capture_epoch']-x.loc[i,'capture_epoch']
                if gap>near_seconds:break
                a,b=fingerprints[i][0],fingerprints[j][0]
                if a is None or b is None:continue
                # Temporal adjacency alone is NOT an equivalence criterion.
                if hamming(a,b)<=near_phash_limit:
                    uf.join(i,j);near_edges+=1
    x['capture_group']=['G'+str(uf.find(i)).zfill(5) for i in range(n)]
    x['capture_group_size']=x.groupby('capture_group').capture_group.transform('size')
    x[['key','serial','date','time','sha256','phash_hex','capture_epoch','capture_group','capture_group_size']].to_csv(out/'capture_groups.csv',index=False)
    pd.DataFrame(errors).to_csv(out/'grouping_errors.csv',index=False)
    (out/'grouping_notes.txt').write_text(f'Exact SHA edges: {exact_edges}; strong-similarity edges: {near_edges}. pHash is computed from inpainted approximate images; do not treat as true duplication ground truth. Groups with both T/F require manual inspection.\n',encoding='utf-8')
    conflicts=x.groupby('capture_group').review_label.nunique()
    suspicious=x[x.capture_group.isin(conflicts[conflicts>1].index)]
    suspicious[['key','serial','review_label','capture_group','phash_hex']].to_csv(out/'group_label_conflicts.csv',index=False)
    return x
