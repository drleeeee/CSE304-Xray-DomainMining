"""Optional smoke test with user-provided input archives, not included in distribution."""
import sys,io,json,zipfile,glob,tempfile
from pathlib import Path
import pandas as pd
from xraydm.features import extract_basic
from xraydm.ingest import canonical_key
from xraydm.candidate_mining import mine_candidates
from xraydm.ablation import grouped_ablations
# Local offline test environment has no pyarrow; use pickle as a test-only surrogate.
pd.DataFrame.to_parquet=lambda self,path,index=False:self.to_pickle(path)
pd.read_parquet=lambda path:pd.read_pickle(path)
src=sorted(glob.glob('/mnt/data/*part00*.zip'))[1]
label='/mnt/data/xray-labels-2026-11-03.zip'
with zipfile.ZipFile(label) as z:
    reviews=pd.read_csv(io.BytesIO(z.read('reviews.csv')))
    ann=json.loads(z.read('annotations.json'))['records']
with tempfile.TemporaryDirectory() as td:
    root=Path(td)
    selected=pd.concat([reviews.loc[reviews.original_boxes_ok=='T'].head(2),reviews.loc[reviews.original_boxes_ok=='F'].head(2)])
    rows=[]
    with zipfile.ZipFile(src) as z:
        files={canonical_key(n):n for n in z.namelist() if n.endswith('.bmp')}
        for _,r in selected.iterrows():
            key=canonical_key(r.image_path)
            if key not in files:raise Exception('missing '+key)
            target=root/Path(key).name;target.write_bytes(z.read(files[key]))
            import hashlib
            rows.append(dict(key=key,local_path=str(target),serial='SN12053',preset='001',
                             sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
                             review_label=r.original_boxes_ok,manual_box_count=r.manual_box_count,
                             boxes=ann[r.image_path]['boxes']))
    f,c=extract_basic(pd.DataFrame(rows),root/'features')
    assert len(f)==4 and len(c)>0
    assert 'domain_band_count_proxy' in f and 'domain_local_cnr' in c
    mine_candidates(c,root/'candidates')
    grouped_ablations(f,root/'ablation')
    print('REAL BMP SMOKE TEST OK',len(f),len(c),'rows')
