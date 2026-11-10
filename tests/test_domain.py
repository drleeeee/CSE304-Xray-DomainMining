import numpy as np
from xraydm.domain import image_domain,candidate_domain,aggregate_candidates
import pandas as pd

def test_domain_synthetic_bands_and_candidate():
    img=np.full((140,200),180,dtype=np.uint8)
    img[55:68,:]=70
    img[100:111,:]=80
    img[61:64,75:78]=18
    boxes=[dict(x=76/200,y=63/140,w=.10,h=.15),dict(x=.7,y=.74,w=.1,h=.14)]
    globalf=image_domain(img)
    cand=candidate_domain(img,boxes)
    assert globalf['domain_band_row_fraction']>=0
    assert len(cand)==2
    assert cand[0]['domain_annotation_density']==2
    assert np.isfinite(cand[0]['domain_local_cnr'])
    frame=pd.DataFrame([dict(key='k',**v) for v in cand])
    agg=aggregate_candidates(frame)
    assert len(agg)==1
    assert any(c.startswith('agg_domain_') for c in agg)
