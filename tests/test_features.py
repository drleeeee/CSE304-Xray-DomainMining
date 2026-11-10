import numpy as np
from xraydm.features import calc_global,candidate_props

def test_features():
    g=np.full((100,120),150,dtype=np.uint8);g[45:55,60:70]=30
    features=calc_global(g,np.zeros_like(g))
    assert 'glcm_contrast' in features
    p=candidate_props(g,dict(x=65/120,y=50/100,w=20/120,h=20/100))
    assert p['local_cnr']>=0
