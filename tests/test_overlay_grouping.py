import numpy as np,cv2
from xraydm.overlay import remove_overlays,overlay_quality
from xraydm.grouping import phash,hamming,UnionFind

def test_red_rectangle_mask_inpaint():
    im=np.full((64,80,3),130,np.uint8)
    cv2.rectangle(im,(20,15),(45,40),(0,0,255),2)
    clean,mask=remove_overlays(im)
    assert np.any(mask)
    assert int(np.max(clean[:,:,2]-clean[:,:,0]))<15
    assert overlay_quality(im,clean,mask)['overlay_mask_fraction']>0

def test_phash_group_helpers():
    gray=np.full((32,32),120,np.uint8)
    assert hamming(phash(gray),phash(gray))==0
    u=UnionFind(3);u.join(0,2)
    assert u.find(0)==u.find(2)
    assert u.find(0)!=u.find(1)
