from xraydm.ingest import canonical_key,parse_metadata

def test_keys():
    a='/top/SN12053_20200622_NgImage/001_20200622_203305(8).bmp'
    b='SN12053_20200622_NgImage/001_20200622_203305(8).bmp'
    assert canonical_key(a)==canonical_key(b)
    assert parse_metadata(a)['preset']=='001'
    assert parse_metadata(a)['serial']=='SN12053'

def test_different_serial_not_same_key():
    a='SN12053_20200622_NgImage/001_20200622_203305(8).bmp'
    b='SN77127_20200622_NgImage/001_20200622_203305(8).bmp'
    assert canonical_key(a)!=canonical_key(b)
