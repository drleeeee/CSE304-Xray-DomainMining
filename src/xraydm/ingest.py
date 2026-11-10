import hashlib, io, json, re, zipfile
from pathlib import Path
import pandas as pd
from tqdm.auto import tqdm

SN_RE = re.compile(r'(SN\d+_\d{8}_NgImage)', re.IGNORECASE)
FN_RE = re.compile(r'^(\d{3})_(\d{8})_(\d{6})\((\d+)\)\.(?:bmp|jpg|jpeg)$',re.I)

def sha(data): return hashlib.sha256(data).hexdigest()

def canonical_key(s):
    s=str(s).replace('\\','/'); sn=SN_RE.search(s)
    return (sn.group(1).upper()+'/'+Path(s).name.lower()) if sn else ''

def parse_metadata(s):
    key=canonical_key(s); n=Path(s).name; match=FN_RE.match(n)
    sn=SN_RE.search(s)
    return dict(key=key,serial=sn.group(1).split('_')[0].upper() if sn else '',
                preset=match.group(1) if match else '',date=match.group(2) if match else '',
                time=match.group(3) if match else '',tenths=int(match.group(4)) if match else -1)

def collect(input_dir,raw_dir,log_dir,max_images=None):
    """Extract only canonical original BMPs. Never extract user-supplied paths directly."""
    raw_dir=Path(raw_dir);raw_dir.mkdir(parents=True,exist_ok=True)
    log_dir=Path(log_dir);log_dir.mkdir(parents=True,exist_ok=True)
    rows=[]; archive_rows=[]; seen={}; count=0
    archives=sorted(Path(input_dir).glob('*.zip'))
    for arc in archives:
        if 'xray-labels' in arc.name.lower():continue
        try: z=zipfile.ZipFile(arc)
        except zipfile.BadZipFile:
            archive_rows.append(dict(archive=arc.name,status='INVALID_ZIP',entries=0,notes='Not a valid complete ZIP; may contain only model weights'))
            continue
        with z:
            names=[i for i in z.infolist() if i.filename.lower().endswith('.bmp') and '/test1/yolov3/' in i.filename.replace('\\','/').lower()]
            archive_rows.append(dict(archive=arc.name,status='OK',entries=len(names),notes='core original BMP only'))
            for item in tqdm(names,desc='BMP '+arc.name[-13:],leave=False):
                k=canonical_key(item.filename)
                if not k:continue
                # Exact collision by canonical machine folder+filename gets separately audited.
                with z.open(item) as fh: b=fh.read()
                digest=sha(b)
                if k in seen:
                    rows.append(dict(source=arc.name,source_member=item.filename,local_path=seen[k][0],sha256=digest,
                                     duplicate_key=True,conflict=(seen[k][1]!=digest),**parse_metadata(item.filename)))
                    continue
                rel=Path(k);target=raw_dir/rel
                target.parent.mkdir(parents=True,exist_ok=True)
                if not target.exists() or sha(target.read_bytes())!=digest:target.write_bytes(b)
                seen[k]=(str(target),digest)
                rows.append(dict(source=arc.name,source_member=item.filename,local_path=str(target),sha256=digest,
                                 duplicate_key=False,conflict=False,**parse_metadata(item.filename)))
                count+=1
                if max_images and count>=max_images:break
        if max_images and count>=max_images:break
    idx=pd.DataFrame(rows)
    pd.DataFrame(archive_rows).to_csv(log_dir/'archive_audit.csv',index=False)
    if idx.empty:raise RuntimeError('No original /test1/yolov3/ BMP images found. Inspect archive_audit.csv')
    idx.to_csv(log_dir/'extracted_images_all.csv',index=False)
    if idx.conflict.any():raise RuntimeError('Conflicting BMP bytes for the same machine/file key; see extracted_images_all.csv')
    idx=idx.loc[~idx.duplicate_key].copy()
    idx['duplicate_sha_group']=idx.sha256
    idx.to_csv(log_dir/'image_index.csv',index=False)
    return idx

def read_review(input_dir):
    """Read *actual* XrayLabels format; unique machine-directory+filename keys only."""
    archives=[p for p in Path(input_dir).glob('*.zip') if 'xray-labels' in p.name.lower()]
    if len(archives)!=1:raise RuntimeError(f'Exactly one XrayLabels ZIP required, found {len(archives)}. Remove duplicate copy.')
    with zipfile.ZipFile(archives[0]) as z:
        reviews=pd.read_csv(io.BytesIO(z.read('reviews.csv')),encoding='utf-8-sig')
        ann=json.loads(z.read('annotations.json'))['records']
    required={'image_path','original_boxes_ok','manual_box_count'}
    if not required.issubset(reviews):raise ValueError('reviews.csv missing required fields')
    reviews['key']=reviews.image_path.map(canonical_key)
    reviews['review_label']=reviews.original_boxes_ok.astype(str).str.strip().str.upper()
    reviews['review_status']=reviews.status.astype(str).str.strip().str.lower() if 'status' in reviews else 'unknown'
    # A draft is not a verified training target even if T/F is populated.
    reviews.loc[reviews.review_status!='done','review_label']='UNREVIEWED'
    reviews['review_subgroup']=reviews.apply(
        lambda r: ('T' if r.review_label=='T' else ('F_no_support_boxes' if r.review_label=='F' and r.manual_box_count==0
                     else 'F_with_support_boxes' if r.review_label=='F' else 'UNREVIEWED')),axis=1)

    if (reviews.key=='').any():raise ValueError('Some label rows lack SN folder; cannot safely match')
    if reviews.key.duplicated().any():raise ValueError('Duplicate review keys; manually resolve first')
    boxes={canonical_key(k):v.get('boxes',[]) for k,v in ann.items()}
    reviews['boxes']=reviews.key.map(lambda k:boxes.get(k,[]))
    reviews['box_count_actual']=reviews.boxes.map(len)
    reviews['count_mismatch']=reviews.box_count_actual!=reviews.manual_box_count
    return reviews

def merge_labels(index,reviews,logs):
    m=index.merge(reviews[['key','review_label','review_subgroup','review_status','manual_box_count','boxes','count_mismatch']],on='key',how='left',validate='one_to_one',indicator=True)
    m['review_label']=m.review_label.fillna('UNREVIEWED')
    m['review_subgroup']=m.review_subgroup.fillna('UNREVIEWED')
    m['review_status']=m.review_status.fillna('unmatched')
    m['manual_box_count']=m.manual_box_count.fillna(0).astype(int)
    m['boxes']=m.boxes.map(lambda x:x if isinstance(x,list) else [])
    m['count_mismatch']=m.count_mismatch.fillna(False).astype(bool)
    m.to_csv(Path(logs)/'matched_images.csv',index=False)
    not_found=reviews.loc[~reviews.key.isin(set(index.key))]
    not_found.to_csv(Path(logs)/'unmatched_review_rows.csv',index=False)
    audit=dict(original_bmp=len(index),matched=int((m._merge=='both').sum()),review_records=len(reviews),unmatched_reviews=len(not_found),
               counts=m.review_label.value_counts().to_dict(), subgroups=m.review_subgroup.value_counts().to_dict(),count_mismatch=int(m.count_mismatch.sum()))
    (Path(logs)/'label_audit.json').write_text(json.dumps(audit,indent=2,ensure_ascii=False),encoding='utf-8')
    if audit['unmatched_reviews']:print('WARNING unmatched review records:',audit['unmatched_reviews'])
    return m
