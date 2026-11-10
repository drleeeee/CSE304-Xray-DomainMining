import argparse,os,json,random
from pathlib import Path
import yaml,pandas as pd,numpy as np
from xraydm.ingest import collect,read_review,merge_labels
from xraydm.features import extract_basic
from xraydm.analyze import analyze
from xraydm.domain import aggregate_candidates
from xraydm.ablation import grouped_ablations
from xraydm.candidate_mining import mine_candidates
from xraydm.grouping import assign_groups
from xraydm.overlay import export_overlay_gallery
from xraydm.statistics_v3 import e4_stats
from xraydm.tableio import write_table

def main():
    p=argparse.ArgumentParser(description='CSE304 review adequacy analysis; Drive-friendly outputs')
    p.add_argument('--input',required=True);p.add_argument('--output',required=True);p.add_argument('--work',default='/content/xray_work')
    p.add_argument('--config',default='configs/default.yaml');p.add_argument('--no-deep',action='store_true')
    p.add_argument('--max-images',type=int,default=None,help='Quick smoke test limit')
    p.add_argument('--with-yolo-roi',action='store_true',help='Optional per-annotation multi-scale YOLO descriptors (GPU recommended)')
    args=p.parse_args()
    config=yaml.safe_load(Path(args.config).read_text())
    root=Path(args.output);root.mkdir(parents=True,exist_ok=True)
    work=Path(args.work);work.mkdir(parents=True,exist_ok=True)
    logs=root/'audit';logs.mkdir(parents=True,exist_ok=True)
    raw=work/'original_bmp'
    index=collect(args.input,raw,logs,max_images=args.max_images)
    reviews=read_review(args.input)
    merged=merge_labels(index,reviews,logs)
    # All three machine units have review labels in the 2026-11-10 ZIP.
    # Capture groups join labels without ever using labels to construct the groups.
    merged=assign_groups(merged,logs/'groups')
    export_overlay_gallery(merged,root/'quality_control'/'overlay',limit=int(config.get('overlay_gallery_limit',18)))
    f,c=extract_basic(merged,root/'features')
    if config.get('use_deep',True) and not args.no_deep:
        from xraydm.deep import extract_deep
        selected=[name for name,field in [('resnet','use_resnet'),('yolo','use_yolo'),('dinov2','use_dinov2')] if config.get(field,True)]
        try:
            emb=extract_deep(merged,root/'features',models=selected,batch_size=int(config.get('batch_size',12)))
            f=f.merge(emb,on='key',how='left',validate='one_to_one')
        except Exception as e:
            (logs/'deep_failure.txt').write_text(repr(e),encoding='utf-8')
            print('Deep extraction failed; classical features remain available:',repr(e))
    if args.with_yolo_roi:
        from xraydm.roi_yolo import extract_yolo_roi
        roi=extract_yolo_roi(merged,root/'features',side=int(config.get('roi_yolo_side',640)))
        if not roi.empty and not c.empty:
            c=c.merge(roi,on=['key','candidate_index'],how='left',validate='one_to_one')
            write_table(c,root/'features'/'candidate_features_with_yolo.parquet')
    # Candidate feature summaries are image-level summaries; missing support boxes are not zero foreign objects.
    ag=aggregate_candidates(c)
    if not ag.empty:
        f=f.merge(ag,on='key',how='left',validate='one_to_one')
    f=f.merge(merged[['key','capture_group','capture_group_size','review_subgroup']].drop(columns=['review_subgroup'],errors='ignore'),on='key',how='left',validate='one_to_one',suffixes=('','_group')) if 'capture_group' not in f else f
    write_table(f,root/'features'/'all_image_features.parquet')
    mine_candidates(c,root/'analysis'/'candidates')
    analyze(f,c,root/'analysis',folds=int(config.get('n_splits',3)))
    e4_stats(f,root/'analysis'/'e4',n_boot=int(config.get('n_group_bootstrap',250)))
    grouped_ablations(f,root/'analysis'/'ablations'/'all_reviewed',folds=int(config.get('n_splits',3)))
    grouped_ablations(f,root/'analysis'/'ablations'/'exclude_no_support_F',folds=int(config.get('n_splits',3)),subset='exclude_F_no_support_boxes')
    for unit in sorted(f.serial.dropna().unique()):
        subset=f.loc[f.serial==unit]
        if (subset.review_label=='F').sum()>=6:
            grouped_ablations(subset,root/'analysis'/'ablations'/f'unit_{unit}',folds=int(config.get('n_splits',3)))
    (root/'run_config.json').write_text(json.dumps(dict(arguments=vars(args),config=config),ensure_ascii=False,indent=2),encoding='utf-8')
    print('COMPLETE:',root)
if __name__=='__main__':main()
