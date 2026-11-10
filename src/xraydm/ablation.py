"""E5 feature-family experiments with group-disjoint CV and fold-fitted transforms."""
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score,roc_auc_score

BLOCK=('manual_box_count','img_h','img_w','annotation_width','annotation_height','bbox_h_pixels','bbox_w_pixels','capture_epoch','capture_group_size')

def family_columns(df):
    alln=[c for c in df.select_dtypes(include=[np.number]).columns if c not in BLOCK]
    def valid(cs):return list(dict.fromkeys(c for c in cs if df[c].notna().sum()>3 and df[c].nunique(dropna=True)>1))
    yolo=valid([c for c in alln if c.startswith('yolo_')])
    classical=valid([c for c in alln if c.startswith(('glcm_','lbp_','log_','blackhat_','edge_','lap_','high_freq','mean','std','p05','p95'))])
    domain=valid([c for c in alln if c.startswith(('domain_','agg_domain_')) and 'annotation_density' not in c])
    overlay=valid([c for c in alln if c.startswith(('overlay_','shortcut_'))])
    roi=valid([c for c in alln if c.startswith('agg_roi_yolo_')])
    return {'A_overlay_shortcut_control':overlay,
            'B_yolo_global':yolo,
            'C_interpretable':valid(classical+domain),
            'D_roi_plus_interpretable':valid(roi+classical+domain),
            'E_full_combination':valid(yolo+roi+classical+domain)}

def grouped_ablations(df,out,seed=42,folds=3,subset='all_reviewed'):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    x=df.loc[df.review_label.isin(['T','F'])].drop_duplicates('key').copy()
    if subset=='exclude_F_no_support_boxes' and 'review_subgroup' in x:
        x=x.loc[x.review_subgroup!='F_no_support_boxes'].copy()
    if x.review_label.nunique()<2 or (x.review_label=='F').sum()<6:
        (out/'ABLATION_NOT_RUN.txt').write_text('Insufficient validated T/F labels.\n');return pd.DataFrame()
    groups=x.capture_group.fillna(x.sha256).astype(str).to_numpy() if 'capture_group' in x else x.sha256.fillna(x.key).astype(str).to_numpy()
    y=(x.review_label=='F').astype(int).to_numpy()
    splits=list(StratifiedGroupKFold(n_splits=min(folds,int(y.sum())),shuffle=True,random_state=seed).split(x,y,groups))
    # Negative-control source: unit/preset + machine-colored overlay proxies only.
    # Human manual annotation counts are deliberately excluded (target-related leakage).
    for unit in sorted(x.serial.dropna().unique()):
        x['shortcut_unit_'+str(unit)]=(x.serial==unit).astype(int)
    if 'preset' in x:
        for preset in sorted(x.preset.dropna().unique()):
            x['shortcut_preset_'+str(preset)]=(x.preset==preset).astype(int)
    families=family_columns(x);rows=[];pred=[]
    if not families['yolo']:families['E_full_combination']=[]
    # ROI annotation availability differs sharply across units and label types.
    # Refuse a ROI-containing experiment when manual ROIs are systematically missing.
    coverage={lab:float((x.loc[x.review_label==lab,'manual_box_count']>0).mean()) for lab in ['T','F']} if 'manual_box_count' in x else {}
    for roi_family in ('D_roi_plus_interpretable','E_full_combination'):
        if roi_family=='D_roi_plus_interpretable' and not families.get('agg_roi_yolo') and not any(c.startswith('agg_roi_yolo_') for c in families[roi_family]):
            families[roi_family]=[]
        if any(c.startswith('agg_roi_yolo_') for c in families[roi_family]) and min(coverage.values(),default=0)<0.8:
            families[roi_family]=[]
    pd.DataFrame([dict(label=lab,roi_annotation_fraction=val) for lab,val in coverage.items()]).to_csv(out/'annotation_coverage_by_label.csv',index=False)
    for family,cols in families.items():
        if not cols:
            rows.append(dict(family=family,status='MISSING_FEATURES',n_features=0));continue
        data=x[cols].replace([np.inf,-np.inf],np.nan)
        for name in ('logistic','random_forest'):
            for fold,(train,test) in enumerate(splits):
                if len(np.unique(y[train]))<2 or len(np.unique(y[test]))<2:
                    rows.append(dict(family=family,model=name,fold=fold,status='SKIPPED_SINGLE_CLASS',n_features=len(cols)));continue
                if name=='logistic':
                    clf=make_pipeline(SimpleImputer(strategy='median'),StandardScaler(),LogisticRegression(max_iter=2500,class_weight='balanced',random_state=seed))
                else:
                    clf=make_pipeline(SimpleImputer(strategy='median'),RandomForestClassifier(n_estimators=200,min_samples_leaf=2,class_weight='balanced_subsample',random_state=seed,n_jobs=-1))
                clf.fit(data.iloc[train],y[train]);p=clf.predict_proba(data.iloc[test])[:,1]
                k=max(1,int(np.ceil(.10*len(test))));idx=np.argsort(p)[-k:]
                rows.append(dict(family=family,model=name,fold=fold,status='OK',n_features=len(cols),n_test=len(test),n_F=int(y[test].sum()),prevalence=float(y[test].mean()),average_precision=float(average_precision_score(y[test],p)),roc_auc=float(roc_auc_score(y[test],p)),recall_at_10pct=float(y[test][idx].sum()/max(1,y[test].sum()))))
                pred.extend(dict(key=x.iloc[i].key,fold=fold,family=family,model=name,true=int(y[i]),score=float(prob)) for i,prob in zip(test,p))
    r=pd.DataFrame(rows);r.to_csv(out/'feature_family_ablation.csv',index=False)
    pd.DataFrame(pred).to_csv(out/'ablation_out_of_fold.csv',index=False)
    if not r.empty and 'average_precision' in r:
        r.loc[r.status=='OK'].groupby(['family','model'])[['average_precision','roc_auc','recall_at_10pct','prevalence']].agg(['mean','std']).to_csv(out/'ablation_summary.csv')
    (out/'ABLATION_LIMITS.txt').write_text('Image-level T/F marking adequacy, NOT material detection. F may contain multiple exception mechanisms. Overlay-only is a shortcut control. Capture grouping is heuristic and requires visual audit. Fold-specific imputation/scaling is fit only on train. ROI features can be missing for images with no manual ROI and must NOT be treated as evidence of absence.\n',encoding='utf-8')
    return r
