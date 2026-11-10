import json
from pathlib import Path
import numpy as np,pandas as pd,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.metrics import average_precision_score,roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
import joblib

META={'manual_box_count','img_h','img_w'}
DROP={'source_file','key','serial','preset','sha256','review_label'}

def analyze(table,candidates,out,folds=3,seed=42):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    # Data are NG-only. F means inadequate markings, not 'foreign object present'.
    table=table.drop_duplicates('key').copy()
    counts=table.review_label.value_counts().to_dict()
    (out/'class_counts.json').write_text(json.dumps(counts,indent=2),encoding='utf-8')
    numerical=[c for c in table.select_dtypes(include=[np.number]) if c not in META and not c.startswith(('annotation_','overlay_')) and 'annotation_density' not in c]
    Xall=table[numerical].replace([np.inf,-np.inf],np.nan)
    Xall=Xall.loc[:,Xall.notna().sum()>1]
    Xall=Xall.loc[:,Xall.nunique(dropna=True)>1]
    if not Xall.empty and len(Xall)>4:
        a=SimpleImputer(strategy='median').fit_transform(Xall)
        a=StandardScaler().fit_transform(a)
        p=PCA(n_components=2,random_state=seed).fit_transform(a)
        clusters=KMeans(n_clusters=min(4,len(table)),random_state=seed,n_init=10).fit_predict(a)
        pd.DataFrame(dict(key=table.key.values,pc1=p[:,0],pc2=p[:,1],cluster=clusters,label=table.review_label.values)).to_csv(out/'pca_clusters.csv',index=False)
        fig,ax=plt.subplots(figsize=(8,5))
        for lab in sorted(table.review_label.unique()):
            m=table.review_label.values==lab
            ax.scatter(p[m,0],p[m,1],s=13,alpha=.55,label=lab)
        ax.set(xlabel='PC1',ylabel='PC2',title='PCA: Image representations');ax.legend()
        fig.tight_layout();fig.savefig(out/'pca_labels.png',dpi=170);plt.close(fig)
    # Only reviewed images can contribute supervised ground truth.
    labeled=table.loc[table.review_label.isin(['T','F'])].copy()
    if labeled.empty or labeled.review_label.nunique()<2:
        (out/'NO_SUPERVISED_EVAL.txt').write_text('Cannot evaluate T/F without reviewed labels of both classes.\n')
        return
    # Feature tests, descriptive not causal; BH correction below.
    results=[]
    features=[c for c in numerical if c in labeled and labeled[c].nunique(dropna=True)>1]
    for col in features:
        t=pd.to_numeric(labeled.loc[labeled.review_label=='T',col],errors='coerce').dropna()
        f=pd.to_numeric(labeled.loc[labeled.review_label=='F',col],errors='coerce').dropna()
        if len(t)<3 or len(f)<3:continue
        u,p=mannwhitneyu(f,t,alternative='two-sided')
        delta=2*u/(len(f)*len(t))-1
        results.append(dict(feature=col,n_T=len(t),n_F=len(f),median_T=float(t.median()),median_F=float(f.median()),
                            cliff_delta=float(delta),p_value=float(p)))
    stats=pd.DataFrame(results)
    if not stats.empty:
        stats=stats.sort_values('p_value').reset_index(drop=True)
        n=len(stats);q=(stats.p_value.values*n/np.arange(1,n+1))
        stats['bh_fdr_q']=np.minimum.accumulate(q[::-1])[::-1].clip(0,1)
        stats.to_csv(out/'feature_statistics.csv',index=False)
    # Prevent perfect duplicate leakage, plus serial-specific report.
    if len(labeled)<30 or sum(labeled.review_label=='F')<6:return
    X=labeled[features].replace([np.inf,-np.inf],np.nan)
    X=X.loc[:,X.nunique(dropna=True)>1]
    if X.empty:return
    y=(labeled.review_label=='F').astype(int).to_numpy()
    groups=(labeled.capture_group.fillna(labeled.sha256) if 'capture_group' in labeled else labeled.sha256.fillna(labeled.key)).to_numpy()
    n_splits=min(folds,int(sum(y==1)),int(sum(y==0)))
    splitter=StratifiedGroupKFold(n_splits=n_splits,shuffle=True,random_state=seed)
    rows=[];predict=[]
    models={
      'logistic':make_pipeline(SimpleImputer(strategy='median'),StandardScaler(),LogisticRegression(max_iter=2500,class_weight='balanced',random_state=seed)),
      'random_forest':make_pipeline(SimpleImputer(strategy='median'),RandomForestClassifier(n_estimators=250,min_samples_leaf=2,
                                        class_weight='balanced_subsample',random_state=seed,n_jobs=-1))}
    for name,model in models.items():
        for fold,(train,test) in enumerate(splitter.split(X,y,groups)):
            if len(np.unique(y[train]))<2 or len(np.unique(y[test]))<2:
                rows.append(dict(model=name,fold=fold,status='SKIPPED_CLASS_MISSING'))
                continue
            model.fit(X.iloc[train],y[train]);prob=model.predict_proba(X.iloc[test])[:,1]
            k=max(1,int(np.ceil(.1*len(test))));top=np.argsort(prob)[-k:]
            rows.append(dict(model=name,fold=fold,status='OK',n_test=len(test),n_positive=int(y[test].sum()),
                     average_precision=float(average_precision_score(y[test],prob)),
                     roc_auc=float(roc_auc_score(y[test],prob)),
                     recall_at_top_10pct=float(y[test][top].sum()/max(1,y[test].sum())),
                     prevalence=float(y[test].mean())))
            for ix,pred in zip(test,prob):predict.append(dict(key=labeled.iloc[ix].key,label=int(y[ix]),model=name,fold=fold,score=float(pred)))
        model.fit(X,y);joblib.dump(dict(model=model,features=X.columns.tolist()),out/f'{name}_descriptive_model.joblib')
    pd.DataFrame(rows).to_csv(out/'group_cv_metrics.csv',index=False)
    pd.DataFrame(predict).to_csv(out/'out_of_fold_predictions.csv',index=False)
    # Separate unit-specific data audit; Unit confounding cannot be excluded via group CV alone.
    labeled.groupby(['serial','review_label']).size().unstack(fill_value=0).to_csv(out/'unit_label_crosstab.csv')
    if not candidates.empty:
        cols=['roi_mean','roi_std','local_cnr','apparent_component_area','apparent_eccentricity','apparent_solidity']
        present=[c for c in cols if c in candidates]
        candidates.groupby('review_label')[present].median(numeric_only=True).to_csv(out/'candidate_roi_summary.csv')
    (out/'INTERPRETATION_LIMITS.txt').write_text(
      'This is exploratory, observational analysis of reviewed NG images.\n'
      'Manual support boxes are NOT exhaustive object truth; box count is NOT object count.\n'
      'F means incomplete machine marking, not a verified foreign object or general machine miss rate.\n'
      'YOLO pretrained embeddings do NOT detect confirmed X-ray foreign objects.\n'
      'Small F class and serial imbalance limit generalization.\n'
      'No evidence of material identity or the internal Ishida decision rule.\n',encoding='utf-8')
