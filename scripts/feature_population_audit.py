"""Exploratory feature-level, conditional-null and population-budget SAE audits.

Uses historical paper embeddings/states; preservation reruns are never substituted.
"""
from pathlib import Path
import argparse,hashlib,io,json,os,time,zipfile
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from scipy.sparse import csr_matrix,vstack
from scipy.stats import rankdata,spearmanr

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/feature_population_audit_v1'
EMBED=ROOT/'data/rank_control_embeddings_v1.zip'
MODELS=ROOT/'dai_analysis_reproduction_20261029/data/sae_models.zip'
COV=ROOT/'results/reviewer_followup_20261029/confound_audit/validation_covariates.csv'
PLAN={'version':'feature-population-audit-v1','scope':'post-review exploratory reused validation; no confirmatory inference',
      'methods':['ncl','simclr','supervised'],'encoder_seeds':[10,42,52],'sae_seeds':[10,42,52],
      'primary_variant':'baseline','top_m':10,'dictionary_size':1536,'k':24,
      'null_permutations':1000,'null_seed':20261030,'strata_bin_counts':[4,3,5],
      'primary_strata_bins':4,'strata':'separate zero tumour area; quantile bins of positive area crossed with axial-index quantile bins',
      'bins_fit':'fixed validation-covariate quantiles, no feature/label-purity optimization',
      'null_scope':'descriptive conditional reference, not causal adjustment or patient-level permutation test; area uses annotation masks',
      'orientation':'univariate AUROC orientation chosen on training tumour labels only; applied frozen to validation',
      'budgets':'every integer J>=8 up to maximum eligible count; common paired domain plus method-specific endpoint',
      'selectors':['validation_eligible_training_usage','training_only_usage'],
      'missing_bounds':'retain selected IDs; [0,1] for unscorable purity; do not replace or renormalize them in bounded mean',
      'random_replicates':1000,'random_seed':783013,'random_rule':'without replacement, nested random permutations of same selector candidate pool',
      'aggregation':'average SAE seeds within encoder first; report 3 encoder means; feature-level p-values not interpreted',
      'supervised_scope':'new native SAE fits on archived supervised encoder vectors, historical SAE initialization/optimizer recipe; encoder objective/augmentation differ'}


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    t=path.with_suffix('.tmp');t.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8');os.replace(t,path)


def auc_sparse(values,indices,y):
    """Exact AUROC for nonnegative scores with sparse nonzero entries and ties."""
    positive=int(y.sum());negative=len(y)-positive
    if not positive or not negative:return np.nan
    zero_count=len(y)-len(values);positive_zero=positive-int(y[indices].sum())
    rank_sum=positive_zero*(zero_count+1)/2
    rank_sum+=float((rankdata(values,method='average')+zero_count)@y[indices])
    return (rank_sum-positive*(positive+1)/2)/(positive*negative)


def sparse_aucs(matrix,y):
    m=matrix.tocsc();result=[]
    for j in range(m.shape[1]):
        lo,hi=m.indptr[j:j+2];result.append(auc_sparse(m.data[lo:hi],m.indices[lo:hi],y))
    return np.asarray(result)


def activations(x,state,device):
    sd=state['model'];w=sd['encoder.weight'].to(device);bias=sd['encoder.bias'].to(device)
    center=(state['transform']['mean']+sd['pre_bias']).to(device);chunks=[]
    with torch.inference_mode():
        for start in range(0,len(x),1024):
            raw=F.relu(F.linear(torch.from_numpy(x[start:start+1024]).to(device)-center,w,bias))
            values,indices=raw.topk(24,dim=1)
            a=torch.zeros_like(raw).scatter_(1,indices,values).cpu().numpy()
            chunks.append(csr_matrix(a))
    return vstack(chunks,format='csr')


def strata(cov,bins):
    area=cov.tumour_area_fraction.to_numpy();index=cov.slice_idx.to_numpy()
    cuts=np.unique(np.quantile(area[area>0],np.arange(1,bins)/bins))
    ac=np.where(area==0,0,1+np.searchsorted(cuts,area,side='right'))
    icuts=np.unique(np.quantile(index,np.arange(1,bins)/bins))
    ic=np.searchsorted(icuts,index,side='right');s=ac*(bins+1)+ic
    return s,{'positive_area_cutpoints':cuts.tolist(),'slice_index_cutpoints':icuts.tolist(),
              'cells':[{'id':int(k),'slices':int((s==k).sum())} for k in np.unique(s)],
              'zero_area_slices':int((area==0).sum())}


def permuted_labels(labels,stratum=None):
    rng=np.random.default_rng(PLAN['null_seed']);result=np.empty((PLAN['null_permutations'],len(labels)),dtype=np.int16)
    groups=[np.arange(len(labels))] if stratum is None else [np.flatnonzero(stratum==v) for v in np.unique(stratum)]
    for b in range(len(result)):
        for ids in groups:result[b,ids]=rng.permutation(labels[ids])
    return result


def null_mean(top,permuted,classes):
    total=np.zeros(len(top),dtype=float)
    for start in range(0,len(permuted),100):
        labels=permuted[start:start+100,top]
        total+=np.stack([(labels==c).sum(2) for c in classes]).max(0).mean(0)*len(labels)
    return total/len(permuted)/10


def safe_rho(x,y):
    good=np.isfinite(x)&np.isfinite(y)
    if good.sum()<3 or np.unique(x[good]).size<2 or np.unique(y[good]).size<2:return None
    return float(spearmanr(x[good],y[good]).statistic)


def budget_records(feature,counts,method,enc,seed):
    observed=feature.purity.to_numpy();valid=np.isfinite(observed);all_ids=np.arange(len(feature))
    candidates={'validation_eligible_training_usage':all_ids[valid],
                'training_only_usage':all_ids[counts>0]}
    # Endpoints are each dictionary's eligible count; paired summaries use common J.
    max_j=int(valid.sum());records=[]
    for selector,pool in candidates.items():
        ordered=pool[np.lexsort((pool,-counts[pool]))]
        nmax=min(len(ordered),max_j)
        if nmax<8:continue
        js=np.arange(8,nmax+1)
        p=observed[ordered];v=valid[ordered];sums=np.cumsum(np.nan_to_num(p,nan=0.));ns=np.cumsum(v)
        adjusted=feature.purity_above_stratified_null.to_numpy()[ordered]
        adj=np.cumsum(np.nan_to_num(adjusted,nan=0.))
        # Same candidate pool and J, but a label-agnostic random order per replicate.
        rng=np.random.default_rng(PLAN['random_seed']+enc*100+seed)
        random_means=[];random_lower=[];random_upper=[]
        for _ in range(PLAN['random_replicates']):
            ids=rng.permutation(pool)[:nmax];good=valid[ids]
            total=np.cumsum(np.nan_to_num(observed[ids],nan=0.));num=np.cumsum(good)
            random_means.append(np.divide(total[js-1],num[js-1],out=np.full(len(js),np.nan),where=num[js-1]>0))
            random_lower.append(total[js-1]/js)
            random_upper.append((total[js-1]+js-num[js-1])/js)
        random_means=np.asarray(random_means);random_lower=np.asarray(random_lower);random_upper=np.asarray(random_upper)
        for i,j in enumerate(js):
            n=int(ns[j-1]);value=float(sums[j-1]/n) if n else np.nan
            records.append({'method':method,'encoder_seed':enc,'sae_seed':seed,'selector':selector,'J':int(j),
             'candidate_pool':len(pool),'validation_eligible_total':max_j,'scorable':n,'coverage':n/j,
             'purity_conditional':value,'purity_lower':sums[j-1]/j,'purity_upper':(sums[j-1]+j-n)/j,
             'stratified_excess_conditional':adj[j-1]/n if n else np.nan,
             'random_mean':float(np.nanmean(random_means[:,i])),
             'random_q025':float(np.nanquantile(random_means[:,i],.025)),
             'random_q975':float(np.nanquantile(random_means[:,i],.975)),
             'usage_minus_random':value-float(np.nanmean(random_means[:,i])),
             'random_lower_mean':float(np.mean(random_lower[:,i])),
             'random_upper_mean':float(np.mean(random_upper[:,i])),
             'usage_percentile_in_random':float(np.mean(random_means[:,i]<=value)) if n else np.nan})
    return pd.DataFrame(records)


def main(methods,force=False):
    OUT.mkdir(parents=True,exist_ok=True);torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    manifest={'plan':PLAN,'inputs':{'embeddings_sha256':sha(EMBED),'historical_sae_archive_sha256':sha(MODELS),'covariates_sha256':sha(COV)}}
    protocol=OUT/'audit_protocol.json'
    if protocol.exists():assert json.loads(protocol.read_text())==manifest,'Audit protocol changed'
    else:atomic_json(protocol,manifest)
    with zipfile.ZipFile(EMBED) as emb,zipfile.ZipFile(MODELS) as models:
        frame=pd.read_csv(io.BytesIO(emb.read('split_manifest.csv')))
        tf=frame[frame.split=='train'].reset_index(drop=True);vf=frame[frame.split=='val'].reset_index(drop=True)
        cov=pd.read_csv(COV);assert np.array_equal(cov.sample_id,vf.sample_id)
        labels=vf.dominant_label.to_numpy();classes=np.unique(labels)
        permutation_sets={'global':permuted_labels(labels)};stratum_records={}
        for n in PLAN['strata_bin_counts']:
            s,info=strata(cov,n);permutation_sets[f'stratified_{n}']=permuted_labels(labels,s)
            for cell in info['cells']:
                ids=np.flatnonzero(s==cell['id']);cell['distinct_patients']=int(vf.iloc[ids].patient_id.nunique())
                cell['label_counts']={str(k):int((labels[ids]==k).sum()) for k in classes}
            stratum_records[str(n)]=info
        atomic_json(OUT/'strata_definition.json',stratum_records)
        for method in methods:
            for enc in [10,42,52]:
                xs={}
                for split in ['train','val']:
                    if method=='supervised':
                        path=OUT/'supervised_embeddings'/f'encoder_{enc}'/f'{split}.npz'
                        if not path.exists():raise FileNotFoundError(f'Finish supervised SAE stage first: {path}')
                        handle=np.load(path,allow_pickle=False)
                    else:handle=np.load(io.BytesIO(emb.read(f'runs/{method}/encoder_{enc}/{split}.npz')),allow_pickle=False)
                    with handle as f:
                        assert np.array_equal(f['sample_ids'],frame[frame.split==split].sample_id.to_numpy())
                        xs[split]=f['x'].copy()
                for seed in [10,42,52]:
                    folder=OUT/'fits'/method/f'encoder_{enc}'/f'sae_{seed}';folder.mkdir(parents=True,exist_ok=True)
                    if (folder/'complete.json').exists() and not force:print('REUSE AUDIT',method,enc,seed,flush=True);continue
                    if method=='supervised':
                        model_path=OUT/'supervised_models'/f'encoder_{enc}'/f'sae_{seed}'/'completed_model.pt'
                        state=torch.load(model_path,map_location='cpu',weights_only=True);model_sha=sha(model_path)
                    else:
                        raw=models.read(f'fits/{method}/encoder_{enc}/baseline_r192/sae_{seed}/completed_model.pt')
                        state=torch.load(io.BytesIO(raw),map_location='cpu',weights_only=True);model_sha=hashlib.sha256(raw).hexdigest()
                    counts=state['training_activity']['positive_counts'].numpy()
                    started=time.monotonic();at=activations(xs['train'],state,device)
                    train_auc=sparse_aucs(at,tf.has_tumor.to_numpy());del at
                    av=activations(xs['val'],state,device);val_auc=sparse_aucs(av,vf.has_tumor.to_numpy())
                    orientation=np.where(train_auc>=.5,1,-1);oriented=np.where(orientation>0,val_auc,1-val_auc)
                    dense=av.toarray();support=np.asarray((av>0).sum(0)).ravel();eligible=np.flatnonzero(support>=10)
                    tops=[]
                    for j in eligible:
                        ids=np.flatnonzero(dense[:,j]>0);tops.append(ids[np.argsort(-dense[ids,j],kind='stable')[:10]])
                    top=np.asarray(tops,dtype=int).reshape(-1,10)
                    top_labels=labels[top];purity=np.stack([(top_labels==c).mean(1) for c in classes]).max(0)
                    feature=pd.DataFrame({'feature':np.arange(1536),'train_cumulative_count':counts,'val_support':support,
                        'train_auc':train_auc,'val_auc_positive_activation':val_auc,'train_fitted_orientation':orientation,
                        'val_auc_train_oriented':oriented,'purity':np.nan})
                    feature.loc[eligible,'purity']=purity
                    for name,permuted in permutation_sets.items():
                        null=null_mean(top,permuted,classes)
                        feature.loc[eligible,f'{name}_null']=null
                    feature['purity_above_global_null']=feature.purity-feature.global_null
                    feature['purity_above_stratified_null']=feature.purity-feature.stratified_4_null
                    feature['method']=method;feature['encoder_seed']=enc;feature['sae_seed']=seed
                    feature.to_csv(folder/'features.csv',index=False)
                    np.save(folder/'top10_indices.npy',top);np.save(folder/'eligible_feature_ids.npy',eligible)
                    correlations={'method':method,'encoder_seed':enc,'sae_seed':seed,'eligible_features':len(eligible),
                     'rho_purity_utility':safe_rho(feature.purity.to_numpy(),oriented),
                     'rho_global_excess_utility':safe_rho(feature.purity_above_global_null.to_numpy(),oriented),
                     'rho_stratified_excess_utility':safe_rho(feature.purity_above_stratified_null.to_numpy(),oriented),
                     'rho_purity_positive_auc':safe_rho(feature.purity.to_numpy(),val_auc),
                     'mean_purity':float(np.mean(purity)),'mean_global_excess':float(feature.purity_above_global_null.mean()),
                     'mean_stratified_excess':float(feature.purity_above_stratified_null.mean()),
                     'mean_stratified3_excess':float((feature.purity-feature.stratified_3_null).mean()),
                     'mean_stratified5_excess':float((feature.purity-feature.stratified_5_null).mean())}
                    atomic_json(folder/'correlations.json',correlations)
                    budget_records(feature,counts,method,enc,seed).to_csv(folder/'budgets.csv',index=False)
                    atomic_json(folder/'complete.json',{'method':method,'encoder_seed':enc,'sae_seed':seed,'model_sha256':model_sha,
                        'seconds':time.monotonic()-started,'test_accessed':False})
                    print('AUDIT COMPLETE',method,enc,seed,'seconds',round(time.monotonic()-started,1),correlations,flush=True)
    summarize()


def summarize():
    correlations=[];budgets=[]
    for completed in OUT.glob('fits/*/encoder_*/sae_*/complete.json'):
        correlations.append(json.loads((completed.parent/'correlations.json').read_text()))
        budgets.append(pd.read_csv(completed.parent/'budgets.csv'))
    if not correlations:return
    c=pd.DataFrame(correlations);c.to_csv(OUT/'fit_summary.csv',index=False)
    measures=[k for k in c.columns if k not in ['method','encoder_seed','sae_seed']]
    per=c.groupby(['method','encoder_seed'])[measures].mean();per.to_csv(OUT/'per_encoder.csv')
    per.groupby('method').agg(['mean','std','count']).to_csv(OUT/'summary.csv')
    b=pd.concat(budgets,ignore_index=True);b.to_csv(OUT/'budget_fits.csv',index=False)
    # Do not silently let large J use fewer SAE fits or fewer encoder seeds.
    numeric=[k for k in b.columns if k not in ['method','encoder_seed','sae_seed','selector','J']]
    n=b.groupby(['method','encoder_seed','selector','J']).sae_seed.nunique()
    bp=b.groupby(['method','encoder_seed','selector','J'])[numeric].mean()
    bp['sae_count']=n;bp=bp[bp.sae_count==3].reset_index();bp.to_csv(OUT/'budget_per_encoder.csv',index=False)
    bp.groupby(['method','selector','J'])[numeric].agg(['mean','std','count']).to_csv(OUT/'budget_summary.csv')
    print(c.groupby('method')[['rho_purity_utility','mean_purity','mean_global_excess','mean_stratified_excess']].mean().to_string(),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--methods',nargs='+',default=['ncl','simclr'],choices=PLAN['methods'])
    p.add_argument('--force-recompute',action='store_true',help='Recompute the same registered audit, reusing saved SAE models.')
    a=p.parse_args();main(a.methods,a.force_recompute)
