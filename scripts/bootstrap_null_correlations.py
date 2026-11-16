"""Paired patient bootstrap of all-eligible null excess and feature correlations.

Historical fitted states only. Expanded patient copies are distinct permutation
positions; the conditional label mix and top-ten lists are recomputed each draw.
"""
from pathlib import Path
import importlib.util, io, json, os, time, zipfile
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('audit',ROOT/'scripts/feature_population_audit.py')
audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)
OUT=audit.OUT/'patient_null_correlation_bootstrap'
B=1000;R=1000
PLAN={'replicates':B,'null_permutations_per_draw':R,'null_seed':2026103019,
      'draws':'first 1000 paired whole-patient draws from patient_budget_bootstrap/patient_draws.npz',
      'models':'fixed historical models; training-fitted AUROC orientations fixed',
      'population':'full frozen dictionary; reapply positive support >=10 in every draw, including originally ineligible units',
      'strata':'original four-bin covariate cutpoints fixed; cell label counts recomputed on expanded patient copies',
      'null':'shuffle expanded slice labels without replacement within each stratum; same permutations across models',
      'top10':'recomputed with multiplicity; descending activation, original slice index, copy index ties',
      'interval':'pointwise percentile 2.5/97.5, conditional on fitted models and original analysis choices',
      'aggregation':'average SAE seeds within encoder, then equal-weight encoder average; no resampling encoder seeds',
      'scope':'exploratory reused validation cohort; excludes model/selection uncertainty; finite inner Monte Carlo error remains'}


def top_lists(scores,patient_codes,counts):
    """Expanded-index top tens and eligibility for each patient draw."""
    weights=counts[:,patient_codes].astype(np.int16)
    offsets=np.cumsum(weights,axis=1,dtype=np.int32)-weights
    tops=np.zeros((len(counts),scores.shape[1],10),dtype=np.int32)
    valid=np.zeros(tops.shape[:2],bool)
    for j in range(scores.shape[1]):
        ids=np.flatnonzero(scores[:,j]>0)
        if not len(ids):continue
        ids=ids[np.argsort(-scores[ids,j],kind='stable')]
        w=weights[:,ids];c=np.cumsum(w,axis=1,dtype=np.int32)
        valid[:,j]=c[:,-1]>=10
        for k in range(10):
            pos=np.argmax(c>k,axis=1)
            original=ids[pos];before=c[np.arange(len(c)),pos]-w[np.arange(len(w)),pos]
            tops[:,j,k]=offsets[np.arange(len(c)),original]+k-before
        tops[~valid[:,j],j]=0
    return tops,valid


def pair_matrices(scores,y,codes,npats):
    """Exact per-patient positive/negative score comparison counts, ties half."""
    h=np.empty((scores.shape[1],npats*npats),np.float32)
    for j in range(scores.shape[1]):
        _,groups=np.unique(scores[:,j],return_inverse=True)
        n=groups.max()+1
        pos=np.zeros((n,npats));neg=np.zeros_like(pos)
        np.add.at(pos,(groups[y==1],codes[y==1]),1)
        np.add.at(neg,(groups[y==0],codes[y==0]),1)
        h[j]=(pos.T@(np.cumsum(neg,axis=0)-.5*neg)).ravel()
    return h


def verify():
    from sklearn.metrics import roc_auc_score
    codes=np.repeat(np.arange(3),5);y=np.array([0,1,1,0,1]*3)
    scores=np.array([0,0,1,1,2,2,1,0,0,3,1,2,0,1,3.])[:,None]
    counts=np.array([[1,1,1],[2,0,1],[0,3,0]])
    h=pair_matrices(scores,y,codes,3)
    outer=(counts[:,:,None]*counts[:,None,:]).reshape(3,-1)
    den=(counts@np.bincount(codes[y==1],minlength=3))*(counts@np.bincount(codes[y==0],minlength=3))
    auc=outer@h.T/den[:,None]
    tops,valid=top_lists(scores,codes,counts)
    for b,w in enumerate(counts):
        expanded=np.repeat(np.arange(len(y)),w[codes])
        assert np.isclose(auc[b,0],roc_auc_score(y[expanded],scores[expanded,0]))
        order=np.flatnonzero(scores[expanded,0]>0)
        order=order[np.argsort(-scores[expanded[order],0],kind='stable')]
        assert valid[b,0]==(len(order)>=10)
        if valid[b,0]:assert np.array_equal(tops[b,0],order[:10])


def main():
    verify();OUT.mkdir(parents=True,exist_ok=True);torch.set_num_threads(2)
    registration={'plan':PLAN,'input_hashes':{
        'embeddings':audit.sha(audit.EMBED),'historical_states':audit.sha(audit.MODELS),
        'covariates':audit.sha(audit.COV),
        'draws':audit.sha(audit.OUT/'patient_budget_bootstrap/patient_draws.npz')}}
    protocol=OUT/'protocol.json'
    if protocol.exists():
        old=json.loads(protocol.read_text())
        assert old in (PLAN,registration),'Bootstrap protocol or inputs changed; use a new output folder'
    audit.atomic_json(protocol,registration)
    draws=np.load(audit.OUT/'patient_budget_bootstrap/patient_draws.npz',allow_pickle=False)
    counts=draws['counts'][:B].astype(np.int32);patients=draws['patient_ids']
    cov=pd.read_csv(audit.COV);cells,_=audit.strata(cov,4)
    with zipfile.ZipFile(audit.EMBED) as emb,zipfile.ZipFile(audit.MODELS) as models:
        frame=pd.read_csv(io.BytesIO(emb.read('split_manifest.csv')))
        vf=frame[frame.split=='val'].reset_index(drop=True)
        assert np.array_equal(vf.sample_id.to_numpy(),cov.sample_id.to_numpy())
        codes=pd.Categorical(vf.patient_id,categories=patients).codes
        labels=vf.dominant_label.to_numpy();classes=np.array([0,1,2,4]);y=vf.has_tumor.to_numpy().astype(int)
        expanded=[np.repeat(np.arange(len(vf)),w[codes]) for w in counts]
        maxn=max(map(len,expanded));cache=ROOT/'tmp/null_bootstrap_permutations.npy'
        cache.parent.mkdir(exist_ok=True)
        metadata=cache.with_suffix('.json')
        cache_protocol={'shape':[B,R,maxn],'null_seed':PLAN['null_seed'],'inputs':registration['input_hashes']}
        if not cache.exists() or not metadata.exists() or json.loads(metadata.read_text())!=cache_protocol:
            partial=cache.with_name('null_bootstrap_permutations.partial.npy')
            perm=np.lib.format.open_memmap(partial,mode='w+',dtype=np.uint8,shape=(B,R,maxn))
            rng=np.random.default_rng(PLAN['null_seed'])
            for b,ids in enumerate(expanded):
                lab=labels[ids];cell=cells[ids]
                for c in np.unique(cell):
                    ii=np.flatnonzero(cell==c)
                    perm[b,:,ii]=np.stack([rng.permutation(lab[ii]) for _ in range(R)]).T
                if b%100==0:print('NULL DRAWS',b,flush=True)
            perm.flush();del perm
            os.replace(partial,cache);audit.atomic_json(metadata,cache_protocol)
        perm=np.load(cache,mmap_mode='r')
        assert perm.shape==(B,R,maxn)
        # Check conditional label counts in independently selected draws/permutations.
        for b in [0,19,999]:
            for c in np.unique(cells):
                ii=np.flatnonzero(cells[expanded[b]]==c)
                for r in [0,499,999]:
                    assert np.array_equal(np.bincount(perm[b,r,ii],minlength=5),np.bincount(labels[expanded[b][ii]],minlength=5))
        device='cuda' if torch.cuda.is_available() else 'cpu'
        outer=(counts[:,:,None]*counts[:,None,:]).reshape(B,-1)
        den=(counts@np.bincount(codes[y==1],minlength=37))*(counts@np.bincount(codes[y==0],minlength=37))
        for method in ['ncl','simclr','supervised']:
            for enc in [10,42,52]:
                if method=='supervised':x=np.load(audit.OUT/f'supervised_embeddings/encoder_{enc}/val.npz')['x']
                else:x=np.load(io.BytesIO(emb.read(f'runs/{method}/encoder_{enc}/val.npz')))['x']
                for seed in [10,42,52]:
                    target=OUT/f'{method}_enc{enc}_sae{seed}.npz'
                    if target.exists():print('REUSE',target.name,flush=True);continue
                    start=time.monotonic()
                    if method=='supervised':state=torch.load(audit.OUT/f'supervised_models/encoder_{enc}/sae_{seed}/completed_model.pt',map_location='cpu',weights_only=True)
                    else:state=torch.load(io.BytesIO(models.read(f'fits/{method}/encoder_{enc}/baseline_r192/sae_{seed}/completed_model.pt')),map_location='cpu',weights_only=True)
                    scores=audit.activations(x,state,device).toarray()
                    f=pd.read_csv(audit.OUT/f'fits/{method}/encoder_{enc}/sae_{seed}/features.csv')
                    tops,valid=top_lists(scores,codes,counts)
                    h=pair_matrices(scores,y,codes,37)
                    auc=(outer@h.T)/den[:,None]
                    original_den=int(y.sum())*int((1-y).sum())
                    assert np.allclose(h.sum(1)/original_den,f.val_auc_positive_activation,atol=1e-7)
                    pt,ok=top_lists(scores,codes,np.ones((1,37),dtype=np.int32))
                    assert np.array_equal(ok[0],f.val_support.to_numpy()>=10)
                    observed=labels[pt[0,ok[0]]]
                    observed_purity=np.stack([(observed==c).sum(1) for c in classes]).max(0)/10
                    assert np.allclose(observed_purity,f.purity.to_numpy()[ok[0]])
                    auc=np.where(f.train_fitted_orientation.to_numpy()[None,:]==1,auc,1-auc)
                    purity=np.full(valid.shape,np.nan);null=np.full(valid.shape,np.nan);half=np.full(valid.shape,np.nan)
                    for b in range(B):
                        ids=np.flatnonzero(valid[b]);top=tops[b,ids]
                        observed=labels[expanded[b][top]]
                        purity[b,ids]=np.stack([(observed==c).sum(1) for c in classes]).max(0)/10
                        # One draw at a time bounds GPU memory even for full dictionaries.
                        p=torch.as_tensor(np.array(perm[b]),device=device)
                        t=torch.as_tensor(top,device=device,dtype=torch.long)
                        shuffled=p[:,t]
                        mx=torch.zeros(shuffled.shape[:2],device=device,dtype=torch.uint8)
                        for c in classes:mx=torch.maximum(mx,(shuffled==int(c)).sum(2).to(torch.uint8))
                        values=mx.float().mean(0).cpu().numpy()/10
                        null[b,ids]=values
                        half[b,ids]=mx[:R//2].float().mean(0).cpu().numpy()/10
                    raw=np.array([audit.safe_rho(purity[b],auc[b]) for b in range(B)],dtype=float)
                    excess=np.array([audit.safe_rho(purity[b]-null[b],auc[b]) for b in range(B)],dtype=float)
                    excess_half=np.array([audit.safe_rho(purity[b]-half[b],auc[b]) for b in range(B)],dtype=float)
                    np.savez_compressed(target,raw_rho=raw,excess_rho=excess,excess_rho_half=excess_half,
                        excess_mean=np.nanmean(purity-null,axis=1),excess_mean_half=np.nanmean(purity-half,axis=1),
                        raw_mean=np.nanmean(purity,axis=1),eligible_count=valid.sum(1),
                        state_sha256=np.array(audit.sha(audit.OUT/f'supervised_models/encoder_{enc}/sae_{seed}/completed_model.pt') if method=='supervised'
                            else __import__('hashlib').sha256(models.read(f'fits/{method}/encoder_{enc}/baseline_r192/sae_{seed}/completed_model.pt')).hexdigest()),
                        original_eligible_retained=(valid & (f.val_support.to_numpy()[None,:]>=10)).sum(1))
                    print('DONE',method,enc,seed,'seconds',round(time.monotonic()-start,1),flush=True)
    summarize()
    verify_outputs()


def summarize():
    point=pd.read_csv(audit.OUT/'fit_summary.csv');rows=[];macro={}
    for method in ['ncl','simclr','supervised']:
        fits=[np.load(OUT/f'{method}_enc{e}_sae{s}.npz') for e in [10,42,52] for s in [10,42,52]]
        macro[method]={k:np.mean([f[k] for f in fits],axis=0) for k in ['excess_mean','excess_mean_half','raw_rho','excess_rho','excess_rho_half']}
        for metric in ['excess_mean','raw_rho','excess_rho']:
            v=macro[method][metric];good=np.isfinite(v)
            lo,hi=np.quantile(v[good],[.025,.975]) if good.mean()>=.95 else (np.nan,np.nan)
            rows.append({'method':method,'metric':metric,'lower':lo,'upper':hi,'finite_fraction':good.mean(),
                         'point_estimate':point.loc[point.method==method,{'excess_mean':'mean_stratified_excess','raw_rho':'rho_purity_utility','excess_rho':'rho_stratified_excess_utility'}[metric]].mean(),
                         'bootstrap_mean':np.nanmean(v),'minimum_eligible':int(min(f['eligible_count'].min() for f in fits)),
                         'maximum_eligible':int(max(f['eligible_count'].max() for f in fits))})
    gap=macro['ncl']['excess_mean']-macro['simclr']['excess_mean']
    half=macro['ncl']['excess_mean_half']-macro['simclr']['excess_mean_half']
    rows.append({'method':'ncl_minus_simclr','metric':'excess_mean','lower':np.quantile(gap,.025),
        'upper':np.quantile(gap,.975),'finite_fraction':np.isfinite(gap).mean(),'bootstrap_mean':gap.mean(),
        'point_estimate':point.loc[point.method=='ncl','mean_stratified_excess'].mean()-point.loc[point.method=='simclr','mean_stratified_excess'].mean()})
    result=pd.DataFrame(rows);result.to_csv(OUT/'intervals.csv',index=False)
    sensitivity={'gap_max_abs_change_half_vs_full':float(np.max(np.abs(gap-half))),
                 'gap_interval_half':np.quantile(half,[.025,.975]).tolist(),
                 'gap_interval_full':np.quantile(gap,[.025,.975]).tolist(),
                 'excess_rho_interval_half':{m:np.quantile(macro[m]['excess_rho_half'],[.025,.975]).tolist() for m in macro}}
    audit.atomic_json(OUT/'SUMMARY.json',{'status':'complete','plan':PLAN,'intervals':rows,
        'inner_Monte_Carlo_sensitivity':sensitivity,'original_point_estimates':'unchanged original 1000-permutation audit; intervals recompute all-eligible functional',
        'bootstrap_bias_note':'resampling duplicates patients/slices and changes eligibility and top-ten order statistics; percentile interval need not center on original estimate'})
    print(result.to_string(index=False),flush=True)


def verify_outputs():
    """Independently replay unresampled metrics and bind all output states."""
    import hashlib
    verify();torch.set_num_threads(2)
    with zipfile.ZipFile(audit.EMBED) as emb,zipfile.ZipFile(audit.MODELS) as models:
        vf=pd.read_csv(io.BytesIO(emb.read('split_manifest.csv'))).query("split=='val'").reset_index(drop=True)
        patients=np.sort(vf.patient_id.unique());codes=pd.Categorical(vf.patient_id,categories=patients).codes
        labels=vf.dominant_label.to_numpy();y=vf.has_tumor.to_numpy().astype(int)
        for method in ['ncl','simclr','supervised']:
            for enc in [10,42,52]:
                x=(np.load(audit.OUT/f'supervised_embeddings/encoder_{enc}/val.npz')['x'] if method=='supervised'
                   else np.load(io.BytesIO(emb.read(f'runs/{method}/encoder_{enc}/val.npz')))['x'])
                for seed in [10,42,52]:
                    raw=(audit.OUT/f'supervised_models/encoder_{enc}/sae_{seed}/completed_model.pt').read_bytes() if method=='supervised' else models.read(f'fits/{method}/encoder_{enc}/baseline_r192/sae_{seed}/completed_model.pt')
                    state=torch.load(io.BytesIO(raw),map_location='cpu',weights_only=True)
                    scores=audit.activations(x,state,'cuda' if torch.cuda.is_available() else 'cpu').toarray()
                    f=pd.read_csv(audit.OUT/f'fits/{method}/encoder_{enc}/sae_{seed}/features.csv')
                    h=pair_matrices(scores,y,codes,37)
                    assert np.allclose(h.sum(1)/(y.sum()*(1-y).sum()),f.val_auc_positive_activation,atol=1e-7)
                    top,ok=top_lists(scores,codes,np.ones((1,37),np.int32))
                    assert np.array_equal(ok[0],f.val_support>=10)
                    observed=labels[top[0,ok[0]]]
                    p=np.stack([(observed==c).sum(1) for c in [0,1,2,4]]).max(0)/10
                    assert np.allclose(p,f.purity.to_numpy()[ok[0]])
                    target=OUT/f'{method}_enc{enc}_sae{seed}.npz'
                    with np.load(target) as z:values={k:z[k] for k in z.files}
                    assert all(values[k].shape==(B,) for k in ['raw_rho','excess_rho','excess_mean','eligible_count'])
                    values['state_sha256']=np.array(hashlib.sha256(raw).hexdigest())
                    np.savez_compressed(target,**values)
    registration={'plan':PLAN,'input_hashes':{'embeddings':audit.sha(audit.EMBED),
        'historical_states':audit.sha(audit.MODELS),'covariates':audit.sha(audit.COV),
        'draws':audit.sha(audit.OUT/'patient_budget_bootstrap/patient_draws.npz')}}
    audit.atomic_json(OUT/'protocol.json',registration)
    cache=ROOT/'tmp/null_bootstrap_permutations.npy'
    perm=np.load(cache,mmap_mode='r');cov=pd.read_csv(audit.COV);cells,_=audit.strata(cov,4)
    counts=np.load(audit.OUT/'patient_budget_bootstrap/patient_draws.npz')['counts'][:B]
    assert np.all(counts.sum(1)==37)
    for b,w in enumerate(counts):
        expanded=np.repeat(np.arange(len(vf)),w[codes])
        for c in np.unique(cells):
            ids=np.flatnonzero(cells[expanded]==c)
            expected=np.bincount(labels[expanded[ids]],minlength=5)
            for r in [0,499,999]:assert np.array_equal(np.bincount(perm[b,r,ids],minlength=5),expected)
    audit.atomic_json(cache.with_suffix('.json'),{'shape':list(perm.shape),'null_seed':PLAN['null_seed'],'inputs':registration['input_hashes']})
    audit.atomic_json(OUT/'VERIFICATION.json',{'status':'passed','dictionaries':27,
        'original_full_dictionary_AUROC_matches_archived_sparse_implementation':True,
        'original_eligibility_and_purity_match_archived_tables':True,
        'literal_expansion_top10_and_sklearn_weighted_AUROC_toys':True,
        'null_cell_label_counts_checked':'all 1000 patient draws; permutations 0,499,999',
        'patient_draw_totals':37,'state_sha256_bound_to_each_fit':True})
    print('VERIFIED all 27 dictionaries and 1000 patient draws',flush=True)


if __name__=='__main__':
    import sys
    if '--summarize' in sys.argv:summarize()
    elif '--verify' in sys.argv:verify_outputs()
    else:main()
