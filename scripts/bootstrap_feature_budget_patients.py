"""Paired whole-patient bootstrap for fixed-feature NCL/SimCLR budget curves."""
from pathlib import Path
import importlib.util,io,json,os,time,zipfile
import numpy as np
import pandas as pd
import torch

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('audit',ROOT/'scripts/feature_population_audit.py')
audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)
OUT=audit.OUT/'patient_budget_bootstrap'
PLAN={'replicates':2000,'seed':2026103007,'unit':'whole validation patient, with replacement',
      'paired':'same 37 patient draws for every method, encoder and SAE seed',
      'budgets':[8,87],'selection':'original validation-eligible training-usage ranking; IDs frozen before bootstrap',
      'models':'fixed historical baseline encoders/SAEs; no refitting or reselection',
      'top10':'recomputed from all resampled positive activations; repeated patients/slices retain multiplicity',
      'ties':'descending activation, original slice order, then repeated-copy index',
      'missing':'do not replace/drop selected IDs; retain [0,1] purity bounds if positive support <10',
      'strict_interval':'2.5/97.5 percentiles on fully scorable replicates, only if >=95% replicates defined',
      'outer_envelope':'2.5th percentile of gap lower bound to 97.5th percentile of gap upper bound; not guaranteed nominal coverage',
      'scope':'conditional exploratory validation uncertainty; excludes fitting/selection uncertainty and does not create new patients'}


def weighted_purity(order,labels,patient_codes,counts,m=10):
    """Top-M purity on an expanded cluster sample, without materializing copies."""
    order=np.asarray(order,dtype=int)
    support=counts@np.bincount(patient_codes[order],minlength=counts.shape[1])
    valid=support>=m;values=np.full(len(counts),np.nan);remaining=np.flatnonzero(valid)
    limit=min(80,len(order));classes=np.unique(labels)
    while len(remaining):
        selected=order[:limit];weights=counts[remaining][:,patient_codes[selected]]
        cumulative=np.cumsum(weights,axis=1,dtype=np.int32)
        sufficient=cumulative[:,-1]>=m
        take=np.minimum(weights,np.maximum(0,m-(cumulative-weights)))
        class_counts=np.stack([(take*(labels[selected]==c)).sum(1) for c in classes])
        values[remaining[sufficient]]=class_counts[:,sufficient].max(0)/m
        remaining=remaining[~sufficient]
        if len(remaining):
            if limit==len(order):raise AssertionError('Support and ordered-list multiplicity disagree')
            limit=min(len(order),limit*2)
    return values,valid,support


def sanity_check():
    rng=np.random.default_rng(73)
    labels=np.array([0,1,2,4,2,0,1,4,4,2,1,0])
    patients=np.repeat(np.arange(4),3)
    scores=np.array([1,1,.9,.8,.7,.7,.6,.5,.4,.3,0,0])
    ids=np.flatnonzero(scores>0);order=ids[np.argsort(-scores[ids],kind='stable')]
    counts=rng.multinomial(4,np.full(4,.25),size=25)
    actual,valid,_=weighted_purity(order,labels,patients,counts,m=5)
    for b,c in enumerate(counts):
        expanded=np.repeat(order,c[patients[order]])
        if len(expanded)>=5:
            expected=np.unique(labels[expanded[:5]],return_counts=True)[1].max()/5
            assert actual[b]==expected
        else:assert not valid[b] and np.isnan(actual[b])


def main():
    sanity_check();OUT.mkdir(parents=True,exist_ok=True);torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    inputs={'embeddings_sha256':audit.sha(audit.EMBED),'models_sha256':audit.sha(audit.MODELS)}
    original=json.loads((audit.OUT/'audit_protocol.json').read_text())['inputs']
    assert inputs['embeddings_sha256']==original['embeddings_sha256']
    assert inputs['models_sha256']==original['historical_sae_archive_sha256']
    registration={'plan':PLAN,'inputs':inputs}
    protocol=OUT/'protocol.json'
    if protocol.exists():assert json.loads(protocol.read_text())==registration
    else:audit.atomic_json(protocol,registration)
    with zipfile.ZipFile(audit.EMBED) as emb,zipfile.ZipFile(audit.MODELS) as models:
        frame=pd.read_csv(io.BytesIO(emb.read('split_manifest.csv')))
        vf=frame[frame.split=='val'].reset_index(drop=True)
        patients=np.sort(vf.patient_id.unique());assert len(patients)==37
        codes=pd.Categorical(vf.patient_id,categories=patients).codes
        labels=vf.dominant_label.to_numpy()
        rng=np.random.default_rng(PLAN['seed'])
        draws=rng.integers(0,len(patients),size=(PLAN['replicates'],len(patients)))
        counts=np.stack([np.bincount(d,minlength=len(patients)) for d in draws]).astype(np.int16)
        np.savez_compressed(OUT/'patient_draws.npz',patient_ids=patients.astype(str),draw_indices=draws,counts=counts)
        # Freeze every selected ID before computing resampled outcomes.
        selections=[]
        for method in ['ncl','simclr']:
            for enc in [10,42,52]:
                for seed in [10,42,52]:
                    f=pd.read_csv(audit.OUT/'fits'/method/f'encoder_{enc}'/f'sae_{seed}'/'features.csv')
                    eligible=f.loc[f.val_support>=10,'feature'].to_numpy(dtype=int)
                    counts_train=f.train_cumulative_count.to_numpy()
                    selected=eligible[np.lexsort((eligible,-counts_train[eligible]))][:87]
                    assert len(selected)==87
                    selections.append({'method':method,'encoder_seed':enc,'sae_seed':seed,'feature_ids':selected.tolist()})
        audit.atomic_json(OUT/'frozen_feature_ids.json',selections)
        device='cuda' if torch.cuda.is_available() else 'cpu'
        for method in ['ncl','simclr']:
            for enc in [10,42,52]:
                with np.load(io.BytesIO(emb.read(f'runs/{method}/encoder_{enc}/val.npz')),allow_pickle=False) as f:
                    assert np.array_equal(f['sample_ids'],vf.sample_id.to_numpy());x=f['x'].copy()
                for seed in [10,42,52]:
                    target=OUT/f'{method}_enc{enc}_sae{seed}.npz'
                    if target.exists():print('REUSE bootstrap',target.name,flush=True);continue
                    started=time.monotonic()
                    raw=models.read(f'fits/{method}/encoder_{enc}/baseline_r192/sae_{seed}/completed_model.pt')
                    state=torch.load(io.BytesIO(raw),map_location='cpu',weights_only=True)
                    selection=next(s for s in selections if (s['method'],s['encoder_seed'],s['sae_seed'])==(method,enc,seed))
                    ids=np.asarray(selection['feature_ids']);a=audit.activations(x,state,device).toarray()
                    feature=pd.read_csv(audit.OUT/'fits'/method/f'encoder_{enc}'/f'sae_{seed}'/'features.csv')
                    purity=np.empty((len(counts),87));scorable=np.empty((len(counts),87),dtype=bool)
                    for j,feature_id in enumerate(ids):
                        positive=np.flatnonzero(a[:,feature_id]>0)
                        order=positive[np.argsort(-a[positive,feature_id],kind='stable')]
                        original=np.unique(labels[order[:10]],return_counts=True)[1].max()/10
                        assert np.isclose(original,feature.loc[feature_id,'purity'],atol=1e-12)
                        purity[:,j],scorable[:,j],_=weighted_purity(order,labels,codes,counts)
                    temp=target.with_suffix('.tmp.npz')
                    np.savez_compressed(temp,purity=purity,scorable=scorable,feature_ids=ids,
                                        point_purity=feature.loc[ids,'purity'].to_numpy())
                    os.replace(temp,target)
                    print('BOOTSTRAP COMPLETE',method,enc,seed,'seconds',round(time.monotonic()-started,1),flush=True)
    summarize()


def summarize():
    methods={};js=np.arange(1,88)
    for method in ['ncl','simclr']:
        lower=[];upper=[];complete=[];point=[];coverage=[]
        for enc in [10,42,52]:
            for seed in [10,42,52]:
                with np.load(OUT/f'{method}_enc{enc}_sae{seed}.npz',allow_pickle=False) as f:
                    p=f['purity'];valid=f['scorable'];n=np.cumsum(valid,axis=1)
                    lo=np.cumsum(np.nan_to_num(p,nan=0.),axis=1)/js
                    lower.append(lo);upper.append(lo+(js-n)/js);complete.append(n==js)
                    coverage.append(n/js);point.append(np.cumsum(f['point_purity'])/js)
        methods[method]={'lower':np.mean(lower,axis=0),'upper':np.mean(upper,axis=0),
                         'complete':np.all(complete,axis=0),'point':np.mean(point,axis=0),
                         'coverage':np.mean(coverage,axis=0)}
    n,s=methods['ncl'],methods['simclr']
    lower=n['lower']-s['upper'];upper=n['upper']-s['lower']
    complete=n['complete']&s['complete'];point=n['point']-s['point']
    rows=[]
    for j in range(8,88):
        k=j-1;defined=complete[:,k];strict=lower[defined,k]
        lo,hi=np.quantile(strict,[.025,.975]) if defined.mean()>=.95 else (np.nan,np.nan)
        rows.append({'J':j,'point_gap':point[k],
                     'strict_complete_fraction':float(defined.mean()),'strict_ci_lower':lo,'strict_ci_upper':hi,
                     'outer_percentile_lower':float(np.quantile(lower[:,k],.025)),
                     'outer_percentile_upper':float(np.quantile(upper[:,k],.975)),
                     'mean_ncl_coverage':float(n['coverage'][:,k].mean()),
                     'mean_simclr_coverage':float(s['coverage'][:,k].mean())})
    result=pd.DataFrame(rows);result.to_csv(OUT/'curve_intervals.csv',index=False)
    np.savez_compressed(OUT/'paired_macro_replicates.npz',J=js,gap_lower=lower,gap_upper=upper,fully_scorable=complete)
    report={'status':'complete','patients':37,'replicates':PLAN['replicates'],'dictionaries':18,
            'paired_patient_resampling':True,'fitted_models_and_feature_ids_fixed':True,
            'literal_expansion_toy_check':'passed','observed_points_match_saved_feature_purity':'passed',
            'intervals_conditional_on_models_and_original_selection':True,
            'patient_intervals_added_for':'Fig 2B budget gap only; not new correlation or null results',
            'selected_budgets':result[result.J.isin([8,32,64,80,87])].replace({np.nan:None}).to_dict('records')}
    audit.atomic_json(OUT/'SUMMARY.json',report)
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
