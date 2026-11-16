"""Frozen native SAE-code probes on archived train/validation embeddings.

No encoder/SAE training. Sparse scaling uses training standard deviations;
centering is absorbed by the unpenalized logistic intercept. Exact feature IDs
for J=64 reproduce the existing exploratory validation-eligible selection.
"""
from pathlib import Path
import io,json,zipfile,itertools,time,hashlib,warnings
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.sparse import csr_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,f1_score,balanced_accuracy_score,roc_auc_score,average_precision_score,recall_score,precision_score
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits

ROOT=Path.cwd();OUT=ROOT/'results/reviewer_followup_20261029/sae_code_probes'
OUT.mkdir(parents=True,exist_ok=True)
PLAN={'version':'native-frozen-SAE-probes-v1','methods':['ncl','simclr'],'encoder_seeds':[10,42,52],
 'sae_seeds':[10,42,52],'sae_variant':'baseline','feature_sets':['full_code','matched64'],
 'probe_C':1.0,'probe_max_iter':2000,'probe_tol':1e-4,'threshold':0.5,
 'standardization':'training standard deviation; sparse centering absorbed by free intercept',
 'matched64_selection':'existing validation-eligible support>=10, ranked by cumulative training activity, feature-index tie break',
 'evaluation':'exploratory inspected validation; no test; no encoder/SAE training'}
reg=OUT/'protocol.json'
if reg.exists():assert json.loads(reg.read_text())==PLAN
else:reg.write_text(json.dumps(PLAN,indent=2))
torch.set_num_threads(2)

def encode(x,state):
    sd=state['model'];mean=state['transform']['mean'];chunks=[]
    with torch.inference_mode():
        for start in range(0,len(x),2048):
            inp=torch.from_numpy(x[start:start+2048])-mean-sd['pre_bias']
            raw=F.relu(F.linear(inp,sd['encoder.weight'],sd['encoder.bias']))
            values,indices=raw.topk(24,dim=1)
            a=csr_matrix((values.numpy().ravel(),indices.numpy().ravel(),np.arange(len(inp)+1)*24),shape=(len(inp),1536))
            a.eliminate_zeros();a.sort_indices();chunks.append(a)
    from scipy.sparse import vstack
    return vstack(chunks,format='csr')

def summarize():
    records=[json.loads(p.read_text()) for p in OUT.glob('*/encoder_*/sae_*/*/metrics.json')]
    if not records:return
    frame=pd.DataFrame(records);frame.to_csv(OUT/'all_fits.csv',index=False)
    metrics=['accuracy','macro_f1','balanced_accuracy','auroc','auprc','patient_macro_slice_accuracy']
    per=frame.groupby(['method','encoder_seed','feature_set'])[metrics].mean().reset_index()
    per.to_csv(OUT/'per_encoder.csv',index=False)
    summary=per.groupby(['method','feature_set'])[metrics].agg(['mean','std','count'])
    summary.to_csv(OUT/'summary.csv')
    return per

bundle=ROOT/'data/rank_control_embeddings_v1.zip'
models=Path.home()/'Downloads/auxk_native_controls_v1_enc10-42-52_ALL_RESULTS.zip'
with zipfile.ZipFile(bundle) as emb,zipfile.ZipFile(models) as mz,threadpool_limits(2):
    manifest=json.loads(emb.read('input_manifest.json'))
    metadata=pd.read_csv(io.BytesIO(emb.read('split_manifest.csv')),dtype={'sample_id':str,'patient_id':str})
    train_meta=metadata[metadata.split=='train'].reset_index(drop=True)
    val_meta=metadata[metadata.split=='val'].reset_index(drop=True)
    yt=train_meta.has_tumor.to_numpy(dtype=int);yv=val_meta.has_tumor.to_numpy(dtype=int)
    for method,enc in itertools.product(PLAN['methods'],PLAN['encoder_seeds']):
        arrays={}
        for split,meta in [('train',train_meta),('val',val_meta)]:
            name=f'runs/{method}/encoder_{enc}/{split}.npz';content=emb.read(name)
            assert hashlib.sha256(content).hexdigest()==manifest['files'][name]['sha256']
            with np.load(io.BytesIO(content),allow_pickle=False) as d:
                assert np.array_equal(d['sample_ids'],meta.sample_id.to_numpy())
                arrays[split]=d['x'].astype(np.float32)
        for sae_seed in PLAN['sae_seeds']:
            prefix=f'fits/{method}/encoder_{enc}/baseline_r192/sae_{sae_seed}/'
            folders={feature_set:OUT/method/f'encoder_{enc}'/f'sae_{sae_seed}'/feature_set for feature_set in PLAN['feature_sets']}
            if all((p/'metrics.json').exists() for p in folders.values()):continue
            state=torch.load(io.BytesIO(mz.read(prefix+'completed_model.pt')),map_location='cpu',weights_only=True)
            assert state['protocol']['manifest_sha256']==manifest['manifest_sha256']
            assert state['protocol']['method']==method and state['protocol']['encoder_seed']==enc
            assert state['transform']['condition'] in ('native','baseline')
            assert np.allclose(state['transform']['basis'].numpy(),np.eye(192))
            assert np.allclose(state['transform']['scale'].numpy(),np.ones(192))
            started=time.monotonic();at=encode(arrays['train'],state);av=encode(arrays['val'],state)
            features=pd.read_csv(io.BytesIO(mz.read(prefix+'validation_features.csv')))
            counts=state['training_activity']['positive_counts'].numpy()
            features['training_count']=counts[features.feature.to_numpy(dtype=int)]
            selected=features.sort_values(['training_count','feature'],ascending=[False,True],kind='stable').head(64).feature.to_numpy(dtype=int)
            assert len(selected)==64
            inferred_support=np.asarray((av>0).sum(axis=0)).ravel()
            recorded_support=features.positive_support.to_numpy(dtype=int)
            assert np.max(np.abs(inferred_support[features.feature.to_numpy(dtype=int)]-recorded_support))<=2
            print('Encoded',method,enc,sae_seed,'seconds',round(time.monotonic()-started,1),flush=True)
            for feature_set,folder in folders.items():
                if (folder/'metrics.json').exists():continue
                folder.mkdir(parents=True,exist_ok=True)
                ids=np.arange(1536) if feature_set=='full_code' else selected
                scaler=StandardScaler(with_mean=False);xt=scaler.fit_transform(at[:,ids]);xv=scaler.transform(av[:,ids])
                lr=LogisticRegression(C=1.,max_iter=2000,tol=1e-4,solver='lbfgs')
                fitted=time.monotonic()
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter('always',ConvergenceWarning);lr.fit(xt,yt)
                prob=lr.predict_proba(xv)[:,list(lr.classes_).index(1)];pred=(prob>=.5).astype(int)
                correct=pd.DataFrame({'patient':val_meta.patient_id,'correct':pred==yv}).groupby('patient').correct.mean()
                record={'method':method,'encoder_seed':enc,'sae_seed':sae_seed,'feature_set':feature_set,
                    'features':len(ids),'solver_iterations':int(lr.n_iter_.max()),
                    'solver_converged':bool(lr.n_iter_.max()<2000 and not any(isinstance(w.message,ConvergenceWarning) for w in caught)),
                    'accuracy':float(accuracy_score(yv,pred)),'macro_f1':float(f1_score(yv,pred,average='macro')),
                    'balanced_accuracy':float(balanced_accuracy_score(yv,pred)),'auroc':float(roc_auc_score(yv,prob)),
                    'auprc':float(average_precision_score(yv,prob)),'tumour_recall':float(recall_score(yv,pred)),
                    'tumour_precision':float(precision_score(yv,pred,zero_division=0)),
                    'patient_macro_slice_accuracy':float(correct.mean()),'fit_seconds':time.monotonic()-fitted,
                    'split':'val','target':'slice-level tumour presence'}
                saved=val_meta[['sample_id','patient_id','has_tumor']].copy();saved['prediction']=pred;saved['probability']=prob
                saved.to_csv(folder/'predictions.csv',index=False)
                np.savez(folder/'probe.npz',coef=lr.coef_,intercept=lr.intercept_,classes=lr.classes_,
                    scale=scaler.scale_,feature_ids=ids)
                (folder/'metrics.json').write_text(json.dumps(record,indent=2));summarize()
                print('Probe',method,enc,sae_seed,feature_set,'F1',round(record['macro_f1'],4),'converged',record['solver_converged'],'fit_s',round(record['fit_seconds'],1),flush=True)
            del at,av,state
        del arrays
per=summarize()
print('Completed',len(list(OUT.glob('*/encoder_*/sae_*/*/metrics.json'))),'probe fits',flush=True)
print(per.groupby(['method','feature_set'])[['macro_f1','auroc']].mean().to_string(),flush=True)
