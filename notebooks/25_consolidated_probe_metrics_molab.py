# /// script
# requires-python = ">=3.11"
# dependencies = ["marimo>=0.24,<1","numpy>=1.26,<3","pandas>=2,<4",
#   "torch>=2.7","scikit-learn>=1.5,<2","psutil>=5.9",
#   "huggingface-hub>=0.34,<3","python-dotenv>=1,<2"]
# ///
import marimo
__generated_with = "0.24.1"
app = marimo.App(width="full")

@app.cell
def _():
    import marimo as mo
    return (mo,)

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Consolidated probe metrics for archived MRI representations

    Refit ONLY small logistic-regression probes. No encoder or SAE training.
    Input: SAME `rank_control_embeddings_v1.zip`, SAME private HF repo and read-only
    HF_TOKEN. No results/model ZIP upload needed. A CPU runtime is sufficient.
    New output folder: `probe_metrics_v1`. Default: all encoder seeds 10/42/52.

    NCL and SimCLR in five conditions: native, PCA rank 8, rank-8 training-spectrum
    matching, >=95% and >=99% training-variance PCA. Total 30 probes; 10 per encoder
    seed if split into sessions. Training-only transforms reproduce the previous
    control recipes; dimensions never use validation outcomes. AuxK does not modify
    backbone embeddings, so baseline/AuxK do not require duplicate raw-embedding probes.
    Supervised is outside this NCL-vs-SimCLR input bundle.

    Same StandardScaler + LogisticRegression(C=1,max_iter=2000), fitted on training
    only. Predict uses the unchanged classifier decision rule; no threshold tuning,
    class-weight changes or hyperparameter search. Save solver iterations and flags.

    Metrics: accuracy, macro-F1 over labels [0,1], balanced accuracy, tumour-class
    precision/recall/F1, specificity, AUROC, tumour-class average precision (AUPRC),
    patient-macro accuracy, and TN/FP/FN/TP confusion counts. Predictions and positive
    class probabilities are saved per slice, as are probe coefficients, standardizer
    and PCA transform. These are slice tumour-presence labels, not patient diagnoses.
    AUPRC here means average precision, not trapezoidal integration of a PR curve.
    Precision/F1 with a zero denominator use zero. Balanced accuracy/AUROC are undefined
    when a resample has only one observed class; those cases are recorded as missing.

    1,000 paired WHOLE-PATIENT bootstrap resamples retain all slices for each drawn
    patient, with the same draws for every representation. Intervals are conditional
    on the fixed encoders/training data and fitted probes. Across-encoder summaries
    require all three seeds; also report encoder mean/SD separately. Do not count SAE
    seeds as new probe replications. Validation has already been inspected; these
    intervals are exploratory and do not establish held-out or clinical performance.

    Apply settings -> Download / verify embedding bundle -> Start / continue probes.
    Progress refreshes every 10 seconds. Complete cases are reused. Stop requests
    finish the current probe/bootstrap case. Export the full results ZIP before
    closing the runtime. Local Jupyter Run All starts all 30 probes by default.
    """)
    return

@app.cell(hide_code=True)
def _():
    import textwrap as _textwrap
    RUNTIME_SOURCE = _textwrap.dedent(r'''
    """Expanded-cohort rank controls using archived CLS embeddings only."""
    from pathlib import Path
    import gc,hashlib,itertools,json,os,random,sys,time,zipfile
    import numpy as np
    import pandas as pd
    import torch
    from torch import nn
    from torch.nn import functional as F

    DEFAULT_PLAN={'version':'expanded-rank-control-v1','methods':['ncl','simclr'],
     'encoder_seeds':[10,42,52],'sae_seeds':[10,42,52],'ranks':[8],
     'conditions':['native','pca_rank','spectrum_matched'],'dictionary_size':1536,'k':24,
     'epochs':30,'batch_size':512,'lr':0.0005,'gradient_clip':1.0,'top_m':10,
     'permutations':100,'probe_C':1.0,'probe_max_iter':2000,
     'spectrum_reference':'NCL training covariance, same encoder seed',
     'save_completed_models':True,'per_epoch_checkpoints':False,
     'scope':'exploratory validation follow-up; no test images/features or spatial maps'}

    def sha(path):
        h=hashlib.sha256()
        with open(path,'rb') as f:
            for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
        return h.hexdigest()

    def replace_file(source,target):
        # Windows readers may briefly deny replacement while the live UI reads CSVs.
        for attempt in range(41):
            try:
                os.replace(source,target);return
            except PermissionError:
                if os.name!='nt' or attempt==40:raise
                time.sleep(.05)

    def environment():
        import sklearn
        return {'python':sys.version,'torch':str(torch.__version__),'numpy':np.__version__,
            'pandas':pd.__version__,'scikit_learn':sklearn.__version__,'cuda':torch.version.cuda,
            'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}

    def atomic_json(path,value):
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
        tmp=path.with_name(path.name+'.'+str(os.getpid())+'.tmp')
        tmp.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8');replace_file(tmp,path)

    def atomic_csv(frame,path):
        path=Path(path);tmp=path.with_name(path.name+'.'+str(os.getpid())+'.tmp')
        frame.to_csv(tmp,index=False);replace_file(tmp,path)

    def bundle_download(repo_id,filename,destination,expected_sha):
        target=Path(destination).expanduser().resolve();target.parent.mkdir(parents=True,exist_ok=True)
        if target.is_file():
            if sha(target)!=expected_sha:raise ValueError('Input bundle SHA256 mismatch')
            return str(target)
        token=os.environ.get('HF_TOKEN','').strip()
        if not token:
            from dotenv import dotenv_values
            token=str(dotenv_values('.env').get('HF_TOKEN') or '').strip()
        if not token:raise RuntimeError('Set read-only HF_TOKEN in runtime secrets or .env')
        from huggingface_hub import hf_hub_download
        downloaded=Path(hf_hub_download(repo_id=repo_id,repo_type='dataset',filename=filename,token=token,local_dir=str(target.parent))).resolve()
        if sha(downloaded)!=expected_sha:raise ValueError('Input bundle SHA256 mismatch')
        if downloaded!=target:os.replace(downloaded,target)
        return str(target)

    class Inputs:
        def __init__(self,bundle,cache,expected_sha):
            self.bundle=Path(bundle).resolve();self.cache=Path(cache).resolve();self.cache.mkdir(parents=True,exist_ok=True)
            signature={'bytes':self.bundle.stat().st_size,'mtime_ns':self.bundle.stat().st_mtime_ns,'sha256':expected_sha}
            receipt=self.cache/'bundle_verified.json'
            if not receipt.exists() or json.loads(receipt.read_text())!=signature:
                if sha(self.bundle)!=expected_sha:raise ValueError('Wrong/corrupt embedding bundle')
                with zipfile.ZipFile(self.bundle) as z:
                    self.registration=json.loads(z.read('input_manifest.json'))
                    if self.registration['format']!='rank-control-embeddings-v1':raise ValueError('Wrong bundle format')
                    for name,identity in self.registration['files'].items():
                        target=(self.cache/name).resolve()
                        if not target.is_relative_to(self.cache):raise ValueError('Unsafe archive path')
                        target.parent.mkdir(parents=True,exist_ok=True)
                        temp=target.with_name(target.name+'.tmp');h=hashlib.sha256();count=0
                        with z.open(name) as src,open(temp,'wb') as dst:
                            while True:
                                b=src.read(8*1024*1024)
                                if not b:break
                                h.update(b);count+=len(b);dst.write(b)
                        if count!=identity['bytes'] or h.hexdigest()!=identity['sha256']:raise ValueError('Corrupt embedding file: '+name)
                        os.replace(temp,target)
                    atomic_json(self.cache/'input_manifest.json',self.registration)
                atomic_json(receipt,signature)
            self.registration=json.loads((self.cache/'input_manifest.json').read_text())
            if sha(self.cache/'split_manifest.csv')!=self.registration['manifest_sha256']:raise ValueError('Manifest changed')
            self.frame=pd.read_csv(self.cache/'split_manifest.csv',dtype={'patient_id':str,'sample_id':str})
            if self.frame.duplicated(['patient_id','slice_idx']).any() or self.frame.groupby('patient_id').split.nunique().max()!=1:
                raise ValueError('Duplicate samples or crossing patient partitions')
            self.pca_cache={}

        def load(self,mode,seed,split):
            if split not in ('train','val'):raise ValueError('No test features available')
            prefix=self.cache/'runs'/mode/f'encoder_{seed}'
            relative=f'runs/{mode}/encoder_{seed}/{split}.npz'
            if sha(prefix/f'{split}.npz')!=self.registration['files'][relative]['sha256']:raise ValueError('Cached input file changed')
            meta=json.loads((prefix/f'{split}_identity.json').read_text())
            expected=self.frame.loc[self.frame.split==split,'sample_id'].astype(str).to_numpy()
            if meta['manifest']!=self.registration['manifest_sha256'] or meta['samples']!=hashlib.sha256('\n'.join(expected).encode()).hexdigest():
                raise ValueError('Embedding identity mismatch')
            with np.load(prefix/f'{split}.npz',allow_pickle=False) as data:
                x=data['x'].astype(np.float32);ids=data['sample_ids']
            if not np.array_equal(ids,expected) or x.shape!=(len(expected),192) or not np.isfinite(x).all():raise ValueError('Invalid embedding cache')
            return x

        def pca(self,mode,seed,x=None):
            key=(mode,seed)
            if key not in self.pca_cache:
                x=self.load(mode,seed,'train') if x is None else x
                mean=x.astype(np.float64).mean(0);cov=covariance(x)
                eig,basis=np.linalg.eigh(cov);order=np.argsort(eig)[::-1]
                self.pca_cache[key]={'mean':mean.astype(np.float32),'eigenvalues':np.maximum(eig[order],0),'basis':basis[:,order]}
            return self.pca_cache[key]

    def covariance(x):
        mean=np.mean(x,axis=0,dtype=np.float64);cov=np.zeros((x.shape[1],x.shape[1]),dtype=np.float64)
        for start in range(0,len(x),8192):
            centered=x[start:start+8192].astype(np.float64)-mean;cov+=centered.T@centered
        return cov/max(1,len(x)-1)

    def spectrum_stats(eigenvalues):
        v=np.maximum(eigenvalues,0);total=float(v.sum());p=v/total if total else v
        p=p[p>0]
        return {'variance':total,'effective_rank':float(np.exp(-np.sum(p*np.log(p)))) if total else 0,
                'participation_ratio':float(total**2/np.sum(v*v)) if total else 0}

    def build_transform(inputs,mode,seed,condition,rank,train,val):
        info=inputs.pca(mode,seed,train);mean=info['mean'];centered_train=train-mean;centered_val=val-mean
        if condition=='native':
            basis=np.eye(192);scale=np.ones(192);rank=192
            ztrain=centered_train;zval=centered_val;target=None;pca_ev=1.0
        else:
            basis=info['basis'][:,:rank];values=info['eigenvalues'][:rank]
            if np.min(values)<=info['eigenvalues'][0]*1e-12:raise ValueError('Requested rank contains numerically singular dimensions')
            scale=np.ones(rank)
            target=None
            if condition=='spectrum_matched':
                target=inputs.pca('ncl',seed)['eigenvalues'][:rank]
                if np.min(target)<=0:raise ValueError('NCL reference covariance is singular')
                scale=np.sqrt(target/values)
            elif condition!='pca_rank':raise ValueError('Unknown control condition')
            ztrain=((centered_train@basis)*scale)@basis.T;zval=((centered_val@basis)*scale)@basis.T
            unscaled=(centered_val@basis)@basis.T
            pca_ev=1-float(np.sum((centered_val.astype(np.float64)-unscaled)**2))/float(np.sum(centered_val.astype(np.float64)**2))
        ztrain=np.asarray(ztrain,dtype=np.float32);zval=np.asarray(zval,dtype=np.float32)
        train_eig=np.maximum(np.linalg.eigvalsh(covariance(ztrain)),0)[::-1]
        val_eig=np.maximum(np.linalg.eigvalsh(covariance(zval)),0)[::-1]
        diagnostics={'condition':condition,'rank':rank,'PCA_only_validation_EV':pca_ev,
            'PCA_train_variance_retained':float(info['eigenvalues'][:rank].sum()/info['eigenvalues'].sum()),
            'train':spectrum_stats(train_eig),'validation':spectrum_stats(val_eig),
            'matching_scope':'train only; validation spectra are measured, never forced'}
        if target is not None:
            error=float(np.linalg.norm(train_eig[:rank]-target)/np.linalg.norm(target))
            if error>1e-4:raise ValueError('Training spectrum matching failed')
            diagnostics.update(target=spectrum_stats(target),relative_training_spectrum_error=error)
        recipe={'mean':mean,'basis':basis.astype(np.float32),'scale':scale.astype(np.float32),'condition':condition,'rank':rank}
        return ztrain,zval,recipe,diagnostics

    class SAE(nn.Module):
        def __init__(self,width,dictionary,k):
            super().__init__();self.k=k;self.pre_bias=nn.Parameter(torch.zeros(width))
            self.encoder=nn.Linear(width,dictionary);self.decoder=nn.Linear(dictionary,width,bias=False)
        def forward(self,x):
            raw=F.relu(self.encoder(x-self.pre_bias));v,i=raw.topk(self.k,dim=-1)
            acts=torch.zeros_like(raw).scatter_(-1,i,v)
            return self.decoder(acts)+self.pre_bias,acts

    def seed_all(seed):
        random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
        if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False

    def purity_scores(top,labels,classes):
        if len(top)==0:return np.empty(0),np.empty(0)
        counts=np.stack([(labels[top]==label).sum(axis=1) for label in classes],axis=1)
        p=counts/top.shape[1];entropy=-np.sum(np.where(p>0,p*np.log(np.maximum(p,1e-300)),0),axis=1)
        return p.max(axis=1),entropy

    def selectivity(acts,labels,patients,m,permutations,seed):
        support=(acts>0).sum(0);alive=np.flatnonzero(support);eligible=np.flatnonzero(support>=m)
        top=np.argsort(acts[:,eligible],axis=0)[-m:].T if len(eligible) else np.empty((0,m),int)
        classes=np.unique(labels);purity,entropy=purity_scores(top,labels,classes)
        patient_top=[];patient_features=[]
        for feature in eligible:
            chosen=[];seen=set()
            for index in np.argsort(acts[:,feature])[::-1]:
                if acts[index,feature]<=0:break
                if patients[index] in seen:continue
                seen.add(patients[index]);chosen.append(index)
                if len(chosen)==m:break
            if len(chosen)==m:patient_features.append(feature);patient_top.append(chosen)
        patient_top=np.asarray(patient_top,dtype=int).reshape(-1,m)
        pp,pe=purity_scores(patient_top,labels,classes)
        rng=np.random.default_rng(seed);null=[];pnull=[]
        for _ in range(permutations):
            shuffled=rng.permutation(labels)
            a,_=purity_scores(top,shuffled,classes);b,_=purity_scores(patient_top,shuffled,classes)
            if len(a):null.append(float(a.mean()))
            if len(b):pnull.append(float(b.mean()))
        def average(v):return float(np.mean(v)) if len(v) else None
        result={'alive_features':len(alive),'eligible_features':len(eligible),'mean_purity':average(purity),
            'mean_entropy_nats':average(entropy),'permutation_mean_purity':average(null),
            'patient_distinct_eligible_features':len(patient_features),'patient_distinct_purity':average(pp),
            'patient_distinct_permutation_mean_purity':average(pnull),'patient_distinct_entropy':average(pe),
            'mean_positive_features_per_slice':float((acts>0).sum(1).mean()),
            'permutation_scope':'descriptive slice-label shuffle; not a patient-level significance test'}
        result['purity_minus_permutation']=result['mean_purity']-result['permutation_mean_purity'] if null else None
        result['patient_distinct_purity_minus_permutation']=result['patient_distinct_purity']-result['patient_distinct_permutation_mean_purity'] if pnull else None
        table=pd.DataFrame({'feature':eligible,'positive_support':support[eligible],'purity':purity,'entropy_nats':entropy})
        patient_table=pd.DataFrame({'feature':patient_features,'purity':pp,'entropy_nats':pe})
        return result,table,patient_table

    def evaluate(model,zval,original_val,recipe,labels,patients,plan,sae_seed):
        device=next(model.parameters()).device;codes=[];sse=0.;original_sse=0.
        mean=recipe['mean'];basis=recipe['basis'];scale=recipe['scale']
        centered=original_val.astype(np.float64)-mean
        with torch.no_grad():
            for start in range(0,len(zval),512):
                x=zval[start:start+512];recon,acts=model(torch.from_numpy(x).to(device))
                r=recon.cpu().numpy();codes.append(acts.cpu().numpy())
                sse+=float(np.sum((x.astype(np.float64)-r)**2))
                restored=r if recipe['condition']=='native' else ((r@basis)/scale)@basis.T
                original_sse+=float(np.sum((centered[start:start+len(x)]-restored)**2))
        acts=np.concatenate(codes);reference=float(np.sum(zval.astype(np.float64)**2))
        metrics={'normalized_SAE_EV':1-sse/reference,'original_coordinate_EV':1-original_sse/float(np.sum(centered**2)),'mse':sse/zval.size}
        concept,features,patient_features=selectivity(acts,labels,patients,plan['top_m'],plan['permutations'],sae_seed)
        metrics.update(concept)
        return metrics,features,patient_features

    def fit_one(ztrain,zval,original_val,recipe,diagnostics,labels,patients,plan,mode,encoder_seed,sae_seed,folder,identity):
        folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
        protocol={**identity,'method':mode,'encoder_seed':encoder_seed,'sae_seed':sae_seed,
            'condition':recipe['condition'],'rank':recipe['rank'],'plan':plan}
        registration=folder/'protocol.json'
        if registration.exists() and json.loads(registration.read_text())!=protocol:raise ValueError('Fit protocol changed; choose new output')
        atomic_json(registration,protocol)
        complete=folder/'validation_metrics.json';model_path=folder/'completed_model.pt'
        if complete.exists():
            r=json.loads(complete.read_text())
            if not model_path.exists() or sha(model_path)!=r['model_sha256']:raise ValueError('Missing/corrupt completed dictionary')
            print('Reusing completed control:',mode,encoder_seed,sae_seed,recipe['condition'],recipe['rank'],flush=True)
            return r
        seed_all(sae_seed);device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model=SAE(ztrain.shape[1],plan['dictionary_size'],plan['k']).to(device)
        if model_path.exists():
            saved=torch.load(model_path,map_location='cpu',weights_only=True)
            if saved['protocol']!=protocol:raise ValueError('Completed model protocol mismatch')
            model.load_state_dict(saved['model']);history=saved['history']
        else:
            optimizer=torch.optim.Adam(model.parameters(),lr=plan['lr']);history=[]
            loader=torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.from_numpy(ztrain)),batch_size=plan['batch_size'],shuffle=True)
            for epoch in range(plan['epochs']):
                model.train();total=count=0
                for (batch,) in loader:
                    batch=batch.to(device);optimizer.zero_grad(set_to_none=True);recon,_=model(batch)
                    loss=F.mse_loss(recon,batch)
                    if not torch.isfinite(loss):raise ValueError('Non-finite loss')
                    loss.backward();nn.utils.clip_grad_norm_(model.parameters(),plan['gradient_clip']);optimizer.step()
                    total+=float(loss.detach())*len(batch);count+=len(batch)
                history.append({'epoch':epoch+1,'train_mse':total/count})
                atomic_json(folder/'history.json',history)
                print(mode,encoder_seed,recipe['condition'],recipe['rank'],'SAE',sae_seed,'epoch',epoch+1,'mse',total/count,flush=True)
            # One final artifact, no optimizer/scaler/RNG or per-epoch model writes.
            payload={'model':{k:v.detach().cpu() for k,v in model.state_dict().items()},'protocol':protocol,'history':history,'training_environment':environment(),
                'transform':{k:torch.from_numpy(v) if isinstance(v,np.ndarray) else v for k,v in recipe.items()}}
            temp=model_path.with_suffix('.tmp');torch.save(payload,temp);replace_file(temp,model_path)
        model.eval();metrics,features,patient_features=evaluate(model,zval,original_val,recipe,labels,patients,plan,sae_seed)
        features.to_csv(folder/'validation_features.csv',index=False);patient_features.to_csv(folder/'patient_distinct_features.csv',index=False)
        result={**metrics,'method':mode,'encoder_seed':encoder_seed,'sae_seed':sae_seed,'condition':recipe['condition'],
            'rank':recipe['rank'],'k':plan['k'],'epochs':plan['epochs'],'split':'val','diagnostics':diagnostics,
            'model_sha256':sha(model_path),'selection_scope':'exploratory controls; no test evaluation'}
        atomic_json(complete,result);del model;gc.collect()
        if torch.cuda.is_available():torch.cuda.empty_cache()
        return result

    def probe(ztrain,zval,labels_train,labels_val,plan):
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LogisticRegression
        model=make_pipeline(StandardScaler(),LogisticRegression(C=plan['probe_C'],max_iter=plan['probe_max_iter']))
        model.fit(ztrain,labels_train);predictions=model.predict(zval)
        return {'slice_accuracy':float(np.mean(predictions==labels_val)),'train_fit_only':True,'C':plan['probe_C'],'max_iter':plan['probe_max_iter']}

    def run(bundle,bundle_sha,output,plan=None,seeds=None,methods=None,max_hours=8,require_cuda=True):
        plan=json.loads(json.dumps(DEFAULT_PLAN if plan is None else plan));seeds=plan['encoder_seeds'] if seeds is None else list(seeds)
        methods=plan['methods'] if methods is None else list(methods)
        if not seeds or set(seeds)-set(plan['encoder_seeds']) or not methods or set(methods)-set(plan['methods']):raise ValueError('Invalid method/seed selection')
        if require_cuda and not torch.cuda.is_available():raise RuntimeError('Attach a CUDA GPU')
        root=Path(output).expanduser().resolve();root.mkdir(parents=True,exist_ok=True)
        started=time.monotonic();inputs=Inputs(bundle,root/'input_cache',bundle_sha)
        identity={'bundle_sha256':bundle_sha,'manifest_sha256':inputs.registration['manifest_sha256'],
            'runtime_sha256':globals().get('RUNTIME_SHA256','development'),'plan':plan}
        registration=root/'protocol.json'
        if registration.exists() and json.loads(registration.read_text())!=identity:raise ValueError('Control protocol/code changed; choose a new output folder')
        atomic_json(registration,identity)
        atomic_json(root/'environment.json',environment())
        val=inputs.frame[inputs.frame.split=='val'];train_frame=inputs.frame[inputs.frame.split=='train']
        labels=val.dominant_label.to_numpy();patients=val.patient_id.to_numpy()
        status='completed selected controls'
        for seed,mode in itertools.product(seeds,methods):
            train=inputs.load(mode,seed,'train');validation=inputs.load(mode,seed,'val')
            cases=[('native',192)]+[(condition,rank) for condition in plan['conditions'] if condition!='native' for rank in plan['ranks']]
            for condition,rank in cases:
                if (root/'stop_after_fit.flag').exists() or time.monotonic()-started>=max_hours*3600:
                    status='stopped between fits; completed dictionaries retained';break
                ztrain,zval,recipe,diagnostics=build_transform(inputs,mode,seed,condition,rank,train,validation)
                case_dir=root/'fits'/mode/f'encoder_{seed}'/f'{condition}_r{rank}'
                case_dir.mkdir(parents=True,exist_ok=True);atomic_json(case_dir/'transform_diagnostics.json',diagnostics)
                probe_path=case_dir/'validation_probe.json'
                if not probe_path.exists():atomic_json(probe_path,probe(ztrain,zval,train_frame.has_tumor.to_numpy(),val.has_tumor.to_numpy(),plan))
                for sae_seed in plan['sae_seeds']:
                    if (root/'stop_after_fit.flag').exists() or time.monotonic()-started>=max_hours*3600:
                        status='stopped between fits; completed dictionaries retained';break
                    fit_one(ztrain,zval,validation,recipe,diagnostics,labels,patients,plan,mode,seed,sae_seed,case_dir/f'sae_{sae_seed}',identity)
                    summarize(root)
                del ztrain,zval;gc.collect()
                if status!='completed selected controls':break
            del train,validation;gc.collect()
            if status!='completed selected controls':break
        report={'status':status,'encoder_seeds':seeds,'methods':methods,'elapsed_hours':(time.monotonic()-started)/3600,
            'test_evaluated':False,'encoders_retrained':False,'per_epoch_checkpoints':False,'completed_models_saved_once':True}
        atomic_json(root/'last_session.json',report);summarize(root)
        return report

    def summarize(root):
        root=Path(root);records=[json.loads(p.read_text()) for p in root.glob('fits/*/encoder_*/*/sae_*/validation_metrics.json')]
        if not records:return pd.DataFrame(),pd.DataFrame(),pd.DataFrame()
        frame=pd.json_normalize(records);group=['method','encoder_seed','condition','rank']
        metrics=['normalized_SAE_EV','original_coordinate_EV','alive_features','eligible_features','mean_purity','purity_minus_permutation',
            'patient_distinct_purity','patient_distinct_purity_minus_permutation','patient_distinct_eligible_features']
        means=frame.groupby(group)[metrics].mean().reset_index()
        means['sae_replicates']=frame.groupby(group).size().to_numpy()
        # Only complete sets of SAE seeds enter the across-encoder summary.
        plan=json.loads((root/'protocol.json').read_text())['plan'];complete=means[means.sae_replicates==len(plan['sae_seeds'])]
        summary=complete.groupby(['method','condition','rank'])[metrics].agg(['mean','std','count']).reset_index()
        summary.columns=['_'.join(str(v) for v in c if str(v)) if isinstance(c,tuple) else c for c in summary.columns]
        atomic_csv(frame,root/'all_control_fits.csv');atomic_csv(means,root/'per_encoder_controls.csv');atomic_csv(summary,root/'control_summary.csv')
        differences=[]
        for (seed,condition,rank),g in complete.groupby(['encoder_seed','condition','rank']):
            if set(g.method)!={'ncl','simclr'}:continue
            a=g[g.method=='ncl'].iloc[0];b=g[g.method=='simclr'].iloc[0]
            differences.append({'encoder_seed':int(seed),'condition':condition,'rank':int(rank),
                **{'ncl_minus_simclr_'+m:float(a[m]-b[m]) for m in metrics}})
        atomic_csv(pd.DataFrame(differences),root/'paired_control_differences.csv')
        return frame,means,summary

    def export_results(root):
        root=Path(root).resolve()
        seeds=sorted({json.loads(p.read_text())['encoder_seed'] for p in root.glob('fits/*/encoder_*/*/sae_*/validation_metrics.json')})
        suffix='_enc'+'-'.join(str(s) for s in seeds) if seeds else ''
        target=root.parent/(root.name+suffix+'_ALL_RESULTS.zip')
        with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED) as z:
            for p in sorted(root.rglob('*')):
                if not p.is_file() or 'input_cache' in p.relative_to(root).parts or p.name in ('.env','worker_request.json'):continue
                if p.suffix in ('.json','.csv','.pt','.log','.txt','.png'):z.write(p,p.relative_to(root))
        return str(target)

    """Append to frozen rank runtime. Refit only linear probes on archived CLS."""
    from sklearn.metrics import roc_auc_score,average_precision_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from threadpoolctl import threadpool_limits
    DEFAULT_PLAN={'version':'expanded-probe-metrics-v1','methods':['ncl','simclr'],
     'encoder_seeds':[10,42,52],'conditions':['native','pca_rank','spectrum_matched','retained_95','retained_99'],
     'probe_C':1.0,'probe_max_iter':2000,'bootstrap_replicates':1000,'bootstrap_seed':271029,
     'scope':'slice-level tumour presence on inspected validation; fixed-model paired patient bootstrap; no test'}
    METRICS=['accuracy','macro_f1','balanced_accuracy','tumour_precision','tumour_recall','tumour_f1','specificity','auroc','auprc','patient_macro_accuracy']

    def binary_metrics(y,prob,pred,weights=None,patient_index=None):
        w=np.ones(len(y)) if weights is None else np.asarray(weights,dtype=float)
        tn=float(w[(y==0)&(pred==0)].sum());fp=float(w[(y==0)&(pred==1)].sum())
        fn=float(w[(y==1)&(pred==0)].sum());tp=float(w[(y==1)&(pred==1)].sum())
        def ratio(a,b):return a/b if b else 0.
        recall=ratio(tp,tp+fn);spec=ratio(tn,tn+fp)
        valid=(tp+fn)>0 and (tn+fp)>0
        f1=ratio(2*tp,2*tp+fp+fn)
        result={'accuracy':ratio(tp+tn,w.sum()),'macro_f1':(f1+ratio(2*tn,2*tn+fp+fn))/2,
            'balanced_accuracy':(recall+spec)/2 if valid else np.nan,'tumour_precision':ratio(tp,tp+fp),
            'tumour_recall':recall,'tumour_f1':f1,'specificity':spec,
            'auroc':float(roc_auc_score(y,prob,sample_weight=w)) if valid else np.nan,
            'auprc':float(average_precision_score(y,prob,sample_weight=w)) if (tp+fn)>0 else np.nan}
        if patient_index is not None:
            numerator=denominator=0.
            for p in np.unique(patient_index):
                mask=patient_index==p;mass=float(w[mask][0]);numerator+=mass*float(np.mean(pred[mask]==y[mask]));denominator+=mass
            result['patient_macro_accuracy']=numerator/denominator
        result.update(tn=tn,fp=fp,fn=fn,tp=tp)
        return result

    def metric_transform(inputs,mode,seed,condition,train,val):
        rank=192 if condition=='native' else 8
        source_condition=condition
        if condition.startswith('retained_'):
            target={'retained_95':.95,'retained_99':.99}[condition]
            eigen=inputs.pca(mode,seed,train)['eigenvalues']
            rank=int(np.searchsorted(np.cumsum(eigen)/eigen.sum(),target))+1
            source_condition='pca_rank'
        return build_transform(inputs,mode,seed,source_condition,rank,train,val)

    def summarize(root):
        root=Path(root);paths=list(root.glob('probes/*/encoder_*/*/metrics.json'))
        if not paths:return pd.DataFrame(),pd.DataFrame(),pd.DataFrame()
        frame=pd.DataFrame([json.loads(p.read_text()) for p in paths])
        atomic_csv(frame,root/'probe_metrics.csv');atomic_csv(frame,root/'per_encoder_controls.csv')
        summary=frame.groupby(['method','condition'])[METRICS].agg(['mean','std','count']).reset_index()
        summary.columns=['_'.join(str(x) for x in c if str(x)) for c in summary.columns]
        atomic_csv(summary,root/'probe_summary.csv')
        plan=json.loads((root/'protocol.json').read_text())['plan'];rows=[];boot={}
        for p in paths:
            record=json.loads(p.read_text());key=(record['method'],record['condition'],int(record['encoder_seed']))
            boot[key]=pd.read_csv(p.parent/'patient_bootstrap.csv')[METRICS].to_numpy()
        aggregate={};points={}
        def interval(label,method,condition,point,replicates):
            for i,m in enumerate(METRICS):
                values=replicates[:,i];finite=values[np.isfinite(values)]
                low,high=np.quantile(finite,[.025,.975]) if len(finite)>=.95*len(values) else (np.nan,np.nan)
                rows.append({'comparison':label,'method':method,'condition':condition,'metric':m,
                    'point':point[i],'ci_low':low,'ci_high':high,'finite_replicates':len(finite),
                    'total_replicates':len(values),'scope':'paired patient percentile interval conditional on fixed training/probes; exploratory validation'})
        for (mode,condition),g in frame.groupby(['method','condition']):
            if set(g.encoder_seed)!=set(plan['encoder_seeds']):continue
            stack=np.stack([boot[(mode,condition,s)] for s in plan['encoder_seeds']])
            values=np.mean(stack,axis=0);point=g[METRICS].mean().to_numpy()
            aggregate[(mode,condition)]=values;points[(mode,condition)]=point
            interval('across_fixed_encoders',mode,condition,point,values)
        for condition in plan['conditions']:
            if all((m,condition) in aggregate for m in ['ncl','simclr']):
                interval('ncl_minus_simclr','paired',condition,points[('ncl',condition)]-points[('simclr',condition)],
                    aggregate[('ncl',condition)]-aggregate[('simclr',condition)])
        atomic_csv(pd.DataFrame(rows),root/'patient_bootstrap_intervals.csv')
        return frame,frame,summary

    def run(bundle,bundle_sha,output,plan=None,seeds=None,methods=None,max_hours=8,require_cuda=False):
        plan=json.loads(json.dumps(DEFAULT_PLAN if plan is None else plan));seeds=plan['encoder_seeds'] if seeds is None else list(seeds)
        methods=plan['methods'] if methods is None else list(methods)
        if not seeds or set(seeds)-set(plan['encoder_seeds']) or not methods or set(methods)-set(plan['methods']):raise ValueError('Invalid selection')
        root=Path(output).resolve();root.mkdir(parents=True,exist_ok=True);started=time.monotonic()
        inputs=Inputs(bundle,root/'input_cache',bundle_sha)
        identity={'bundle_sha256':bundle_sha,'manifest_sha256':inputs.registration['manifest_sha256'],
            'runtime_sha256':globals().get('RUNTIME_SHA256','development'),'plan':plan}
        if (root/'protocol.json').exists() and json.loads((root/'protocol.json').read_text())!=identity:raise ValueError('Protocol changed; choose a fresh output')
        atomic_json(root/'protocol.json',identity);atomic_json(root/'environment.json',environment())
        tf=inputs.frame[inputs.frame.split=='train'];vf=inputs.frame[inputs.frame.split=='val']
        y=tf.has_tumor.to_numpy().astype(int);yv=vf.has_tumor.to_numpy().astype(int)
        if set(np.unique(y))!={0,1} or not set(np.unique(yv))<={0,1}:raise ValueError('Expected binary tumour-presence labels')
        patient_order=sorted(vf.patient_id.unique());patient_index=np.array([patient_order.index(p) for p in vf.patient_id])
        draws=np.random.default_rng(plan['bootstrap_seed']).integers(0,len(patient_order),(plan['bootstrap_replicates'],len(patient_order)))
        atomic_csv(pd.DataFrame(draws),root/'patient_bootstrap_draw_indices.csv');atomic_json(root/'patient_order.json',patient_order)
        status='completed selected controls'
        with threadpool_limits(2):
            for seed,mode in itertools.product(seeds,methods):
                train=inputs.load(mode,seed,'train');val=inputs.load(mode,seed,'val')
                for condition in plan['conditions']:
                    if (root/'stop_after_fit.flag').exists() or time.monotonic()-started>max_hours*3600:
                        status='stopped between probes; completed outputs retained';break
                    folder=root/'probes'/mode/f'encoder_{seed}'/condition;folder.mkdir(parents=True,exist_ok=True)
                    complete=folder/'metrics.json'
                    if complete.exists():
                        saved=json.loads(complete.read_text())
                        if saved['predictions_sha256']!=sha(folder/'predictions.csv') or saved['bootstrap_sha256']!=sha(folder/'patient_bootstrap.csv'):
                            raise ValueError('Corrupt saved probe outputs')
                        continue
                    zt,zv,recipe,diagnostics=metric_transform(inputs,mode,seed,condition,train,val)
                    model=make_pipeline(StandardScaler(),LogisticRegression(C=plan['probe_C'],max_iter=plan['probe_max_iter']))
                    model.fit(zt,y);pred=model.predict(zv);prob=model.predict_proba(zv)[:,list(model.classes_).index(1)]
                    metrics=binary_metrics(yv,prob,pred,patient_index=patient_index)
                    predictions=vf[['sample_id','patient_id','has_tumor']].copy();predictions['prediction']=pred;predictions['tumour_probability']=prob
                    atomic_csv(predictions,folder/'predictions.csv')
                    replicates=[]
                    for draw in draws:
                        frequency=np.bincount(draw,minlength=len(patient_order));weights=frequency[patient_index]
                        replicates.append(binary_metrics(yv,prob,pred,weights,patient_index))
                    atomic_csv(pd.DataFrame(replicates),folder/'patient_bootstrap.csv')
                    scaler=model.named_steps['standardscaler'];lr=model.named_steps['logisticregression']
                    np.savez(folder/'probe_and_transform.npz',coef=lr.coef_,intercept=lr.intercept_,classes=lr.classes_,
                        standardizer_mean=scaler.mean_,standardizer_scale=scaler.scale_,
                        input_mean=recipe['mean'],basis=recipe['basis'],pca_scale=recipe['scale'])
                    atomic_json(folder/'transform_diagnostics.json',diagnostics)
                    atomic_json(complete,{'method':mode,'encoder_seed':seed,'condition':condition,'rank':recipe['rank'],
                        'solver_iterations':int(lr.n_iter_.max()),'solver_hit_iteration_limit':bool(lr.n_iter_.max()>=plan['probe_max_iter']),
                        **{k:float(v) if np.isfinite(v) else None for k,v in metrics.items()},
                        'split':'val','target':'slice-level tumour presence','predictions_sha256':sha(folder/'predictions.csv'),
                        'bootstrap_sha256':sha(folder/'patient_bootstrap.csv')})
                    summarize(root);print('Probe complete:',mode,seed,condition,metrics,flush=True)
                    del zt,zv;gc.collect()
                del train,val;gc.collect()
                if status!='completed selected controls':break
        report={'status':status,'encoder_seeds':seeds,'methods':methods,'test_evaluated':False,
            'encoders_retrained':False,'saes_retrained':False,'small_probes_refitted':True}
        atomic_json(root/'last_session.json',report);summarize(root);return report

    def export_results(root):
        root=Path(root).resolve();target=root.parent/(root.name+'_ALL_PROBE_RESULTS.zip')
        with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED) as z:
            for p in sorted(root.rglob('*')):
                if p.is_file() and 'input_cache' not in p.relative_to(root).parts and p.suffix in ('.json','.csv','.npz','.log') and p.name!='worker_request.json':z.write(p,p.relative_to(root))
        return str(target)
    ''').lstrip('\n')
    WORKER_SOURCE = _textwrap.dedent(r'''
    import os,sys,json,time,types,hashlib,traceback
    from pathlib import Path
    request=json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
    root=Path(request['output_dir']).resolve();root.mkdir(parents=True,exist_ok=True)
    lock=open(root/'control_worker.lock','a+b')
    if lock.tell()==0:lock.write(b'0');lock.flush()
    lock.seek(0)
    try:
        if os.name=='nt':
            import msvcrt
            msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        else:
            import fcntl
            fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    except OSError:
        print('Duplicate control worker refused.',flush=True);sys.exit(2)
    def state(status,**extra):
        path=root/'worker_state.json';tmp=path.with_name('worker_state.'+str(os.getpid())+'.tmp')
        tmp.write_text(json.dumps({'pid':os.getpid(),'status':status,'time':time.time(),**extra},indent=2))
        for attempt in range(41):
            try:os.replace(tmp,path);break
            except PermissionError:
                if os.name!='nt' or attempt==40:raise
                time.sleep(.05)
    try:
        state('initializing')
        engine=types.ModuleType('rank_control_runtime')
        exec(compile(request['runtime_source'],'rank_control_runtime','exec'),engine.__dict__)
        engine.RUNTIME_SHA256=hashlib.sha256(request['runtime_source'].encode()).hexdigest()
        state('running')
        result=engine.run(request['bundle_path'],request['bundle_sha'],root,request['plan'],
            request['encoder_seeds'],request['methods'],request['max_hours'])
        state(result['status'],report=result)
    except BaseException as exc:
        state('error',error=f'{type(exc).__name__}: {exc}');traceback.print_exc();raise
    finally:lock.close()
    ''').lstrip('\n')
    CONTROLLER_SOURCE = _textwrap.dedent(r'''
    from pathlib import Path
    import json,os,sys,subprocess,time
    import pandas as pd
    import psutil

    def status(root):
        root=Path(root).expanduser().resolve();p=root/'worker_state.json'
        r=json.loads(p.read_text()) if p.exists() else {'status':'not started'}
        alive=False
        try:
            process=psutil.Process(r['pid'])
            alive=process.is_running() and process.status()!=psutil.STATUS_ZOMBIE and any('rank_control_worker.py' in str(v) for v in process.cmdline())
        except (KeyError,psutil.NoSuchProcess,psutil.AccessDenied):pass
        r['alive']=alive
        if not alive and r['status'] in ('running','initializing','starting'):r['status']='worker exited; inspect log before continuing'
        return r

    def launch(settings,bundle_path,bundle_sha,plan,runtime_source,worker_source):
        root=Path(settings['output_dir']).expanduser().resolve();root.mkdir(parents=True,exist_ok=True)
        old=status(root)
        if old['alive']:return {**old,'launch':'already running; duplicate refused'}
        (root/'stop_after_fit.flag').unlink(missing_ok=True)
        worker=root/'rank_control_worker.py';worker.write_text(worker_source,encoding='utf-8')
        request=root/'worker_request.json'
        request.write_text(json.dumps({'bundle_path':str(Path(bundle_path).resolve()),'bundle_sha':bundle_sha,
            'output_dir':str(root),'plan':plan,'runtime_source':runtime_source,'encoder_seeds':settings['encoder_seeds'],
            'methods':settings['methods'],'max_hours':settings['max_hours']},indent=2),encoding='utf-8')
        flags={'start_new_session':True} if os.name!='nt' else {'creationflags':subprocess.CREATE_NO_WINDOW}
        with open(root/'worker.log','ab',buffering=0) as log:
            process=subprocess.Popen([sys.executable,'-u',str(worker),str(request)],cwd=root,
                stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,**flags)
        state=root/'worker_state.json'
        if not state.exists() or json.loads(state.read_text()).get('pid')!=process.pid:
            temp=root/'launcher_state.tmp';temp.write_text(json.dumps({'pid':process.pid,'status':'starting'}));os.replace(temp,state)
        return {'pid':process.pid,'status':'background control worker launched'}

    def snapshot(root):
        root=Path(root).expanduser().resolve()
        rows=[json.loads(p.read_text()) for p in root.glob('probes/*/encoder_*/*/metrics.json')]
        df=pd.json_normalize(rows)
        table=root/'per_encoder_controls.csv'
        per=pd.read_csv(table) if table.exists() and table.stat().st_size>1 else pd.DataFrame()
        log=root/'worker.log';tail=''
        if log.exists():
            with open(log,'rb') as f:
                f.seek(max(0,log.stat().st_size-12000));tail=f.read().decode('utf-8',errors='replace')
        return status(root),df,per,tail

    def request_stop(root):
        root=Path(root).expanduser().resolve();root.mkdir(parents=True,exist_ok=True)
        (root/'stop_after_fit.flag').write_text('Stop after the current completed fit')
        return 'Stop requested. Wait for the current fit/evaluation to complete and its final model to save.'
    ''').lstrip('\n')
    BUNDLE_SHA256 = 'e41d4ae24932127a5747073d7225bdfd2d20022ca8a22027222674b7494cf9eb'
    return RUNTIME_SOURCE, WORKER_SOURCE, CONTROLLER_SOURCE, BUNDLE_SHA256

@app.cell(hide_code=True)
def _(CONTROLLER_SOURCE,RUNTIME_SOURCE):
    import types as _types
    import hashlib as _hashlib
    engine = _types.ModuleType('rank_control_engine')
    exec(compile(RUNTIME_SOURCE,'rank_control_runtime','exec'),engine.__dict__)
    engine.RUNTIME_SHA256 = _hashlib.sha256(RUNTIME_SOURCE.encode()).hexdigest()
    control = _types.ModuleType('rank_control_controller')
    exec(compile(CONTROLLER_SOURCE,'rank_control_controller','exec'),control.__dict__)
    return control, engine

@app.cell
def _(mo):
    settings = mo.ui.dictionary({
        'input_source':mo.ui.dropdown(options=['huggingface','local'],value='huggingface',label='Embedding bundle source'),
        'repo_id':mo.ui.text(value='YOUR_USERNAME/YOUR_PRIVATE_DATASET',label='Private HF repository'),
        'bundle_filename':mo.ui.text(value='rank_control_embeddings_v1.zip',label='Upload this file to HF first'),
        'bundle_path':mo.ui.text(value='data/rank_control_embeddings_v1.zip',label='Local embedding bundle path'),
        'output_dir':mo.ui.text(value='probe_metrics_v1',label='Results folder (same folder to reuse completed fits)'),
        'encoder_seeds':mo.ui.multiselect(options=[10,42,52],value=[10,42,52],label='Encoder seeds to evaluate this session'),
        'methods':mo.ui.multiselect(options=['ncl','simclr'],value=['ncl','simclr'],label='Methods (keep both for paired comparison)'),
        'max_hours':mo.ui.slider(start=1,stop=10,step=.5,value=8,label='Stop between fits after this many hours'),
    }).form(submit_button_label='Apply settings')
    settings
    return (settings,)

@app.cell
def _(mo,settings):
    mo.stop(settings.value is None,mo.md('Upload the embedding bundle to HF, add HF_TOKEN to .env, then apply settings.'))
    applied = dict(settings.value)
    if not applied['encoder_seeds'] or not applied['methods']:raise ValueError('Choose encoder seeds and methods')
    mo.md(f"**{len(applied['encoder_seeds'])*len(applied['methods'])*5} small probe fits** selected. No encoders retrained. Predictions and fitted probe parameters are saved.")
    return (applied,)

@app.cell
def _(mo):
    import_inputs = mo.ui.run_button(label='Download / verify embedding bundle')
    import_inputs
    return (import_inputs,)

@app.cell
def _(BUNDLE_SHA256,applied,engine,import_inputs,mo):
    from pathlib import Path as _InputPath
    _target = _InputPath(applied['bundle_path']).expanduser().resolve()
    mo.stop(not import_inputs.value,mo.md('Click Download / verify embedding bundle. The raw 12 GB MRI PKL is not needed.'))
    if applied['input_source']=='huggingface':
        bundle_path = engine.bundle_download(applied['repo_id'],applied['bundle_filename'],_target,BUNDLE_SHA256)
    else:
        if not _target.is_file() or engine.sha(_target)!=BUNDLE_SHA256:raise ValueError('Local bundle is missing or has the wrong SHA256')
        bundle_path = str(_target)
    mo.md('Embedding bundle SHA256 verified.')
    return (bundle_path,)

@app.cell
def _(mo):
    start_controls = mo.ui.run_button(label='Start / continue probes')
    stop_controls = mo.ui.run_button(label='Stop after completed probe')
    make_zip = mo.ui.run_button(label='Create full results ZIP')
    mo.hstack([start_controls,stop_controls,make_zip])
    return make_zip, start_controls, stop_controls

@app.cell
def _(BUNDLE_SHA256,RUNTIME_SOURCE,WORKER_SOURCE,applied,bundle_path,control,engine,mo,start_controls):
    mo.stop(not start_controls.value)
    launch_status = control.launch(applied,bundle_path,BUNDLE_SHA256,engine.DEFAULT_PLAN,RUNTIME_SOURCE,WORKER_SOURCE)
    mo.md(str(launch_status))
    return (launch_status,)

@app.cell
def _(applied,control,mo,stop_controls):
    mo.stop(not stop_controls.value)
    mo.md(control.request_stop(applied['output_dir']))
    return

@app.cell
def _(applied,control,engine,make_zip,mo):
    mo.stop(not make_zip.value)
    mo.stop(control.status(applied['output_dir'])['alive'],mo.md('Wait until the current worker stops before archiving final models.'))
    export_path = engine.export_results(applied['output_dir'])
    mo.md('ZIP ready: `'+export_path+'`. Refresh Files, then download this ZIP. It includes predictions, metrics, intervals and fitted probe parameters; input cache is excluded.')
    return (export_path,)

@app.cell
def _(mo):
    refresh = mo.ui.refresh(default_interval='10s',label='Live controls progress')
    refresh
    return (refresh,)

@app.cell
def _(applied,control,mo,refresh):
    _ = refresh.value
    _state,_fits,_per,_tail = control.snapshot(applied['output_dir'])
    _items = [mo.md('**Worker:** '+str(_state)),mo.md(f'**Completed probes in output folder: {len(_fits)} / 30** for the full three-seed plan.')]
    if len(_per):_items.append(mo.ui.table(_per,selection=None))
    if len(_fits):_items.append(mo.download(data=_fits.to_csv(index=False).encode(),filename='rank_control_metrics.csv',mimetype='text/csv',label='Download metrics CSV'))
    _items.append(mo.md('```text\n'+_tail+'\n```'))
    mo.vstack(_items)
    return

if __name__=='__main__':
    app.run()
