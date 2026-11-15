# /// script
# requires-python = ">=3.11"
# dependencies = ["marimo>=0.24,<1","numpy>=1.26,<3","pandas>=2,<4",
#   "torch>=2.7","scikit-learn>=1.5,<2","scipy>=1.11,<2","psutil>=5.9",
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
    # Patient bootstrap and SAE feature stability

    No model training. Analyze the 36 completed native baseline/AuxK SAE models.
    Inputs: the SAME `rank_control_embeddings_v1.zip`, plus
    `auxk_native_controls_v1_enc10-42-52_ALL_RESULTS.zip` (79.7 MB).
    Upload this completed-model ZIP ONCE to the root of your private
    `YOUR_USERNAME/YOUR_PRIVATE_DATASET` repository, keeping its filename. Then use the same
    read-only HF_TOKEN. The notebook verifies both source archive hashes.

    Default: all encoder seeds 10/42/52, both methods, baseline and AuxK. Output:
    `patient_stability_v1`. No new encoder/saved-model analyses and no raw MRI PKL needed.

    Patient bootstrap: 1,000 whole-patient resamples, paired across all methods/models.
    Resample 37 validation patients with replacement, retaining all their slices.
    Recompute feature eligibility and top-10 label purity in every resample; do not
    bootstrap slices independently. Patient-distinct top-10 treats repeated copies of
    one original patient as one identity. Recompute eligible coverage in both metrics.
    Shared patient draws give paired NCL-minus-SimCLR and AuxK-minus-baseline intervals.
    Average SAE seeds within encoder, then average the three fixed encoder seeds.
    Intervals are descriptive 95% percentile intervals conditional on these trained
    models; they do not include encoder-training uncertainty or undo validation reuse.
    No bootstrap-adjusted-purity intervals or spatial Dice are calculated here.

    Feature stability: within each encoder/method/variant, compare all three pairs of
    SAE seeds. Select at most 4,096 training slices by patient-balanced round robin,
    without labels. Fit one-to-one feature matches to training activation correlations
    only, among features with >=10 positive training-subset examples and nonzero
    variance. Assess matched activation correlations and support Jaccard on validation;
    also report decoder cosine, matching coverage and a random-matching reference.
    Thresholds 0.7/0.9 are descriptive sensitivity cutoffs, not validated quality levels.
    Unmatched, constant and ineligible features remain visible in coverage counts.
    This tests within-encoder dictionary reproducibility, not cross-encoder neuron
    identity, causal concepts or clinical relevance.

    Use Apply settings -> Download / verify embedding bundle -> Start / continue
    controls. Live progress refreshes every 10 seconds. Create full results ZIP after
    the worker stops. Export contains analysis tables/replicates, not the input models.
    Keep your original model archive. Interrupted analyses reuse committed outputs.
    All results use previously inspected validation data; no held-out test is read.
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

    """Append to rank runtime: paired patient-cluster bootstrap and train-only matching."""
    import io
    from scipy.optimize import linear_sum_assignment
    DEFAULT_PLAN={'version':'patient-stability-v1','encoder_seeds':[10,42,52],
     'methods':['ncl','simclr'],'sae_seeds':[10,42,52],'conditions':['baseline','auxk'],
     'bootstrap_replicates':1000,'bootstrap_seed':271027,'bootstrap_batch':8,'top_m':10,
     'alignment_train_slices':4096,'alignment_seed':271028,'alignment_min_positive_support':10,
     'correlation_thresholds':[0.7,0.9],'random_alignment_replicates':100,
     'models_filename':'auxk_native_controls_v1_enc10-42-52_ALL_RESULTS.zip',
     'models_sha256':'724f0a4e5d81b1bcebca019bdf17bb1eeb673ce7d1d8640960909e82fd165856',
     'scope':'exploratory validation; percentile patient-bootstrap conditional on fixed trained models; no test'}
    BOOT_METRICS=['mean_purity','patient_distinct_purity','normalized_SAE_EV','alive_features','eligible_features','patient_distinct_eligible_features']

    def balanced_train_indices(frame,limit,seed):
        rng=np.random.default_rng(seed)
        groups=[rng.permutation(np.flatnonzero(frame.patient_id.to_numpy()==p)).tolist() for p in sorted(frame.patient_id.unique())]
        rng.shuffle(groups);selected=[]
        while len(selected)<min(limit,len(frame)):
            for group in groups:
                if group:selected.append(group.pop())
                if len(selected)==min(limit,len(frame)):break
        return np.asarray(selected,dtype=int)

    def encode(model,x):
        device=next(model.parameters()).device;codes=[];errors=[]
        with torch.no_grad():
            for start in range(0,len(x),512):
                batch=torch.from_numpy(x[start:start+512]).to(device)
                recon,acts=model(batch)
                codes.append(acts.cpu().numpy())
                errors.append(((batch-recon).double().square().sum(1)).cpu().numpy())
        return np.concatenate(codes),np.concatenate(errors)

    def patient_pool(codes,labels,patients,errors,reference,m,device):
        unique=sorted(set(patients));width=codes.shape[1]
        values=np.full((len(unique),m,width),-np.inf,dtype=np.float32)
        lab=np.zeros((len(unique),m,width),dtype=np.int64)
        support=[];sse=[];ref=[]
        # Per-patient top M is sufficient for every bootstrap global top M.
        for p,patient in enumerate(unique):
            indices=np.flatnonzero(patients==patient);a=codes[indices]
            order=np.argsort(a,axis=0,kind='stable')[-m:][::-1]
            v=np.take_along_axis(a,order,axis=0)
            count=len(v);values[p,:count]=np.where(v>0,v,-np.inf)
            lab[p,:count]=labels[indices[order]]
            support.append((a>0).sum(0));sse.append(errors[indices].sum());ref.append(reference[indices].sum())
        def tensor(x,dtype=None):return torch.as_tensor(x,device=device,dtype=dtype)
        return {'values':tensor(values),'labels':tensor(lab),'support':tensor(np.asarray(support),torch.float64),
            'sse':tensor(sse,torch.float64),'reference':tensor(ref,torch.float64),
            'classes':np.unique(labels).tolist(),'patients':unique}

    def bootstrap_statistics(pool,draws,m,batch_size=8):
        values=pool['values'];labels=pool['labels'];device=values.device;P=values.shape[0]
        rows=[]
        for start in range(0,len(draws),batch_size):
            index=torch.as_tensor(draws[start:start+batch_size],device=device,dtype=torch.long);B=len(index)
            weights=torch.zeros((B,P),device=device,dtype=torch.float64)
            weights.scatter_add_(1,index,torch.ones_like(index,dtype=torch.float64))
            support=weights@pool['support'];eligible=support>=m
            candidates=values[index].reshape(B,-1,values.shape[-1])
            chosen=candidates.topk(m,dim=1).indices
            selected_labels=labels[index].reshape(B,-1,values.shape[-1]).gather(1,chosen)
            counts=torch.stack([(selected_labels==c).sum(1) for c in pool['classes']],dim=1)
            purity=counts.amax(1).double()/m
            raw=(purity*eligible).sum(1)/eligible.sum(1).clamp_min(1)
            raw=raw.masked_fill(eligible.sum(1)==0,float('nan'))
            # Repeated copies of one patient do not count as different identities.
            present=weights>0
            best=values[:,0,:][None,:,:].expand(B,-1,-1).masked_fill(~present[:,:,None],-np.inf)
            distinct_support=torch.isfinite(best).sum(1);distinct_eligible=distinct_support>=m
            picked=best.topk(m,dim=1).indices
            pl=labels[:,0,:][None,:,:].expand(B,-1,-1).gather(1,picked)
            pc=torch.stack([(pl==c).sum(1) for c in pool['classes']],dim=1)
            dp=(pc.amax(1).double()/m*distinct_eligible).sum(1)/distinct_eligible.sum(1).clamp_min(1)
            dp=dp.masked_fill(distinct_eligible.sum(1)==0,float('nan'))
            ev=1-(weights@pool['sse'])/(weights@pool['reference'])
            rows.append(torch.stack([raw,dp,ev,(support>0).sum(1),eligible.sum(1),distinct_eligible.sum(1)],dim=1).cpu().numpy())
        return np.concatenate(rows)

    def normalized_columns(a):
        a=a.astype(np.float64);a=a-a.mean(0);norm=np.sqrt(np.sum(a*a,axis=0))
        return (a/np.maximum(norm,1e-30)).astype(np.float32),norm>1e-12

    def feature_match(train_a,train_b,val_a,val_b,decoder_a,decoder_b,plan,device):
        na,oka=normalized_columns(train_a);nb,okb=normalized_columns(train_b)
        ia=np.flatnonzero(oka&((train_a>0).sum(0)>=plan['alignment_min_positive_support']))
        ib=np.flatnonzero(okb&((train_b>0).sum(0)>=plan['alignment_min_positive_support']))
        if not len(ia) or not len(ib):return pd.DataFrame(),{'eligible_left':len(ia),'eligible_right':len(ib),'matches':0}
        score=(torch.from_numpy(na[:,ia]).to(device).T@torch.from_numpy(nb[:,ib]).to(device)).cpu().numpy()
        left,right=linear_sum_assignment(score,maximize=True);a=ia[left];b=ib[right]
        va,valid_a=normalized_columns(val_a);vb,valid_b=normalized_columns(val_b)
        valid=valid_a[a]&valid_b[b]
        corr=np.sum(va[:,a]*vb[:,b],axis=0).astype(float);corr[~valid]=np.nan
        da=decoder_a[:,a];db=decoder_b[:,b]
        cosine=np.sum(da*db,axis=0)/np.maximum(np.linalg.norm(da,axis=0)*np.linalg.norm(db,axis=0),1e-30)
        pa=val_a[:,a]>0;pb=val_b[:,b]>0;union=(pa|pb).sum(0)
        jaccard=np.divide((pa&pb).sum(0),union,out=np.full(len(a),np.nan),where=union>0)
        table=pd.DataFrame({'left_feature':a,'right_feature':b,'train_correlation':score[left,right],
            'validation_correlation':corr,'decoder_cosine':cosine,'validation_activation_jaccard':jaccard,
            'validation_positive_left':pa.sum(0),'validation_positive_right':pb.sum(0)})
        rng=np.random.default_rng(plan['alignment_seed']);null=[]
        for _ in range(plan['random_alignment_replicates']):
            ra=rng.choice(ia,len(a),replace=False);rb=rng.choice(ib,len(b),replace=False)
            good=valid_a[ra]&valid_b[rb]
            null.append(float(np.sum(va[:,ra[good]]*vb[:,rb[good]],axis=0).mean()) if good.any() else np.nan)
        summary={'eligible_left':len(ia),'eligible_right':len(ib),'matches':len(a),
            'validation_variable_matches':int(valid.sum()),'train_correlation_mean':float(score[left,right].mean()),
            'validation_correlation_mean':float(np.nanmean(corr)) if valid.any() else None,
            'decoder_cosine_mean':float(cosine.mean()),'validation_jaccard_mean':float(np.nanmean(jaccard)) if np.isfinite(jaccard).any() else None,
            'random_alignment_validation_correlation_mean':float(np.nanmean(null)) if np.isfinite(null).any() else None}
        for threshold in plan['correlation_thresholds']:
            n=int(np.sum(corr>=threshold));key=str(threshold).replace('.','p')
            summary.update({f'matches_val_corr_ge_{key}':n,f'left_fraction_val_corr_ge_{key}':n/len(ia),f'right_fraction_val_corr_ge_{key}':n/len(ib)})
        return table,summary

    def analysis_summary(root):
        root=Path(root);records=[];points=[];stability=[]
        plan=json.loads((root/'protocol.json').read_text())['plan']
        for p in root.glob('analysis_fits/*/encoder_*/*/sae_*/completed.json'):
            r=json.loads(p.read_text());identity={k:r[k] for k in ['method','encoder_seed','condition','sae_seed']}
            df=pd.read_csv(p.parent/'bootstrap_replicates.csv');df=df.assign(**identity);records.append(df)
            points.append({**identity,**r['point']})
        for p in root.glob('stability/*/encoder_*/*/pair_*/summary.json'):stability.append(json.loads(p.read_text()))
        if stability:
            sf=pd.DataFrame(stability);atomic_csv(sf,root/'feature_stability_pairs.csv')
            numeric=[c for c in sf.select_dtypes(include='number').columns if c not in ('encoder_seed','left_sae_seed','right_sae_seed')]
            sg=['method','encoder_seed','condition']
            sm=sf.groupby(sg)[numeric].mean().reset_index();sm['sae_pairs']=sf.groupby(sg).size().to_numpy()
            atomic_csv(sm,root/'feature_stability_per_encoder.csv')
            full=sm[sm.sae_pairs==len(list(itertools.combinations(plan['sae_seeds'],2)))]
            if len(full):
                ss=full.groupby(['method','condition'])[numeric].agg(['mean','std','count']).reset_index()
                ss.columns=['_'.join(str(x) for x in c if str(x)) for c in ss.columns]
                atomic_csv(ss,root/'feature_stability_summary.csv')
        if not records:return pd.DataFrame(),pd.DataFrame(),pd.DataFrame()
        all_reps=pd.concat(records,ignore_index=True);pt=pd.DataFrame(points)
        group=['method','encoder_seed','condition']
        sizes=pt.groupby(group).size();complete=sizes[sizes==len(plan['sae_seeds'])].index
        mask=pd.MultiIndex.from_frame(all_reps[group]).isin(complete)
        per=all_reps[mask].groupby([*group,'replicate'])[BOOT_METRICS].mean().reset_index()
        per_point=pt[ pd.MultiIndex.from_frame(pt[group]).isin(complete)].groupby(group)[BOOT_METRICS].mean().reset_index()
        output=[]
        def interval(label,data,point,extra):
            for metric in BOOT_METRICS:
                values=data[metric].to_numpy();finite=values[np.isfinite(values)]
                if len(finite)<len(values)*.95:low=high=None
                else:low,high=[float(v) for v in np.quantile(finite,[.025,.975])]
                output.append({'comparison':label,**extra,'metric':metric,'point':float(point[metric]),
                    'ci_low':low,'ci_high':high,'finite_replicates':len(finite),'total_replicates':len(values),
                    'scope':'paired patient-cluster percentile interval conditional on fixed models; exploratory validation'})
        for key,g in per.groupby(group):
            p=per_point.set_index(group).loc[key];interval('per_encoder',g,p,dict(zip(group,key)))
        # Across-encoder intervals require every registered encoder, never partial groups.
        available=per_point.groupby(['method','condition']).encoder_seed.nunique()
        valid_groups=available[available==len(plan['encoder_seeds'])].index
        selected=per[pd.MultiIndex.from_frame(per[['method','condition']]).isin(valid_groups)]
        overall=selected.groupby(['method','condition','replicate'])[BOOT_METRICS].mean().reset_index()
        overall_pt=per_point.groupby(['method','condition'])[BOOT_METRICS].mean()
        for key,g in overall.groupby(['method','condition']):interval('across_fixed_encoders',g,overall_pt.loc[key],dict(zip(['method','condition'],key)))
        for condition,g in overall.groupby('condition'):
            if set(g.method)!={'ncl','simclr'}:continue
            diff=g[g.method=='ncl'].set_index('replicate')[BOOT_METRICS]-g[g.method=='simclr'].set_index('replicate')[BOOT_METRICS]
            point=overall_pt.loc[('ncl',condition)]-overall_pt.loc[('simclr',condition)]
            interval('ncl_minus_simclr',diff,point,{'condition':condition})
        for method,g in overall.groupby('method'):
            if set(g.condition)!={'baseline','auxk'}:continue
            diff=g[g.condition=='auxk'].set_index('replicate')[BOOT_METRICS]-g[g.condition=='baseline'].set_index('replicate')[BOOT_METRICS]
            interval('auxk_minus_baseline',diff,overall_pt.loc[(method,'auxk')]-overall_pt.loc[(method,'baseline')],{'method':method})
        atomic_csv(pt,root/'bootstrap_fit_points.csv');atomic_csv(pd.DataFrame(output),root/'patient_bootstrap_intervals.csv')
        atomic_csv(per_point,root/'per_encoder_controls.csv')
        return pt,per_point,pd.DataFrame(output)

    def run(bundle,bundle_sha,output,plan=None,seeds=None,methods=None,max_hours=8,require_cuda=True):
        plan=json.loads(json.dumps(DEFAULT_PLAN if plan is None else plan));seeds=plan['encoder_seeds'] if seeds is None else list(seeds)
        methods=plan['methods'] if methods is None else list(methods)
        if not seeds or set(seeds)-set(plan['encoder_seeds']) or not methods or set(methods)-set(plan['methods']):raise ValueError('Invalid selection')
        if require_cuda and not torch.cuda.is_available():raise RuntimeError('Attach a CUDA GPU')
        models_path=Path(plan.pop('models_path',str(Path(bundle).parent/plan['models_filename']))).resolve()
        if sha(models_path)!=plan['models_sha256']:raise ValueError('Wrong/incomplete AuxK models ZIP')
        root=Path(output).resolve();root.mkdir(parents=True,exist_ok=True)
        inputs=Inputs(bundle,root/'input_cache',bundle_sha);device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        identity={'bundle_sha256':bundle_sha,'manifest_sha256':inputs.registration['manifest_sha256'],
            'runtime_sha256':globals().get('RUNTIME_SHA256','development'),'plan':plan}
        protocol=root/'protocol.json'
        if protocol.exists() and json.loads(protocol.read_text())!=identity:raise ValueError('Analysis protocol changed; choose new output')
        atomic_json(protocol,identity);atomic_json(root/'environment.json',environment())
        vf=inputs.frame[inputs.frame.split=='val'];tf=inputs.frame[inputs.frame.split=='train']
        labels=vf.dominant_label.to_numpy();patients=vf.patient_id.to_numpy();unique=sorted(set(patients))
        if len(unique)<plan['top_m']:raise ValueError('Too few validation patients')
        draws=np.random.default_rng(plan['bootstrap_seed']).integers(0,len(unique),size=(plan['bootstrap_replicates'],len(unique)))
        draw_table=pd.DataFrame(draws,columns=[f'draw_{i}' for i in range(len(unique))]);atomic_csv(draw_table,root/'patient_bootstrap_draw_indices.csv')
        atomic_json(root/'patient_order.json',unique)
        train_idx=balanced_train_indices(tf,plan['alignment_train_slices'],plan['alignment_seed'])
        atomic_csv(tf.iloc[train_idx][['sample_id','patient_id']],root/'alignment_training_samples.csv')
        started=time.monotonic();status='completed selected controls'
        with zipfile.ZipFile(models_path) as archive:
            for enc,mode in itertools.product(seeds,methods):
                train=inputs.load(mode,enc,'train');val=inputs.load(mode,enc,'val')
                for condition in plan['conditions']:
                    activations={}
                    for sae_seed in plan['sae_seeds']:
                        if (root/'stop_after_fit.flag').exists() or time.monotonic()-started>=max_hours*3600:
                            status='stopped between analyses; completed outputs retained';break
                        prefix=f'fits/{mode}/encoder_{enc}/{condition}_r192/sae_{sae_seed}/'
                        metric=json.loads(archive.read(prefix+'validation_metrics.json'))
                        raw=archive.read(prefix+'completed_model.pt')
                        if hashlib.sha256(raw).hexdigest()!=metric['model_sha256']:raise ValueError('Model hash mismatch')
                        saved=torch.load(io.BytesIO(raw),map_location='cpu',weights_only=True)
                        sp=saved['protocol']
                        if sp['bundle_sha256']!=bundle_sha or sp['manifest_sha256']!=inputs.registration['manifest_sha256']:
                            raise ValueError('Model/input identity mismatch')
                        if (sp['method'],sp['encoder_seed'],sp['sae_seed'],sp['condition'])!=(mode,enc,sae_seed,condition):raise ValueError('Model identity mismatch')
                        mean=saved['transform']['mean'].numpy()
                        if not np.allclose(mean,train.astype(np.float64).mean(0),atol=1e-7,rtol=1e-5):raise ValueError('Unexpected centering')
                        model=SAE(192,sp['plan']['dictionary_size'],sp['plan']['k']).to(device)
                        model.load_state_dict(saved['model']);model.eval()
                        codes,errors=encode(model,val-mean);train_codes,_=encode(model,train[train_idx]-mean)
                        activations[sae_seed]=(train_codes,codes,saved['model']['decoder.weight'].numpy())
                        folder=root/'analysis_fits'/mode/f'encoder_{enc}'/condition/f'sae_{sae_seed}'
                        folder.mkdir(parents=True,exist_ok=True);done=folder/'completed.json'
                        if not done.exists():
                            reference=((val.astype(np.float64)-mean)**2).sum(1)
                            pool=patient_pool(codes,labels,patients,errors,reference,plan['top_m'],device)
                            point=bootstrap_statistics(pool,np.arange(len(unique))[None,:],plan['top_m'],1)[0]
                            stats=bootstrap_statistics(pool,draws,plan['top_m'],plan['bootstrap_batch'])
                            frame=pd.DataFrame(stats,columns=BOOT_METRICS);frame.insert(0,'replicate',np.arange(len(draws)))
                            atomic_csv(frame,folder/'bootstrap_replicates.csv')
                            record={'method':mode,'encoder_seed':enc,'condition':condition,'sae_seed':sae_seed,
                                'point':dict(zip(BOOT_METRICS,point.tolist())),
                                'point_minus_archived_purity':float(point[0]-metric['mean_purity']),
                                'model_sha256':metric['model_sha256'],'replicates_sha256':sha(folder/'bootstrap_replicates.csv')}
                            atomic_json(done,record);del pool
                        else:
                            record=json.loads(done.read_text())
                            if record['model_sha256']!=metric['model_sha256'] or record['replicates_sha256']!=sha(folder/'bootstrap_replicates.csv'):
                                raise ValueError('Analysis result identity/integrity mismatch')
                        print('Bootstrap complete:',mode,enc,condition,sae_seed,flush=True)
                        analysis_summary(root);del model;gc.collect()
                    if status!='completed selected controls':break
                    for a,b in itertools.combinations(plan['sae_seeds'],2):
                        folder=root/'stability'/mode/f'encoder_{enc}'/condition/f'pair_{a}_{b}'
                        if (folder/'summary.json').exists():continue
                        ta,va,da=activations[a];tb,vb,db=activations[b]
                        table,summary=feature_match(ta,tb,va,vb,da,db,plan,device)
                        folder.mkdir(parents=True,exist_ok=True);atomic_csv(table,folder/'matched_features.csv')
                        atomic_json(folder/'summary.json',{'method':mode,'encoder_seed':enc,'condition':condition,'left_sae_seed':a,'right_sae_seed':b,**summary})
                    analysis_summary(root);del activations;gc.collect()
                del train,val;gc.collect()
                if status!='completed selected controls':break
        report={'status':status,'encoder_seeds':seeds,'methods':methods,'test_evaluated':False,'models_retrained':False,
            'elapsed_hours':(time.monotonic()-started)/3600}
        atomic_json(root/'last_session.json',report);analysis_summary(root);return report

    def summarize(root):return analysis_summary(root)

    def export_results(root):
        root=Path(root).resolve();target=root.parent/(root.name+'_ALL_ANALYSIS_RESULTS.zip')
        with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED) as z:
            for p in sorted(root.rglob('*')):
                if p.is_file() and 'input_cache' not in p.relative_to(root).parts and p.suffix in ('.csv','.json','.log') and p.name!='worker_request.json':
                    z.write(p,p.relative_to(root))
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
        rows=[json.loads(p.read_text()) for p in root.glob('analysis_fits/*/encoder_*/*/sae_*/completed.json')]
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
        'models_path':mo.ui.text(value='data/auxk_native_controls_v1_enc10-42-52_ALL_RESULTS.zip',label='Completed AuxK models ZIP (additional input)'),
        'output_dir':mo.ui.text(value='patient_stability_v1',label='Results folder (same folder to reuse completed fits)'),
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
    mo.md(f"**{len(applied['encoder_seeds'])*len(applied['methods'])*6} saved-model analyses** selected. No encoders retrained. No model training.")
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
    _models_target = _InputPath(applied['models_path']).expanduser().resolve()
    if applied['input_source']=='huggingface':
        models_path = engine.bundle_download(applied['repo_id'],engine.DEFAULT_PLAN['models_filename'],_models_target,engine.DEFAULT_PLAN['models_sha256'])
    else:
        if not _models_target.is_file() or engine.sha(_models_target)!=engine.DEFAULT_PLAN['models_sha256']:raise ValueError('Missing/wrong completed AuxK models ZIP')
        models_path = str(_models_target)
    mo.md('Embedding bundle and completed-model ZIP SHA256 verified.')
    return bundle_path, models_path

@app.cell
def _(mo):
    start_controls = mo.ui.run_button(label='Start / continue controls')
    stop_controls = mo.ui.run_button(label='Stop after completed fit')
    make_zip = mo.ui.run_button(label='Create full results ZIP')
    mo.hstack([start_controls,stop_controls,make_zip])
    return make_zip, start_controls, stop_controls

@app.cell
def _(BUNDLE_SHA256,RUNTIME_SOURCE,WORKER_SOURCE,applied,bundle_path,control,engine,models_path,mo,start_controls):
    mo.stop(not start_controls.value)
    launch_status = control.launch(applied,bundle_path,BUNDLE_SHA256,{**engine.DEFAULT_PLAN,'models_path':models_path},RUNTIME_SOURCE,WORKER_SOURCE)
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
    mo.md('ZIP ready: `'+export_path+'`. Refresh Files, then download this ZIP. It includes intervals, bootstrap replicates and feature matches; input models are excluded.')
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
    _items = [mo.md('**Worker:** '+str(_state)),mo.md(f'**Completed model analyses in output folder: {len(_fits)} / 36** for the full three-seed plan.')]
    if len(_per):_items.append(mo.ui.table(_per,selection=None))
    if len(_fits):_items.append(mo.download(data=_fits.to_csv(index=False).encode(),filename='rank_control_metrics.csv',mimetype='text/csv',label='Download metrics CSV'))
    _items.append(mo.md('```text\n'+_tail+'\n```'))
    mo.vstack(_items)
    return

if __name__=='__main__':
    app.run()
