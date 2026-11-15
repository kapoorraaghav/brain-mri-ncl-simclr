"""Appended to the frozen rank runtime; overrides only the SAE experiment loop."""
DEFAULT_PLAN = {**DEFAULT_PLAN, 'version':'expanded-auxk-control-v1',
 'conditions':['baseline','auxk'], 'ranks':[], 'auxk_coefficient':1/32,
 'auxk_k':192, 'dead_after_training_passes':1.0,
 'auxk_target':'detached main reconstruction residual; no decoder bias',
 'initialization':'unchanged torch-default untied initialization in both arms',
 'decoder_normalization':False,
 'scope':'native centered archived CLS; exploratory validation; not full Gao recipe'}

def auxk_objective(model,batch,ages,threshold,aux_k):
    """Top-K positive dead-latent residual reconstruction, sharing the encoder."""
    raw=F.relu(model.encoder(batch-model.pre_bias))
    values,indices=raw.topk(model.k,dim=-1)
    acts=torch.zeros_like(raw).scatter_(-1,indices,values)
    recon=model.decoder(acts)+model.pre_bias
    # Main activity only resets dead status. Auxiliary activations never reset it.
    rows=torch.arange(len(batch),device=batch.device,dtype=torch.long)[:,None]+1
    last=torch.where(acts>0,rows,0).amax(0)
    updated=torch.where(last>0,len(batch)-last,ages+len(batch))
    dead=updated>=threshold
    n_dead=int(dead.sum().item())
    aux=recon.sum()*0
    positive_aux=0
    if n_dead:
        masked=raw.masked_fill(~dead[None,:],float('-inf'))
        v,i=masked.topk(min(aux_k,n_dead),dim=-1)
        code=torch.zeros_like(raw).scatter_(-1,i,v)
        residual=(batch-recon).detach()
        aux=F.mse_loss(model.decoder(code),residual)
        positive_aux=int((code>0).sum().item())
    return recon,acts,aux,updated,n_dead,positive_aux

def fit_one(ztrain,zval,original_val,recipe,diagnostics,labels,patients,plan,mode,encoder_seed,sae_seed,folder,identity):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    condition=recipe['condition']
    protocol={**identity,'method':mode,'encoder_seed':encoder_seed,'sae_seed':sae_seed,
        'condition':condition,'rank':192,'plan':plan}
    registration=folder/'protocol.json'
    if registration.exists() and json.loads(registration.read_text())!=protocol:
        raise ValueError('Fit protocol changed; choose a fresh output folder')
    atomic_json(registration,protocol)
    complete=folder/'validation_metrics.json';model_path=folder/'completed_model.pt'
    if complete.exists():
        record=json.loads(complete.read_text())
        if not model_path.exists() or sha(model_path)!=record['model_sha256']:raise ValueError('Missing/corrupt completed model')
        return record
    seed_all(sae_seed);device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model=SAE(192,plan['dictionary_size'],plan['k']).to(device)
    threshold=max(1,int(np.ceil(len(ztrain)*plan['dead_after_training_passes'])))
    if model_path.exists():
        saved=torch.load(model_path,map_location='cpu',weights_only=True)
        if saved['protocol']!=protocol:raise ValueError('Saved model protocol mismatch')
        model.load_state_dict(saved['model']);history=saved['history'];train_activity=saved['training_activity']
    else:
        optimizer=torch.optim.Adam(model.parameters(),lr=plan['lr'])
        loader=torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.from_numpy(ztrain)),
            batch_size=plan['batch_size'],shuffle=True)
        ages=torch.zeros(plan['dictionary_size'],dtype=torch.long,device=device)
        counts=torch.zeros_like(ages);history=[]
        for epoch in range(plan['epochs']):
            total=aux_total=objective_total=0.;seen=aux_batches=aux_positive=0;max_dead=0
            model.train()
            for (batch,) in loader:
                batch=batch.to(device);optimizer.zero_grad(set_to_none=True)
                if condition=='auxk':
                    recon,acts,aux,ages,n_dead,n_positive=auxk_objective(model,batch,ages,threshold,plan['auxk_k'])
                    aux_batches+=int(n_dead>0);aux_positive+=n_positive;max_dead=max(max_dead,n_dead)
                else:
                    recon,acts=model(batch);aux=recon.sum()*0
                    rows=torch.arange(len(batch),device=device,dtype=torch.long)[:,None]+1
                    last=torch.where(acts>0,rows,0).amax(0)
                    ages=torch.where(last>0,len(batch)-last,ages+len(batch))
                counts+=(acts.detach()>0).sum(0)
                main=F.mse_loss(recon,batch)
                objective=main+plan['auxk_coefficient']*aux
                if not torch.isfinite(objective):raise ValueError('Non-finite AuxK objective')
                objective.backward();nn.utils.clip_grad_norm_(model.parameters(),plan['gradient_clip']);optimizer.step()
                total+=float(main.detach())*len(batch);aux_total+=float(aux.detach())*len(batch)
                objective_total+=float(objective.detach())*len(batch);seen+=len(batch)
            item={'epoch':epoch+1,'train_mse':total/seen,'auxiliary_mse':aux_total/seen,
                'objective':objective_total/seen,'auxiliary_batches':aux_batches,
                'positive_auxiliary_activations':aux_positive,'max_dead_candidates':max_dead,
                'dead_at_epoch_end':int((ages>=threshold).sum()),'alive_training_cumulative':int((counts>0).sum())}
            history.append(item);atomic_json(folder/'history.json',history)
            print(mode,encoder_seed,condition,'SAE',sae_seed,item,flush=True)
        train_activity={'positive_counts':counts.cpu(),'samples_since_main_activation':ages.cpu(),
            'dead_threshold_samples':threshold,'auxiliary_batches':sum(h['auxiliary_batches'] for h in history),
            'positive_auxiliary_activations':sum(h['positive_auxiliary_activations'] for h in history)}
        payload={'model':{k:v.detach().cpu() for k,v in model.state_dict().items()},'protocol':protocol,
            'history':history,'training_activity':train_activity,'training_environment':environment(),
            'transform':{k:torch.from_numpy(v) if isinstance(v,np.ndarray) else v for k,v in recipe.items()}}
        temp=model_path.with_suffix('.tmp');torch.save(payload,temp);replace_file(temp,model_path)
    model.eval()
    metrics,features,patient_features=evaluate(model,zval,original_val,recipe,labels,patients,plan,sae_seed)
    atomic_csv(features,folder/'validation_features.csv');atomic_csv(patient_features,folder/'patient_distinct_features.csv')
    counts=train_activity['positive_counts'];ages=train_activity['samples_since_main_activation']
    metrics.update(training_alive_features=int((counts>0).sum()),training_dead_at_end=int((ages>=threshold).sum()),
        auxiliary_batches=train_activity['auxiliary_batches'],positive_auxiliary_activations=train_activity['positive_auxiliary_activations'])
    result={**metrics,'method':mode,'encoder_seed':encoder_seed,'sae_seed':sae_seed,'condition':condition,
        'rank':192,'k':plan['k'],'epochs':plan['epochs'],'split':'val','diagnostics':diagnostics,
        'model_sha256':sha(model_path),'selection_scope':'exploratory AuxK control; no test evaluation'}
    atomic_json(complete,result);del model;gc.collect()
    if torch.cuda.is_available():torch.cuda.empty_cache()
    return result

_base_summarize=summarize
def summarize(root):
    frame,means,summary=_base_summarize(root)
    differences=[]
    if len(means):
        for (mode,seed),g in means.groupby(['method','encoder_seed']):
            expected=len(json.loads((Path(root)/'protocol.json').read_text())['plan']['sae_seeds'])
            g=g[g.sae_replicates==expected]
            if set(g.condition)!={'baseline','auxk'}:continue
            a=g[g.condition=='auxk'].iloc[0];b=g[g.condition=='baseline'].iloc[0]
            metrics=['alive_features','eligible_features','normalized_SAE_EV','mean_purity','purity_minus_permutation',
                'patient_distinct_purity','patient_distinct_purity_minus_permutation']
            differences.append({'method':mode,'encoder_seed':int(seed),
                **{'auxk_minus_baseline_'+m:float(a[m]-b[m]) for m in metrics}})
    atomic_csv(pd.DataFrame(differences),Path(root)/'auxk_minus_baseline.csv')
    return frame,means,summary

def run(bundle,bundle_sha,output,plan=None,seeds=None,methods=None,max_hours=8,require_cuda=True):
    plan=json.loads(json.dumps(DEFAULT_PLAN if plan is None else plan))
    seeds=plan['encoder_seeds'] if seeds is None else list(seeds)
    methods=plan['methods'] if methods is None else list(methods)
    if not seeds or set(seeds)-set(plan['encoder_seeds']) or not methods or set(methods)-set(plan['methods']):raise ValueError('Invalid method/seed selection')
    if plan['conditions']!=['baseline','auxk'] or plan['auxk_k']<1 or plan['auxk_coefficient']<=0 or plan['dead_after_training_passes']<=0:
        raise ValueError('Invalid registered AuxK plan')
    if require_cuda and not torch.cuda.is_available():raise RuntimeError('Attach a CUDA GPU')
    root=Path(output).expanduser().resolve();root.mkdir(parents=True,exist_ok=True)
    started=time.monotonic();inputs=Inputs(bundle,root/'input_cache',bundle_sha)
    identity={'bundle_sha256':bundle_sha,'manifest_sha256':inputs.registration['manifest_sha256'],
        'runtime_sha256':globals().get('RUNTIME_SHA256','development'),'plan':plan}
    registration=root/'protocol.json'
    if registration.exists() and json.loads(registration.read_text())!=identity:raise ValueError('Protocol changed; use a fresh AuxK output folder')
    atomic_json(registration,identity);atomic_json(root/'environment.json',environment())
    val=inputs.frame[inputs.frame.split=='val'];train_frame=inputs.frame[inputs.frame.split=='train']
    status='completed selected controls'
    for seed,mode in itertools.product(seeds,methods):
        train=inputs.load(mode,seed,'train');validation=inputs.load(mode,seed,'val')
        ztrain,zval,base_recipe,diagnostics=build_transform(inputs,mode,seed,'native',192,train,validation)
        probe_result=None
        for condition in plan['conditions']:
            recipe={**base_recipe,'condition':condition};case_dir=root/'fits'/mode/f'encoder_{seed}'/f'{condition}_r192'
            case_dir.mkdir(parents=True,exist_ok=True)
            atomic_json(case_dir/'transform_diagnostics.json',diagnostics)
            probe_path=case_dir/'validation_probe.json'
            if not probe_path.exists():
                if probe_result is None:probe_result=probe(ztrain,zval,train_frame.has_tumor.to_numpy(),val.has_tumor.to_numpy(),plan)
                atomic_json(probe_path,probe_result)
            for sae_seed in plan['sae_seeds']:
                if (root/'stop_after_fit.flag').exists() or time.monotonic()-started>=max_hours*3600:
                    status='stopped between fits; completed dictionaries retained';break
                fit_one(ztrain,zval,validation,recipe,diagnostics,val.dominant_label.to_numpy(),val.patient_id.to_numpy(),
                    plan,mode,seed,sae_seed,case_dir/f'sae_{sae_seed}',identity)
                summarize(root)
            if status!='completed selected controls':break
        del train,validation,ztrain,zval;gc.collect()
        if status!='completed selected controls':break
    report={'status':status,'encoder_seeds':seeds,'methods':methods,'elapsed_hours':(time.monotonic()-started)/3600,
        'test_evaluated':False,'encoders_retrained':False,'per_epoch_checkpoints':False,'completed_models_saved_once':True}
    atomic_json(root/'last_session.json',report);summarize(root);return report
