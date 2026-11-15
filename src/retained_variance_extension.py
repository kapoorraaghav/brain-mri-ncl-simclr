"""Appended to the frozen rank runtime. Ranks are selected from training only."""
DEFAULT_PLAN={**DEFAULT_PLAN,'version':'expanded-retained-variance-v1',
 'conditions':['retained_95','retained_99'],'ranks':[],
 'variance_targets':{'retained_95':0.95,'retained_99':0.99},
 'primary_condition':'retained_95',
 'scope':'exploratory validation sensitivity; comparable TRAIN variance fraction, not equal information or rank'}
_rank_transform=build_transform
def retained_rank(eigenvalues,fraction):
    values=np.maximum(np.asarray(eigenvalues,dtype=float),0)
    if not 0<fraction<1 or values.sum()<=0:raise ValueError('Invalid variance target or covariance')
    return min(len(values),int(np.searchsorted(np.cumsum(values)/values.sum(),fraction,side='left'))+1)

def build_transform(inputs,mode,seed,condition,rank,train,val):
    if condition not in DEFAULT_PLAN['variance_targets']:
        raise ValueError('Unknown registered retained-variance condition')
    fraction=DEFAULT_PLAN['variance_targets'][condition]
    values=inputs.pca(mode,seed,train)['eigenvalues']
    selected=retained_rank(values,fraction)
    zt,zv,recipe,diagnostics=_rank_transform(inputs,mode,seed,'pca_rank',selected,train,val)
    recipe['condition']=condition
    diagnostics.update(condition=condition,requested_training_variance_fraction=fraction,
        rank_selection_scope='minimal dimension meeting target on training covariance only',
        previous_rank_train_variance_fraction=float(values[:selected-1].sum()/values.sum()))
    assert diagnostics['PCA_train_variance_retained']>=fraction-1e-10
    assert selected==1 or diagnostics['previous_rank_train_variance_fraction']<fraction
    return zt,zv,recipe,diagnostics

def summarize(root):
    root=Path(root)
    records=[json.loads(p.read_text()) for p in root.glob('fits/*/encoder_*/*/sae_*/validation_metrics.json')]
    if not records:return pd.DataFrame(),pd.DataFrame(),pd.DataFrame()
    frame=pd.json_normalize(records)
    metrics=['normalized_SAE_EV','original_coordinate_EV','alive_features','eligible_features',
        'mean_purity','purity_minus_permutation','patient_distinct_purity',
        'patient_distinct_purity_minus_permutation','patient_distinct_eligible_features',
        'diagnostics.PCA_train_variance_retained','diagnostics.PCA_only_validation_EV']
    group=['method','encoder_seed','condition']
    per=frame.groupby(group)[['rank',*metrics]].mean().reset_index()
    per['sae_replicates']=frame.groupby(group).size().to_numpy()
    plan=json.loads((root/'protocol.json').read_text())['plan']
    complete=per[per.sae_replicates==len(plan['sae_seeds'])]
    summary=complete.groupby(['method','condition'])[['rank',*metrics]].agg(['mean','std','count']).reset_index()
    summary.columns=['_'.join(str(v) for v in c if str(v)) for c in summary.columns]
    differences=[]
    for (seed,condition),g in complete.groupby(['encoder_seed','condition']):
        if set(g.method)!={'ncl','simclr'}:continue
        a=g[g.method=='ncl'].iloc[0];b=g[g.method=='simclr'].iloc[0]
        differences.append({'encoder_seed':int(seed),'condition':condition,
            'ncl_rank':int(a['rank']),'simclr_rank':int(b['rank']),
            **{'ncl_minus_simclr_'+m:float(a[m]-b[m]) for m in metrics}})
    atomic_csv(frame,root/'all_control_fits.csv');atomic_csv(per,root/'per_encoder_controls.csv')
    atomic_csv(summary,root/'control_summary.csv')
    atomic_csv(pd.DataFrame(differences),root/'paired_control_differences.csv')
    return frame,per,summary

def run(bundle,bundle_sha,output,plan=None,seeds=None,methods=None,max_hours=8,require_cuda=True):
    plan=json.loads(json.dumps(DEFAULT_PLAN if plan is None else plan))
    seeds=plan['encoder_seeds'] if seeds is None else list(seeds)
    methods=plan['methods'] if methods is None else list(methods)
    if not seeds or set(seeds)-set(plan['encoder_seeds']) or not methods or set(methods)-set(plan['methods']):raise ValueError('Invalid selection')
    if plan['variance_targets']!=DEFAULT_PLAN['variance_targets'] or plan['conditions']!=DEFAULT_PLAN['conditions']:
        raise ValueError('Variance targets/conditions differ from registered protocol')
    if require_cuda and not torch.cuda.is_available():raise RuntimeError('Attach a CUDA GPU')
    root=Path(output).expanduser().resolve();root.mkdir(parents=True,exist_ok=True)
    started=time.monotonic();inputs=Inputs(bundle,root/'input_cache',bundle_sha)
    identity={'bundle_sha256':bundle_sha,'manifest_sha256':inputs.registration['manifest_sha256'],
        'runtime_sha256':globals().get('RUNTIME_SHA256','development'),'plan':plan}
    registration=root/'protocol.json'
    if registration.exists() and json.loads(registration.read_text())!=identity:raise ValueError('Protocol changed; use a fresh output folder')
    atomic_json(registration,identity);atomic_json(root/'environment.json',environment())
    vf=inputs.frame[inputs.frame.split=='val'];tf=inputs.frame[inputs.frame.split=='train']
    status='completed selected controls'
    for seed,mode in itertools.product(seeds,methods):
        train=inputs.load(mode,seed,'train');val=inputs.load(mode,seed,'val')
        for condition in plan['conditions']:
            if (root/'stop_after_fit.flag').exists() or time.monotonic()-started>=max_hours*3600:
                status='stopped between fits; completed dictionaries retained';break
            zt,zv,recipe,diag=build_transform(inputs,mode,seed,condition,None,train,val)
            case_dir=root/'fits'/mode/f'encoder_{seed}'/f'{condition}_r{recipe["rank"]}'
            case_dir.mkdir(parents=True,exist_ok=True);atomic_json(case_dir/'transform_diagnostics.json',diag)
            if not (case_dir/'validation_probe.json').exists():
                atomic_json(case_dir/'validation_probe.json',probe(zt,zv,tf.has_tumor.to_numpy(),vf.has_tumor.to_numpy(),plan))
            for sae_seed in plan['sae_seeds']:
                if (root/'stop_after_fit.flag').exists() or time.monotonic()-started>=max_hours*3600:
                    status='stopped between fits; completed dictionaries retained';break
                fit_one(zt,zv,val,recipe,diag,vf.dominant_label.to_numpy(),vf.patient_id.to_numpy(),
                    plan,mode,seed,sae_seed,case_dir/f'sae_{sae_seed}',identity)
                summarize(root)
            del zt,zv;gc.collect()
            if status!='completed selected controls':break
        del train,val;gc.collect()
        if status!='completed selected controls':break
    report={'status':status,'encoder_seeds':seeds,'methods':methods,'elapsed_hours':(time.monotonic()-started)/3600,
        'test_evaluated':False,'encoders_retrained':False,'per_epoch_checkpoints':False,'completed_models_saved_once':True}
    atomic_json(root/'last_session.json',report);summarize(root);return report
