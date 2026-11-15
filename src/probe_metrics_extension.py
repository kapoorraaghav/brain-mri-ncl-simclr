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
