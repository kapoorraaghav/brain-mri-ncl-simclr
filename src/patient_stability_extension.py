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
