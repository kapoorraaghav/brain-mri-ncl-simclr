"""Check audit arithmetic, saved rankings, input alignment and model fingerprints."""
from pathlib import Path
import importlib.util,io,json,zipfile
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/feature_population_audit_v1'
spec=importlib.util.spec_from_file_location('feature_audit',ROOT/'scripts/feature_population_audit.py')
audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)


def main():
    protocol=json.loads((OUT/'audit_protocol.json').read_text())
    for key,path in [('embeddings_sha256',audit.EMBED),('historical_sae_archive_sha256',audit.MODELS),('covariates_sha256',audit.COV)]:
        assert audit.sha(path)==protocol['inputs'][key],key
    rng=np.random.default_rng(14)
    for n in [20,99,203]:
        for keep in [0,.1,.7,1.]:
            y=np.arange(n)%2;x=rng.integers(0,6,n).astype(float)*(rng.random(n)<keep)
            ids=np.flatnonzero(x>0)
            assert np.isclose(audit.auc_sparse(x[ids],ids,y),roc_auc_score(y,x),atol=1e-12)
    with zipfile.ZipFile(audit.EMBED) as e,zipfile.ZipFile(audit.MODELS) as m:
        frame=pd.read_csv(io.BytesIO(e.read('split_manifest.csv')))
        val=frame[frame.split=='val'].reset_index(drop=True)
        cov=pd.read_csv(audit.COV);assert np.array_equal(cov.sample_id,val.sample_id)
        assert np.array_equal(cov.tumour_area_fraction.to_numpy()==0,val.has_tumor.to_numpy()==0)
        for bins in [3,4,5]:
            s,_=audit.strata(cov,bins);original=audit.PLAN['null_permutations'];audit.PLAN['null_permutations']=5
            perms=audit.permuted_labels(val.dominant_label.to_numpy(),s);audit.PLAN['null_permutations']=original
            for p in perms:
                for cell in np.unique(s):
                    assert np.array_equal(np.sort(p[s==cell]),np.sort(val.dominant_label.to_numpy()[s==cell]))
        complete=list(OUT.glob('fits/*/encoder_*/sae_*/complete.json'));assert len(complete)==27
        for marker in complete:
            r=json.loads(marker.read_text());method,enc,seed=r['method'],r['encoder_seed'],r['sae_seed']
            if method=='supervised':
                path=OUT/'supervised_models'/f'encoder_{enc}'/f'sae_{seed}'/'completed_model.pt'
                assert audit.sha(path)==r['model_sha256']
                state=torch.load(path,map_location='cpu',weights_only=True)
            else:
                raw=m.read(f'fits/{method}/encoder_{enc}/baseline_r192/sae_{seed}/completed_model.pt')
                import hashlib
                assert hashlib.sha256(raw).hexdigest()==r['model_sha256']
                state=torch.load(io.BytesIO(raw),map_location='cpu',weights_only=True)
            assert len(state['history'])==30
            f=pd.read_csv(marker.parent/'features.csv')
            ids=np.load(marker.parent/'eligible_feature_ids.npy');tops=np.load(marker.parent/'top10_indices.npy')
            assert np.array_equal(ids,np.flatnonzero(f.val_support.to_numpy()>=10))
            assert tops.shape==(len(ids),10)
            labels=val.dominant_label.to_numpy()[tops]
            p=np.stack([(labels==c).mean(1) for c in [0,1,2,4]]).max(0)
            assert np.allclose(p,f.loc[ids,'purity'],atol=1e-12)
            oriented=np.where(f.train_auc>=.5,f.val_auc_positive_activation,1-f.val_auc_positive_activation)
            assert np.allclose(oriented,f.val_auc_train_oriented,atol=1e-12)
            b=pd.read_csv(marker.parent/'budgets.csv')
            assert ((b.coverage>=0)&(b.coverage<=1)).all()
            assert (b.purity_lower<=b.purity_conditional+1e-12).all()
            assert (b.purity_upper+1e-12>=b.purity_conditional).all()
            eligible=b[b.selector=='validation_eligible_training_usage'].iloc[-1]
            assert int(eligible.J)==len(ids)
            assert np.isclose(eligible.purity_conditional,p.mean(),atol=1e-12)
            assert np.isclose(eligible.random_mean,p.mean(),atol=1e-12)
        result={'status':'passed','dictionary_states_checked':27,'new_supervised_states_checked':9,
                'registered_input_hashes':'passed',
                'sparse_auroc_sklearn_tie_cases':12,'strata_count_preservation':'passed',
                'saved_top10_purity_recalculation':'passed','orientation_training_only':'passed',
                'bounds_and_full_pool_random_endpoint':'passed','test_images_accessed':False}
        (OUT/'VERIFICATION.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
        print(json.dumps(result,indent=2))


if __name__=='__main__':main()
