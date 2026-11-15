"""Verify preserved rerun archives; optionally check real image-to-embedding replay.

Run inside the preserved reproduction bundle with --bundle .; the raw dataset is
optional and is never distributed by this package. No test images are accessed.
"""
from pathlib import Path
import argparse
import hashlib
import io
import json
import sys
import types
import zipfile
import numpy as np
import pandas as pd
import torch


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def check_close(actual, expected, rtol=1e-4, atol=1e-5):
    actual, expected = np.asarray(actual), np.asarray(expected)
    return {'passed': bool(np.allclose(actual, expected, rtol=rtol, atol=atol)),
            'max_abs_error': float(np.max(np.abs(actual-expected))),
            'rtol': rtol, 'atol': atol}


def verify(bundle, dataset=None):
    bundle = Path(bundle).resolve()
    manifest = json.loads((bundle/'ARTIFACTS.json').read_text())
    torch.set_num_threads(2)
    reports = []
    store = None
    for entry in manifest['runs']:
        archive_path = bundle/entry['archive']
        if sha(archive_path) != entry['sha256']:
            raise ValueError('Archive SHA256 mismatch: '+entry['archive'])
        with zipfile.ZipFile(archive_path) as archive:
            if archive.testzip() is not None:
                raise ValueError('ZIP CRC failure')
            names = archive.namelist()
            completion_name = next(n for n in names if n.endswith('/completion_verified.json'))
            prefix = completion_name.rsplit('/', 1)[0]+'/'
            completion = json.loads(archive.read(completion_name))
            method, seed = completion['method'], completion['seed']
            for filename, key in [('encoder_final.pt','final_sha256'),('best_loss.pt','best_sha256')]:
                if hashlib.sha256(archive.read(prefix+filename)).hexdigest() != completion[key]:
                    raise ValueError('Checkpoint fingerprint mismatch')
            protocol = json.loads(archive.read('protocol.json'))
            prepared = json.loads(archive.read(prefix+'prepared.json'))
            assert prepared['encoder_sha'] == completion['final_sha256']
            assert prepared['registration'] == hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
            history = json.loads(archive.read(prefix+'history.json'))
            assert [r['epoch'] for r in history] == list(range(1,51))
            frame = pd.read_csv(io.BytesIO(archive.read('split_manifest.csv')))
            assert frame.groupby('patient_id').split.nunique().max() == 1
            assert not frame.sample_id.duplicated().any()
            assert hashlib.sha256(frame.to_csv(index=False, lineterminator='\n').encode()).hexdigest() == protocol['manifest_sha']
            # Load the archived definition, not a potentially changed workspace module.
            runtime = types.ModuleType('archived_runtime')
            exec(compile(archive.read('portable_runtime.py'), 'archived_runtime', 'exec'), runtime.__dict__)
            core = types.ModuleType('archived_core')
            exec(compile(runtime.CORE_SOURCE, 'archived_core', 'exec'), core.__dict__)
            final = torch.load(io.BytesIO(archive.read(prefix+'encoder_final.pt')), map_location='cpu', weights_only=True)
            model = core.Encoder({'img_size':128}, method).eval()
            model.load_state_dict(final['model'], strict=True)
            reference = torch.load(io.BytesIO(archive.read(prefix+'verification_encoder_final/reference.pt')), map_location='cpu', weights_only=True)
            with torch.inference_mode():
                h = model.backbone(reference['x'])
                z = model(reference['x'])
            synthetic = {'h':check_close(h.numpy(),reference['h'].numpy()),
                         'z':check_close(z.numpy(),reference['z'].numpy())}
            assert all(v['passed'] for v in synthetic.values()), synthetic
            arrays = {}
            for split in ['train','val']:
                with np.load(io.BytesIO(archive.read(prefix+f'{split}_h_u_z.npz')), allow_pickle=False) as f:
                    part = frame[frame.split==split]
                    assert np.array_equal(f['sample_ids'],part.sample_id.to_numpy(dtype=str))
                    assert f['h'].shape == (len(part),192)
                    for key in ['h','u','z']:
                        assert np.isfinite(f[key]).all()
                    if split == 'val':
                        arrays = {k:f[k].copy() for k in ['h','u','z','sample_ids']}
            transform_path = prefix+'evaluation/native/transform.npz'
            with np.load(io.BytesIO(archive.read(transform_path)),allow_pickle=False) as tr:
                x = runtime.transform_h(arrays['h'][:32],tr['mean'],tr['basis'])
            fits=[]
            for sae_seed in [10,42,52]:
                fit = prefix+f'evaluation/native/baseline/sae_{sae_seed}/'
                state = torch.load(io.BytesIO(archive.read(fit+'model.pt')),map_location='cpu',weights_only=True)
                assert len(state['history']) == protocol['plan']['sae_epochs'] == 30
                sae=runtime.SAE(dictionary=protocol['plan']['dictionary_size'],k=protocol['plan']['k']).eval()
                sae.load_state_dict(state['model'],strict=True)
                with torch.inference_mode():
                    reconstruction,codes=sae(torch.from_numpy(x))
                assert torch.isfinite(reconstruction).all() and torch.isfinite(codes).all()
                assert int((codes>0).sum(1).max()) <= protocol['plan']['k']
                subset=json.loads(archive.read(fit+'subset_status.json'))
                assert fit+'validation_full.json' in names
                if subset['feasible']:
                    assert fit+'validation_matched64.json' in names
                fits.append({'seed':sae_seed,'epochs':len(state['history']),
                             'state_and_inference':'passed','matched64_feasible':subset['feasible']})
            replay=None
            if dataset:
                if store is None:
                    assert sha(dataset)==protocol['dataset_sha'], 'Dataset checksum differs'
                    index={};exec(runtime.INDEX_SOURCE,index)
                    store=index['IndexedArchive'](dataset)
                assert hashlib.sha256(store.frame[['sample_id','patient_id','slice_idx','split','has_tumor','dominant_label']].to_csv(index=False,lineterminator='\n').encode()).hexdigest()==protocol['manifest_sha']
                part=frame[frame.split=='val'].reset_index(drop=True)
                indices=np.linspace(0,len(part)-1,3,dtype=int)
                source_indices={v:i for i,v in enumerate(store.frame.sample_id)}
                images=np.stack([store.item(f'pkl:{source_indices[part.iloc[i].sample_id]}')['image'] for i in indices])
                with torch.inference_mode():
                    actual_h=model.backbone(torch.from_numpy(images).float().unsqueeze(1))
                    actual_u=model.head(actual_h)
                    actual_z=torch.relu(actual_u) if method=='ncl' else actual_u
                replay={'split':'val','sample_ids':part.iloc[indices].sample_id.tolist(),
                        'h':check_close(actual_h.numpy(),arrays['h'][indices],rtol=1e-3,atol=5e-4),
                        'u':check_close(actual_u.numpy(),arrays['u'][indices],rtol=1e-3,atol=5e-4),
                        'z':check_close(actual_z.numpy(),arrays['z'][indices],rtol=1e-3,atol=5e-4)}
                assert all(replay[k]['passed'] for k in ['h','u','z']), replay
            report={'method':method,'encoder_seed':seed,'archive_sha256':'passed',
                    'encoder_hashes':'passed','epochs':50,'manifest_and_embeddings':'passed',
                    'synthetic_reference_inference':synthetic,'sae_fits':fits,
                    'real_image_replay':replay}
            reports.append(report)
            print(f'PASS {method} seed {seed}: encoder, embeddings, 3 SAE states'+('; real MRI replay' if dataset else ''),flush=True)
    if store is not None:
        store.close()
    result={'scope':'New preserved reruns only; original reported encoders remain unavailable.',
            'status':'passed','encoder_runs':len(reports),'sae_fits':sum(len(r['sae_fits']) for r in reports),
            'test_images_accessed':False,'full_training_rerun_performed':False,
            'full_probe_refit_performed':False,'torch':str(torch.__version__),'python':sys.version,
            'runs':reports}
    (bundle/'VERIFICATION_REPORT.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--bundle',type=Path,required=True)
    parser.add_argument('--dataset',type=Path)
    args=parser.parse_args()
    verify(args.bundle,args.dataset)
