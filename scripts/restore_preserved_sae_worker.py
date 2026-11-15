"""Restore a preserved encoder ZIP and prepare native SAEs without MRI access."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import shutil
import sys
import time
import types
import zipfile


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def run(source, output, verify_only=False):
    source = Path(source).expanduser().resolve()
    output = Path(output).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f'Upload the encoder ZIP first. Missing input: {source}')
    output.mkdir(parents=True, exist_ok=True)
    fingerprint = sha(source)
    receipt_path = output / 'restore_input.json'
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        if receipt['zip_sha256'] != fingerprint:
            raise ValueError('Another input ZIP used this output folder. Choose a new OUTPUT_DIR.')
    else:
        if any(output.iterdir()):
            raise ValueError('Choose an empty output folder for the first restore.')
        receipt_path.write_text(json.dumps({'zip_sha256': fingerprint,
                                           'source_name': source.name}, indent=2))
    print(f'INPUT ZIP: {source}\nOUTPUT FOLDER: {output}', flush=True)
    with zipfile.ZipFile(source) as archive:
        if archive.testzip() is not None:
            raise ValueError('Input ZIP integrity check failed')
        runtimes = [n for n in archive.namelist() if n.endswith('portable_runtime.py')]
        if len(runtimes) != 1:
            raise ValueError('Upload one preserved encoder run ZIP with portable_runtime.py.')
        prefix = runtimes[0][:-len('portable_runtime.py')]
        for member in archive.infolist():
            if not member.filename.startswith(prefix):
                continue
            relative = member.filename[len(prefix):]
            if not relative or member.is_dir():
                continue
            destination = (output / relative).resolve()
            if not destination.is_relative_to(output):
                raise ValueError('Unsafe archive member')
            if destination.name in ['STOP', 'export_manifest.json']:
                continue
            if destination.exists():
                continue  # Resume files are never overwritten by the input ZIP.
            destination.parent.mkdir(parents=True, exist_ok=True)
            temp = destination.with_name(destination.name + '.restore_tmp')
            with archive.open(member) as src, temp.open('wb') as dst:
                shutil.copyfileobj(src, dst)
            os.replace(temp, destination)
    completions = list(output.glob('final/*/*/seed_*/completion_verified.json'))
    if len(completions) != 1:
        raise ValueError('ZIP must contain exactly one completed encoder run.')
    folder = completions[0].parent
    complete = json.loads(completions[0].read_text())
    for filename, key in [('encoder_final.pt', 'final_sha256'), ('best_loss.pt', 'best_sha256')]:
        if sha(folder / filename) != complete[key]:
            raise ValueError(f'Checkpoint hash mismatch: {filename}')
    registration = json.loads((output / 'protocol.json').read_text())
    method, seed = complete['method'], int(complete['seed'])
    print(f'VERIFIED ENCODER: {method}, seed {seed}', flush=True)
    if verify_only:
        print('Restore and checkpoint hashes passed. No training performed.', flush=True)
        return

    # Execute the archived training recipe, with only a documented numerical
    # iteration ceiling adjustment. No original encoder is trained or changed.
    runtime = (output / 'portable_runtime.py').read_text(encoding='utf-8')
    old = 'if int(lr.n_iter_.max())>=2000:'
    runtime = runtime.replace(old, "if int(lr.n_iter_.max())>=plan['probe_max_iter']:")
    engine = types.ModuleType('restored_sae_runtime')
    exec(compile(runtime, 'archived_portable_runtime', 'exec'), engine.__dict__)
    if not engine.torch.cuda.is_available():
        raise RuntimeError('Attach a CUDA GPU to this Molab runtime.')
    engine.torch.set_num_threads(2)
    engine.torch.use_deterministic_algorithms(True)
    runner = engine.Runner.__new__(engine.Runner)
    runner.root = output
    runner.plan = engine.copy.deepcopy(registration['plan'])
    original_ceiling = runner.plan['probe_max_iter']
    runner.plan['probe_max_iter'] = max(20000, original_ceiling)
    runner.registration = registration
    runner.started = time.monotonic()
    runner.max_seconds = 72 * 3600
    runner.backup = None
    runner.device = engine.torch.device('cuda')
    runner.core = types.ModuleType('restored_core')
    exec(compile(engine.CORE_SOURCE, 'archived_core', 'exec'), runner.core.__dict__)
    runner.frame = engine.pd.read_csv(output / 'split_manifest.csv')
    manifest = runner.frame.to_csv(index=False, lineterminator='\n')
    if hashlib.sha256(manifest.encode()).hexdigest() != registration['manifest_sha']:
        raise ValueError('Manifest does not match registered metadata.')
    for split in ['train', 'val']:
        with engine.np.load(folder / f'{split}_h_u_z.npz', allow_pickle=False) as arrays:
            frame = runner.frame[runner.frame.split == split]
            if not engine.np.array_equal(arrays['sample_ids'], frame.sample_id.to_numpy(dtype=str)):
                raise ValueError('Embedding sample ordering mismatch')
            if not engine.np.isfinite(arrays['h']).all():
                raise ValueError('Nonfinite embeddings')
    engine.atomic_json(output / 'restored_sae_execution.json', {
        'source_zip_sha256': fingerprint, 'method': method, 'encoder_seed': seed,
        'torch': str(engine.torch.__version__), 'python': sys.version,
        'gpu': engine.torch.cuda.get_device_name(0),
        'probe_max_iter_original': original_ceiling,
        'probe_max_iter_execution': runner.plan['probe_max_iter'],
        'numerical_override': 'Convergence ceiling only; C grid, folds, tolerance and objective unchanged',
        'executed_runtime_sha256': hashlib.sha256(runtime.encode()).hexdigest(),
        'encoder_retrained': False, 'raw_dataset_required': False, 'test_images_accessed': False})
    original_probe = engine.fit_probe

    def visible_probe(*args, **kwargs):
        print(f'CPU PROBE START: {args[2]}', flush=True)
        started = time.monotonic()
        result = original_probe(*args, **kwargs)
        print(f'CPU PROBE FINISHED: {time.monotonic()-started:.1f}s', flush=True)
        return result

    engine.fit_probe = visible_probe
    print('Preparing 3 SAE seeds, 30 epochs each. Saved SAE models are reused.', flush=True)
    engine.prepare(runner, [method], [seed])
    for sae_seed in runner.plan['sae_seeds']:
        fit = folder / 'evaluation' / 'native' / 'baseline' / f'sae_{sae_seed}'
        for filename in ['model.pt', 'validation_full.json', 'subset_status.json']:
            if not (fit / filename).is_file():
                raise FileNotFoundError(fit / filename)
        subset = json.loads((fit / 'subset_status.json').read_text())
        if subset['feasible'] and not (fit / 'validation_matched64.json').is_file():
            raise FileNotFoundError(fit / 'validation_matched64.json')
    target = engine.export_preserved(output)
    engine.atomic_json(output.parent / (output.name + '_DOWNLOAD_READY.json'), {
        'zip_path': target, 'zip_sha256': sha(target),
        'status': 'complete', 'sae_fits': len(runner.plan['sae_seeds'])})
    print(f'ALL DONE. DOWNLOAD THIS ZIP: {target}', flush=True)
    print('Download before closing the Molab runtime.', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    run(args.input, args.output, args.verify_only)
