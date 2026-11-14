"""CLI for the curated repository. Copied to scripts/run_embedding_experiment.py."""
from pathlib import Path
import argparse,hashlib,json,types
ROOT=Path(__file__).resolve().parents[1]
EXTENSIONS={'rank':None,'auxk':'auxk_control_extension.py','patient':'patient_stability_extension.py',
    'retention':'retained_variance_extension.py','probe':'probe_metrics_extension.py'}
def main():
    p=argparse.ArgumentParser(description='Run fixed-embedding MRI SAE controls; no encoder training.')
    p.add_argument('--stage',choices=EXTENSIONS,required=True)
    p.add_argument('--bundle',type=Path,default=ROOT/'data/rank_control_embeddings_v1.zip')
    p.add_argument('--models',type=Path,help='Required for patient stage: complete 36-model AuxK ZIP')
    p.add_argument('--output',type=Path)
    p.add_argument('--encoder-seeds',nargs='+',type=int,choices=[10,42,52],default=[10,42,52])
    p.add_argument('--methods',nargs='+',choices=['ncl','simclr'],default=['ncl','simclr'])
    p.add_argument('--max-hours',type=float,default=8)
    p.add_argument('--allow-cpu',action='store_true',help='Allow CPU for SAE/analysis stages; may be slow')
    p.add_argument('--dry-run',action='store_true',help='Inspect paths/plan size only; no data load or training')
    args=p.parse_args();output=(args.output or ROOT/'results'/f'{args.stage}_v1').resolve()
    if args.max_hours<=0:p.error('--max-hours must be positive')
    counts={'rank':9,'auxk':6,'patient':6,'retention':6,'probe':5}
    if args.dry_run:
        print(json.dumps({'stage':args.stage,'bundle':str(args.bundle.resolve()),'bundle_exists':args.bundle.is_file(),
            'output':str(output),'encoder_seeds':args.encoder_seeds,'methods':args.methods,
            'work_units':len(args.encoder_seeds)*len(args.methods)*counts[args.stage],
            'requires_models_archive':args.stage=='patient','encoders_trained':False},indent=2));return
    if not args.bundle.is_file():p.error('Embedding bundle not found; see data/README.md')
    if args.stage=='patient' and (args.models is None or not args.models.is_file()):p.error('--models must point to the completed AuxK archive')
    source=(ROOT/'src/rank_control_runtime.py').read_text(encoding='utf-8')
    if EXTENSIONS[args.stage]:source+='\n'+(ROOT/'src'/EXTENSIONS[args.stage]).read_text(encoding='utf-8')
    engine=types.ModuleType('experiment_runtime');exec(compile(source,'experiment_runtime','exec'),engine.__dict__)
    engine.RUNTIME_SHA256=hashlib.sha256(source.encode()).hexdigest()
    plan=json.loads(json.dumps(engine.DEFAULT_PLAN))
    if args.models:plan['models_path']=str(args.models.resolve())
    receipt=json.loads((ROOT/'configs/rank_control_embeddings_v1_receipt.json').read_text())
    report=engine.run(str(args.bundle.resolve()),receipt['sha256'],str(output),plan,
        seeds=args.encoder_seeds,methods=args.methods,max_hours=args.max_hours,
        require_cuda=args.stage!='probe' and not args.allow_cpu)
    print(json.dumps(report,indent=2));print('Export:',engine.export_results(output))
if __name__=='__main__':main()
