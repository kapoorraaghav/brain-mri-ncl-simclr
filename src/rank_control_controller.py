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
    rows=[json.loads(p.read_text()) for p in root.glob('fits/*/encoder_*/*/sae_*/validation_metrics.json')]
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
