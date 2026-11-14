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
