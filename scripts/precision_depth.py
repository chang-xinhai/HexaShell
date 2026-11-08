"""Five lanes, native RoboDojo evaluation; dry-run by default.

Sensor/env YAML is explicit input: do not reimplement RobotManager's camera merge here.
"""
import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import threading
import urllib.request

LANES=('rgb','gt','dav2','tradsl','nsl')
STOP=threading.Event()


def cleanup(processes, grace=15):
    for process in reversed(processes):
        try:os.killpg(process.pid,signal.SIGTERM)
        except ProcessLookupError:pass
    deadline=time.monotonic()+grace
    while time.monotonic()<deadline:
        alive=False
        for process in processes:
            process.poll()
            try:os.killpg(process.pid,0);alive=True
            except ProcessLookupError:pass
        if not alive:break
        time.sleep(.1)
    for process in reversed(processes):
        try:os.killpg(process.pid,signal.SIGKILL)
        except ProcessLookupError:pass
        process.wait()


def ready(process, port, backend):
    deadline=time.monotonic()+300
    while time.monotonic()<deadline:
        if STOP.is_set():raise InterruptedError('launcher interrupted')
        if process.poll() is not None:raise RuntimeError(f'{backend} service exited')
        try:
            with urllib.request.urlopen(f'http://127.0.0.1:{port}/health',timeout=1) as response:
                health=json.load(response)
            if health.get('initialized') and health.get('backend')==backend:return
        except OSError:pass
        time.sleep(.5)
    raise TimeoutError(f'{backend} service not ready')


def lane(args, name, gpu, env_config, tasks, output):
    env=dict(os.environ,HEXASHELL_DEPTH='0' if name=='rgb' else '1',HEXASHELL_DEPTH_BACKEND='gt' if name=='rgb' else name)
    env['EVAL_ENV_TYPE']='sim'
    env.pop('PYTHONPATH',None);env.pop('HEXASHELL_DEPTH_URL',None);env.pop('CUDA_VISIBLE_DEVICES',None);env.pop('HEXASHELL_SMOKE_STEPS',None)
    directory=output/name;directory.mkdir()
    processes=[];logs=[];rows=[];services=[]
    def start(argv,cwd,label,*,model=False):
        log=(directory/(label+'.log')).open('w');logs.append(log)
        process=subprocess.Popen(argv,cwd=cwd,env=dict(env,CUDA_VISIBLE_DEVICES=gpu) if model else env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        processes.append(process);return process
    try:
        if name not in ('rgb','gt'):
            import socket
            for port in ([19124,19130] if name=='nsl' else [19120+LANES.index(name)]):
                with socket.socket() as sock:
                    sock.bind(('127.0.0.1',port))
            python=str(args.data_root/'envs'/('nsl-infer' if name=='nsl' else name)/'bin/python')
            port=19120+LANES.index(name)
            if name=='nsl':
                command=[python,'-m','record.depth.nsl_v1_5.inference_server','--host','127.0.0.1','--port','19130','--device','cuda:0',
                         '--weights',str(args.data_root/'models/nsl/nsl1.5_lingbot_giant_detail_global_200k.pt'),
                         '--config',str(args.data_root/'models/nsl/nsl1.5_lingbot_giant_detail_global_200k.yaml'),'--precision','mixed','--no-conf','--startup-timeout','45','--inference-timeout','15']
                native=start(command,args.depth_root/'nsl_infer_server/server','nsl-native',model=True);services.append(native)
                # Native socket/provenance is checked by the proxy; retry while native initializes.
                for _ in range(300):
                    if STOP.is_set():raise InterruptedError("launcher interrupted")
                    if native.poll() is not None:raise RuntimeError('native NSL exited')
                    import socket
                    with socket.socket() as sock:
                        if sock.connect_ex(('127.0.0.1',19130))==0:break
                    time.sleep(1)
                else:raise TimeoutError('native NSL did not bind')
            command=[python,str(args.here/'services/depth_server.py'),'--backend',name,'--depth-root',str(args.depth_root),'--port',str(port)]
            if name=='dav2':command+=['--weights',str(args.data_root/'models/dav2-metric-indoor/depth_anything_v2_metric_hypersim_vitl.pth')]
            service=start(command,args.here,'depth',model=True);services.append(service);ready(service,port,name)
            env['HEXASHELL_DEPTH_URL']=f'http://127.0.0.1:{port}'
        for task in tasks:
            if STOP.is_set():raise InterruptedError("launcher interrupted")
            env['HEXASHELL_RESULT_DIR']=str(directory/task)
            run_id=f'{output.name}-{name}-{task}';env['ROBODOJO_RUN_ID']=run_id
            config=os.path.relpath(Path(env_config).resolve().with_suffix(''),args.robo/'env_cfg')
            command=['bash',str(args.robo/'scripts/robodojo.sh'),'eval','--policy-dir',str(args.policy_dir),'--task',task,
                     '--ckpt','no-checkpoint','--action-type','joint','--policy-env',env['POLICY_ENV'],'--eval-env',env['EVAL_ENV'],
                     '--env-cfg',config,'--policy-gpu',gpu,'--env-gpu',gpu,'--seed',str(args.seed),'--eval-num',str(args.eval_num)]
            process=start(command,args.robo,task)
            while process.poll() is None:
                if STOP.is_set():raise InterruptedError('launcher interrupted')
                if any(p.poll() is not None for p in services):raise RuntimeError('depth service exited during evaluation')
                time.sleep(1)
            code=process.returncode;rows.append(dict(task=task,exit_code=code,run_id=run_id))
            if code:raise RuntimeError(f'{name}/{task} exited {code}')
    finally:
        cleanup(processes)
        for log in logs:log.close()
        (directory/'exits.json').write_text(json.dumps(rows,indent=2))
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--robo',type=Path,required=True);p.add_argument('--depth-root',type=Path,required=True)
    p.add_argument('--data-root',type=Path,required=True);p.add_argument('--policy-dir',type=Path,required=True)
    p.add_argument('--env-configs',nargs=5,metavar='YAML')
    p.add_argument('--gpus',nargs=5,required=True);p.add_argument('--tasks',nargs='+')
    p.add_argument('--seed',type=int,default=0);p.add_argument('--eval-num',type=int,default=1)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--execute',action='store_true')
    args=p.parse_args();args.here=Path(__file__).resolve().parents[1]
    if args.env_configs is None:args.env_configs=[str(args.here/'config/precision-depth'/(n+'.yml')) for n in LANES]
    import sys
    tasks=subprocess.check_output([sys.executable,str(args.robo/'scripts/internal/task_inventory.py'),
                                   '--dimension','precision','--only-runnable'],text=True).split()
    if args.tasks:
        if any(t not in tasks for t in args.tasks):p.error('task outside precision category')
        tasks=args.tasks
    if len(set(args.gpus))!=5 or args.eval_num<1 or args.seed<0:p.error('five distinct GPUs and valid seed/eval count required')
    import yaml
    for name,path in zip(LANES,args.env_configs):
        if not Path(path).is_file():p.error(f'missing explicit sensor configuration: {path}')
        config=yaml.safe_load(Path(path).read_text())
        vision=config.get('observation',{}).get('vision',{})
        if config.get('config_name')!='arx_x5':p.error('preserve native arx_x5 layout namespace')
        if bool(vision.get('depth'))!=(name=='gt'):p.error('only GT lane may request GT depth')
        if name!='rgb' and not all(vision.get(k) for k in ('intrinsic_matrix','extrinsic_matrix')):
            p.error('depth modes require observation intrinsics/extrinsics')
    task_cfg=yaml.safe_load((args.robo/'task/RoboDojo/config/_task.yml').read_text())
    for task in tasks:
        effective=dict(task_cfg.get('common',{}),**task_cfg.get('tasks',{}).get(task,{}))
        if effective.get('camera_config','camera_config')!='camera_config' or effective.get('robot_config','dual_x5')!='dual_x5':
            p.error('explicit sensor configs only support native dual_x5/default camera task settings')
    for name,path in zip(LANES,args.env_configs):
        config=yaml.safe_load(Path(path).read_text())
        camera_ref=config['config']['camera']
        camera_path=(Path(path).parent/camera_ref if camera_ref.startswith('./') else args.robo/'env_cfg/camera'/camera_ref).with_suffix('.yml')
        if not camera_path.is_file():p.error(f'missing camera input: {camera_path}')
        cameras=yaml.safe_load(camera_path.read_text())
        required=['cam_head','cam_left_wrist','cam_right_wrist']
        if name in ('tradsl','nsl'):required += [n+'_stereo_right' for n in required]
        if any(n not in cameras for n in required):p.error(f'incomplete camera roles for {name}')
        if name=='gt' and not any(isinstance(v,dict) and v.get('type')=='distance_to_image_plane' for v in cameras.get('annotator',{}).get('common',{}).values()):
            p.error('GT requires optical-axis distance_to_image_plane annotations')
    args.output.mkdir(parents=True,exist_ok=False)
    generated=args.output/'configs';generated.mkdir()
    configs=[]
    for name,path in zip(LANES,args.env_configs):
        config=yaml.safe_load(Path(path).read_text());ref=config['config']['camera']
        if ref.startswith('./'):config['config']['camera']=str((Path(path).parent/ref).resolve())
        target=generated/(name+'.yml');target.write_text(yaml.safe_dump(config,sort_keys=False));configs.append(str(target))
    args.env_configs=configs
    plan=dict(lanes=[dict(name=n,gpu=g,env_config=c) for n,g,c in zip(LANES,args.gpus,args.env_configs)],tasks=tasks,seed=args.seed,eval_num=args.eval_num,invocations=5*len(tasks))
    (args.output/'plan.json').write_text(json.dumps(plan,indent=2));print(json.dumps(plan,indent=2))
    if args.execute:
        pins=json.loads((args.here/'config/precision-depth/sources.json').read_text())
        for name,pin in pins['sources'].items():
            actual=subprocess.check_output(['git','-C',str(args.depth_root/name),'rev-parse','HEAD'],text=True).strip()
            if actual!=pin['commit']:p.error(f'upstream source mismatch: {name}')
        import hashlib
        for relative,expected in pins['native_sensor_inputs'].items():
            if hashlib.sha256((args.robo/relative).read_bytes()).hexdigest()!=expected:p.error(f'native sensor metadata changed: {relative}')
        for relative,key in [('dav2-metric-indoor/depth_anything_v2_metric_hypersim_vitl.pth','dav2_weights_sha256'),
                             ('nsl/nsl1.5_lingbot_giant_detail_global_200k.pt','nsl_weights_sha256'),
                             ('nsl/nsl1.5_lingbot_giant_detail_global_200k.yaml','nsl_config_sha256')]:
            with (args.data_root/'models'/relative).open('rb') as model:
                if hashlib.file_digest(model,'sha256').hexdigest()!=pins[key]:p.error(f'model artifact mismatch: {relative}')
        for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,lambda *_: STOP.set())
        if args.policy_dir.parent.resolve()!=(args.robo/'XPolicyLab/policy').resolve():
            p.error('policy-dir must be a prepared alias under RoboDojo/XPolicyLab/policy')
        if args.policy_dir.resolve()!=(args.here/'xpolicylab/HexaShell').resolve():
            p.error('policy-dir must resolve to this checkout; old policy alias would run old code')
        if not all(os.environ.get(k) for k in ('POLICY_ENV','EVAL_ENV')):
            p.error('source machine env first: POLICY_ENV/EVAL_ENV required')
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
            futures=[pool.submit(lane,args,n,g,c,tasks,args.output) for n,g,c in zip(LANES,args.gpus,args.env_configs)]
            for future in futures:future.result()

if __name__=='__main__':main()
