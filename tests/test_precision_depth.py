import importlib.util
from pathlib import Path
from types import SimpleNamespace
import json
import pytest

spec=importlib.util.spec_from_file_location('precision_depth',Path(__file__).parents[1]/'scripts/precision_depth.py')
runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)


def test_two_native_tasks_no_gpu_visibility_override(monkeypatch,tmp_path):
    commands=[];signals=[]
    class Process:
        next_pid=10
        def __init__(self,argv,**kw):
            self.pid=Process.next_pid;Process.next_pid+=1;self.returncode=0
            commands.append((argv,kw));self.calls=0
        def poll(self):self.calls+=1;return None if self.calls==1 else 0
        def wait(self,**kw):return 0
    monkeypatch.setenv('POLICY_ENV','policy');monkeypatch.setenv('EVAL_ENV','sim')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','7')
    monkeypatch.setenv('EVAL_ENV_TYPE','debug')
    monkeypatch.setattr(runner.subprocess,'Popen',Process)
    def kill(pid,sig):
        signals.append((pid,sig))
        if sig==0:raise ProcessLookupError
    monkeypatch.setattr(runner.os,'killpg',kill);monkeypatch.setattr(runner.time,'sleep',lambda x:None)
    runner.STOP.clear()
    args=SimpleNamespace(robo=tmp_path,policy_dir=tmp_path/'policy',seed=1,eval_num=1)
    rows=runner.lane(args,'rgb','4',str(tmp_path/'config.yml'),['a','b'],tmp_path)
    assert len(rows)==2 and all(r['exit_code']==0 for r in rows)
    assert len(commands)==2
    for argv,kw in commands:
        assert 'CUDA_VISIBLE_DEVICES' not in kw['env']
        assert kw['env']['EVAL_ENV_TYPE']=='sim'
        assert '--env-gpu' in argv and argv[argv.index('--env-gpu')+1]=='4'
        assert kw['start_new_session']
        assert 'HEXASHELL_SMOKE_STEPS' not in kw['env']
    assert len(json.loads((tmp_path/'rgb/exits.json').read_text()))==2
    assert {p for p,s in signals}=={10,11}


def test_interrupt_before_first_task_creates_no_process(monkeypatch,tmp_path):
    monkeypatch.setattr(runner.subprocess,'Popen',lambda *a,**k:pytest.fail('must not launch'))
    runner.STOP.set()
    try:
        with pytest.raises(InterruptedError):runner.lane(SimpleNamespace(),'rgb','1','config.yml',['a'],tmp_path)
        assert json.loads((tmp_path/'rgb/exits.json').read_text())==[]
    finally:runner.STOP.clear()


def test_real_owned_group_cleanup_kills_term_ignoring_descendant(tmp_path):
    import subprocess,sys,time,os,signal
    if sys.platform!='linux':pytest.skip('Linux launcher')
    marker=tmp_path/'child.pid'
    code="import os,signal,time,pathlib; pid=os.fork(); signal.signal(signal.SIGTERM,signal.SIG_IGN); pathlib.Path("+repr(str(marker))+ ").write_text(str(os.getpid())) if pid==0 else None; time.sleep(60)"
    process=subprocess.Popen([sys.executable,'-c',code],start_new_session=True)
    outsider=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)'],start_new_session=True)
    try:
        deadline=time.monotonic()+5
        while not marker.exists() and time.monotonic()<deadline:time.sleep(.02)
        assert marker.exists()
        child=int(marker.read_text())
        runner.cleanup([process],grace=.2)
        assert process.poll() is not None
        # PID1 may retain a reparented zombie; it must not be an executing process.
        stat=Path(f'/proc/{child}/stat')
        try:state=stat.read_text().split()[2]
        except (FileNotFoundError,ProcessLookupError):state='gone'
        assert state in ('gone','Z','X')
        assert outsider.poll() is None
    finally:
        runner.cleanup([process,outsider],grace=.2)
