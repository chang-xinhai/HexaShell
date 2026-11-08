import importlib.util
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest

spec=importlib.util.spec_from_file_location('depth_server',Path(__file__).parents[1]/'services/depth_server.py')
server=importlib.util.module_from_spec(spec);spec.loader.exec_module(server)


def payload():
    vision={}
    for name in server.CAMS:
        left={'color':np.zeros((8,10,3),np.uint8),'intrinsic_matrix':np.array([[10.,0,5],[0,10,4],[0,0,1]]),'extrinsic_matrix':np.eye(4)}
        right={**left,'extrinsic_matrix':np.eye(4)};right['extrinsic_matrix'][0,3]=.05
        vision[name]=left;vision[name+'_stereo_right']=right
    return dict(sequence=7,vision=vision)


def test_stereo_contract():
    p=payload()['vision'];l,r,k,b=server.stereo(p['cam_head'],p['cam_head_stereo_right'])
    assert l.shape==(8,10) and b==.05
    p['cam_head_stereo_right']['extrinsic_matrix'][0,3]=.1
    with pytest.raises(ValueError):server.stereo(p['cam_head'],p['cam_head_stereo_right'])


def test_tradsl_uses_existing_pipeline_and_converts_mm_to_m():
    model=object.__new__(server.Model);model.args=SimpleNamespace(backend='tradsl');model.pipelines={};model.cfg={}
    model.params=lambda **kw:kw
    class Pipeline:
        def __init__(self,*args,**kw):pass
        def run(self,l,r):return {'depth':np.full(l.shape,1200,np.uint16)}
    model.pipeline=Pipeline
    out=model.predict(payload())
    assert out['sequence']==7 and out['backend']=='tradsl'
    for d in out['depths'].values():assert d.dtype==np.float32 and np.allclose(d,1.2)
    assert len(model.pipelines)==3


def test_nsl_uses_native_client_protocol():
    model=object.__new__(server.Model);model.args=SimpleNamespace(backend='nsl')
    class Client:
        def close(self):pass
        def connect(self):pass
        def infer(self,**kw):return SimpleNamespace(sequence=kw['sequence'],stamp_ns=kw['stamp_ns'],depth_left_m=np.ones(kw['ir_left'].shape,np.float32))
    model.client=Client();out=model.predict(payload())
    assert all(np.all(d==1.) for d in out['depths'].values())
