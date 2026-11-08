"""Native depth tests: no external bootstrap, model service, or simulator required."""
import copy
import numpy as np
import pytest
from hexashell.xpl import depth
from hexashell.runtime.card import system_prompt
from hexashell.xpl.session import RoboDojoSession
from hexashell.state import parse_obs as parse_rgb
from hexashell.robot.host import RobotHost
from hexashell.vision.cameras import HEAD_POSE

def observation():
    # Native policy schema: xyz + wxyz, camera-to-world OpenGL.
    pose=[0.,0.,.9,1.,0.,0.,0.]
    state={}
    for arm in ['left','right']:
        state[arm+'_ee_pose']=pose.copy()
        state[arm+'_ee_joint_state']=1.
        state[arm+'_arm_joint_state']=[0.]*6
    return {'state':state,'vision':{k:{'color':np.zeros((480,640,3),np.uint8),'depth':np.ones((480,640),np.float32),'intrinsic_matrix':np.array([[288.,0,319.5],[0,290.,239.5],[0,0,1]]),'extrinsic_matrix':HEAD_POSE.copy()} for k in depth.state.CAMERA_KEYS},'additional_info':{'camera_extrinsic_convention':'opengl_to_world'}}

def test_pose_and_depth_roundtrip():
    obs=observation();snap=depth.parse_obs(obs,1)
    for key,cam in depth.state.CAMERA_KEYS.items():
        camera=depth.observed_cameras(snap)[cam]
        np.testing.assert_array_equal(camera.pose,obs['vision'][key]['extrinsic_matrix'])
        assert snap.depths[cam].dtype==np.float32
        assert not np.shares_memory(snap.depths[cam],obs['vision'][key]['depth'])
    assert depth.observed_cameras(snap)['head'].axis[2]<0

@pytest.mark.parametrize('kind',['missing','integer','shape','zero','nan','camera'])
def test_bad_depth_fails_closed(kind):
    obs=observation();v=obs['vision']['cam_head']
    if kind=='missing':v.pop('depth')
    elif kind=='integer':v['depth']=v['depth'].astype(np.uint16)
    elif kind=='shape':v['depth']=v['depth'][:4]
    elif kind=='zero':v['depth'][:]=0
    elif kind=='nan':v['depth'][:]=np.nan
    else:obs['additional_info']['camera_extrinsic_convention']='optical_ros_to_world'
    with pytest.raises(ValueError):depth.parse_obs(obs,1)

def test_parser_injection_survives_actions():
    obs=observation();snap=depth.parse_obs(obs,1)
    class Bridge:
        def act(self,action):return 2,obs
    host=RobotHost(Bridge(),snap,parser=depth.parse_obs)
    assert host.act({}).depths
    assert host.snap.seq==2

def test_rgb_default_and_prompt_do_not_leak():
    obs=observation();obs['vision']['cam_head']['depth']='invalid'
    assert not hasattr(parse_rgb(obs,1),'depths')
    session=RoboDojoSession(None,{})
    assert session.prompt_variant() is None
    original=system_prompt('robodojo')
    session.depth=True
    assert session.prompt_variant()=='rgbd'
    assert system_prompt('robodojo','rgbd')!=original
    session.depth=False
    assert system_prompt('robodojo',session.prompt_variant())==original


@pytest.mark.parametrize('backend',['dav2','tradsl','nsl'])
def test_prediction_preserves_observation_and_strips_gt(backend,monkeypatch):
    import io,pickle
    obs=observation();before=copy.deepcopy(obs);events=[]
    provider=depth.DepthProvider(backend,'http://127.0.0.1:18401',events.append)
    def reply(request,timeout):
        payload=pickle.loads(request.data)
        assert all('depth' not in camera for camera in payload['vision'].values())
        return io.BytesIO(pickle.dumps({'backend':backend,'sequence':1,'depths':{k:np.full((480,640),.8,np.float32) for k in depth.state.CAMERA_KEYS}}))
    monkeypatch.setattr(depth.urllib.request,'urlopen',reply)
    snap=provider.parse(obs,1)
    for key,cam in depth.state.CAMERA_KEYS.items():
        for field in ['color','depth','intrinsic_matrix','extrinsic_matrix']:
            np.testing.assert_array_equal(obs['vision'][key][field],before['vision'][key][field])
        np.testing.assert_array_equal(snap.observed_camera_extrinsics[cam],before['vision'][key]['extrinsic_matrix'])
        assert np.allclose(snap.depths[cam],.8)
    assert events[-1]['gt_fallback'] is False

@pytest.mark.parametrize('url',['https://example.com','http://localhost:1','http://user@127.0.0.1:1'])
def test_untrusted_predictor_rejected(url):
    with pytest.raises(ValueError):depth.DepthProvider('nsl',url)


def test_off_axis_axial_projection_roundtrip():
    snap=depth.parse_obs(observation(),1)
    camera=depth.observed_cameras(snap)['head']
    xy=np.array([[120.,150.],[500.,350.]])
    z=np.array([.7,1.2])
    # Inverse of OpenGL projection for unequal fx/fy; z is axial, not ray length.
    local=np.column_stack(((xy[:,0]-camera.cx)*z/camera.fx,-(xy[:,1]-camera.cy)*z/camera.fy,-z))
    world=local@camera.pose[:3,:3].T+camera.position
    projected,depths=camera.project(world)
    np.testing.assert_allclose(projected,xy,atol=1e-10)
    np.testing.assert_allclose(depths,z,atol=1e-10)


def test_shown_snapshot_depth_is_not_latest_depth():
    old=depth.parse_obs(observation(),1)
    obs=observation()
    for v in obs['vision'].values():v['depth'][:]=1.5
    new=depth.parse_obs(obs,2,previous=old)
    host=depth.RGBDHost(None,old,parser=depth.parse_obs)
    host.snap=new
    point_old=host._to_world('head',(100,100))
    host.shown['head']=new
    point_new=host._to_world('head',(100,100))
    assert str(point_old)!=str(point_new)
    assert len(host._rgbd_cache)<=6


def test_wrong_image_resolution_rejected():
    obs=observation();obs['vision']['cam_head']['color']=np.zeros((10,10,3),np.uint8)
    with pytest.raises(ValueError,match='640x480'):depth.parse_obs(obs,1)


def test_depth_session_start_close_and_no_global_parser_change(monkeypatch):
    from types import SimpleNamespace
    import hexashell.state as native_state
    import hexashell.robot.host as native_host
    events=[];closed=[]
    original=native_state.parse_obs
    class Scene:
        def __init__(self,*args,**kwargs):pass
        def bind(self,*args,**kwargs):pass
        def close(self):closed.append(True)
    class Bridge:
        def latest(self):return 1,observation()
    monkeypatch.setattr(depth,'DepthScene',Scene)
    rec=SimpleNamespace(event=events.append)
    session=RoboDojoSession(Bridge(),{})
    monkeypatch.setattr(session,'_log_state',lambda *args:None)
    monkeypatch.delenv('HEXASHELL_DEPTH_BACKEND',raising=False)
    settings=SimpleNamespace(depth=True,sam3_url='fake',motion='v1')
    session.open(settings,rec,None)
    assert session.first.depths and session.prompt_variant()=='rgbd'
    assert native_state.parse_obs is original and native_host.parse_obs is original
    session.close();session.close()
    assert len(closed)==1


def test_depth_session_missing_sam_fails_before_observation():
    from types import SimpleNamespace
    session=RoboDojoSession(None,{})
    with pytest.raises(RuntimeError,match='requires SAM3'):
        session.open(SimpleNamespace(depth=True,sam3_url=''),None,None)


def test_prediction_wrong_identity_and_integer_depth_refused(monkeypatch):
    import io,pickle
    provider=depth.DepthProvider('dav2','http://127.0.0.1:18401')
    result={'backend':'nsl','sequence':1,'depths':{k:np.ones((480,640),np.float32) for k in depth.state.CAMERA_KEYS}}
    monkeypatch.setattr(depth.urllib.request,'urlopen',lambda *args,**kwargs:io.BytesIO(pickle.dumps(result)))
    with pytest.raises(ValueError,match='identity'):provider.parse(observation(),1)
    result['backend']='dav2'
    result['depths']={k:np.ones((480,640),np.uint16) for k in depth.state.CAMERA_KEYS}
    with pytest.raises(ValueError,match='float32'):provider.parse(observation(),1)


def test_prediction_cache_is_bounded(monkeypatch):
    import io,pickle
    provider=depth.DepthProvider('nsl','http://127.0.0.1:18401')
    calls=[]
    def reply(request,timeout):
        seq=pickle.loads(request.data)['sequence'];calls.append(seq)
        return io.BytesIO(pickle.dumps({'backend':'nsl','sequence':seq,'depths':{k:np.ones((480,640),np.float32) for k in depth.state.CAMERA_KEYS}}))
    monkeypatch.setattr(depth.urllib.request,'urlopen',reply)
    for seq in range(7):provider.parse(observation(),seq)
    provider.parse(observation(),6)
    assert len(provider.cache)==4 and len(calls)==7
