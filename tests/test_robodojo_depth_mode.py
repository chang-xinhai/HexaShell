"""Baseline-native depth wiring: no simulator, algorithm service or RGBDHost."""
import numpy as np
import pytest
from hexashell.state import parse_obs
from hexashell.robot.host import RobotHost
from hexashell.vision.cameras import HEAD_POSE, cameras_for
from hexashell.xpl.session import closing_yaw


def observation(distance=1.):
    return {'additional_info':{'camera_extrinsic_convention':'opengl_to_world'},'vision':{k:{'color':np.zeros((480,640,3),np.uint8),
            'depth':np.full((480,640),distance,np.float32),
            'intrinsic_matrix':np.array([[288.,0,320.],[0,290.,240.],[0,0,1]]),
            'extrinsic_matrix':HEAD_POSE.copy()}
            for k in ['cam_head','cam_left_wrist','cam_right_wrist']}}


def test_rgb_unchanged_and_depth_uses_same_host():
    snap=parse_obs(observation(),1,depth=True)
    rgb=RobotHost(None,snap)
    host=RobotHost(None,snap,depth=True)
    assert rgb.cameras_now()['head'].cx==319.5
    assert host.cameras_now()['head'].cx==320.
    assert host.cameras_now()['head'].fy==290.
    view=host.views(['head'])['head']
    assert view.image is snap.images['head']
    xy=np.array([[80.,100.],[450.,350.]])
    xyz=view.back_project(xy[:,0],xy[:,1],np.ones(2))
    projected,z=view.camera.project(xyz)
    np.testing.assert_allclose(projected,xy,atol=1e-9)
    np.testing.assert_allclose(z,1.)


def test_old_shown_frame_survives_action_without_new_cache():
    old=parse_obs(observation(1.),1,depth=True)
    class Bridge:
        def act(self,action):return 2,observation(1.5)
    host=RobotHost(Bridge(),old,depth=True)
    old_point=host._to_world('head',(100,100)).xyz
    host.act({})
    assert host.snap.seq==2
    assert host._to_world('head',(100,100)).xyz==old_point
    host.looked['head']=host.snap
    host.pictures_shown()
    assert host._to_world('head',(100,100)).xyz!=old_point
    assert not hasattr(host,'_rgbd_cache')


@pytest.mark.parametrize('value',[np.zeros((480,640),np.float32),np.ones((4,4),np.float32),np.ones((480,640),np.uint16)])
def test_invalid_depth_rejected(value):
    obs=observation();obs['vision']['cam_head']['depth']=value
    with pytest.raises(ValueError):RobotHost(None,parse_obs(obs,1,depth=True),depth=True).views(['head'])


def test_closing_yaw_matches_robodojo_axes():
    assert closing_yaw([1.,0.])==0.
    assert closing_yaw([0.,1.])==90.


def test_prediction_connection_strips_gt_and_preserves_input(monkeypatch):
    import io,pickle,copy
    from hexashell.xpl import depth_source
    obs=observation();before=copy.deepcopy(obs)
    def response(req,timeout):
        assert timeout==110  # Native simulator RPC deadline is120s.
        payload=pickle.loads(req.data)
        assert all('depth' not in v for v in payload['vision'].values())
        return io.BytesIO(pickle.dumps({'sequence':3,'backend':'dav2','token':payload['token'],'depths':{k:np.full((480,640),.7,np.float32) for k in obs['vision']}}))
    monkeypatch.setattr(depth_source.urllib.request,'build_opener',lambda *a: type('Opener',(),{'open':staticmethod(response)})())
    out=depth_source.predict('http://127.0.0.1:19122',obs,3,backend='dav2')
    for key in obs['vision']:
        np.testing.assert_array_equal(obs['vision'][key]['depth'],before['vision'][key]['depth'])
        assert np.allclose(out['vision'][key]['depth'],.7)
    with pytest.raises(ValueError):depth_source.predict('https://example.com',obs,3,backend='dav2')


def test_depth_and_calibration_are_snapshot_copies():
    obs=observation();snap=parse_obs(obs,1,depth=True)
    obs['vision']['cam_head']['depth'][:]=2.
    obs['vision']['cam_head']['extrinsic_matrix'][0,3]=99.
    assert snap.vision['cam_head']['depth'][0,0]==1.
    assert snap.vision['cam_head']['extrinsic_matrix'][0,3]!=99.
    snap.camera_convention='optical_to_world'
    with pytest.raises(ValueError):cameras_for(snap,observed=True)


def test_rgb_parser_does_not_touch_depth_payload():
    class Unreadable:
        def __array__(self,*args,**kwargs):raise AssertionError('RGB read depth')
    obs=observation();obs['vision']['cam_head']['depth']=Unreadable()
    assert parse_obs(obs,1).vision=={}


def test_depth_snapshot_owns_color_too():
    obs=observation();snap=parse_obs(obs,1,depth=True)
    obs['vision']['cam_head']['color'][:]=255
    assert not snap.images['head'].any()


@pytest.mark.parametrize('pair',[(),42])
def test_invalid_pair_refused(pair):
    from hexashell.errors import Refused
    host=RobotHost(None,parse_obs(observation(),1,depth=True),depth=True)
    with pytest.raises(Refused):host._to_world('head',(1,1),pair=pair)


def test_gt_session_uses_original_host_and_shared_scene(monkeypatch):
    from types import SimpleNamespace
    from hexashell.xpl.session import RoboDojoSession
    from hexashell.runtime.episode import Settings
    from hexashell.vision import rgbd
    instances=[]
    class Scene:
        def __init__(self,*a,**kw):instances.append(self);self.closed=False
        def bind(self,get_views,**kw):self.get_views=get_views;self.hands=kw['hands']
        def close(self):self.closed=True
    monkeypatch.setattr(rgbd,'DepthScene',Scene)
    bridge=SimpleNamespace(latest=lambda:(1,observation()))
    session=RoboDojoSession(bridge,{})
    session.open(Settings(depth=True,sam3_url='mock'),SimpleNamespace(event=lambda e:None,state=lambda e:None),lambda *a:None)
    assert type(session.host) is RobotHost and session.prompt_variant()=='rgbd'
    assert set(instances[0].hands)=={'left','right'}
    assert instances[0].get_views(['head'])['head'].frame_id=='robodojo:1:head'
    session.close();session.close();assert instances[0].closed


def test_adapter_prediction_failure_abandons_episode(monkeypatch,tmp_path):
    from hexashell.xpl.adapter import PolicyHost
    from hexashell.xpl import depth_source
    monkeypatch.setenv('HEXASHELL_DEPTH','1');monkeypatch.setenv('HEXASHELL_DEPTH_BACKEND','dav2')
    monkeypatch.setenv('HEXASHELL_DEPTH_URL','http://127.0.0.1:12345')
    monkeypatch.setenv('HEXASHELL_RESULT_DIR',str(tmp_path))
    host=PolicyHost({});called=[];monkeypatch.setattr(host.bridge,'abandon',lambda:called.append(True))
    def failed(*a,**kw):raise TimeoutError('model unavailable')
    monkeypatch.setattr(depth_source,'predict',failed)
    with pytest.raises(TimeoutError):host.update_obs(observation())
    assert called and host.thread is None


def test_predicted_backend_cannot_implicitly_use_gt(monkeypatch,tmp_path):
    from hexashell.xpl.adapter import PolicyHost
    monkeypatch.setenv('HEXASHELL_DEPTH','1');monkeypatch.setenv('HEXASHELL_DEPTH_BACKEND','nsl')
    monkeypatch.delenv('HEXASHELL_DEPTH_URL',raising=False)
    monkeypatch.setenv('HEXASHELL_RESULT_DIR',str(tmp_path))
    with pytest.raises(ValueError):PolicyHost({})


def test_stale_mask_refused_before_shared_scene_fuses_views():
    from types import SimpleNamespace
    from hexashell.vision.rgbd import Mask
    from hexashell.errors import Refused
    host=RobotHost(None,parse_obs(observation(),2,depth=True),depth=True)
    mask=object.__new__(Mask)
    # Dataclass constructor is deliberately avoided: only camera/id are needed for dispatch.
    object.__setattr__(mask,'cam','head')
    host.scene=SimpleNamespace(known=lambda m:(m,SimpleNamespace(frame_id='robodojo:1:head')),
                               measure=lambda *a,**k:pytest.fail('must not fuse old/current views'))
    with pytest.raises(Refused):host._measure(mask)


def test_unsupported_intrinsics_refused():
    obs=observation();obs['vision']['cam_head']['intrinsic_matrix'][0,1]=2
    with pytest.raises(ValueError):cameras_for(parse_obs(obs,1,depth=True),observed=True)
