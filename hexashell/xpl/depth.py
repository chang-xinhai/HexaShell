"""Explicit RoboDojo RGBD consumer. No startup injection or simulator-version conversions."""
from __future__ import annotations
import math
import pickle
import urllib.request
import hashlib
from urllib.parse import urlsplit
import numpy as np
from .. import state
from ..robot.host import RobotHost
from ..vision.cameras import Camera, check_cam
from ..vision.robotmask import robot_mask
from ..vision.rgbd import DepthScene, DepthView, spot

def parse_obs(obs,seq,previous=None):
    snap=state.parse_obs(obs,seq,previous)
    if (obs.get('additional_info') or {}).get('camera_extrinsic_convention')!='opengl_to_world':
        raise ValueError('RGBD requires RoboDojo policy camera convention opengl_to_world; no consumer conversion')
    snap.depths={};snap.rgbd_intrinsics={};snap.observed_camera_extrinsics={}
    for key,cam in state.CAMERA_KEYS.items():
        if cam not in snap.images or snap.images[cam].shape != (480,640,3):
            raise ValueError('RGBD requires 640x480 RGB: '+cam)
        data=obs['vision'][key]
        raw=np.asarray(data.get('depth'))
        if raw.ndim==3 and raw.shape[-1]==1:raw=raw[...,0]
        if raw.dtype.kind!='f' or raw.shape!=snap.images[cam].shape[:2]:
            raise ValueError('RGBD requires float optical-axis metres aligned with RGB: '+cam)
        if not ((np.isfinite(raw)) & (raw>.02) & (raw<8)).any():raise ValueError('No valid depth: '+cam)
        k=np.asarray(data.get('intrinsic_matrix'),float);t=np.asarray(data.get('extrinsic_matrix'),float)
        if k.shape!=(3,3) or not np.isfinite(k).all() or min(k[0,0],k[1,1])<=0:raise ValueError('Invalid camera intrinsics: '+cam)
        if t.shape!=(4,4) or not np.isfinite(t).all() or not np.allclose(t[3],[0,0,0,1]) or not np.allclose(t[:3,:3].T@t[:3,:3],np.eye(3),atol=1e-4) or not np.isclose(np.linalg.det(t[:3,:3]),1,atol=1e-4):raise ValueError('Invalid camera extrinsics: '+cam)
        snap.depths[cam]=np.array(raw,np.float32,copy=True,order='C')
        snap.rgbd_intrinsics[cam]=k.copy();snap.observed_camera_extrinsics[cam]=t.copy()
    return snap

def observed_cameras(snap):
    return {cam:Camera(cam,float(snap.rgbd_intrinsics[cam][0,0]),t.copy(),snap.images[cam].shape[1],snap.images[cam].shape[0],k=snap.rgbd_intrinsics[cam].copy()) for cam,t in snap.observed_camera_extrinsics.items()}

def closing_yaw(direction_xy):
    yaw=math.degrees(math.atan2(float(direction_xy[1]),float(direction_xy[0])))
    while yaw<=-90: yaw+=180
    while yaw>90: yaw-=180
    return round(yaw,1)

class EvidenceDepthView(DepthView):
    def back_project(self,xs,ys,depth):
        out=super().back_project(xs,ys,depth)
        self._evidence('back_project',int(len(out)))
        return out
    def depth_at(self,xy,*,robot_too=False):
        out=super().depth_at(xy,robot_too=robot_too)
        self._evidence('depth_at',int(out is not None))
        return out
    def _evidence(self,operation,n):
        self._rgbd_log({'kind':'rgbd_depth_consumed','cam':self.cam,'seq':self._rgbd_seq,
                        'step':self.step,'operation':operation,'count':n,
                        'source':'vision.depth','units':'m','convention':'optical_axis'})

class RGBDHost(RobotHost):
    def __init__(self,*args,**kwargs):
        self._rgbd_cache={}
        super().__init__(*args,**kwargs)
    def bind_scene(self,scene):
        scene.bind(self.views,hands={arm:(lambda arm=arm:self.driver.grasp_pose(arm)) for arm in self.rig.arms})
        self.carry=scene
    def cameras_now(self,cams=None):
        models=observed_cameras(self.snap)
        for cam,k in getattr(self.snap,"rgbd_intrinsics",{}).items():
            models[cam].k=k
        return models
    def views(self,cams=None,*,snapshot=None):
        snap=self.snap if snapshot is None else snapshot
        if not hasattr(snap,'depths'):
            raise RuntimeError('RGBD host requires a depth-parsed snapshot')
        names=list(state.CAMERAS) if cams is None else [check_cam(c) for c in cams]
        models=observed_cameras(snap)
        out={}
        for cam in names:
            key=(id(snap),cam)
            if key not in self._rgbd_cache:
                camera=models[cam]
                if cam in snap.rgbd_intrinsics:
                    camera.k=snap.rgbd_intrinsics[cam]
                arms=snap.arms_in_pictures
                openings={a:min(1.,float(arms[a].gripper)+.2) for a in self.rig.arms if self.driver.squeezing(a)}
                mask=robot_mask(camera,arms,openings or None)
                d=snap.depths[cam]
                v=EvidenceDepthView(cam,snap.images[cam],d,camera,mask,step=snap.step or 0,
                                    frame_id=f'robodojo:{snap.seq}:{cam}')
                v._rgbd_log=self.log;v._rgbd_seq=snap.seq
                self._rgbd_cache[key]=v
                valid=v.valid()
                self.log({'kind':'rgbd_view','cam':cam,'seq':snap.seq,'step':snap.step,
                          'valid_pixels':int(valid.sum()),'robot_pixels':int(mask.sum()),
                          'depth_min_m':float(d[valid].min()) if valid.any() else None,'depth_max_m':float(d[valid].max()) if valid.any() else None,
                          'camera_pose_source':'policy observation opengl_to_world (unchanged)',
                          'intrinsics_source':'observation' if cam in snap.rgbd_intrinsics else 'RoboDojo calibrated pinhole',
                          'source':'vision.depth','units':'m','convention':'optical_axis'})
            out[cam]=self._rgbd_cache[key]
        # Keep current and shown snapshots; a selected pixel refers to the picture shown to the model.
        keep={id(self.snap),id(snap)}|{id(s) for s in self.shown.values()}|{id(s) for s in self.looked.values()}
        self._rgbd_cache={k:v for k,v in self._rgbd_cache.items() if k[0] in keep}
        return out
    def _to_world(self,cam,xy,*,z=None,pair=None):
        cam=check_cam(cam)
        # Match the existing RoboDojo selected-pixel convention: use the last delivered image.
        picked=self.shown.get(cam,self.snap)
        views=self.views([cam],snapshot=picked)
        if pair is not None:
            if not isinstance(pair,(tuple,list)) or len(pair)!=2:
                raise ValueError('pair must be (camera,(x,y))')
            other=check_cam(pair[0]);views.update(self.views([other],snapshot=self.shown.get(other,self.snap)))
        return spot(views,cam,xy,z=z,pair=pair,check_cam=check_cam)


class DepthProvider:
    """Per-session depth source. Predictions never receive GT depth or mutate observations."""
    def __init__(self, backend="gt", url=None, log=None):
        if backend not in ("gt", "dav2", "tradsl", "nsl"):
            raise ValueError("Unknown RoboDojo depth backend: " + backend)
        self.backend, self.url, self.log = backend, url, log or (lambda event: None)
        self.cache = {}
        if backend != "gt":
            parsed = urlsplit(url or "")
            if parsed.scheme != "http" or parsed.hostname != "127.0.0.1" or parsed.username or parsed.password:
                raise ValueError("Depth service must be a trusted http://127.0.0.1 endpoint")

    def parse(self, obs, seq, previous=None):
        if self.backend == "gt":
            return parse_obs(obs, seq, previous)
        if seq not in self.cache:
            vision = {key: {field: value for field, value in camera.items()
                            if field in ("color", "intrinsic_matrix", "extrinsic_matrix")}
                      for key, camera in obs["vision"].items()}
            request = urllib.request.Request(self.url.rstrip("/") + "/predict",
                data=pickle.dumps({"sequence": seq, "vision": vision}, protocol=pickle.HIGHEST_PROTOCOL),
                headers={"Content-Type": "application/octet-stream"})
            # Local service is trusted code; never deserialize remote/untrusted endpoints.
            with urllib.request.urlopen(request, timeout=360) as response:
                result = pickle.loads(response.read())
            if result.get("sequence") != seq or result.get("backend") != self.backend:
                raise ValueError("Depth response identity mismatch")
            self.cache[seq] = result["depths"]
            while len(self.cache) > 4:
                self.cache.pop(next(iter(self.cache)))
        depths = self.cache[seq]
        replaced = dict(obs)
        replaced["vision"] = dict(obs["vision"])
        hashes = {}
        for key in state.CAMERA_KEYS:
            value = np.asarray(depths[key])
            if value.dtype != np.float32:
                raise ValueError("Prediction must be float32 metres")
            replaced["vision"][key] = dict(obs["vision"][key], depth=value)
            hashes[key] = hashlib.sha256(value.tobytes()).hexdigest()
        snap = parse_obs(replaced, seq, previous)
        self.log({"kind": "prediction_used", "seq": seq, "backend": self.backend,
                  "gt_fallback": False, "depth_hashes": hashes})
        return snap
