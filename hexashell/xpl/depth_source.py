"""Trusted local model endpoint -> ordinary RoboDojo depth observations."""
import pickle
import uuid
import urllib.request
from urllib.parse import urlsplit
import numpy as np
from ..state import CAMERA_KEYS


def predict(url, obs, sequence, *, backend):
    address=urlsplit(url)
    if address.scheme!='http' or address.hostname!='127.0.0.1' or address.username or address.password:
        raise ValueError('depth endpoint must be trusted loopback HTTP')
    vision={k:{f:v for f,v in camera.items() if f in ('color','intrinsic_matrix','extrinsic_matrix')}
            for k,camera in obs['vision'].items()}
    token=uuid.uuid4().hex
    request=urllib.request.Request(url.rstrip('/')+'/predict',
        data=pickle.dumps({'sequence':sequence,'token':token,'vision':vision}),
        headers={'Content-Type':'application/octet-stream'})
    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self,*args,**kwargs):raise ValueError('depth endpoint must not redirect')
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}),NoRedirect)
    with opener.open(request,timeout=110) as response:
        result=pickle.loads(response.read())
    if result['sequence']!=sequence or result.get('backend')!=backend or result.get('token')!=token:
        raise ValueError('depth response does not match observation')
    replaced=dict(obs,vision=dict(obs['vision']))
    for key in CAMERA_KEYS:
        d=np.asarray(result['depths'][key])
        if d.dtype!=np.float32 or d.shape!=np.asarray(obs['vision'][key]['color']).shape[:2]:
            raise ValueError('predicted depth must be aligned float32 metres')
        replaced['vision'][key]=dict(obs['vision'][key],depth=d)
    return replaced
