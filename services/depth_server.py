"""Trusted loopback depth endpoint. Model algorithms remain in their upstream repositories."""
import argparse
import json
import pickle
import sys
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
import cv2
import numpy as np

CAMS = ('cam_head', 'cam_left_wrist', 'cam_right_wrist')


def stereo(left, right):
    k = np.asarray(left['intrinsic_matrix'], float)
    relation = np.linalg.inv(left['extrinsic_matrix']) @ right['extrinsic_matrix']
    if not np.allclose(k, right['intrinsic_matrix']) or not np.allclose(relation[:3,:3], np.eye(3), atol=1e-5) or not np.allclose(relation[:3,3], [.05,0,0], atol=1e-5):
        raise ValueError('requires rectified identical-intrinsics stereo, +5cm local-x')
    return cv2.cvtColor(left['color'], cv2.COLOR_RGB2GRAY), cv2.cvtColor(right['color'], cv2.COLOR_RGB2GRAY), k, float(relation[0,3])


class Model:
    def __init__(self, args):
        self.args, self.pipelines = args, {}
        if args.backend == 'nsl':
            sys.path.insert(0, str(args.depth_root/'nsl_infer_server/server'))
            from record.depth.nsl.client import NslInferenceClient
            from record.depth.nsl.wire import fetch_server_provenance
            doc, identity = fetch_server_provenance('127.0.0.1', args.nsl_port)
            if doc['method'] != 'nsl_v1_5': raise ValueError('requires native NSL v1.5 server')
            self.client = NslInferenceClient(host='127.0.0.1',port=args.nsl_port,timeout_s=100,
                                            expected_server_provenance_sha256=identity)
        elif args.backend == 'dav2':
            import torch
            sys.path.insert(0, str(args.depth_root/'Depth-Anything-V2/metric_depth'))
            from depth_anything_v2.dpt import DepthAnythingV2
            self.model = DepthAnythingV2(encoder='vitl',features=256,out_channels=[256,512,1024,1024],max_depth=20.)
            self.model.load_state_dict(torch.load(args.weights,map_location='cpu',weights_only=True),strict=True)
            self.model = self.model.to('cuda:0').eval()
        else:
            import yaml
            sys.path.insert(0, str(args.depth_root/'TradSL'))
            from tradsl.pipeline import StereoPipeline
            from tradsl.io import CameraParams
            self.pipeline, self.params = StereoPipeline, CameraParams
            self.cfg = yaml.safe_load((args.depth_root/'TradSL/configs/d435.yaml').read_text())

    def predict(self, request):
        depths = {}
        for name in CAMS:
            left = request['vision'][name]
            if self.args.backend == 'dav2':
                import torch
                with torch.inference_mode():
                    d = self.model.infer_image(cv2.cvtColor(left['color'],cv2.COLOR_RGB2BGR),518)
            else:
                l,r,k,b = stereo(left,request['vision'][name+'_stereo_right'])
                if self.args.backend == 'nsl':
                    stamp = time.time_ns()
                    self.client.close(); self.client.connect()
                    result = self.client.infer(sequence=request['sequence'],stamp_ns=stamp,ir_left=l,ir_right=r,
                                               fx_px=float(k[0,0]),baseline_m=b)
                    if result.sequence != request['sequence'] or result.stamp_ns != stamp: raise ValueError('NSL response identity')
                    d = result.depth_left_m
                else:
                    key = (name,k.tobytes(),b)
                    if key not in self.pipelines:
                        h,w=l.shape
                        params=self.params(fx=k[0,0],fy=k[1,1],cx=k[0,2],cy=k[1,2],baseline_mm=b*1000,width=w,height=h,K=k)
                        self.pipelines[key]=self.pipeline(self.cfg,params,device='cuda:0')
                    d = self.pipelines[key].run(l,r)['depth'].astype(np.float32)*.001
            d=np.ascontiguousarray(d,dtype=np.float32)
            if d.shape!=left['color'].shape[:2] or not np.isfinite(d).all() or (d<0).any(): raise ValueError('invalid model depth')
            depths[name]=d
        return dict(depths=depths,backend=self.args.backend,sequence=request['sequence'],token=request.get('token'))


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        body=json.dumps(dict(backend=self.server.model.args.backend,initialized=True)).encode()
        self.send_response(200 if self.path=='/health' else 404);self.end_headers();self.wfile.write(body)

    def do_POST(self):
        try:
            if self.path!='/predict':raise ValueError('unknown endpoint')
            n=int(self.headers['Content-Length'])
            if not 0<n<32*1024*1024:raise ValueError('invalid payload size')
            result=self.server.model.predict(pickle.loads(self.rfile.read(n)))
            body=pickle.dumps(result);self.send_response(200)
        except Exception as exc:
            body=str(exc).encode();self.send_response(500)
        self.end_headers();self.wfile.write(body)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--backend',choices=['dav2','tradsl','nsl'],required=True)
    p.add_argument('--depth-root',type=Path,required=True)
    p.add_argument('--weights',type=Path)
    p.add_argument('--port',type=int,required=True)
    p.add_argument('--nsl-port',type=int,default=19130)
    a=p.parse_args()
    with HTTPServer(('127.0.0.1',a.port),Handler) as server:
        server.model=Model(a)
        try:server.serve_forever()
        finally:
            if a.backend=='nsl':server.model.client.close()

if __name__=='__main__':main()
