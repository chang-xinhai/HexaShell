#!/usr/bin/env python3
"""Persistent RGB-only depth worker (DAV2, TradSL, or external NSL).

SECURITY: /predict deserializes pickle, which can execute arbitrary code. This
service binds ONLY to 127.0.0.1 and is for trusted local clients. Never expose or
forward its port to untrusted clients. There is no authentication or TLS.
Models/pipelines persist across requests; NSL is independently managed and is
never started here. No simulator, ground-truth depth input, or CPU fallback.
"""
from __future__ import annotations
import argparse
import hashlib
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import os
from pathlib import Path
import pickle
import signal
import sys
import time
import traceback

import cv2
import numpy as np

DEFAULT_DEPTH_ROOT = Path(__file__).resolve().parents[1] / 'third_party'
CAMS = ('cam_head', 'cam_left_wrist', 'cam_right_wrist')
WEIGHTS_SHA = 'a2109e05290bf627608e71786c250bd81df6c255e625d17ddba346e9dd570ca9'
CONFIG_SHA = 'cf0632ed32e912d65413b841dfaf3cb5ea03ce014f1a6fc75447e3542d539b4a'


def sha_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def sha_array(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def require(ok, message):
    if not ok:
        raise ValueError(message)


def rgb(view):
    a = view['color']
    require(isinstance(a, np.ndarray) and a.dtype == np.uint8 and a.shape == (480, 640, 3),
            'color must be RGB uint8 [480,640,3]')
    return a


def stereo(left, right):
    l, r = rgb(left), rgb(right)
    K = np.asarray(left['intrinsic_matrix'], dtype=np.float64)
    Kr = np.asarray(right['intrinsic_matrix'], dtype=np.float64)
    Tl = np.asarray(left['extrinsic_matrix'], dtype=np.float64)
    Tr = np.asarray(right['extrinsic_matrix'], dtype=np.float64)
    require(K.shape == Kr.shape == (3, 3) and Tl.shape == Tr.shape == (4, 4), 'calibration shapes')
    require(all(np.isfinite(a).all() for a in (K, Kr, Tl, Tr)), 'nonfinite calibration')
    require(K[0, 0] > 0 and K[1, 1] > 0, 'nonpositive focal length')
    require(np.allclose(K, Kr, atol=1e-5, rtol=0), 'left/right intrinsics differ')
    require(np.allclose(Tl[3], [0, 0, 0, 1], atol=1e-5, rtol=0) and
            np.allclose(Tr[3], [0, 0, 0, 1], atol=1e-5, rtol=0), 'invalid homogeneous transforms')
    T = np.linalg.inv(Tl) @ Tr
    require(np.allclose(T[:3, :3], np.eye(3), atol=1e-5, rtol=0), 'stereo rotations differ')
    require(np.allclose(T[:3, 3], [.05, 0, 0], atol=1e-5, rtol=0), 'baseline must be +0.05m local-x')
    return cv2.cvtColor(l, cv2.COLOR_RGB2GRAY), cv2.cvtColor(r, cv2.COLOR_RGB2GRAY), K, float(T[0, 3]), T


class Worker:
    def __init__(self, backend, *, depth_root, data_root, nsl_port=17639):
        require(backend in ('dav2', 'tradsl', 'nsl'), 'unknown backend')
        require(type(nsl_port) is int and 1 <= nsl_port <= 65535, 'invalid NSL port')
        depth_root = Path(depth_root).expanduser().resolve()
        data_root = Path(data_root).expanduser().resolve()
        self.backend = backend
        self.client = None
        self.pipelines = {}
        self.rpc_sequence = 0
        self.info = {'backend': backend, 'initialized': False, 'pid': os.getpid(),
                     'python': sys.executable, 'rgb_input_only': True, 'gt_depth_arrays_read': False}
        started = time.perf_counter()
        if backend == 'nsl':
            # This proxy neither imports torch nor allocates a GPU; server starts externally.
            sys.path.insert(0, str(depth_root/'nsl_infer_server/server'))
            from record.depth.nsl.client import NslInferenceClient
            from record.depth.nsl.wire import fetch_server_provenance
            doc, identity = fetch_server_provenance('127.0.0.1', nsl_port)
            require(doc.get('method') == 'nsl_v1_5', 'Unexpected NSL server method')
            require(doc['sha256']['weights'] == WEIGHTS_SHA, 'Unexpected NSL checkpoint')
            require(doc['sha256']['config'] == CONFIG_SHA, 'Unexpected NSL YAML')
            self.client = NslInferenceClient(host='127.0.0.1', port=nsl_port, timeout_s=360,
                                            expected_server_provenance_sha256=identity)
            self.client.connect()
            self.info.update(model_fingerprint=identity, provenance=doc, gpu_id=doc.get('gpu_uuid'),
                             device='external-nsl-server', proxy_allocates_gpu=False, endpoint=f'127.0.0.1:{nsl_port}')
        else:
            import torch
            self.torch = torch
            require(torch.cuda.is_available(), 'CUDA required; no CPU fallback')
            torch.cuda.set_device(0)
            prop = torch.cuda.get_device_properties(0)
            self.info.update(device='cuda:0', gpu_id=str(getattr(prop, 'uuid', torch.cuda.get_device_name(0))),
                             gpu_name=torch.cuda.get_device_name(0), cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'))
            if backend == 'dav2':
                sys.path.insert(0, str(depth_root/'Depth-Anything-V2/metric_depth'))
                from depth_anything_v2.dpt import DepthAnythingV2
                weights = data_root/'models/dav2-metric-indoor/depth_anything_v2_metric_hypersim_vitl.pth'
                self.model = DepthAnythingV2(encoder='vitl', features=256,
                                            out_channels=[256, 512, 1024, 1024], max_depth=20.)
                self.model.load_state_dict(torch.load(weights, map_location='cpu', weights_only=True), strict=True)
                self.model = self.model.to('cuda:0').eval()
                source_dir = depth_root/'Depth-Anything-V2/metric_depth/depth_anything_v2'
                sources = {str(f.relative_to(depth_root.parent)): sha_file(f) for f in sorted(source_dir.rglob('*.py'))}
                self.info.update(weights=str(weights), weights_sha256=sha_file(weights), source_sha256=sources,
                                 input_size=518, max_depth_m=20., strict_load=True)
            else:
                import yaml
                sys.path.insert(0, str(depth_root/'TradSL'))
                from tradsl.pipeline import StereoPipeline
                from tradsl.io import CameraParams
                self.pipeline_type, self.params_type = StereoPipeline, CameraParams
                config = depth_root/'TradSL/configs/d435.yaml'
                self.cfg = yaml.safe_load(config.read_text())
                sources = {str(f.relative_to(depth_root.parent)): sha_file(f) for f in sorted((depth_root/'TradSL/tradsl').glob('*.py'))}
                self.info.update(config=str(config), config_sha256=sha_file(config), source_sha256=sources,
                                 checkpoint='none (traditional stereo)', config_modified=False)
            fingerprint = {k: self.info[k] for k in ('weights_sha256', 'config_sha256', 'source_sha256') if k in self.info}
            self.info['model_fingerprint'] = hashlib.sha256(json.dumps(fingerprint, sort_keys=True).encode()).hexdigest()
            torch.cuda.synchronize()
        self.info.update(initialized=True, initialization_s=time.perf_counter() - started)

    def predict(self, payload, record):
        require(isinstance(payload, dict), 'request must be dict')
        sequence = payload['sequence']
        require(type(sequence) is int, 'sequence must be int (not bool)')
        record['sequence'] = sequence
        vision = payload['vision']
        require(isinstance(vision, dict), 'vision must be dict')
        depths = {}
        for cam in CAMS:
            t = time.perf_counter()
            meta = record['cameras'][cam] = {}
            left = vision[cam]
            color = rgb(left)
            meta['source_rgb_sha256'] = sha_array(color)
            if self.backend == 'dav2':
                self.torch.cuda.synchronize()
                infer_start = time.perf_counter()
                with self.torch.inference_mode():
                    depth = self.model.infer_image(cv2.cvtColor(color, cv2.COLOR_RGB2BGR), 518)
                self.torch.cuda.synchronize()
            else:
                right = vision[cam+'_stereo_right']
                meta['right_source_rgb_sha256'] = sha_array(rgb(right))
                lg, rg, K, baseline, T = stereo(left, right)
                meta.update(K=K.tolist(), baseline_m=baseline, right_in_left=T.tolist(),
                            left_gray_sha256=sha_array(lg), right_gray_sha256=sha_array(rg))
                if self.backend == 'nsl':
                    infer_start = time.perf_counter()
                    # RPC has uint64 IDs; caller sequence is preserved independently, including negative values.
                    rpc_seq = self.rpc_sequence
                    self.rpc_sequence += 1
                    stamp = time.time_ns()
                    self.client.close()
                    self.client.connect()
                    result = self.client.infer(sequence=rpc_seq, stamp_ns=stamp, ir_left=lg, ir_right=rg,
                                               fx_px=float(K[0, 0]), baseline_m=baseline)
                    require(result.sequence == rpc_seq and result.stamp_ns == stamp, 'NSL response IDs differ')
                    depth = result.depth_left_m
                    meta.update(inference_latency_ms=result.inference_latency_ms,
                                rpc_round_trip_ms=result.request_round_trip_ms, rpc_sequence=rpc_seq,
                                provenance_sha256=result.server_provenance_sha256)
                else:
                    params = self.params_type(fx=K[0, 0], fy=K[1, 1], cx=K[0, 2], cy=K[1, 2],
                                              baseline_mm=baseline*1000, width=640, height=480, K=K)
                    key = (K.tobytes(), baseline)
                    if cam not in self.pipelines or self.pipelines[cam][0] != key:
                        self.pipelines[cam] = (key, self.pipeline_type(self.cfg, params, device='cuda:0'))
                    self.torch.cuda.synchronize()
                    infer_start = time.perf_counter()
                    result = self.pipelines[cam][1].run(lg, rg)
                    self.torch.cuda.synchronize()
                    mm = result['depth']
                    require(mm.dtype == np.uint16, 'TradSL must return uint16 millimeters')
                    depth = mm.astype(np.float32) * .001
            meta['inference_s'] = time.perf_counter() - infer_start
            depth = np.ascontiguousarray(depth, dtype=np.float32)
            require(depth.shape == (480, 640), 'unexpected predicted depth shape')
            require(np.isfinite(depth).all() and (depth >= 0).all(), 'nonfinite or negative predicted depth')
            meta.update(predicted_sha256=sha_array(depth), dtype=str(depth.dtype), shape=list(depth.shape),
                        pixel_count=int(depth.size), finite_count=int(np.isfinite(depth).sum()),
                        positive_count=int((depth > 0).sum()), zero_count=int((depth == 0).sum()),
                        valid_002_8m_count=int(((depth > .02) & (depth < 8)).sum()),
                        min_m=float(depth.min()), max_m=float(depth.max()), wall_s=time.perf_counter() - t)
            depths[cam] = depth
        return {'depths': depths, 'backend': self.backend, 'sequence': sequence}

    def close(self):
        if self.client is not None:
            self.client.close()


class Handler(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.0'

    def reply(self, status, body, content_type):
        self.send_response(status)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path != '/health':
            return self.reply(404, b'not found\n', 'text/plain')
        self.reply(200, json.dumps(self.server.worker.info, sort_keys=True).encode(), 'application/json')

    def do_POST(self):
        if self.path != '/predict':
            return self.reply(404, b'not found\n', 'text/plain')
        t = time.perf_counter()
        record = {'time_ns': time.time_ns(), 'backend': self.server.worker.backend,
                  'model_fingerprint': self.server.worker.info['model_fingerprint'], 'sequence': None, 'cameras': {}}
        try:
            require(self.headers.get('Transfer-Encoding') is None, 'chunked bodies unsupported')
            length = int(self.headers['Content-Length'])
            require(0 < length <= 128*1024*1024, 'body length out of range')
            self.connection.settimeout(360)
            body = self.rfile.read(length)
            require(len(body) == length, 'truncated body')
            decoded = time.perf_counter()
            payload = pickle.loads(body)  # TRUSTED LOOPBACK ONLY: pickle executes code.
            record.update(body_bytes=length, receive_s=decoded-t, decode_s=time.perf_counter()-decoded)
            result = self.server.worker.predict(payload, record)
            encoded_at = time.perf_counter()
            response = pickle.dumps(result, protocol=pickle.HIGHEST_PROTOCOL)
            record.update(encode_s=time.perf_counter()-encoded_at, wall_s=time.perf_counter()-t,
                          response_bytes=len(response), status=200)
            self.server.write_log(record)
        except Exception as exc:
            record.update(status=500, error_type=type(exc).__name__, error=str(exc), wall_s=time.perf_counter()-t)
            traceback.print_exc()
            try:
                self.server.write_log(record)
            except Exception:
                traceback.print_exc()
            error = json.dumps({'error': type(exc).__name__, 'message': str(exc), 'sequence': record['sequence']}).encode()
            return self.reply(500, error, 'application/json')
        self.reply(200, response, 'application/octet-stream')


class Server(HTTPServer):
    allow_reuse_address = True

    def write_log(self, record):
        self.logfile.write(json.dumps(record, sort_keys=True, allow_nan=False)+'\n')
        self.logfile.flush()


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--backend', choices=('dav2', 'tradsl', 'nsl'), required=True)
    p.add_argument('--port', type=int, default=18401)
    p.add_argument('--log', type=Path, default=Path(__file__).with_name('requests.jsonl'))
    p.add_argument('--depth-root', type=Path, default=DEFAULT_DEPTH_ROOT,
                   help='Depth checkout containing TradSL, Depth-Anything-V2 and nsl_infer_server (not their parent)')
    p.add_argument('--data-root', type=Path, required=True,
                   help='Data directory containing models/dav2-metric-indoor checkpoint assets')
    p.add_argument('--nsl-port', type=int, default=17639,
                   help='Existing loopback NSL server port (default: 17639); never starts that server')
    return p


def main(argv=None):
    a = build_parser().parse_args(argv)
    require(1 <= a.port <= 65535, 'invalid port')
    require(1 <= a.nsl_port <= 65535, 'invalid NSL port')
    # Bind before heavyweight initialization so an occupied port cannot allocate a model.
    server = Server(('127.0.0.1', a.port), Handler)
    worker = None
    signal.signal(signal.SIGTERM, lambda signum, frame: sys.exit(0))
    try:
        with a.log.open('a', encoding='utf-8') as logfile:
            server.logfile = logfile
            worker = Worker(a.backend, depth_root=a.depth_root, data_root=a.data_root, nsl_port=a.nsl_port)
            server.worker = worker
            print(json.dumps({'ready': True, 'host': '127.0.0.1', 'port': a.port, **worker.info}), flush=True)
            server.serve_forever(poll_interval=.2)
    except KeyboardInterrupt:
        pass
    finally:
        if worker is not None:
            worker.close()
        server.server_close()


if __name__ == '__main__':
    main()
