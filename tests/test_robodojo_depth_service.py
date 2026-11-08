"""CPU-only contracts; never start a real HTTP service or initialize models."""
import importlib.util
import io
import json
from pathlib import Path
import pickle
import subprocess
import sys
import types
import unittest
from unittest.mock import patch

import cv2
import numpy as np

SERVICE = Path(__file__).resolve().parents[1] / 'services' / 'robodojo_depth.py'
spec = importlib.util.spec_from_file_location('maintained_robodojo_depth', SERVICE)
service = importlib.util.module_from_spec(spec)
spec.loader.exec_module(service)


def view():
    return {'color': np.zeros((480, 640, 3), np.uint8),
            'intrinsic_matrix': np.array([[288., 0, 319.5], [0, 290., 239.5], [0, 0, 1]]),
            'extrinsic_matrix': np.eye(4)}


def pair():
    left, right = view(), view()
    right['extrinsic_matrix'][0, 3] = .05
    return left, right


def payload():
    vision = {}
    for cam in service.CAMS:
        vision[cam], vision[cam + '_stereo_right'] = pair()
    return {'sequence': -12, 'vision': vision}


class ContractTests(unittest.TestCase):
    def test_stereo_contract_and_rgb_gray(self):
        left, right = pair()
        left['color'][0, 0] = [255, 0, 0]
        right['color'][0, 0] = [0, 255, 0]
        # Camera-to-world rotation/translation cancels; no convention conversion.
        world = np.array([[0., -1, 0, 2], [1, 0, 0, 3], [0, 0, 1, 4], [0, 0, 0, 1]])
        left['extrinsic_matrix'] = world
        right['extrinsic_matrix'] = world @ right['extrinsic_matrix']
        lg, rg, K, baseline, relative = service.stereo(left, right)
        np.testing.assert_array_equal(lg, cv2.cvtColor(left['color'], cv2.COLOR_RGB2GRAY))
        np.testing.assert_array_equal(rg, cv2.cvtColor(right['color'], cv2.COLOR_RGB2GRAY))
        self.assertEqual(lg.shape, (480, 640))
        self.assertEqual(lg.dtype, np.uint8)
        self.assertAlmostEqual(baseline, .05)
        np.testing.assert_array_equal(K, left['intrinsic_matrix'])
        np.testing.assert_allclose(relative[:3, 3], [.05, 0, 0], atol=1e-12)

    def test_invalid_rgb(self):
        for color in [None, np.zeros((480, 640, 3), np.float32), np.zeros((640, 480, 3), np.uint8), np.zeros((480, 640), np.uint8)]:
            with self.subTest(color_type=type(color)), self.assertRaisesRegex(ValueError, 'RGB uint8'):
                service.rgb({'color': color})

    def test_invalid_stereo(self):
        for kind in ['shape', 'nan', 'focal', 'intrinsics', 'homogeneous', 'rotation', 'baseline_sign', 'baseline_y', 'baseline_size']:
            left, right = pair()
            if kind == 'shape': left['intrinsic_matrix'] = np.eye(4)
            elif kind == 'nan': right['extrinsic_matrix'][0, 0] = np.nan
            elif kind == 'focal': left['intrinsic_matrix'][0, 0] = -1
            elif kind == 'intrinsics': right['intrinsic_matrix'][0, 0] += 1
            elif kind == 'homogeneous': right['extrinsic_matrix'][3, 0] = 1
            elif kind == 'rotation': right['extrinsic_matrix'][0, 0] = -1
            elif kind == 'baseline_sign': right['extrinsic_matrix'][0, 3] = -.05
            elif kind == 'baseline_y': right['extrinsic_matrix'][1, 3] = .01
            else: right['extrinsic_matrix'][0, 3] = .06
            with self.subTest(kind=kind), self.assertRaises(ValueError): service.stereo(left, right)

    def test_parser_paths_and_help_without_models(self):
        args = service.build_parser().parse_args(['--backend', 'tradsl', '--depth-root', '/custom/Depth', '--data-root', '/custom/data'])
        self.assertEqual(args.depth_root / 'TradSL', Path('/custom/Depth/TradSL'))
        self.assertEqual(args.data_root, Path('/custom/data'))
        self.assertEqual(args.nsl_port, 17639)
        self.assertEqual(args.port, 18401)
        result = subprocess.run([sys.executable, str(SERVICE), '--help'], capture_output=True, text=True, timeout=20)
        self.assertEqual(result.returncode, 0, result.stderr)
        for term in ['--depth-root', '--data-root', '--nsl-port', '17639', 'pickle', '127.0.0.1', 'dav2', 'tradsl', 'nsl']:
            self.assertIn(term, result.stdout)

    def test_invalid_cli_ports_bind_nothing(self):
        for flag in ['--port', '--nsl-port']:
            for port in ['0', '65536']:
                with self.subTest(flag=flag, port=port), patch.object(service, 'Server') as server, patch.object(service, 'Worker') as worker:
                    with self.assertRaises(ValueError): service.main(['--backend', 'nsl', '--data-root', '/unused', flag, port])
                    server.assert_not_called(); worker.assert_not_called()

    def test_fake_nsl_configuration_and_persistence(self):
        # Stub only the external NSL modules. No network, torch or GPU code.
        doc = {'method': 'nsl_v1_5', 'sha256': {'weights': service.WEIGHTS_SHA, 'config': service.CONFIG_SHA}}
        calls = []
        class Client:
            def __init__(self, **kwargs): self.options = kwargs; self.closed = 0
            def connect(self): pass
            def close(self): self.closed += 1
            def infer(self, **kwargs):
                calls.append(kwargs)
                return types.SimpleNamespace(sequence=kwargs['sequence'], stamp_ns=kwargs['stamp_ns'],
                    depth_left_m=np.ones((480, 640), np.float32), inference_latency_ms=1,
                    request_round_trip_ms=2, server_provenance_sha256='identity')
        client_mod = types.ModuleType('record.depth.nsl.client'); client_mod.NslInferenceClient = Client
        wire_mod = types.ModuleType('record.depth.nsl.wire')
        wire_mod.fetch_server_provenance = lambda host, port: (calls.append((host, port)) or (doc, 'identity'))
        modules = {name: types.ModuleType(name) for name in ['record', 'record.depth', 'record.depth.nsl']}
        modules.update({'record.depth.nsl.client': client_mod, 'record.depth.nsl.wire': wire_mod})
        with patch.dict(sys.modules, modules), patch.object(sys, 'path', list(sys.path)):
            worker = service.Worker('nsl', depth_root='/custom/Depth', data_root='/custom/data', nsl_port=19000)
            self.assertEqual(sys.path[0], '/custom/Depth/nsl_infer_server/server')
            self.assertEqual(calls[0], ('127.0.0.1', 19000))
            self.assertEqual(worker.client.options['port'], 19000)
            self.assertEqual(worker.info['endpoint'], '127.0.0.1:19000')
            client = worker.client
            for seq in [-12, 999]:
                data = payload(); data['sequence'] = seq
                result = worker.predict(data, {'cameras': {}})
                self.assertEqual(result['sequence'], seq)
                self.assertIs(worker.client, client)
                self.assertEqual(set(result['depths']), set(service.CAMS))
                for depth in result['depths'].values():
                    self.assertEqual(depth.dtype, np.float32)
                    self.assertEqual(depth.shape, (480, 640))
                    self.assertTrue(depth.flags.c_contiguous)
            self.assertEqual([c['sequence'] for c in calls[1:]], list(range(6)))
            for bad in [None, [], {'sequence': True}, {'sequence': 1, 'vision': []}]:
                with self.subTest(bad=bad), self.assertRaises(ValueError): worker.predict(bad, {'cameras': {}})
            worker.close()
            self.assertEqual(client.closed, 7)

    def test_tradsl_pipeline_cache_and_millimeters(self):
        worker = service.Worker.__new__(service.Worker)
        worker.backend = 'tradsl'; worker.pipelines = {}; worker.cfg = {}
        worker.client = None
        worker.torch = types.SimpleNamespace(cuda=types.SimpleNamespace(synchronize=lambda: None))
        worker.params_type = lambda **kwargs: kwargs
        built = []
        class Pipeline:
            def __init__(self, cfg, params, device):
                built.append((params, device))
            def run(self, left, right): return {'depth': np.full((480, 640), 1234, np.uint16)}
        worker.pipeline_type = Pipeline
        for _ in range(2):
            result = worker.predict(payload(), {'cameras': {}})
            for depth in result['depths'].values(): np.testing.assert_allclose(depth, 1.234)
        self.assertEqual(len(built), 3)
        self.assertEqual(built[0][0]['baseline_mm'], 50.)
        self.assertEqual(built[0][1], 'cuda:0')
        data = payload()
        for suffix in ['', '_stereo_right']:
            data['vision']['cam_head' + suffix]['intrinsic_matrix'][0, 0] += 1
        worker.predict(data, {'cameras': {}})
        self.assertEqual(len(built), 4)
        self.assertEqual(built[-1][0]['fx'], 289.)
        worker.pipelines['cam_head'][1].run = lambda l, r: {'depth': np.ones((480, 640), np.float32)}
        with self.assertRaisesRegex(ValueError, 'uint16 millimeters'):
            worker.predict(data, {'cameras': {}})

    def test_dav2_rgb_only_and_output_validation(self):
        from contextlib import nullcontext
        worker = service.Worker.__new__(service.Worker)
        worker.backend = 'dav2'
        worker.torch = types.SimpleNamespace(cuda=types.SimpleNamespace(synchronize=lambda: None), inference_mode=nullcontext)
        calls = []
        def infer(image, size):
            calls.append((image[0, 0].tolist(), size))
            return np.ones((480, 640), np.float64)
        worker.model = types.SimpleNamespace(infer_image=infer)
        # DAV2 requires no calibration, right view, or GT depth.
        data = {'sequence': 0, 'vision': {cam: {'color': view()['color']} for cam in service.CAMS}}
        for v in data['vision'].values(): v['color'][0, 0] = [1, 2, 3]
        result = worker.predict(data, {'cameras': {}})
        self.assertEqual(calls, [([3, 2, 1], 518)] * 3)
        self.assertEqual(result['depths']['cam_head'].dtype, np.float32)
        for bad in [np.zeros((10, 10)), np.full((480, 640), np.nan), np.full((480, 640), -1.)]:
            worker.model.infer_image = lambda *args: bad
            with self.subTest(shape=bad.shape), self.assertRaises(ValueError):
                worker.predict(data, {'cameras': {}})

    def test_http_contract_in_memory_no_sockets(self):
        handler = service.Handler.__new__(service.Handler)
        handler.path = '/health'
        handler.server = types.SimpleNamespace(worker=types.SimpleNamespace(info={'initialized': True}))
        replies = []
        handler.reply = lambda status, body, content_type: replies.append((status, body, content_type))
        handler.do_GET()
        self.assertEqual(replies[-1][0], 200)
        self.assertEqual(json.loads(replies[-1][1]), {'initialized': True})
        handler.path = '/wrong'; handler.do_GET(); self.assertEqual(replies[-1][0], 404)
        handler.do_POST(); self.assertEqual(replies[-1][0], 404)
        data = {'sequence': 3, 'vision': {}}
        body = pickle.dumps(data)
        logs = []
        worker = types.SimpleNamespace(backend='nsl', info={'model_fingerprint': 'test'},
            predict=lambda p, record: {'sequence': p['sequence'], 'backend': 'nsl', 'depths': {}})
        handler.server = types.SimpleNamespace(worker=worker, write_log=logs.append)
        handler.path = '/predict'; handler.headers = {'Content-Length': str(len(body))}
        handler.rfile = io.BytesIO(body); handler.connection = types.SimpleNamespace(settimeout=lambda timeout: None)
        handler.do_POST()
        self.assertEqual(replies[-1][0], 200)
        self.assertEqual(pickle.loads(replies[-1][1])['sequence'], 3)
        self.assertEqual(logs[-1]['status'], 200)
        for headers, raw in [({'Content-Length': '0'}, b''), ({'Content-Length': '10'}, b'x'),
                             ({'Transfer-Encoding': 'chunked'}, b''), ({}, b'')]:
            handler.headers = headers; handler.rfile = io.BytesIO(raw)
            with patch.object(service.traceback, 'print_exc'): handler.do_POST()
            self.assertEqual(replies[-1][0], 500)
            self.assertIn('error', json.loads(replies[-1][1]))

    def test_main_fixed_loopback_and_cleanup(self):
        with patch.object(service, 'Server') as server, patch.object(service, 'Worker') as worker, patch.object(service.signal, 'signal'), patch('builtins.open'):
            # /dev/null avoids creating a log file; all services/models mocked.
            service.main(['--backend', 'nsl', '--data-root', '/custom/data', '--depth-root', '/custom/Depth', '--nsl-port', '19000', '--log', '/dev/null'])
            self.assertEqual(server.call_args.args[0], ('127.0.0.1', 18401))
            worker.assert_called_once_with('nsl', depth_root=Path('/custom/Depth'), data_root=Path('/custom/data'), nsl_port=19000)
            worker.return_value.close.assert_called_once()
            server.return_value.server_close.assert_called_once()


if __name__ == '__main__':
    unittest.main()
