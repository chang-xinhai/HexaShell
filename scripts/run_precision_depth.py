#!/usr/bin/env python3
"""Five isolated precision lanes. Planning is the default; --execute starts workloads.

Only YAML/JSON run artifacts are generated. No setup hooks, import-path injection,
source edits, model overrides, or task motion-budget overrides are used. The native
RoboDojo eval CLI remains responsible for its policy/simulator environments.
"""
from __future__ import annotations

import argparse
import ast
import concurrent.futures
import copy
import csv
import io
import json
import math
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import uuid
import urllib.request

LANES = ("rgb", "gt", "dav2", "tradsl", "nsl")


def tasks_from_inventory(root):
    """Read the canonical literal without executing Isaac-dependent code."""
    path = root / "RoboDojo/scripts/internal/task_inventory.py"
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == "DIMENSION_TASKS":
            tasks = ast.literal_eval(node.value)["precision"]
            if len(tasks) != 8 or len(set(tasks)) != 8:
                raise ValueError("native precision inventory must contain eight distinct tasks")
            return list(tasks)
    raise ValueError(f"canonical precision inventory not found in {path}")


def gpu_ids(value):
    values = value.split(",")
    if len(values) != 5 or len(set(values)) != 5 or any(not v.isdigit() for v in values):
        raise argparse.ArgumentTypeError("--gpus requires five distinct numeric GPU IDs, comma-separated")
    return values


def positive(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def yaml_load(path):
    import yaml
    with path.open() as stream:
        return yaml.safe_load(stream)


def yaml_save(path, value):
    import yaml
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(value, sort_keys=False))


def native_cameras(root, tasks):
    """Resolve the supported native dual-X5 merge, refusing unknown semantics.

    RobotManager replaces wrist roles from Assets metadata after reading camera
    YAML. Supplemental *_stereo_right roles survive that merge with resolved mount links.
    """
    robo = root / "RoboDojo"
    base = yaml_load(robo / "env_cfg/arx_x5.yml")
    index = yaml_load(robo / "task/RoboDojo/config/_task.yml")
    for task in tasks:
        settings = dict(index.get("common", {}), **index.get("tasks", {}).get(task, {}))
        if settings.get("robot_config", "dual_x5") != "dual_x5" or settings.get("camera_config", "camera_config") != "camera_config":
            raise ValueError(f"cannot safely resolve {task}: native per-task robot/camera override")
    robots = yaml_load(robo / "env_cfg/robot/dual_x5.yml")["robots"]
    if len(robots) != 2 or any(r.get("robot_name") != "x5" or r.get("coupled", False) or r.get("type", "target") != "target" for r in robots):
        raise ValueError("cannot safely resolve native wrist cameras: expected two uncoupled target X5 robots")
    camera = yaml_load(robo / "env_cfg/camera/camera_config.yml")
    metadata = yaml_load(robo / "Assets/Robots/x5/robot_config.yml").get("camera")
    if not metadata or len(metadata) != 1 or metadata[0].get("name") != "cam_wrist":
        raise ValueError("cannot safely resolve native X5 cam_wrist metadata")
    wrist = metadata[0]
    for i, side in enumerate(("left", "right")):
        required = ("link", "pos", "ori", "type")
        if any(key not in wrist for key in required):
            raise ValueError("cannot safely resolve native wrist pose/type: incomplete metadata")
        pose = {key: copy.deepcopy(wrist[key]) for key in ("pos", "ori", "type", "mesh") if key in wrist}
        pose["mount_link"] = f"robot{i}/{wrist['link']}"
        if len(pose["ori"]) == 4:  # Native RobotManager converts historical WXYZ metadata.
            w, x, y, z = pose["ori"]
            pose["ori"] = [x, y, z, w]
        camera[f"cam_{side}_wrist"] = {"camera": pose}
    if set(k for k, v in camera.items() if isinstance(v, dict) and "camera" in v) != {"cam_head", "cam_left_wrist", "cam_right_wrist"}:
        raise ValueError("cannot safely resolve unexpected native camera roles")
    for role in ("cam_head", "cam_left_wrist", "cam_right_wrist"):
        pose = camera[role]["camera"]
        if len(pose.get("pos", [])) != 3 or len(pose.get("ori", [])) not in (3, 4):
            raise ValueError(f"cannot safely resolve native camera pose: {role}")
        if not all(math.isfinite(float(v)) for v in pose["pos"] + pose["ori"]):
            raise ValueError(f"nonfinite native camera pose: {role}")
    return base, camera


def camera_variant(camera, lane):
    result = copy.deepcopy(camera)
    roles = ("cam_head", "cam_left_wrist", "cam_right_wrist")
    result["annotator"] = {"common": {"enabled": True, "rgb_capture": {"type": "rgb", "device": "cpu"}}}
    if lane == "gt":
        result["annotator"]["common"]["distance_to_image_plane_capture"] = {"type": "distance_to_image_plane", "device": "cpu"}
    if lane in ("tradsl", "nsl"):
        if result.get("random"):
            raise ValueError("cannot safely resolve stereo randomization correspondence")
        for role in roles:
            right = copy.deepcopy(result[role])
            orientation = right["camera"]["ori"]
            if len(orientation)==3:
                x,y,z=map(math.radians,orientation)
                cx,sx,cy,sy,cz,sz=math.cos(x),math.sin(x),math.cos(y),math.sin(y),math.cos(z),math.sin(z)
                axis=(cy*cz,cx*sz+sx*sy*cz,sx*sz-cx*sy*cz)
            else:
                norm=math.sqrt(sum(v*v for v in orientation))
                if norm==0:raise ValueError("Zero camera quaternion")
                x,y,z,w=(v/norm for v in orientation)
                axis=(1-2*(y*y+z*z),2*(x*y+z*w),2*(x*z-y*w))
            right["camera"]["pos"]=[float(p+.05*v) for p,v in zip(right["camera"]["pos"],axis)]
            result[role + "_stereo_right"] = right
    return result


def build_plan(args, output):
    root = args.root.resolve()
    tasks = tasks_from_inventory(root)
    selected = getattr(args, "tasks", None)
    if selected:
        requested = selected.split(",")
        if len(set(requested)) != len(requested) or any(t not in tasks for t in requested):
            raise ValueError("--tasks must contain unique canonical precision task names")
        tasks = [t for t in tasks if t in requested]
    native = root / "RoboDojo"
    records = []
    lanes = []
    for index, (lane, gpu) in enumerate(zip(LANES, args.gpus)):
        lane_dir = output / lane
        config_path = lane_dir / "env.yml"
        # Shell eval_policy.sh concatenates env_cfg/name.yml; traversal is intentional.
        config_stem = os.path.relpath(config_path.with_suffix(""), native / "env_cfg")
        commands = []
        for task in tasks:
            task_dir = lane_dir / task
            invocation = {
                "lane": lane, "gpu": gpu, "task": task,
                "run_id": f"{output.name}-{lane}-{task}",
                "config_name": yaml_load(native / "env_cfg/arx_x5.yml")["config_name"],
                "log": str(task_dir / "eval.log"), "result_dir": str(task_dir / "policy"),
                "argv": ["bash", str(native / "scripts/robodojo.sh"), "eval",
                         "--policy-dir", str(native / "XPolicyLab/policy/HexaShell"),
                         "--task", task, "--ckpt", "default", "--action-type", "joint",
                         "--policy-env", "${POLICY_ENV:-hexashell}", "--eval-env", "${EVAL_ENV:-RoboDojo}",
                         "--env-cfg", config_stem, "--policy-gpu", gpu, "--env-gpu", gpu,
                         "--seed", str(args.seed), "--eval-num", str(args.eval_num)],
                "env": {"HEXASHELL_DEPTH": "0" if lane == "rgb" else "1",
                        "HEXASHELL_DEPTH_BACKEND": "gt" if lane == "rgb" else lane,
                        "HEXASHELL_DEPTH_URL": "" if lane in ("rgb", "gt") else f"http://127.0.0.1:{19120 + index}",
                        "HEXASHELL_RESULT_DIR": str(task_dir / "policy"),
                        "ROBODOJO_RUN_ID": f"{output.name}-{lane}-{task}"},
            }
            commands.append(invocation)
            records.append(invocation)
        lanes.append({"name": lane, "gpu": gpu, "port": 19120 + index,
                      "nsl_port": 19130, "config": str(config_path), "invocations": commands})
    return {"schema": 1, "mode": "execute" if args.execute else "dry-run", "root": str(root),
            "data_root": str(args.data_root.resolve()), "output": str(output),
            "seed": args.seed, "eval_num": args.eval_num, "tasks": tasks,
            "lanes": lanes, "invocations": records,
            "budget": "native per-task budgets; no overrides", "camera_preflight": "pending"}


def materialize(plan):
    root = Path(plan["root"])
    base, camera = native_cameras(root, plan["tasks"])
    sim = yaml_load(root / "RoboDojo/env_cfg/sim/sim_config.yml")
    sim.setdefault("scene", {})["num_envs"] = 1
    for lane in plan["lanes"]:
        directory = Path(lane["config"]).parent
        yaml_save(directory / "camera.yml", camera_variant(camera, lane["name"]))
        yaml_save(directory / "sim.yml", sim)
        cfg = copy.deepcopy(base)
        cfg["config_name"] = lane["invocations"][0]["config_name"]
        cfg["config"]["camera"] = str(directory / "camera")
        cfg["config"]["sim"] = os.path.relpath(directory / "sim", root / "RoboDojo/env_cfg/sim")
        vision = cfg.setdefault("observation", {}).setdefault("vision", {})
        vision.update(depth=lane["name"] == "gt", approximate_depth=False,
                      intrinsic_matrix=True, extrinsic_matrix=True, shape=True)
        yaml_save(Path(lane["config"]), cfg)
    plan["camera_preflight"] = "resolved native head and dual-X5 wrists; stereo baseline 0.05m camera-local +x"


def sourced_environment(root):
    """Capture inherited credentials only during execution, never persist/print them."""
    result = subprocess.run(["bash", "-c", 'set -ae; source "$1" >/dev/null 2>&1; env -0',
                             "precision-env", str(root / "HexaShell/config/env.sh")],
                            check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    env = dict(item.split("=", 1) for item in result.stdout.decode().split("\0") if "=" in item)
    conda_exe = Path(env.get("CONDA_EXE", ""))
    if not conda_exe.is_file():
        candidate = Path(sys.executable).resolve().parents[3] / "bin/conda"
        if not candidate.is_file():
            raise ValueError("Conda executable missing; activate policy environment or set CONDA_EXE")
        conda_exe = candidate
    env["PATH"] = str(conda_exe.parent) + os.pathsep + env.get("PATH", "")
    env["CONDA_EXE"] = str(conda_exe)
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONSTARTUP", None)
    for key in ("HEXASHELL_RGBD_ADAPTER","EXACT_ROBODOJO_LAYOUT","DEPTH_BACKEND","DEPTH_PREDICTOR_URL","HEXASHELL_ARCHIVE_DIR","HEXASHELL_DEPTH_RECORD_DIR"):
        env.pop(key,None)
    return env


def resource_preflight(plan, env):
    if os.name != "posix":
        raise ValueError("--execute requires POSIX process groups")
    root = Path(plan["root"])
    for path in (root / "HexaShell/services/robodojo_depth.py", root / "HexaShell/config/env.sh",
                 root / "RoboDojo/scripts/robodojo.sh", Path(plan["data_root"])):
        if not path.exists():
            raise ValueError(f"execute preflight missing: {path}")
    server = root / "HexaShell/third_party/nsl_infer_server/server"
    checkpoint = Path(plan["data_root"]) / "models/dav2-metric-indoor/depth_anything_v2_metric_hypersim_vitl.pth"
    if not checkpoint.is_file():
        raise ValueError(f"execute preflight missing DAV2 metric checkpoint: {checkpoint}")
    for path in (server / "run.sh", server / "record/depth/nsl_v1_5/inference_server.py",
                 Path(plan["data_root"])/"models/nsl/nsl1.5_lingbot_giant_detail_global_200k.pt",
                 server / "third_party/nslv1.5/model.py"):
        if not path.is_file():
            raise ValueError(f"execute preflight missing native NSL asset: {path}")
    result = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,memory.free", "--format=csv,noheader,nounits"],
                            check=True, capture_output=True, text=True, env=env)
    available = {row[0].strip(): (row[1].strip(), int(row[2])) for row in csv.reader(io.StringIO(result.stdout))}
    selected = [lane["gpu"] for lane in plan["lanes"]]
    if any(gpu not in available for gpu in selected):
        raise ValueError("execute preflight: requested GPU does not exist")
    if any(available[gpu][1] < 16000 for gpu in selected):
        raise ValueError("execute preflight: each lane needs at least 16000 MiB free GPU memory")
    active = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"],
                            check=True, capture_output=True, text=True, env=env)
    uuids = {available[gpu][0] for gpu in selected}
    if any(row and row[0].strip() in uuids for row in csv.reader(io.StringIO(active.stdout))):
        raise ValueError("execute preflight: selected GPU already has compute processes")
    for port in [lane["port"] for lane in plan["lanes"] if lane["name"] not in ("rgb", "gt")] + [19130]:
        with socket.socket() as sock:
            try:
                sock.bind(("127.0.0.1", port))
            except OSError as exc:
                raise ValueError(f"execute preflight: port {port} unavailable") from exc
    deploy = yaml_load(root / "HexaShell/xpolicylab/HexaShell/deploy.yml")
    url = env.get("HEXASHELL_SAM3_URL", deploy.get("sam3_url", ""))
    if not url:
        raise ValueError("execute preflight: SAM3 URL is required")
    from urllib.parse import urlsplit
    address = urlsplit(url)
    with socket.create_connection((address.hostname, address.port or 80), timeout=3):
        pass


class Processes:
    def __init__(self):
        self.lock = threading.Lock()
        self.children = set()
        self.stop = threading.Event()

    def start(self, argv, cwd, env, log):
        if self.stop.is_set():
            raise RuntimeError("launcher interrupted")
        with self.lock:
            process = subprocess.Popen(argv, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=True)
            self.children.add(process)
        return process

    def terminate(self, process):
        # Always signal the owned group, including descendants after leader exit.
        for sig, timeout in ((signal.SIGTERM, 10), (signal.SIGKILL, 3)):
            try:
                os.killpg(process.pid, sig)
            except ProcessLookupError:
                break
            try:
                process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                continue
            if sig == signal.SIGTERM:
                # A leader can exit while its policy descendants remain.
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            break
        with self.lock:
            self.children.discard(process)

    def close(self):
        self.stop.set()
        with self.lock:
            children = list(self.children)
        for process in children:
            self.terminate(process)


def wait_service(process, port, processes, timeout=600, backend=None):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline and not processes.stop.is_set():
        if process.poll() is not None:
            raise RuntimeError(f"service exited with code {process.returncode}; inspect service log")
        try:
            if backend is not None:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1) as response:
                    info = json.load(response)
                if info.get("backend") != backend:
                    raise RuntimeError("depth service readiness backend mismatch")
            else:
                with socket.create_connection(("127.0.0.1", port), timeout=1):
                    pass
            return
        except (OSError, ValueError):
            processes.stop.wait(0.25)
    raise RuntimeError(f"service readiness timeout on port {port}")


def result_scores(plan, item):
    native = Path(plan["root"]) / "RoboDojo/eval_result/RoboDojo" / item["task"] / "HexaShell" / item["config_name"]
    paths = list(native.glob(f"*/{item['run_id']}/_result.json"))
    if len(paths) != 1:
        raise RuntimeError(f"expected one native _result.json for {item['run_id']}; found {len(paths)}")
    data = json.loads(paths[0].read_text())
    details = data.get("details")
    entries = list(details.values()) if isinstance(details, dict) else details
    if not isinstance(entries, list) or len(entries) != plan["eval_num"]:
        raise RuntimeError("native result episode count does not match requested eval-num")
    if any(not isinstance(e.get("success"), bool) or not isinstance(e.get("score"), (float, int)) for e in entries):
        raise RuntimeError("native result missing official success/score fields")
    destination = Path(item["log"]).parent / "_result.json"
    shutil.copy2(paths[0], destination)
    return {"native_result": str(paths[0]), "episodes": len(entries),
            "success_rate_percent": 100 * sum(e["success"] for e in entries) / len(entries),
            "official_score_percent": 100 * sum(e["score"] for e in entries) / len(entries)}


def run_lane(plan, lane, inherited, processes):
    root = Path(plan["root"])
    directory = Path(lane["config"]).parent
    directory.mkdir(parents=True, exist_ok=True)
    env = dict(inherited, CUDA_VISIBLE_DEVICES=lane["gpu"])
    services, handles, results = [], [], []
    try:
        python = str(Path(plan["data_root"]) / "envs" / {"dav2":"dav2","tradsl":"tradsl","nsl":"nsl-infer"}.get(lane["name"],"dav2") / "bin/python")
        if lane["name"] == "nsl":
            server = root / "HexaShell/third_party/nsl_infer_server/server"
            # Native run.sh's v1.5 weights/alt/conf, no comparison gateway, setup or path injection.
            nsl_python = env.get("NSLV1_5_PYTHON") or env.get("DEPTH_SERVER_PYTHON") or python
            argv = [nsl_python, "-m", "record.depth.nsl_v1_5.inference_server", "--host", "127.0.0.1",
                    "--port", str(lane["nsl_port"]), "--device", "cuda:0",
                    "--weights", str(Path(plan["data_root"])/"models/nsl/nsl1.5_lingbot_giant_detail_global_200k.pt"),
                    "--config", str(Path(plan["data_root"])/"models/nsl/nsl1.5_lingbot_giant_detail_global_200k.yaml"),
                    "--precision","mixed","--no-conf","--startup-timeout","240","--inference-timeout","120"]
            log = (directory / "nsl.log").open("w"); handles.append(log)
            service = processes.start(argv, server, env, log); services.append(service)
            wait_service(service, lane["nsl_port"], processes)
        if lane["name"] not in ("rgb", "gt"):
            argv = [python, str(root / "HexaShell/services/robodojo_depth.py"), "--backend", lane["name"],
                    "--port", str(lane["port"]), "--log", str(directory / "depth.jsonl"),
                    "--depth-root", str(root / "HexaShell/third_party"), "--data-root", plan["data_root"],
                    "--nsl-port", str(lane["nsl_port"])]
            log = (directory / "service.log").open("w"); handles.append(log)
            service = processes.start(argv, root / "HexaShell", env, log); services.append(service)
            wait_service(service, lane["port"], processes, backend=lane["name"])
        for item in lane["invocations"]:
            record = {"lane": lane["name"], "task": item["task"], "status": "failed"}
            results.append(record)
            task_env = dict(env, **item["env"])
            argv = [inherited.get("POLICY_ENV", "hexashell") if value == "${POLICY_ENV:-hexashell}" else
                    inherited.get("EVAL_ENV", "RoboDojo") if value == "${EVAL_ENV:-RoboDojo}" else value for value in item["argv"]]
            Path(item["log"]).parent.mkdir(parents=True, exist_ok=True)
            try:
                if any(service.poll() is not None for service in services):
                    raise RuntimeError("lane depth service died")
                with Path(item["log"]).open("w") as log:
                    child = processes.start(argv, root / "RoboDojo", task_env, log)
                    try:
                        while child.poll() is None:
                            if processes.stop.wait(0.5):
                                raise RuntimeError("launcher interrupted")
                            if any(service.poll() is not None for service in services):
                                raise RuntimeError("lane depth service died during evaluation")
                        record["returncode"] = child.returncode
                        if child.returncode:
                            raise RuntimeError(f"native eval exited {child.returncode}; see {item['log']}")
                    finally:
                        processes.terminate(child)
                record.update(result_scores(plan, item), status="completed")
            except Exception as exc:
                record["error"] = str(exc)
                print(f"[{lane['name']}/{item['task']}] ERROR: {exc}", file=sys.stderr)
    except Exception as exc:
        print(f"[{lane['name']}] ERROR: {exc}", file=sys.stderr)
        completed = {record["task"] for record in results}
        results.extend({"lane": lane["name"], "task": item["task"], "status": "failed", "error": str(exc)}
                       for item in lane["invocations"] if item["task"] not in completed)
    finally:
        for service in reversed(services):
            processes.terminate(service)
        for handle in handles:
            handle.close()
    return results


def execute(plan):
    env = sourced_environment(Path(plan["root"]))
    resource_preflight(plan, env)
    processes = Processes()
    previous = {}
    def stop(signum, frame):
        processes.stop.set()
    for signum in (signal.SIGINT, signal.SIGTERM):
        previous[signum] = signal.signal(signum, stop)
    records = []
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
            futures = [pool.submit(run_lane, plan, lane, env, processes) for lane in plan["lanes"]]
            for future in concurrent.futures.as_completed(futures):
                records.extend(future.result())
    finally:
        processes.close()
        for signum, handler in previous.items():
            signal.signal(signum, handler)
        summary = {"results": records, "completed": sum(r["status"] == "completed" for r in records),
                   "failed": sum(r["status"] != "completed" for r in records),
                   "scoring": "native _result.json detail success and score; partial experiment, not full benchmark"}
        summary["lanes"] = {}
        for lane in LANES:
            done = [r for r in records if r["lane"] == lane and r["status"] == "completed"]
            count = sum(r["episodes"] for r in done)
            summary["lanes"][lane] = {
                "completed_tasks": len(done), "expected_tasks": len(plan["tasks"]), "episodes": count,
                "success_rate_percent": sum(r["success_rate_percent"] * r["episodes"] for r in done) / count if count else None,
                "official_score_percent": sum(r["official_score_percent"] * r["episodes"] for r in done) / count if count else None,
                "partial": len(done) != len(plan["tasks"]),
            }
        (Path(plan["output"]) / "summary.json").write_text(json.dumps(summary, indent=2))
    return 0 if len(records) == len(plan["invocations"]) and all(r["status"] == "completed" for r in records) else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--gpus", type=gpu_ids, required=True, help="five distinct IDs, rgb,gt,dav2,tradsl,nsl order")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-num", type=positive, default=1)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tasks", help="optional comma-separated precision subset for acceptance; default all eight")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    if args.seed < 0:
        parser.error("seed must be nonnegative")
    output = args.output.expanduser().resolve() / (time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:10])
    output.mkdir(parents=True, exist_ok=False)
    plan = build_plan(args, output)
    try:
        materialize(plan)
    except (ValueError, OSError, ImportError, KeyError, TypeError) as exc:
        plan["camera_preflight"] = f"FAILED: {exc}"
    (output / "plan.json").write_text(json.dumps(plan, indent=2))
    print(f"{'Execute' if args.execute else 'Dry-run'} plan: {output / 'plan.json'} ({len(plan['invocations'])} invocations)")
    print(f"Camera preflight: {plan['camera_preflight']}")
    if not args.execute:
        return 0
    if plan["camera_preflight"].startswith("FAILED"):
        print("Execute refused: " + plan["camera_preflight"], file=sys.stderr)
        return 1
    try:
        return execute(plan)
    except Exception as exc:
        (output / "summary.json").write_text(json.dumps({"preflight_error": str(exc), "completed": 0}, indent=2))
        print(f"Execute preflight failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
