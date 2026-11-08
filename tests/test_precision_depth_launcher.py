"""Offline launcher tests: never start evaluation, services or GPU workloads."""
import argparse
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import yaml

SPEC = importlib.util.spec_from_file_location("precision_launcher", Path(__file__).resolve().parents[1] / "scripts/run_precision_depth.py")
launcher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(launcher)

TASKS = ["fasten_screws", "plug_in_charger", "insert_tubes", "pour_balls_into_vase", "play_Xylophone", "deposit_coin", "insert_key", "build_tower"]


class LauncherTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.robo = self.root / "RoboDojo"
        self.put("scripts/internal/task_inventory.py", "DIMENSION_TASKS: dict = " + repr({"precision": tuple(TASKS)}) + "\nraise RuntimeError('must not import inventory')\n")
        self.put_yaml("env_cfg/arx_x5.yml", {"config_name": "arx_x5", "config": {"camera": "camera_config", "sim": "sim_config", "robot": "dual_x5"}, "observation": {"vision": {"depth": False}}})
        self.put_yaml("env_cfg/camera/camera_config.yml", {"cam_head": {"camera": {"pos": [0., -.41, 1.308], "ori": [30., 0., 0.], "type": "Gemini_345Lg", "mesh": "pinhole"}}})
        self.put_yaml("env_cfg/robot/dual_x5.yml", {"robots": [{"robot_name": "x5"}, {"robot_name": "x5"}]})
        self.put_yaml("Assets/Robots/x5/robot_config.yml", {"camera": [{"name": "cam_wrist", "link": "camera", "pos": [0., 0., .001], "ori": [0., -80., -90.], "type": "d435", "mesh": "pinhole"}]})
        self.put_yaml("task/RoboDojo/config/_task.yml", {"common": {"robot_config": "dual_x5", "camera_config": "camera_config"}, "tasks": {}})
        self.put_yaml("env_cfg/sim/sim_config.yml", {"scene": {"num_envs": 5}, "render_interval": 10})
        self.args = argparse.Namespace(root=self.root, data_root=self.root / "data", gpus=list("01234"), seed=0, eval_num=1, execute=False)

    def put(self, name, content):
        path = self.robo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)

    def put_yaml(self, name, content):
        self.put(name, yaml.safe_dump(content))

    def plan(self):
        return launcher.build_plan(self.args, self.root / "output/run-unique")

    def test_inventory_is_literal_and_canonical(self):
        self.assertEqual(launcher.tasks_from_inventory(self.root), TASKS)

    def test_exact_five_distinct_gpus_required(self):
        self.assertEqual(launcher.gpu_ids("1,2,3,4,5"), list("12345"))
        for invalid in ("0,1,2,3", "0,1,2,3,3", "0,1,2,3,4,5", "0,1,2,3,-1", "0,1,2,3,"):
            with self.assertRaises(argparse.ArgumentTypeError):
                launcher.gpu_ids(invalid)

    def test_plan_has_forty_serial_native_invocations(self):
        plan = self.plan()
        self.assertEqual(len(plan["invocations"]), 40)
        self.assertEqual([v["name"] for v in plan["lanes"]], list(launcher.LANES))
        self.assertEqual(len({v["run_id"] for v in plan["invocations"]}), 40)
        for lane in plan["lanes"]:
            self.assertEqual([v["task"] for v in lane["invocations"]], TASKS)
            for item in lane["invocations"]:
                argv = item["argv"]
                self.assertEqual(argv[argv.index("--policy-gpu") + 1], lane["gpu"])
                self.assertEqual(argv[argv.index("--env-gpu") + 1], lane["gpu"])
                self.assertNotIn("500", argv)
                self.assertNotIn("PYTHONPATH", item["env"])
        self.assertEqual(plan["lanes"][0]["invocations"][0]["env"]["HEXASHELL_DEPTH"], "0")
        self.assertEqual(plan["lanes"][4]["invocations"][0]["env"]["HEXASHELL_DEPTH_BACKEND"], "nsl")

    def test_acceptance_subset_preserves_native_layout_namespace(self):
        args = argparse.Namespace(root=self.root, data_root=self.root / "data", gpus=["0","1","2","3","4"], seed=1, eval_num=1, execute=False, tasks="insert_tubes")
        plan = launcher.build_plan(args, self.root / "output/acceptance")
        self.assertEqual(len(plan["invocations"]), 5)
        self.assertEqual(plan["tasks"], ["insert_tubes"])
        native = yaml.safe_load((self.robo / "env_cfg/arx_x5.yml").read_text())["config_name"]
        self.assertTrue(all(item["config_name"] == native for item in plan["invocations"]))

    def test_actual_native_wrist_resolution_and_stereo_geometry(self):
        plan = self.plan()
        launcher.materialize(plan)
        for name in ("tradsl", "nsl"):
            camera = yaml.safe_load((self.root / "output/run-unique" / name / "camera.yml").read_text())
            for role in ("cam_head", "cam_left_wrist", "cam_right_wrist"):
                left, right = camera[role]["camera"], camera[role + "_stereo_right"]["camera"]
                delta=[r-l for r,l in zip(right["pos"],left["pos"])]
                self.assertAlmostEqual(sum(v*v for v in delta),.05**2)
                if role=="cam_head":self.assertAlmostEqual(delta[0],.05)
                else:self.assertAlmostEqual(delta[1],-.05)
                self.assertEqual(right["ori"], left["ori"])
                self.assertEqual(right.get("mount_link"), left.get("mount_link"))
            self.assertEqual(camera["cam_left_wrist"]["camera"]["mount_link"], "robot0/camera")
            self.assertEqual(camera["cam_right_wrist"]["camera"]["mount_link"], "robot1/camera")
            self.assertEqual(camera["cam_left_wrist"]["camera"]["ori"], [0., -80., -90.])
        gt = yaml.safe_load((self.root / "output/run-unique/gt/camera.yml").read_text())
        self.assertEqual(gt["annotator"]["common"]["distance_to_image_plane_capture"]["type"], "distance_to_image_plane")
        for lane in plan["lanes"]:
            cfg = yaml.safe_load(Path(lane["config"]).read_text())
            self.assertEqual(cfg["observation"]["vision"]["depth"], lane["name"] == "gt")
            stem = lane["invocations"][0]["argv"][lane["invocations"][0]["argv"].index("--env-cfg") + 1]
            self.assertEqual((self.robo / "env_cfg" / (stem + ".yml")).resolve(), Path(lane["config"]))
            self.assertTrue((self.robo / "env_cfg/sim" / (cfg["config"]["sim"] + ".yml")).is_file())

    def test_dry_run_never_sources_credentials_or_starts_processes(self):
        before = {p.relative_to(self.robo): p.read_bytes() for p in self.robo.rglob("*") if p.is_file()}
        with mock.patch.object(launcher.subprocess, "run", side_effect=AssertionError("process forbidden")), mock.patch.object(launcher.subprocess, "Popen", side_effect=AssertionError("process forbidden")), mock.patch.object(launcher, "sourced_environment", side_effect=AssertionError("credentials forbidden")):
            rc = launcher.main(["--root", str(self.root), "--data-root", str(self.root / "data"), "--gpus", "0,1,2,3,4", "--output", str(self.root / "dry")])
        self.assertEqual(rc, 0)
        plans = list((self.root / "dry").glob("*/plan.json"))
        self.assertEqual(len(plans), 1)
        self.assertEqual(len(json.loads(plans[0].read_text())["invocations"]), 40)
        self.assertEqual(before, {p.relative_to(self.robo): p.read_bytes() for p in self.robo.rglob("*") if p.is_file()})

    def test_unresolvable_camera_still_plans_but_refuses_execute(self):
        self.put_yaml("env_cfg/robot/dual_x5.yml", {"robots": [{"robot_name": "unknown"}]})
        args = ["--root", str(self.root), "--data-root", str(self.root), "--gpus", "0,1,2,3,4", "--output", str(self.root / "bad")]
        with mock.patch.object(launcher, "execute", side_effect=AssertionError("must refuse")):
            self.assertEqual(launcher.main(args), 0)
            self.assertEqual(launcher.main(args + ["--execute"]), 1)
        self.assertTrue(all(len(json.loads(p.read_text())["invocations"]) == 40 for p in (self.root / "bad").glob("*/plan.json")))

    def test_native_task_camera_override_refused(self):
        self.put_yaml("task/RoboDojo/config/_task.yml", {"tasks": {TASKS[0]: {"camera_config": "other"}}})
        with self.assertRaisesRegex(ValueError, TASKS[0]):
            launcher.materialize(self.plan())

    def test_official_native_scores_not_proxy_verdicts(self):
        plan = self.plan()
        item = plan["invocations"][0]
        result = self.robo / "eval_result/RoboDojo" / item["task"] / "HexaShell" / item["config_name"] / "0_ckpt_name=default,action_type=joint" / item["run_id"] / "_result.json"
        result.parent.mkdir(parents=True)
        result.write_text(json.dumps({"details": {"0": {"success": False, "score": .35}}}))
        Path(item["log"]).parent.mkdir(parents=True)
        self.assertEqual(launcher.result_scores(plan, item)["official_score_percent"], 35)

    def test_lane_service_failure_is_visible_for_all_eight_tasks(self):
        plan = self.plan()
        processes = mock.Mock()
        processes.start.side_effect = RuntimeError("specific service failure")
        result = launcher.run_lane(plan, plan["lanes"][2], {}, processes)
        self.assertEqual(len(result), 8)
        self.assertTrue(all(r["status"] == "failed" and "specific service failure" in r["error"] for r in result))

    def test_termination_targets_only_owned_group(self):
        processes = launcher.Processes()
        child = mock.Mock(pid=12345)
        processes.children.add(child)
        with mock.patch.object(launcher.os, "killpg") as kill:
            processes.terminate(child)
        self.assertEqual(kill.call_args_list[0].args, (12345, launcher.signal.SIGTERM))
        self.assertNotIn(child, processes.children)


if __name__ == "__main__":
    unittest.main()
