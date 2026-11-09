"""Offline tests for the DeepForest repeated-execution experiment (lowalt_repro_v1). No inference: the runner is
exercised with a fake model, and the driver with a fake process runner.

    .venv/bin/python -m unittest discover -s tests -v
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))

import run_deepforest_lowalt_v1 as rd  # noqa: E402
import run_deepforest_repeatability_v1 as rep  # noqa: E402
from test_lowalt_pilot_v1_deepforest import FakeModel, Workspace  # noqa: E402


def parse(cmd: list) -> dict:
    """Read the runner arguments the driver would pass."""
    ap = argparse.ArgumentParser()
    for a in ("--model", "--manifest", "--image-id", "--experiment-id", "--output-root"):
        ap.add_argument(a)
    ap.add_argument("--run", type=rd.run_label)
    ap.add_argument("--native-gsd-cm", type=float)
    ap.add_argument("--target-gsd-cm", type=float)
    return vars(ap.parse_args(cmd[2:]))


class FakeProcess:
    """Stands in for one runner process: writes a run_record.json with the given status, records the call."""

    def __init__(self, fail_at=None, detections=7):
        self.calls, self.fail_at, self.detections = [], fail_at, detections

    def __call__(self, cmd, cwd):
        a = parse(cmd)
        self.calls.append(a)
        cfg = rd.build_config(a["model"], a["native_gsd_cm"], a["target_gsd_cm"])
        d = rd.run_dir_for(Path(a["output_root"]), cfg, a["experiment_id"], a["image_id"], a["run"])
        d.mkdir(parents=True)
        ok = len(self.calls) != self.fail_at
        (d / "run_record.json").write_text(json.dumps({
            "status": "success" if ok else "error", "detections": self.detections,
            "predictions_hash": "same-hash", "run": a["run"]}))
        return 0 if ok else 1


class TestRepeatLabels(unittest.TestCase):
    def test_labels_and_folders(self):
        cfg = rd.build_config("M01")
        self.assertEqual(rd.run_label("repeat_01"), "repeat_01")
        self.assertEqual(rd.run_label("repeat_10"), "repeat_10")
        for bad in ("repeat_00", "repeat_1", "repeat_100", "run_01", "Primary"):
            with self.assertRaises(argparse.ArgumentTypeError):
                rd.run_label(bad)
        root = Path("/tmp/x")
        self.assertEqual(rd.run_dir_for(root, cfg, "e", "I", "repeat_07").name, "run_07_repeat")
        self.assertEqual(rd.run_dir_for(root, cfg, "e", "I", "primary").name, "run_01_primary")       # unchanged
        self.assertEqual(rd.run_dir_for(root, cfg, "e", "I", "verification").name, "run_02_verification")

    def test_repeat_run_writes_its_own_record(self):
        ws = Workspace()
        try:
            d = ws.run(FakeModel(), run="repeat_03")
            self.assertEqual(d.name, "run_03_repeat")
            rec = json.loads((d / "run_record.json").read_text())
            self.assertEqual((rec["run"], rec["status"]), ("repeat_03", "success"))
            self.assertEqual(rec["model_config_id"], rd.config_id(rd.build_config("M01")))
            with self.assertRaises(FileExistsError):
                ws.run(FakeModel(), run="repeat_03")
        finally:
            ws.close()

    def test_repeat_runs_refused_in_pilot_root(self):
        ws = Workspace()
        try:
            with self.assertRaisesRegex(ValueError, "pilot output root"):
                rd.run("M01", ws.manifest, ws.image_id, "exp", "repeat_01", output_root=rd.OUTPUT_ROOT,
                       model_loader=FakeModel, cache=ws.cache, verify_weights=False)
        finally:
            ws.close()


class TestDriverPlan(unittest.TestCase):
    def test_configurations_equal_the_pilot(self):
        cfgs = rep.check_configurations()
        self.assertEqual([c["config_id"] for c in cfgs],
                         ["M01_9c192941f5", "M01_9c192941f5", "M02v2_b25eccbcc7", "M02v2_1e05a4037f"])
        self.assertEqual([(c["native_gsd_cm"], c["target_gsd_cm"]) for c in cfgs],
                         [(None, None), (None, None), (0.472, 10.0), (1.132, 10.0)])

    def test_forty_interleaved_runs_in_a_dedicated_root(self):
        with tempfile.TemporaryDirectory() as t:
            items = rep.plan(Path(t) / "lowalt_repro_v1")
            self.assertEqual(len(items), 40)
            self.assertEqual([(i["repeat"], i["experiment_id"]) for i in items[:4]],
                             [(1, c[2]) for c in rep.COMBINATIONS])
            self.assertEqual(items[4]["repeat"], 2)
            self.assertEqual(len({i["run_dir"] for i in items}), 40)
            self.assertTrue(all(i["status"] == "pending" for i in items))
            per = {}
            for i in items:
                per.setdefault(i["experiment_id"], []).append(i["run"])
            self.assertEqual({k: len(v) for k, v in per.items()}, {c[2]: 10 for c in rep.COMBINATIONS})
            self.assertTrue(all(not str(i["run_dir"]).startswith(str(rd.OUTPUT_ROOT)) for i in items))
            ids = {c[2] for c in rep.COMBINATIONS}
            self.assertTrue(ids.isdisjoint({c[5] for c in rep.COMBINATIONS}))      # never a pilot experiment id

    def test_commands_reuse_the_runner_with_frozen_arguments(self):
        items = rep.plan(Path("/tmp/r"))
        m02 = next(i for i in items if i["model"] == "M02v2" and i["image_id"] == "LA_PT-02_64m")
        a = parse(rep.command(m02, Path("/m.csv"), Path("/tmp/r")))
        self.assertEqual((a["model"], a["native_gsd_cm"], a["target_gsd_cm"], a["run"]),
                         ("M02v2", 1.132, 10.0, "repeat_01"))
        self.assertTrue(rep.command(m02, Path("/m.csv"), Path("/tmp/r"))[1].endswith("run_deepforest_lowalt_v1.py"))
        m01 = rep.command(items[0], Path("/m.csv"), Path("/tmp/r"))
        self.assertNotIn("--native-gsd-cm", m01)

    def test_driver_refuses_pilot_root(self):
        with self.assertRaises(SystemExit):
            rep.main(["--dry-run", "--output-root", str(rd.OUTPUT_ROOT)])


class TestDriverExecution(unittest.TestCase):
    def test_execute_all_then_nothing_left(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t) / "lowalt_repro_v1"
            fake = FakeProcess()
            res = rep.execute(Path("/m.csv"), root, Path(t), runner=fake, log=lambda *a: None)
            self.assertEqual((res["completed_now"], res["stopped"], len(fake.calls)), (40, None, 40))
            self.assertEqual(len((root / "driver_log.jsonl").read_text().splitlines()), 40)
            design = json.loads((root / "experiment_design.json").read_text())
            self.assertEqual((design["total_runs"], design["runs_per_combination"]), (40, 10))
            again = FakeProcess()
            res = rep.execute(Path("/m.csv"), root, Path(t), runner=again, log=lambda *a: None)
            self.assertEqual((res["completed_now"], len(again.calls)), (0, 0))
            rows = rep.quick_summary(root)
            self.assertTrue(all(r["completed"] == 10 and r["distinct_predictions_hashes"] == 1 for r in rows))

    def test_stops_at_first_failure_and_never_overwrites(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t) / "lowalt_repro_v1"
            res = rep.execute(Path("/m.csv"), root, Path(t), runner=FakeProcess(fail_at=3), log=lambda *a: None)
            self.assertEqual(res["completed_now"], 2)
            self.assertIn("error", res["stopped"])
            with self.assertRaisesRegex(SystemExit, "Inspect"):
                rep.execute(Path("/m.csv"), root, Path(t), runner=FakeProcess(), log=lambda *a: None)

    def test_max_runs_and_dry_run_write_nothing_extra(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t) / "lowalt_repro_v1"
            rep.main(["--dry-run", "--output-root", str(root)])
            self.assertFalse(root.exists())
            fake = FakeProcess()
            rep.execute(Path("/m.csv"), root, Path(t), runner=fake, max_runs=4, log=lambda *a: None)
            self.assertEqual(len(fake.calls), 4)
            self.assertEqual({c["run"] for c in fake.calls}, {"repeat_01"})

    def test_prepare_refuses_a_different_design(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t) / "lowalt_repro_v1"
            p = rep.prepare(Path("/m.csv"), root)
            self.assertEqual(rep.prepare(Path("/m.csv"), root), p)              # same design: accepted
            with self.assertRaises(SystemExit):
                rep.prepare(Path("/other.csv"), root)


if __name__ == "__main__":
    unittest.main()
