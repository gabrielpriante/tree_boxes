"""Offline tests for the lowalt_pilot_v1 DeepForest scripts (no inference; a fake model stands in).

    .venv/bin/python -m unittest discover -s tests -v
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import compare_deepforest_runs_v1 as cmp  # noqa: E402
import resample_to_target_gsd_v2 as rs  # noqa: E402
import run_deepforest_lowalt_v1 as rd  # noqa: E402

PILOT_SHAS = {"LA_PT-R1_27m": "f3ba7ede", "LA_PT-02_64m": "25f3c331"}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class FakeModel:
    """Records predict_tile calls and returns fixed boxes in processed-image pixels."""

    def __init__(self, boxes=None):
        self.boxes = boxes if boxes is not None else [(10.5, 20.25, 60.0, 80.0, 0.9), (100, 100, 150, 140, 0.4)]
        self.calls = []

    def predict_tile(self, path=None, image=None, **kw):
        self.calls.append({"path": path, "image": None if image is None else image.copy(), **kw})
        if not self.boxes:
            return None
        return pd.DataFrame([{"xmin": b[0], "ymin": b[1], "xmax": b[2], "ymax": b[3], "score": b[4], "label": "Tree"}
                             for b in self.boxes])


class Workspace:
    """Temporary manifest + image + fake HF cache, with checkpoint hashes patched to the fake files."""

    def __init__(self, size=(1200, 900), image_id="IMG_A"):
        self.tmp = tempfile.TemporaryDirectory()
        base = Path(self.tmp.name)
        rng = np.random.default_rng(0)
        img = base / "imgs" / "a.jpg"
        img.parent.mkdir()
        Image.fromarray(rng.integers(0, 255, (size[1], size[0], 3), dtype=np.uint8)).save(img, quality=95)
        (base / "manifests").mkdir()
        self.manifest = base / "manifests" / "manifest.csv"
        with open(self.manifest, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["image_id", "image_path", "site", "altitude_m", "sha256"])
            w.writerow([image_id, "imgs/a.jpg", "PT-X", "26.8", sha(img)])
        self.image, self.image_id = img, image_id
        snap = base / "hub" / "models--weecology--deepforest-tree" / "snapshots" / rd.MODEL_REVISION
        snap.mkdir(parents=True)
        (snap / "model.safetensors").write_bytes(b"weights")
        (snap / "config.json").write_bytes(b"{}")
        self.cache = base / "hub"
        self.out = base / "out"
        self.patches = [mock.patch.object(rd, "CHECKPOINT_SHA256", sha(snap / "model.safetensors")),
                        mock.patch.object(rd, "CONFIG_JSON_SHA256", sha(snap / "config.json"))]
        for p in self.patches:
            p.start()

    def run(self, model, code="M01", run="primary", exp="exp_x", **gsd):
        return rd.run(code, self.manifest, self.image_id, exp, run, gsd.get("native"), gsd.get("target"),
                      output_root=self.out, model_loader=lambda: model, cache=self.cache, verify_weights=False)

    def close(self):
        for p in self.patches:
            p.stop()
        self.tmp.cleanup()


# ------------------------------------------------------------------ checkpoint

class TestCheckpoint(unittest.TestCase):
    def test_pinned_constants(self):
        self.assertEqual(rd.MODEL_NAME, "weecology/deepforest-tree")
        self.assertEqual(rd.MODEL_REVISION, "cc21436bc5d572dde8ff5f93c1e71a32f563cace")
        self.assertEqual(rd.CHECKPOINT_SHA256, "d37a7af0b5ba2754282a80d41b4c7b66e8c7149234df5bf3f41bdaf57d329fc8")

    def test_offline_flag_set_on_import(self):
        import os
        self.assertEqual(os.environ.get("HF_HUB_OFFLINE"), "1")

    @unittest.skipUnless((rd.hub_cache() / "models--weecology--deepforest-tree").exists(), "no local HF cache")
    def test_real_cache_verifies(self):
        info = rd.verify_checkpoint()
        self.assertEqual(info["model_safetensors_sha256"], rd.CHECKPOINT_SHA256)
        self.assertEqual(info["revision"], rd.MODEL_REVISION)

    def test_hash_mismatch_refused(self):
        with tempfile.TemporaryDirectory() as t:
            snap = Path(t) / "models--weecology--deepforest-tree" / "snapshots" / rd.MODEL_REVISION
            snap.mkdir(parents=True)
            (snap / "model.safetensors").write_bytes(b"tampered")
            (snap / "config.json").write_bytes(b"{}")
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                rd.verify_checkpoint(Path(t))

    def test_missing_revision_refused(self):
        with tempfile.TemporaryDirectory() as t:
            with self.assertRaises(FileNotFoundError):
                rd.verify_checkpoint(Path(t))


# ------------------------------------------------------------------ M01 fidelity

class TestM01Fidelity(unittest.TestCase):
    def test_constants_equal_historical_scripts(self):
        for script in ("run_deepforest_single.py", "run_deepforest_gsd_7_89.py"):
            src = (ROOT / "scripts" / script).read_text(encoding="utf-8")
            hist = {k: float(v) for k, v in re.findall(r"^(PATCH_SIZE|PATCH_OVERLAP|IOU_THRESHOLD|SCORE_THRESHOLD|NMS_THRESHOLD)\s*=\s*([\d.]+)", src, re.M)}
            self.assertEqual(len(hist), 5, script)
            for k, v in hist.items():
                self.assertEqual(float(getattr(rd, k)), v, f"{script}:{k}")
            self.assertIn('dataloader_strategy="single"', src)
        self.assertEqual(rd.DATALOADER_STRATEGY, "single")

    def test_m01_passes_original_path_and_frozen_kwargs(self):
        ws = Workspace()
        try:
            model = FakeModel()
            ws.run(model)
            (call,) = model.calls
            self.assertEqual(call["path"], str(ws.image))
            self.assertIsNone(call["image"])
            self.assertEqual({k: call[k] for k in ("patch_size", "patch_overlap", "iou_threshold", "dataloader_strategy")},
                             {"patch_size": 400, "patch_overlap": 0.05, "iou_threshold": 0.15, "dataloader_strategy": "single"})
        finally:
            ws.close()

    def test_m01_rejects_gsd_arguments(self):
        with self.assertRaises(ValueError):
            rd.build_config("M01", 0.47, 10.0)

    def test_config_ids_stable_and_distinct(self):
        a, b = rd.build_config("M01"), rd.build_config("M02v2", 0.472, 10.0)
        self.assertEqual(rd.config_id(a), rd.config_id(rd.build_config("M01")))
        self.assertNotEqual(rd.config_id(a), rd.config_id(b))
        self.assertTrue(rd.config_id(a).startswith("M01_") and rd.config_id(b).startswith("M02v2_"))
        self.assertNotEqual(rd.config_id(b), rd.config_id(rd.build_config("M02v2", 1.132, 10.0)))


# ------------------------------------------------------------------ M02 scaling

class TestGsdScaling(unittest.TestCase):
    def test_pilot_dimensions(self):
        p = rs.plan_resample(8192, 6144, 0.472, 10.0)
        self.assertEqual(p["resampled_size"], [387, 290])
        self.assertAlmostEqual(p["resize_factor_exact"], 0.0472)
        q = rs.plan_resample(8192, 6144, 1.132, 10.0)
        self.assertEqual(q["resampled_size"], [927, 696])  # 6144 * 0.1132 = 695.5008 -> 696
        self.assertAlmostEqual(q["applied_scale_y"], 696 / 6144)
        self.assertNotEqual(q["applied_scale_x"], q["applied_scale_y"])

    def test_inverse_roundtrip(self):
        p = rs.plan_resample(8192, 6144, 1.132, 10.0)
        box = [12.3, 45.6, 78.9, 101.1]
        np.testing.assert_allclose(rs.to_resampled(rs.to_original(box, p), p), box, rtol=1e-12)
        self.assertAlmostEqual(rs.to_original([927, 696, 927, 696], p)[0], 8192)
        self.assertAlmostEqual(rs.to_original([927, 696, 927, 696], p)[1], 6144)

    def test_invalid_gsd(self):
        with self.assertRaises(ValueError):
            rs.plan_resample(100, 100, 0, 10)

    def test_resample_writes_record_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as t:
            src = Path(t) / "a.jpg"
            Image.new("RGB", (1000, 750), (40, 120, 40)).save(src, quality=95)
            rec = rs.resample(src, 1.0, 4.0, Path(t) / "out")
            self.assertEqual(rec["resampled_size"], [250, 188])
            with Image.open(rec["output_path"]) as im:
                self.assertEqual(im.size, (250, 188))
            self.assertEqual(rec["source_sha256"], sha(src))
            self.assertEqual(rec["jpeg_quality"], 95)
            with self.assertRaises(FileExistsError):
                rs.resample(src, 1.0, 4.0, Path(t) / "out")


# ------------------------------------------------------------------ small image / padding

class TestPadding(unittest.TestCase):
    def test_pad_bottom_right_only(self):
        arr = np.full((290, 387, 3), 7, np.uint8)
        out, rec = rd.pad_to_patch(arr)
        self.assertEqual(out.shape, (400, 400, 3))
        np.testing.assert_array_equal(out[:290, :387], arr)
        self.assertTrue((out[290:] == 0).all() and (out[:, 387:] == 0).all())
        self.assertEqual((rec["pad_right"], rec["pad_bottom"]), (13, 110))

    def test_no_padding_when_large_enough(self):
        arr = np.zeros((696, 927, 3), np.uint8)
        out, rec = rd.pad_to_patch(arr)
        self.assertIs(out, arr)
        self.assertIsNone(rec)

    def test_partial_padding_only_short_side(self):
        out, rec = rd.pad_to_patch(np.zeros((300, 500, 3), np.uint8))
        self.assertEqual(out.shape, (400, 500, 3))
        self.assertEqual(rec["pad_right"], 0)

    def test_padding_artifacts_removed_and_clipped(self):
        rows = [{"xmin": 10, "ymin": 10, "xmax": 20, "ymax": 20, "score": .5, "label": "Tree"},
                {"xmin": 380, "ymin": 10, "xmax": 395, "ymax": 30, "score": .5, "label": "Tree"},
                {"xmin": 390, "ymin": 300, "xmax": 399, "ymax": 399, "score": .5, "label": "Tree"}]
        kept, dropped, clipped = rd.remove_padding_artifacts(rows, 387, 290)
        self.assertEqual((len(kept), dropped, clipped), (2, 1, 1))
        self.assertEqual(kept[1]["xmax"], 387.0)

    def test_m02_small_image_end_to_end(self):
        ws = Workspace(size=(8192 // 8, 6144 // 8))  # 1024x768 -> factor 0.25 -> 256x192 (< 400)
        try:
            model = FakeModel([(10, 10, 50, 50, 0.8), (250, 100, 300, 150, 0.6), (300, 300, 350, 350, 0.5)])
            d = ws.run(model, code="M02v2", native=2.5, target=10.0)
            (call,) = model.calls
            self.assertIsNone(call["path"])
            self.assertEqual(call["image"].shape, (400, 400, 3))
            self.assertEqual(call["image"].dtype, np.uint8)
            rec = json.loads((d / "run_record.json").read_text())
            self.assertEqual(rec["padding"]["boxes_dropped_in_padding"], 1)
            self.assertEqual(rec["padding"]["boxes_clipped_at_padding"], 1)
            self.assertEqual(rec["model_input_size"], [400, 400])
            trees = json.loads((d / "detections.json").read_text())["trees"]
            self.assertEqual(len(trees), 2)
            np.testing.assert_allclose(trees[0]["bbox_px"], [40, 40, 200, 200])
            self.assertEqual(trees[1]["bbox_processed_px"][2], 256.0)  # clipped at the image edge
            self.assertTrue(trees[1]["clipped_at_padding"])
            # the padded array equals the decoded resampled JPEG in its top-left corner
            with Image.open(rec["preprocessing"]["output_path"]) as im:
                np.testing.assert_array_equal(call["image"][:192, :256], np.array(im.convert("RGB")))
        finally:
            ws.close()


# ------------------------------------------------------------------ coordinates, outputs, overwrite

class TestRunOutputs(unittest.TestCase):
    def test_m02_unpadded_uses_resampled_path_and_restores_coordinates(self):
        ws = Workspace(size=(2000, 1500))  # factor 0.4 -> 800x600, no padding
        try:
            model = FakeModel([(100.0, 60.0, 200.0, 120.0, 0.7)])
            d = ws.run(model, code="M02v2", native=4.0, target=10.0)
            (call,) = model.calls
            self.assertTrue(call["path"].endswith("_gsd10cm.jpg"))
            rec = json.loads((d / "run_record.json").read_text())
            self.assertIsNone(rec["padding"])
            (t,) = json.loads((d / "detections.json").read_text())["trees"]
            np.testing.assert_allclose(t["bbox_px"], [250, 150, 500, 300])
            self.assertEqual(t["bbox_processed_px"], [100.0, 60.0, 200.0, 120.0])
            self.assertTrue((d / "annotated_processed.jpg").exists() and (d / "annotated_original.jpg").exists())
        finally:
            ws.close()

    def test_float_boxes_scores_and_metadata_preserved(self):
        ws = Workspace()
        try:
            d = ws.run(FakeModel())
            det = json.loads((d / "detections.json").read_text())
            self.assertEqual(det["coordinate_system"], "original_pixels")
            self.assertEqual(det["trees"][0]["bbox_px"], [10.5, 20.25, 60.0, 80.0])
            self.assertEqual([t["score"] for t in det["trees"]], [0.9, 0.4])
            self.assertEqual([t["tree_id"] for t in det["trees"]], ["D00001", "D00002"])
            rec = json.loads((d / "run_record.json").read_text())
            self.assertEqual(rec["status"], "success")
            self.assertEqual(rec["source_sha256"], sha(ws.image))
            self.assertEqual(rec["experiment_id"], "exp_x")
            self.assertEqual(rec["model_config_id"], rd.config_id(rd.build_config("M01")))
            self.assertIn("environment", rec)
            with open(d / "predictions_raw.csv") as f:
                raw = list(csv.DictReader(f))
            self.assertEqual(float(raw[0]["score"]), 0.9)
            self.assertTrue((d / "run.log").exists())
            parts = d.relative_to(ws.out).parts
            self.assertEqual(parts[0], "m01_native")
            self.assertEqual(parts[-1], "run_01_primary")
        finally:
            ws.close()

    def test_no_detections(self):
        ws = Workspace()
        try:
            d = ws.run(FakeModel([]))
            self.assertEqual(json.loads((d / "detections.json").read_text())["trees"], [])
        finally:
            ws.close()

    def test_overwrite_refused(self):
        ws = Workspace()
        try:
            ws.run(FakeModel())
            with self.assertRaises(FileExistsError):
                ws.run(FakeModel())
        finally:
            ws.close()

    def test_experiments_isolated(self):
        ws = Workspace()
        try:
            a, b = ws.run(FakeModel(), exp="exp_a"), ws.run(FakeModel(), exp="exp_b")
            self.assertNotEqual(a, b)
            self.assertFalse(str(b).startswith(str(a.parent.parent.parent)))
        finally:
            ws.close()

    def test_source_hash_mismatch_refused(self):
        ws = Workspace()
        try:
            ws.image.write_bytes(ws.image.read_bytes() + b"x")
            with self.assertRaisesRegex(ValueError, "does not match the manifest"):
                ws.run(FakeModel())
        finally:
            ws.close()

    def test_error_is_recorded(self):
        class Boom(FakeModel):
            def predict_tile(self, **kw):
                raise RuntimeError("cuda gone")
        ws = Workspace()
        try:
            with self.assertRaises(RuntimeError):
                ws.run(Boom())
            (d,) = list(ws.out.rglob("run_01_primary"))
            rec = json.loads((d / "run_record.json").read_text())
            self.assertEqual(rec["status"], "error")
            self.assertIn("cuda gone", rec["error"])
            self.assertTrue((d / "error.txt").exists())
        finally:
            ws.close()

    def test_canonical_order_independent_of_model_order(self):
        rows = [{"xmin": 5., "ymin": 1., "xmax": 9., "ymax": 3., "score": .3, "label": "Tree"},
                {"xmin": 1., "ymin": 1., "xmax": 4., "ymax": 3., "score": .8, "label": "Tree"}]
        self.assertEqual(rd.predictions_hash(rows), rd.predictions_hash(rows[::-1]))
        self.assertNotEqual(rd.predictions_hash(rows), rd.predictions_hash([{**rows[0], "score": .30000001}, rows[1]]))


# ------------------------------------------------------------------ determinism

class TestDeterminism(unittest.TestCase):
    def test_identical_runs(self):
        ws = Workspace()
        try:
            a = ws.run(FakeModel())
            ws.run(FakeModel(), run="verification")
            res = cmp.compare(a.parent)
            self.assertTrue(res["bitwise_identical"])
            self.assertTrue(all(res["inputs_identical"].values()))
            self.assertTrue((a.parent / "determinism.json").exists())
            with self.assertRaises(FileExistsError):
                cmp.compare(a.parent)
        finally:
            ws.close()

    def test_differences_quantified(self):
        ws = Workspace()
        try:
            a = ws.run(FakeModel())
            ws.run(FakeModel([(10.5, 20.25, 60.001, 80.0, 0.9), (100, 100, 150, 140, 0.41), (1, 1, 2, 2, .2)]),
                   run="verification")
            res = cmp.compare(a.parent)
            self.assertFalse(res["bitwise_identical"])
            diff = res["difference"]
            self.assertEqual(diff["count_difference"], 1)
            self.assertAlmostEqual(diff["max_coordinate_abs_diff_px"], 0.001, places=6)
            self.assertAlmostEqual(diff["max_score_abs_diff"], 0.01, places=6)
        finally:
            ws.close()


# ------------------------------------------------------------------ real pilot inputs (read-only)

TREE_BOXES = Path("/mnt/c/Users/gabpe/OneDrive/Documents/src/tree_boxes")


@unittest.skipUnless((TREE_BOXES / "manifests" / "manifest.csv").exists(), "tree_boxes repo not mounted")
class TestPilotInputs(unittest.TestCase):
    def test_pilot_images_match_manifest(self):
        for image_id, prefix in PILOT_SHAS.items():
            path, digest, row = rd.resolve_source(TREE_BOXES / "manifests" / "manifest.csv", image_id)
            self.assertTrue(digest.startswith(prefix), image_id)
            with Image.open(path) as im:
                self.assertEqual(im.size, (8192, 6144))


if __name__ == "__main__":
    unittest.main()
