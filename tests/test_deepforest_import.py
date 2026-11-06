"""
Offline tests for importing DeepForest (lowalt_pilot_v1) outputs and evaluating them beside LLM trials.
Synthetic images and synthetic DeepForest run folders; no inference, no network, no API charges.

    python -m pytest tests -q
"""

import csv
import hashlib
import json
import random
import shutil
import sys
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import benchmark_runner as br  # noqa: E402
import evaluate_consistency as ev  # noqa: E402
import evaluation_core as ec  # noqa: E402
import evaluation_geometry as eg  # noqa: E402
import import_deepforest as idf  # noqa: E402
from model_adapters import MockAdapter  # noqa: E402

W, H = 320, 240


def quiet(*_a, **_k):
    pass


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def tree_hashes(root: Path) -> dict:
    return {p.relative_to(root).as_posix(): sha(p) for p in sorted(root.rglob("*")) if p.is_file()}


DF_BOXES = [  # original-image pixels, floats with full precision, and scores
    (10.123456789012345, 20.5, 60.25, 80.75, 0.912345678901),
    (100.0, 100.0, 150.33333333333334, 140.1, 0.41),
    (200.5, 30.0, 250.0, 90.0, 0.15),
]


def make_df_run(base: Path, image_id: str, src_sha: str, experiment_id: str, code="M01", boxes=DF_BOXES,
                identical=True) -> Path:
    """A folder shaped like urban-tree-vision-baselines outputs/lowalt_pilot_v1/<model>/<exp>/<cfg>/<image>."""
    cfg_id = f"{code}_abcdef1234"
    image_dir = base / "m01_native" / experiment_id / cfg_id / image_id
    primary = image_dir / "run_01_primary"
    primary.mkdir(parents=True)
    (image_dir.parent / "model_config.json").write_text(json.dumps({
        "model_code": code, "inference": {"patch_size": 400, "patch_overlap": 0.05, "iou_threshold": 0.15,
                                          "score_thresh": 0.1, "nms_thresh": 0.05, "dataloader_strategy": "single"}}))
    trees = [{"tree_id": f"D{i:05d}", "bbox_px": list(b[:4]), "bbox_processed_px": list(b[:4]), "score": b[4],
              "label": "Tree"} for i, b in enumerate(boxes, start=1)]
    (primary / "detections.json").write_text(json.dumps({
        "coordinate_system": "original_pixels", "image_id": image_id, "image_size": [W, H], "processed_size": [W, H],
        "model_config_id": cfg_id, "trees": trees}))
    (primary / "predictions_raw.csv").write_text("xmin,ymin,xmax,ymax,score,label\n")
    (primary / "run.log").write_text("log\n")
    Image.new("RGB", (W, H)).save(primary / "annotated_original.jpg")
    record = {"status": "success", "run": "primary", "experiment_id": experiment_id, "image_id": image_id,
              "model_config_id": cfg_id, "model_code": code, "source_sha256": src_sha, "original_size": [W, H],
              "processed_size": [W, H], "model_input_size": [W, H], "raw_detections": len(boxes),
              "predictions_hash": "h1", "preprocessing": None, "padding": None,
              "checkpoint": {"revision": "cc21436bc5d572dde8ff5f93c1e71a32f563cace"},
              "started_utc": "2026-11-05T00:00:00+00:00", "finished_utc": "2026-11-05T00:00:10+00:00"}
    (primary / "run_record.json").write_text(json.dumps(record))
    (image_dir / "determinism.json").write_text(json.dumps({
        "bitwise_identical": identical, "verdict": "identical" if identical else "differs", "counts": [len(boxes)] * 2,
        "predictions_hash": ["h1", "h1" if identical else "h2"]}))
    return image_dir


class DeepForestFixture(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        rows = []
        for i, (image_id, site) in enumerate([("IMG_A", "PT-02"), ("IMG_B", "PT-R1")]):
            path = self.tmp / "imgs" / f"{image_id}.jpg"
            path.parent.mkdir(exist_ok=True)
            Image.new("RGB", (W, H), (40 * i, 120, 60)).save(path, quality=90)
            rows.append({"image_id": image_id, "image_path": f"imgs/{image_id}.jpg", "site": site,
                         "week": "NOT_APPLICABLE", "condition": "ALTITUDE_TEST", "altitude_m": "26.8",
                         "coverage_scope": "UNKNOWN", "collection": "low_altitude", "width": W, "height": H,
                         "sha256": sha(path), "capture_datetime": "", "gimbal_pitch_deg": "-90.00",
                         "gimbal_yaw_deg": "+0.00", "needs_review": "", "notes": ""})
        self.rows = rows
        self.manifest = self.tmp / "manifest.csv"
        with self.manifest.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        self.results = self.tmp / "results"
        br.run_benchmark(br.load_manifest(self.manifest, self.tmp), MockAdapter(), self.results, "EXP_LLM", runs=4,
                         max_usd=100, dry_run=False, manifest_path=self.manifest, ledger_path=self.tmp / "l.jsonl",
                         log=quiet)
        self.baselines = self.tmp / "baselines"
        self.review = self.tmp / "human_review"

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def df_run(self, image_id="IMG_B", exp="EXP_DF", **kw):
        src = next(r["sha256"] for r in self.rows if r["image_id"] == image_id)
        return make_df_run(self.baselines, image_id, kw.pop("src_sha", src), exp, **kw)

    def do_import(self, image_dir):
        return idf.import_run(image_dir, self.results, self.manifest, self.tmp, self.baselines)

    def evaluate(self, out="eval", **kw):
        kw.setdefault("figures", False)
        return ev.run(self.results, None, self.tmp / out, self.review, self.manifest, self.tmp, log=quiet, **kw)

    def read(self, name, out="eval"):
        with (self.tmp / out / name).open(newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))


class TestImport(DeepForestFixture):
    def test_layout_and_float_boxes_and_scores_preserved(self):
        exp = self.do_import(self.df_run())
        self.assertEqual(exp.name, "EXP_DF")
        rec_path = exp / "M01_abcdef1234" / "IMG_B" / "trial_01" / "attempt_01" / "record.json"
        rec = json.loads(rec_path.read_text())
        self.assertEqual(rec["coordinate_system"], "original_pixels")
        self.assertEqual(rec["model_config_id"], "M01_abcdef1234")
        self.assertEqual(rec["trial"], 1)
        self.assertTrue(rec["determinism"]["bitwise_identical"])
        self.assertEqual(rec["llm_experiments_same_image_hash"], ["EXP_LLM"])
        self.assertEqual(rec["api_requests_made"], 0)
        boxes = list(csv.DictReader((exp / "boxes_long.csv").open()))
        self.assertEqual(float(boxes[0]["x_min_px"]), DF_BOXES[0][0])          # exact float round trip
        self.assertEqual(float(boxes[0]["score"]), DF_BOXES[0][4])
        self.assertNotIn("x_min", boxes[0])                                    # no 0-1000 grid columns
        summary = list(csv.DictReader((exp / "counts_summary.csv").open()))[0]
        self.assertEqual((summary["successful_trials"], summary["standard_deviation"]), ("1", ""))

    def test_loader_reads_pixel_boxes_exactly(self):
        self.do_import(self.df_run())
        (u,) = ev.load_experiment(self.results / "EXP_DF", self.tmp, self.tmp)
        self.assertEqual(u.coordinate_system, "original_pixels")
        self.assertTrue(u.single_output)
        self.assertEqual(u.model, "DeepForest M01")
        d = u.sets[0].detections[0]
        self.assertEqual(d.box, tuple(Fraction(str(v)) for v in DF_BOXES[0][:4]))
        self.assertEqual(u.scores[d.uid], DF_BOXES[0][4])
        self.assertEqual(len(u.scores), 3)

    def test_tampered_score_detected(self):
        exp = self.do_import(self.df_run())
        p = exp / "boxes_long.csv"
        p.write_text(p.read_text().replace("0.41", "0.42"))
        with self.assertRaises(ev.InputError):
            ev.load_experiment(exp, self.tmp, self.tmp)

    def test_overwrite_refused(self):
        self.do_import(self.df_run())
        with self.assertRaises(idf.DeepForestImportError):
            self.do_import(self.baselines / "m01_native" / "EXP_DF" / "M01_abcdef1234" / "IMG_B")

    def test_source_hash_mismatch_refused(self):
        with self.assertRaisesRegex(idf.DeepForestImportError, "hash"):
            self.do_import(self.df_run(src_sha="0" * 64))
        self.assertFalse((self.results / "EXP_DF").exists())

    def test_llm_records_with_different_hash_refused(self):
        rec = next(self.results.glob("EXP_LLM/*/IMG_B/trial_01/attempt_01/record.json"))
        r = json.loads(rec.read_text())
        r["image"]["file_sha256"] = "f" * 64
        rec.write_text(json.dumps(r))
        with self.assertRaisesRegex(idf.DeepForestImportError, "LLM experiments"):
            self.do_import(self.df_run())

    def test_missing_determinism_refused(self):
        d = self.df_run()
        (d / "determinism.json").unlink()
        with self.assertRaisesRegex(idf.DeepForestImportError, "determinism"):
            self.do_import(d)

    def test_non_identical_verification_is_recorded_not_hidden(self):
        exp = self.do_import(self.df_run(identical=False))
        rec = json.loads(next(exp.rglob("record.json")).read_text())
        self.assertFalse(rec["determinism"]["bitwise_identical"])

    def test_config_ids_and_experiments_isolated(self):
        a = self.do_import(self.df_run(exp="EXP_DF_M01"))
        b = self.do_import(self.df_run(exp="EXP_DF_M02", code="M02v2"))
        self.assertNotEqual(a, b)
        units = [u for d in (a, b) for u in ev.load_experiment(d, self.tmp, self.tmp)]
        self.assertEqual({u.model_config_id for u in units}, {"M01_abcdef1234", "M02v2_abcdef1234"})
        self.assertEqual({u.model for u in units}, {"DeepForest M01", "DeepForest M02v2"})


class TestEvaluationWithSingleOutput(DeepForestFixture):
    def test_frozen_llm_records_unchanged(self):
        before = tree_hashes(self.results / "EXP_LLM")
        self.do_import(self.df_run())
        self.evaluate()
        self.assertEqual(tree_hashes(self.results / "EXP_LLM"), before)

    def test_single_output_handled_and_compared_with_every_trial(self):
        self.do_import(self.df_run())
        res = self.evaluate(figures=True, full_res_maps=False)
        cross = self.read("cross_model_agreement.csv")
        df_rows = [r for r in cross if "M01_abcdef1234" in (r["model_config_a"], r["model_config_b"])]
        self.assertEqual(len(df_rows), 3)                                    # 3 thresholds
        self.assertTrue(all(r["trial_pairs"] == "4" for r in df_rows))     # 1 output x 4 LLM trials, not faked
        pairs = self.read("pairwise_iou.csv")
        self.assertFalse(any(r["experiment_id"] == "EXP_DF" for r in pairs))
        template = self.read("manual_review_template.csv")
        self.assertFalse(any(r["experiment_id"] == "EXP_DF" for r in template))
        dets = [r for r in self.read("detections.csv") if r["experiment_id"] == "EXP_DF"]
        self.assertEqual(len(dets), 3)
        self.assertTrue(all(r["x_min_norm"] == "" for r in dets))          # never put on the 0-1000 grid
        self.assertEqual(float(dets[0]["score"]), DF_BOXES[0][4])
        self.assertEqual(dets[0]["group_status_iou050"], "NOT_APPLICABLE_SINGLE_OUTPUT")
        geo = [r for r in self.read("geometry_summary.csv") if r["image_id"] == "IMG_B"]
        self.assertEqual(len(geo), 4 + 1)
        cov = self.read("cross_model_coverage.csv")
        self.assertEqual(len([r for r in cov if r["image_id"] == "IMG_B"]), 4)
        report = (self.tmp / "eval" / "evaluation_report.md").read_text(encoding="utf-8")
        self.assertIn("single deterministic output; not applicable", report)
        self.assertIn("Box geometry, coverage and overlap", report)
        names = [p.name for p in res["figures"]["comparison/IMG_B"]]
        self.assertEqual(names, ["K_model_comparison.png", "L_models_side_by_side_iou050.png",
                                 "M_agreement_matrices.png", "N_geometry_coverage.png", "O_coverage_maps.png"])
        self.assertNotIn("EXP_DF/IMG_B", res["figures"])                    # no repeatability figures A-E
        counts = {r["experiment_id"]: r for r in self.read("count_consistency.csv")}
        self.assertEqual(counts["EXP_DF"]["matches_counts_summary_csv"], "True")

    def test_accuracy_against_reference_for_single_output(self):
        self.do_import(self.df_run())
        self.evaluate()
        with (self.review / "reference_boxes.csv").open("a", newline="") as f:
            csv.writer(f).writerows([["H01", "IMG_B", 10, 20, 60, 80, "TREE", "", "", ""],
                                     ["H02", "IMG_B", 280, 180, 310, 230, "TREE", "", "", ""]])
        rows = list(csv.DictReader((self.review / "reference_status.csv").open()))
        for r in rows:
            r["reference_status"] = "COMPLETE" if r["image_id"] == "IMG_B" else "NONE"
        with (self.review / "reference_status.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        res = self.evaluate("eval2", figures=True, full_res_maps=False)
        acc = [r for r in self.read("accuracy_vs_reference.csv", "eval2") if r["experiment_id"] == "EXP_DF"]
        self.assertEqual(len(acc), 3)
        r50 = next(r for r in acc if r["iou_threshold"] == "0.50")
        self.assertEqual((r50["trial"], r50["true_positives"], r50["false_positives"], r50["false_negatives"]),
                         ("1", "1", "2", "1"))
        freq = [r for r in self.read("reference_detection_frequency.csv", "eval2")
                if r["experiment_id"] == "EXP_DF" and r["iou_threshold"] == "0.50"]
        self.assertEqual({r["reference_id"]: r["trials_matched"] for r in freq}, {"H01": "1", "H02": "0"})
        self.assertTrue(any(p.name.startswith("J_") for p in res["figures"]["EXP_DF/IMG_B"]))


class TestMatchingPrefilter(unittest.TestCase):
    @staticmethod
    def brute(A, B, t):
        """The pre-prefilter implementation: exact IoU for every pair."""
        ious = [[ec.iou(a, b) for b in B] for a in A]
        ok = [[v > 0 and v >= t for v in row] for row in ious]
        if not A or not B or not any(any(r) for r in ok):
            return []
        k = min(len(A), len(B)) + 1
        w = np.array([[k + float(v) if o else 0.0 for v, o in zip(r, os_)] for r, os_ in zip(ious, ok)])
        rows, cols = linear_sum_assignment(w, maximize=True)
        return sorted((int(i), int(j), ious[i][j]) for i, j in zip(rows, cols) if ok[i][j])

    def test_identical_to_exhaustive_matching(self):
        rng = random.Random(3)
        for _ in range(60):
            def boxes(n):
                out = []
                for _ in range(n):
                    x, y = rng.uniform(0, 80), rng.uniform(0, 80)
                    out.append(ec.box_from_pixels([x, y, x + rng.uniform(1, 30), y + rng.uniform(1, 30)]))
                return out
            A, B = boxes(rng.randint(0, 25)), boxes(rng.randint(0, 25))
            for t in ec.THRESHOLDS:
                self.assertEqual(ec.match_boxes(A, B, t), self.brute(A, B, t))

    def test_boundary_cases_kept(self):
        a, b = ec.box_from_pixels([0, 0, 30, 10]), ec.box_from_pixels([10, 0, 40, 10])   # IoU exactly 1/2
        self.assertEqual(len(ec.match_boxes([a], [b], Fraction(1, 2))), 1)
        c = ec.box_from_pixels(["0.1", "0.2", "33.3", "10.7"])
        self.assertEqual(ec.match_boxes([c], [c], Fraction(1)), [(0, 0, Fraction(1))])


class TestGeometry(unittest.TestCase):
    def det(self, *boxes):
        return [ec.Detection(f"u{i}", f"T{i}", ec.box_from_pixels(b)) for i, b in enumerate(boxes)]

    def test_coverage_and_overlap(self):
        g = eg.set_geometry(self.det([0, 0, 400, 400], [0, 0, 400, 400]), 800, 800, 0.01)
        self.assertAlmostEqual(g["coverage_fraction"], 0.25)
        self.assertAlmostEqual(g["overlap_ratio_summed_over_union"], 2.0)
        self.assertEqual(g["near_duplicate_pairs_iou030"], 1)
        self.assertEqual(g["boxes_in_near_duplicate_pairs_fraction"], 1.0)
        self.assertAlmostEqual(g["box_side_m_median"], 4.0)
        self.assertAlmostEqual(g["boxes_per_100_m2"], 2 / 64 * 100)

    def test_empty_output(self):
        g = eg.set_geometry([], 100, 100, None)
        self.assertEqual((g["n_boxes"], g["coverage_fraction"], g["overlap_ratio_summed_over_union"]), (0, 0.0, None))

    def test_fragmentation(self):
        fine = self.det([0, 0, 10, 10], [20, 0, 30, 10], [500, 500, 510, 510])
        coarse = self.det([0, 0, 40, 40], [200, 200, 300, 300])
        f = eg.fragmentation(fine, coarse)
        self.assertEqual((f["coarse_boxes_with_2plus_fine"], f["coarse_boxes_with_0_fine"]), (1, 1))
        self.assertAlmostEqual(f["fine_centres_inside_any_coarse_fraction"], 2 / 3)
        self.assertAlmostEqual(f["fine_centres_per_coarse_box_mean"], 1.0)

    def test_mask_iou(self):
        a, b = self.det([0, 0, 80, 80]), self.det([0, 0, 40, 80])
        self.assertAlmostEqual(eg.mask_iou(a, b, 80, 80), 0.5)


if __name__ == "__main__":
    unittest.main()
