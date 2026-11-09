"""
Offline tests for the DeepForest repeated-execution analysis (evaluate_deepforest_repeatability.py). Synthetic run
folders in the runner's record layout stand in for real runs: no inference, no network, no API charges.

    python -m pytest tests -q
"""

import csv
import hashlib
import json
import random
import shutil
import sys
import tempfile
import time
import unittest
from fractions import Fraction
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import evaluate_consistency as ev  # noqa: E402
import evaluate_deepforest_repeatability as er  # noqa: E402
import evaluation_core as ec  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
PROTECTED = ["benchmark_results", "spend_ledger", "human_review", "manifests"]
RESULT_DIRS = ["evaluation_results", "evaluation_results_lowalt_v1_deepforest"]
W, H = 800, 600
SHA = {"IMG_R1": "a" * 64, "IMG_02": "b" * 64}


def _snapshot():
    out = {}
    for d in PROTECTED + RESULT_DIRS:
        for p in sorted((REPO / d).rglob("*")):
            if p.is_file() and (d in PROTECTED or p.suffix in (".csv", ".json", ".md")):
                out[p.relative_to(REPO).as_posix()] = hashlib.sha256(p.read_bytes()).hexdigest()
    return out


_BEFORE = {}


def setUpModule():
    _BEFORE.update(_snapshot())


def tearDownModule():
    after = _snapshot()
    if after != _BEFORE:
        changed = sorted(k for k in set(after) | set(_BEFORE) if after.get(k) != _BEFORE.get(k))
        raise AssertionError(f"tests modified protected project data: {changed[:10]}")


def quiet(*_a, **_k):
    pass


def boxes(n, seed):
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        x, y = rng.uniform(0, W - 60), rng.uniform(0, H - 60)
        out.append(([x, y, x + rng.uniform(8, 50), y + rng.uniform(8, 50)], rng.uniform(0.1, 0.99)))
    return out


COMBOS = [("M01", "m01_native", "repro_r1_m01", "M01_9c192941f5", "IMG_R1", 40),
          ("M01", "m01_native", "repro_02_m01", "M01_9c192941f5", "IMG_02", 55),
          ("M02v2", "m02v2_gsd", "repro_r1_m02", "M02v2_b25eccbcc7", "IMG_R1", 6),
          ("M02v2", "m02v2_gsd", "repro_02_m02", "M02v2_1e05a4037f", "IMG_02", 12)]


def write_run(d: Path, model, cfg, image_id, dets, status="success", raw_order=None, run="repeat_01"):
    d.mkdir(parents=True)
    trees = [{"tree_id": f"D{i:05d}", "bbox_px": b, "bbox_processed_px": b, "score": s, "label": "Tree"}
             for i, (b, s) in enumerate(sorted(dets, key=lambda x: (x[0], -x[1])), start=1)]
    (d / "detections.json").write_text(json.dumps({"coordinate_system": "original_pixels", "image_id": image_id,
                                                   "image_size": [W, H], "processed_size": [W, H],
                                                   "model_config_id": cfg, "trees": trees}))
    raw = list(reversed(dets)) if raw_order is None else raw_order
    with open(d / "predictions_raw.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["xmin", "ymin", "xmax", "ymax", "score", "label"])
        w.writerows([[repr(v) for v in b] + [repr(s), "Tree"] for b, s in raw])
    (d / "annotated_original.jpg").write_bytes(hashlib.sha256(json.dumps(trees).encode()).digest())
    rec = {"status": status, "run": run, "model_code": model, "model_config_id": cfg, "image_id": image_id,
           "source_sha256": SHA[image_id], "detections": len(trees), "raw_detections": len(dets),
           "predictions_hash": hashlib.sha256(json.dumps(trees).encode()).hexdigest(),
           "raw_predictions_hash": "r", "loaded_weights": {"all_equal": True},
           "checkpoint": {"model_safetensors_sha256": "d37a7af0"}, "inference_seconds": 1.0,
           "preprocessing": {"output_sha256": "pre"} if model == "M02v2" else None,
           "environment": {"deepforest": "2.1.0", "torch": "2.14.0+cu130", "cudnn": 92400, "gpu": "RTX 5070",
                           "cudnn_deterministic": False, "cudnn_benchmark": False},
           "started_utc": "2026-11-09T00:00:00+00:00"}
    (d / "run_record.json").write_text(json.dumps(rec))


class Fixture(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.runs_root = self.tmp / "baselines" / "outputs" / "lowalt_repro_v1"
        self.pilot_root = self.tmp / "baselines" / "outputs" / "lowalt_pilot_v1"
        self.manifest = self.tmp / "manifest.csv"
        with self.manifest.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["image_id", "site", "sha256"])
            w.writerows([["IMG_R1", "PT-R1", SHA["IMG_R1"]], ["IMG_02", "PT-02", SHA["IMG_02"]]])
        self.llm = self.tmp / "llm_eval"
        self.llm.mkdir()
        ev.write_csv(self.llm / "iou_summary.csv",
                     ["experiment_id", "model_config_id", "site", "image_id", "model", "iou_threshold", "n_trials",
                      "n_pairs", "agreement_mean", "agreement_sd"],
                     [{"experiment_id": f"e_{m}", "model_config_id": f"{m}_high_x", "site": "PT-R1", "image_id": "IMG_R1",
                       "model": m, "iou_threshold": t, "n_trials": 10, "n_pairs": 45, "agreement_mean": a,
                       "agreement_sd": 0.1} for m, a0 in (("M08", 0.9), ("M09", 0.8))
                      for t, a in (("0.50", a0), ("0.75", a0 - 0.1), ("1.00", 0.1))])
        ev.write_csv(self.llm / "count_consistency.csv", ["experiment_id", "image_id", "mean_count", "sd_count", "cv_count"],
                     [{"experiment_id": f"e_{m}", "image_id": "IMG_R1", "mean_count": 6.0, "sd_count": 0.5,
                       "cv_count": 0.08} for m in ("M08", "M09")])
        self.dets = {c[2]: boxes(c[5], c[2]) for c in COMBOS}

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def make(self, n_runs=10, mutate=None, pilot=True):
        """mutate(exp, run_index, dets) -> (dets, raw_order or None, status)"""
        for model, mdir, exp, cfg, image_id, _ in COMBOS:
            base = self.runs_root / mdir / exp / cfg / image_id
            for k in range(1, n_runs + 1):
                dets, raw, status = list(self.dets[exp]), None, "success"
                if mutate:
                    dets, raw, status = mutate(exp, k, dets)
                write_run(base / f"run_{k:02d}_repeat", model, cfg, image_id, dets, status, raw, f"repeat_{k:02d}")
            if pilot:
                pbase = self.pilot_root / mdir / f"pilot_{exp}" / cfg / image_id
                for name in ("run_01_primary", "run_02_verification"):
                    write_run(pbase / name, model, cfg, image_id, list(self.dets[exp]), run=name.split("_")[-1])

    def analyse(self, out="out", **kw):
        return er.run(self.runs_root, self.pilot_root, self.llm, self.tmp / out, self.manifest, self.tmp, log=quiet, **kw)

    def read(self, name, out="out"):
        with (self.tmp / out / name).open(newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))


class TestIdenticalRuns(Fixture):
    def test_identical_runs_verified_with_all_checks(self):
        self.make()
        res = self.analyse(figures=True)
        self.assertEqual(len(res["combos"]), 4)
        for c in res["combos"]:
            self.assertEqual(c["n_runs"], 10)
            self.assertTrue(c["all_raw_files_identical"] and c["all_detection_files_identical"]
                            and c["all_canonical_identical"])
            self.assertEqual(c["stats"]["sd"], 0.0)
            self.assertEqual(c["discrepancies"], [])
            self.assertEqual(len(c["pairs"]), 45)
            for t in ec.THRESHOLDS:
                self.assertEqual(c["summaries"][t]["agreement_mean"], 1.0)
                self.assertEqual(c["summaries"][t]["pairs_with_full_agreement"], 45)
            self.assertTrue(all(v["canonical_identical_to_repeat_runs"] for v in c["pilot"].values()))
            self.assertEqual(len(c["pilot"]), 2)
        summary = self.read("repro_summary.csv")
        self.assertEqual({r["identical_to_pilot_runs"] for r in summary}, {"True"})
        self.assertEqual(len(self.read("repro_runs.csv")), 40)
        self.assertEqual(len(self.read("repro_pairwise.csv")), 4 * 45)
        self.assertEqual(len(self.read("llm_within_model_reference.csv")), 6)
        names = sorted(p.name for p in res["figures"])
        self.assertEqual(names, ["P_counts_by_run.png", "Q_pairwise_agreement_iou050.png",
                                 "Q_pairwise_agreement_iou100.png", "R_repeatability_vs_llm.png"])
        self.assertTrue(all(p.exists() for p in res["figures"]))
        report = (self.tmp / "out" / "repro_report.md").read_text(encoding="utf-8")
        self.assertIn("not independent observations", report)
        self.assertIn("independent stochastic trials", report)
        cfg = json.loads((self.tmp / "out" / "evaluation_config.json").read_text())
        self.assertEqual((cfg["inference_run"], cfg["api_requests_made"]), (False, 0))
        self.assertEqual(len(cfg["inputs"]["files"]), 40 * 3)


class TestDiscrepancies(Fixture):
    def test_raw_order_only_difference_is_separated(self):
        def mutate(exp, k, dets):
            return dets, (sorted(dets, key=lambda d: d[1]) if k == 3 else None), "success"
        self.make(mutate=mutate)
        c = next(c for c in self.analyse(figures=False)["combos"] if c["experiment_id"] == "repro_r1_m01")
        self.assertFalse(c["all_raw_files_identical"])
        self.assertTrue(c["all_detection_files_identical"] and c["all_canonical_identical"])
        order_only = [p for p in c["pairs"] if p["order_only_difference"]]
        self.assertEqual(len(order_only), 9)                         # every pair involving run 3
        self.assertIn("distinct predictions_raw.csv hashes", "; ".join(c["discrepancies"]))

    def test_coordinate_score_and_count_differences_reported(self):
        def mutate(exp, k, dets):
            if exp == "repro_02_m01" and k == 5:
                (b, s) = dets[0]
                dets = [([b[0] + 0.25, b[1], b[2], b[3]], s + 0.001)] + dets[1:-1]   # one moved, one dropped
            return dets, None, "success"
        self.make(mutate=mutate)
        c = next(c for c in self.analyse(figures=False)["combos"] if c["experiment_id"] == "repro_02_m01")
        self.assertFalse(c["all_canonical_identical"])
        self.assertEqual(sorted(set(c["counts"])), [54, 55])
        bad = [p for p in c["pairs"] if not p["canonical_detections_identical"]]
        self.assertEqual(len(bad), 9)
        self.assertAlmostEqual(max(p["max_coordinate_abs_diff_px"] for p in bad), 0.25, places=9)
        self.assertAlmostEqual(max(p["max_score_abs_diff"] for p in bad), 0.001, places=9)
        self.assertLess(min(p["agreement_iou100"] for p in bad), 1.0)
        self.assertTrue(any("counts differ" in d for d in c["discrepancies"]))
        self.assertGreater(c["stats"]["sd"], 0)
        others = [c2 for c2 in self.analyse("out2", figures=False)["combos"] if c2["experiment_id"] != "repro_02_m01"]
        self.assertTrue(all(c2["all_canonical_identical"] for c2 in others))

    def test_failed_run_is_excluded_and_listed(self):
        self.make(mutate=lambda exp, k, d: (d, None, "error" if (exp == "repro_r1_m02" and k == 7) else "success"))
        c = next(c for c in self.analyse(figures=False)["combos"] if c["experiment_id"] == "repro_r1_m02")
        self.assertEqual((c["n_runs"], c["excluded"]), (9, [("run_07_repeat", "error")]))
        self.assertEqual(len(c["pairs"]), 36)


class TestInputChecks(Fixture):
    def test_no_runs(self):
        with self.assertRaises(ev.InputError):
            self.analyse()

    def test_source_hash_and_config_checks(self):
        self.make(n_runs=2)
        rec = next(self.runs_root.rglob("run_02_repeat/run_record.json"))
        r = json.loads(rec.read_text())
        r["source_sha256"] = "f" * 64
        rec.write_text(json.dumps(r))
        with self.assertRaisesRegex(ev.InputError, "manifest"):
            self.analyse()

    def test_output_dir_not_created_by_this_tool_refused(self):
        self.make(n_runs=2)
        (self.tmp / "out").mkdir()
        (self.tmp / "out" / "keep.txt").write_text("x")
        with self.assertRaises(ev.InputError):
            self.analyse(figures=False)
        self.assertTrue((self.tmp / "out" / "keep.txt").exists())


class TestComponentMatching(unittest.TestCase):
    def test_component_path_matches_dense_path(self):
        rng = random.Random(11)
        for _ in range(25):
            def bx(n):
                out = []
                for _ in range(n):
                    x, y = rng.uniform(0, 200), rng.uniform(0, 200)
                    out.append(ec.box_from_pixels([x, y, x + rng.uniform(2, 40), y + rng.uniform(2, 40)]))
                return out
            A, B = bx(rng.randint(1, 60)), bx(rng.randint(1, 60))
            for t in ec.THRESHOLDS:
                dense = ec.match_boxes(A, B, t)
                with mock.patch.object(ec, "COMPONENT_MATCHING_MIN_PAIRS", 0):
                    comp = ec.match_boxes(A, B, t)
                self.assertEqual(len(comp), len(dense))
                self.assertEqual(sum((v for *_, v in comp), Fraction(0)), sum((v for *_, v in dense), Fraction(0)))
                self.assertEqual(len({i for i, _, _ in comp}), len(comp))
                self.assertEqual(len({j for _, j, _ in comp}), len(comp))

    def test_existing_evaluations_stay_on_the_dense_path(self):
        self.assertGreater(ec.COMPONENT_MATCHING_MIN_PAIRS, 6320 * 183)   # largest pilot comparison (M01 vs M02v2)

    def test_large_identical_outputs_are_fast(self):
        rng = random.Random(3)
        bxs = []
        for _ in range(2100):
            x, y = rng.uniform(0, 8000), rng.uniform(0, 6000)
            bxs.append(ec.box_from_pixels([x, y, x + rng.uniform(5, 60), y + rng.uniform(5, 60)]))
        t0 = time.perf_counter()
        res = ec.match_boxes(bxs, list(bxs), Fraction(1))
        self.assertEqual(len(res), 2100)
        self.assertLess(time.perf_counter() - t0, 60)


if __name__ == "__main__":
    unittest.main()
