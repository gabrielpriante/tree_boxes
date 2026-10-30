"""
Offline tests for the evaluation pipeline: synthetic boxes with known answers, plus end-to-end runs
on mock benchmark output. No network access and no API charges.

    python -m pytest tests -q
"""

import csv
import itertools
import json
import random
import shutil
import statistics
import sys
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import benchmark_runner as br  # noqa: E402
import evaluate_consistency as ev  # noqa: E402
import evaluation_core as ec  # noqa: E402
from model_adapters import MockAdapter  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
HALF, THREE_Q, ONE = Fraction(1, 2), Fraction(3, 4), Fraction(1)


def quiet(*_a, **_k):
    pass


def strip(x0, x1, y0=0, y1=10):
    """Boxes of equal height: IoU reduces to interval overlap / union."""
    return ec.box_from_pixels([x0, y0, x1, y1])


def dset(trial, boxes, image="IMG", sha="s", w=1000, h=1000):
    tag = "ref" if trial is None else f"t{trial:02d}"
    dets = tuple(ec.Detection(f"{image}:{tag}:T{i + 1:03d}", f"T{i + 1:03d}", b) for i, b in enumerate(boxes))
    return ec.DetectionSet("test", image, sha, w, h, trial, dets)


class TestIoU(unittest.TestCase):
    def test_zero_when_disjoint_or_touching(self):
        self.assertEqual(ec.iou(strip(0, 10), strip(20, 30)), 0)
        self.assertEqual(ec.iou(strip(0, 10), strip(10, 20)), 0)  # shared edge only

    def test_partial_overlap(self):
        self.assertEqual(ec.iou(strip(0, 10), strip(5, 15)), Fraction(1, 3))

    def test_exactly_half(self):
        a, b = strip(0, 30), strip(10, 40)
        self.assertEqual(ec.iou(a, b), HALF)
        self.assertEqual(len(ec.match_boxes([a], [b], HALF)), 1)   # threshold is inclusive
        self.assertEqual(ec.match_boxes([a], [b], THREE_Q), [])

    def test_exactly_three_quarters(self):
        a, b = strip(0, 100), strip(0, 75)
        self.assertEqual(ec.iou(a, b), THREE_Q)
        self.assertEqual(len(ec.match_boxes([a], [b], THREE_Q)), 1)
        self.assertEqual(ec.match_boxes([a], [b], ONE), [])

    def test_one_requires_identical_geometry(self):
        a = ec.box_from_normalized([100, 100, 200, 200], 8192, 6144)
        same = ec.box_from_normalized([100, 100, 200, 200], 8192, 6144)
        off = ec.box_from_normalized([100, 100, 200, 201], 8192, 6144)
        self.assertEqual(ec.iou(a, same), 1)
        self.assertGreater(ec.iou(a, off), Fraction(98, 100))
        self.assertEqual(len(ec.match_boxes([a], [same], ONE)), 1)
        self.assertEqual(ec.match_boxes([a], [off], ONE), [])

    def test_exact_arithmetic_on_real_image_grid(self):
        # 0-1000 -> 8192x6144 pixels gives non-integer pixel coordinates; IoU stays exact.
        a = ec.box_from_normalized([0, 0, 400, 300], 8192, 6144)
        b = ec.box_from_normalized([0, 0, 300, 300], 8192, 6144)
        self.assertIsInstance(ec.iou(a, b), Fraction)
        self.assertEqual(ec.iou(a, b), THREE_Q)

    def test_normalized_conversion_rejects_non_integers_and_degenerate_boxes(self):
        with self.assertRaises(ValueError):
            ec.box_from_normalized([1.5, 0, 10, 10], 100, 100)
        with self.assertRaises(ValueError):
            ec.box_from_normalized([10, 0, 10, 10], 100, 100)


class TestMatching(unittest.TestCase):
    def test_maximizes_number_of_matches_not_greedy(self):
        a1, a2 = strip(1, 11), strip(-3, 7)
        b1, b2 = strip(0, 10), strip(4, 14)
        # Greedy best-IoU-first would take a1-b1 (0.82) and leave a2 unmatched.
        self.assertGreater(ec.iou(a1, b1), ec.iou(a1, b2))
        pairs = ec.match_boxes([a1, a2], [b1, b2], HALF)
        self.assertEqual([(i, j) for i, j, _ in pairs], [(0, 1), (1, 0)])

    def test_ties_in_count_broken_by_total_iou(self):
        a = strip(0, 10)
        pairs = ec.match_boxes([a], [strip(3, 13), strip(1, 11)], HALF)
        self.assertEqual([(i, j) for i, j, _ in pairs], [(0, 1)])

    def test_one_to_one_and_optimal_against_brute_force(self):
        rng = random.Random(7)
        for _ in range(150):
            def boxes(n):
                out = []
                for _ in range(n):
                    x, y = rng.randint(0, 30), rng.randint(0, 30)
                    out.append(ec.box_from_pixels([x, y, x + rng.randint(4, 14), y + rng.randint(4, 14)]))
                return out
            A, B = boxes(rng.randint(0, 5)), boxes(rng.randint(0, 5))
            for t in (HALF, THREE_Q):
                got = ec.match_boxes(A, B, t)
                self.assertEqual(len({i for i, _, _ in got}), len(got))
                self.assertEqual(len({j for _, j, _ in got}), len(got))
                best = (0, Fraction(0))
                for k in range(min(len(A), len(B)), 0, -1):
                    for ia in itertools.combinations(range(len(A)), k):
                        for jb in itertools.permutations(range(len(B)), k):
                            vals = [ec.iou(A[i], B[j]) for i, j in zip(ia, jb)]
                            if all(v > 0 and v >= t for v in vals):
                                best = max(best, (k, sum(vals)))
                    if best[0]:
                        break
                self.assertEqual(len(got), best[0])
                self.assertEqual(sum((v for _, _, v in got), Fraction(0)), best[1])

    def test_duplicate_detections_match_once(self):
        r = ec.compare_sets(dset(1, [strip(0, 10), strip(0, 10)]), dset(2, [strip(0, 10)]), HALF)
        self.assertEqual((r["matched"], r["unmatched_a"], r["unmatched_b"]), (1, 1, 0))
        self.assertEqual(r["agreement"], Fraction(2, 3))

    def test_unmatched_detections_counted(self):
        r = ec.compare_sets(dset(1, [strip(0, 10), strip(50, 60)]), dset(2, [strip(0, 10), strip(100, 110), strip(200, 210)]), HALF)
        self.assertEqual((r["matched"], r["unmatched_a"], r["unmatched_b"]), (1, 1, 2))
        self.assertEqual(r["agreement"], Fraction(2, 5))
        self.assertEqual(r["mean_iou"], 1.0)

    def test_empty_sets(self):
        both = ec.compare_sets(dset(1, []), dset(2, []), HALF)
        self.assertIsNone(both["agreement"])
        self.assertEqual(both["matched"], 0)
        one = ec.compare_sets(dset(1, []), dset(2, [strip(0, 10)] * 3), HALF)
        self.assertEqual((one["agreement"], one["unmatched_b"]), (0, 3))
        self.assertIsNone(one["mean_iou"])
        s = ec.summarize_pairs([both, one])
        self.assertEqual((s["n_pairs"], s["n_pairs_undefined"], s["agreement_mean"]), (2, 1, 0.0))

    def test_pairwise_count_is_n_choose_two(self):
        sets = [dset(t, [strip(0, 10)]) for t in range(1, 11)]
        self.assertEqual(len(ec.pairwise_trials(sets, HALF)), 45)


class TestPersistence(unittest.TestCase):
    def groups(self, sets, t=HALF):
        return ec.persistence_groups(sets, ec.pairwise_trials(sets, t))

    def test_object_found_in_every_trial(self):
        sets = [dset(t, [strip(0, 10), strip(100, 110)] if t == 1 else [strip(0, 10)]) for t in range(1, 4)]
        gs = self.groups(sets)
        self.assertEqual([(g.group_id, g.status, g.n_trials) for g in gs],
                         [("G01", "CONSISTENT", 3), ("G02", "CONSISTENT", 1)])
        self.assertEqual(ec.persistence_bin(3, 3), "all 3")

    def test_bins_for_ten_trials(self):
        self.assertEqual([b[0] for b in ec.persistence_bins(10)], ["all 10", "8-9", "5-7", "1-4"])
        self.assertEqual([ec.persistence_bin(n, 10) for n in (10, 9, 8, 7, 5, 4, 1)],
                         ["all 10", "8-9", "8-9", "5-7", "5-7", "1-4", "1-4"])

    def test_chain_is_not_merged_transitively(self):
        sets = [dset(1, [strip(0, 10)]), dset(2, [strip(3, 13)]), dset(3, [strip(6, 16)])]
        self.assertGreaterEqual(ec.iou(strip(0, 10), strip(3, 13)), HALF)
        self.assertLess(ec.iou(strip(0, 10), strip(6, 16)), HALF)
        gs = self.groups(sets)
        self.assertEqual(len(gs), 1)
        self.assertEqual(gs[0].status, "AMBIGUOUS")
        self.assertEqual(len(gs[0].missing_links), 1)
        self.assertEqual(gs[0].group_id, "A01")

    def test_two_boxes_from_one_trial_flagged(self):
        sets = [dset(1, [strip(0, 10), strip(500, 510)]), dset(2, [strip(0, 10)]), dset(3, [strip(0, 10)])]
        uid = {(s.trial, i): d.uid for s in sets for i, d in enumerate(s.detections)}
        # Hand-built matches: t1 box 0 ~ t2, t2 ~ t3, t3 ~ t1 box 1 (as a mismatched assignment could produce)
        pairs = [{"pairs": [(uid[(1, 0)], uid[(2, 0)], ONE)]}, {"pairs": [(uid[(2, 0)], uid[(3, 0)], ONE)]},
                 {"pairs": [(uid[(1, 1)], uid[(3, 0)], ONE)]}]
        gs = ec.persistence_groups(sets, pairs)
        self.assertEqual(len(gs), 1)
        self.assertEqual(gs[0].status, "AMBIGUOUS")
        self.assertEqual(gs[0].duplicate_trials, [1])

    def test_object_never_counted_twice_in_a_trial(self):
        sets = [dset(t, [strip(0, 10), strip(1, 11)]) for t in range(1, 5)]
        for g in self.groups(sets):
            if g.status == "CONSISTENT":
                self.assertEqual(len(g.members), g.n_trials)

    def test_match_frequency(self):
        sets = [dset(1, [strip(0, 10), strip(100, 110)]), dset(2, [strip(0, 10)]), dset(3, [strip(0, 10)])]
        freq = ec.detection_match_frequency(sets, ec.pairwise_trials(sets, HALF))
        self.assertEqual(freq["IMG:t01:T001"], 2)
        self.assertEqual(freq["IMG:t01:T002"], 0)


class TestSeparationOfMetrics(unittest.TestCase):
    def test_cross_image_comparison_refused(self):
        a, b = dset(1, [strip(0, 10)], image="LA_PT-R1_27m", sha="x"), dset(1, [strip(0, 10)], image="LA_PT-R1_28m", sha="y")
        with self.assertRaises(ec.CrossImageComparisonError):
            ec.compare_sets(a, b, HALF)
        with self.assertRaises(NotImplementedError):
            ec.compare_sets(a, b, HALF, alignment="homography")
        with self.assertRaises(ec.CrossImageComparisonError):
            ec.pairwise_trials([a, dset(2, [strip(0, 10)], image="LA_PT-R1_28m", sha="y")], HALF)
        with self.assertRaises(ec.CrossImageComparisonError):
            ec.compare_sets(a, dset(2, [strip(0, 10)], image="LA_PT-R1_27m", sha="x", w=500), HALF)

    def test_accuracy_requires_complete_reference(self):
        pred, ref = dset(1, [strip(0, 10), strip(50, 60)]), dset(None, [strip(0, 10), strip(100, 110)])
        for status in ("NONE", "PARTIAL", ""):
            with self.assertRaises(ec.ReferenceNotAvailableError):
                ec.accuracy_vs_reference(pred, ref, status, HALF)
        a = ec.accuracy_vs_reference(pred, ref, "COMPLETE", HALF)
        self.assertEqual((a["true_positives"], a["false_positives"], a["false_negatives"]), (1, 1, 1))
        self.assertEqual((a["precision"], a["recall"], a["f1"]), (0.5, 0.5, 0.5))


# ------------------------------------------------------------------ end to end on mock benchmark output

class MockBenchmark(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        rows = []
        for i, (image_id, site) in enumerate([("IMG_A", "PT-02"), ("IMG_B", "PT-R1")]):
            path = self.tmp / "imgs" / f"{image_id}.jpg"
            path.parent.mkdir(exist_ok=True)
            Image.new("RGB", (320, 240), (40 * i, 120, 60)).save(path, quality=90)
            rows.append({"image_id": image_id, "image_path": f"imgs/{image_id}.jpg", "site": site, "week": "WEEK 3",
                         "condition": "BEFORE", "altitude_m": "95.0", "coverage_scope": "UNKNOWN", "collection": "standard"})
        self.manifest = self.tmp / "manifest.csv"
        with self.manifest.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        self.results = self.tmp / "results"
        br.run_benchmark(br.load_manifest(self.manifest, self.tmp), MockAdapter(), self.results, "EXP", runs=4,
                         max_usd=100, dry_run=False, manifest_path=self.manifest, ledger_path=self.tmp / "l.jsonl", log=quiet)
        self.review = self.tmp / "human_review"

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def evaluate(self, out="eval", **kw):
        kw.setdefault("figures", False)
        return ev.run(self.results, None, self.tmp / out, self.review, self.manifest, self.tmp, log=quiet, **kw)

    def read(self, name, out="eval"):
        with (self.tmp / out / name).open(newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))


class EvaluationRun(MockBenchmark):
    def test_outputs_and_pair_counts(self):
        res = self.evaluate()
        for name in ("pairwise_iou.csv", "iou_summary.csv", "detection_persistence.csv", "manual_review_template.csv",
                     "count_consistency.csv", "evaluation_report.md", "evaluation_config.json", "detections.csv"):
            self.assertTrue((self.tmp / "eval" / name).exists(), name)
        pairs = self.read("pairwise_iou.csv")
        self.assertEqual(len(pairs), 2 * 3 * 6)  # 2 images x 3 thresholds x C(4,2)
        self.assertEqual({p["iou_threshold"] for p in pairs}, {"0.50", "0.75", "1.00"})
        self.assertTrue(all(r["matches_counts_summary_csv"] == "True" for r in self.read("count_consistency.csv")))
        cfg = json.loads((self.tmp / "eval" / "evaluation_config.json").read_text())
        self.assertEqual(cfg["iou_thresholds"], ["1/2", "3/4", "1"])
        self.assertEqual(cfg["api_requests_made"], 0)
        self.assertTrue(cfg["inputs"] and all(len(i["sha256"]) == 64 for i in cfg["inputs"]))
        self.assertEqual(len(res["units"]), 2)

    def test_reproducible_outputs(self):
        self.evaluate("e1")
        self.evaluate("e2")
        files = sorted(p.relative_to(self.tmp / "e1") for p in (self.tmp / "e1").rglob("*") if p.is_file())
        self.assertTrue(files)
        for f in files:
            if f.name == "evaluation_config.json":
                a = json.loads((self.tmp / "e1" / f).read_text())
                b = json.loads((self.tmp / "e2" / f).read_text())
                for k in ("generated_utc", "command"):
                    a.pop(k), b.pop(k)
                self.assertEqual(a, b)
            else:
                self.assertEqual((self.tmp / "e1" / f).read_bytes(), (self.tmp / "e2" / f).read_bytes(), f)

    def test_no_accuracy_without_complete_reference(self):
        self.evaluate()
        self.assertFalse((self.tmp / "eval" / "accuracy_vs_reference.csv").exists())
        report = (self.tmp / "eval" / "evaluation_report.md").read_text(encoding="utf-8")
        self.assertIn("not computed", report)
        for name in ("pairwise_iou.csv", "iou_summary.csv", "detection_persistence.csv", "count_consistency.csv"):
            header = (self.tmp / "eval" / name).read_text(encoding="utf-8").splitlines()[0].lower()
            for word in ("precision", "recall", "f1", "accuracy"):
                self.assertNotIn(word, header, name)

    def test_partial_reference_blocks_and_complete_reference_enables_accuracy(self):
        self.evaluate()  # creates human_review files
        det = [r for r in self.read("detections.csv") if r["image_id"] == "IMG_A" and r["trial"] == "1"]
        with (self.review / "reference_boxes.csv").open("a", newline="") as f:
            w = csv.writer(f)
            for i, r in enumerate(det):
                w.writerow([f"R{i}", "IMG_A", r["x_min_px"], r["y_min_px"], r["x_max_px"], r["y_max_px"], "TREE", "", "", ""])
        status = self.read_status()
        self.write_status(status, "PARTIAL")
        self.evaluate()
        for name in ("accuracy_vs_reference.csv", "accuracy_summary.csv", "accuracy_report.md", "prediction_outcomes.csv"):
            self.assertFalse((self.tmp / "eval" / name).exists(), name)
        self.write_status(status, "COMPLETE")
        res = self.evaluate(figures=True)
        for name in ("accuracy_summary.csv", "reference_detection_frequency.csv", "prediction_outcomes.csv",
                     "accuracy_report.md"):
            self.assertTrue((self.tmp / "eval" / name).exists(), name)
        names = {p.name for ps in res["figures"].values() for p in ps}
        for prefix in ("F_", "G_", "H_", "I_reference_vs_predictions_overview", "I_reference_vs_predictions_iou050", "J_"):
            self.assertTrue(any(n.startswith(prefix) for n in names), prefix)
        self.assertTrue((self.tmp / "eval" / "review" / "human_reference_IMG_A.jpg").exists())
        freq = [r for r in self.read("reference_detection_frequency.csv") if r["iou_threshold"] == "0.50"]
        self.assertEqual(len(freq), len(det))
        acc = [r for r in self.read("accuracy_vs_reference.csv") if r["trial"] == "1"]
        self.assertEqual(len(acc), 3)
        for r in acc:
            self.assertEqual((r["precision"], r["recall"], r["false_positives"], r["false_negatives"]), ("1", "1", "0", "0"))
        self.assertEqual({r["image_id"] for r in self.read("accuracy_vs_reference.csv")}, {"IMG_A"})

    def read_status(self):
        with (self.review / "reference_status.csv").open(newline="") as f:
            return list(csv.DictReader(f))

    def write_status(self, rows, status):
        for r in rows:
            r["reference_status"] = status if r["image_id"] == "IMG_A" else "NONE"
        with (self.review / "reference_status.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)

    def test_human_labels_preserved_and_summarized(self):
        self.evaluate()
        path = self.review / "manual_review.csv"
        with path.open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        rows[0]["human_label"], rows[0]["human_note"] = "NON_TREE", "a shrub"
        rows[1]["human_label"] = "bogus"
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        before = path.read_bytes()
        res = self.evaluate()
        self.assertEqual(path.read_bytes(), before)
        self.assertEqual(res["review"]["labels"][rows[0]["review_unit_id"]]["label"], "NON_TREE")
        self.assertEqual(len(res["review"]["label_errors"]), 1)
        report = (self.tmp / "eval" / "evaluation_report.md").read_text(encoding="utf-8")
        self.assertIn("| NON_TREE | 1 |", report)
        self.assertIn("Input problem", report)

    def test_tampered_detections_rejected(self):
        det = next(self.results.rglob("detections.json"))
        data = json.loads(det.read_text())
        data["trees"][0]["bbox"][2] += 1
        det.write_text(json.dumps(data))
        with self.assertRaises(ev.InputError):
            self.evaluate()

    def test_never_writes_inside_benchmark_results(self):
        with self.assertRaises(ev.InputError):
            ev.run(self.results, None, self.results / "eval", self.review, self.manifest, self.tmp, figures=False, log=quiet)
        before = sorted(p for p in self.results.rglob("*"))
        self.evaluate()
        self.assertEqual(before, sorted(p for p in self.results.rglob("*")))

    def test_refuses_to_clear_foreign_directory(self):
        (self.tmp / "eval").mkdir()
        (self.tmp / "eval" / "mine.txt").write_text("x")
        with self.assertRaises(ev.InputError):
            self.evaluate()

    def test_figures_and_review_images(self):
        res = self.evaluate(figures=True)
        paths = [p for ps in res["figures"].values() for p in ps]
        names = {p.name for p in paths}
        for prefix in ("A_", "B_persistence_overview", "B_persistence_map_iou050", "B_persistence_map_iou075",
                       "B_persistence_map_iou100", "C_", "D_", "E_"):
            self.assertTrue(any(n.startswith(prefix) for n in names), prefix)
        self.assertTrue(all(p.exists() for p in paths))
        template = self.read("manual_review_template.csv")
        self.assertTrue(all((self.tmp / "eval" / r["crop_path"]).exists() for r in template))
        self.assertEqual(len(list((self.tmp / "eval" / "review").glob("contact_sheet_*.png"))), 2)


class TestAccuracyAgainstReference(unittest.TestCase):
    """Synthetic trials vs a complete reference, with hand-computed answers."""

    def setUp(self):
        import evaluation_accuracy as ea
        from types import SimpleNamespace
        self.ea = ea
        A, B, C = strip(0, 10), strip(100, 110), strip(200, 210)
        self.ref = ec.DetectionSet("human_reference", "IMG", "s", 1000, 1000, None,
                                   tuple(ec.Detection(f"REF:IMG:H0{i}", f"H0{i}", b) for i, b in enumerate((A, B, C), 1)))
        t1 = dset(1, [A, strip(102, 112), strip(500, 510)])   # exact A; B at IoU 2/3; box on nothing
        t2 = dset(2, [A, strip(1, 11)])                       # A, plus a second box on A (IoU 9/11)
        self.u = SimpleNamespace(experiment_id="EXP", image_id="IMG", sets=[t1, t2])
        self.res = ea.evaluate_accuracy(self.u, self.ref, "COMPLETE", ec.THRESHOLDS)

    def trial(self, t, k):
        return self.res.per_trial[t][k]

    def outcome(self, t, uid):
        lab = ec.threshold_label(t)
        return next(r for r in self.res.outcome_rows if r["iou_threshold"] == lab and r["detection_uid"] == uid)

    def test_counts_per_trial_and_threshold(self):
        got = {(ec.threshold_label(t), k): tuple(self.trial(t, k)[m] for m in ("true_positives", "false_positives", "false_negatives"))
               for t in ec.THRESHOLDS for k in (1, 2)}
        self.assertEqual(got, {("0.50", 1): (2, 1, 1), ("0.50", 2): (1, 1, 2),
                               ("0.75", 1): (1, 2, 2), ("0.75", 2): (1, 1, 2),
                               ("1.00", 1): (1, 2, 2), ("1.00", 2): (1, 1, 2)})
        a = self.trial(HALF, 1)
        self.assertAlmostEqual(a["precision"], 2 / 3)
        self.assertAlmostEqual(a["recall"], 2 / 3)
        self.assertAlmostEqual(a["f1"], 2 / 3)
        self.assertAlmostEqual(a["mean_iou_tp"], (1 + 2 / 3) / 2)

    def test_false_positive_types(self):
        self.assertEqual(self.outcome(HALF, "IMG:t01:T003")["fp_type"], "no_reference_overlap")
        self.assertEqual(self.outcome(THREE_Q, "IMG:t01:T002")["fp_type"], "localization_below_threshold")
        self.assertEqual(self.outcome(HALF, "IMG:t02:T002")["fp_type"], "reference_matched_by_other_detection")
        tp = self.outcome(HALF, "IMG:t01:T002")
        self.assertEqual((tp["outcome"], tp["matched_reference"]), ("TP", "H02"))

    def test_iou_one_needs_identical_box(self):
        self.assertEqual(self.outcome(ONE, "IMG:t01:T001")["outcome"], "TP")
        self.assertEqual(self.outcome(ONE, "IMG:t01:T002")["outcome"], "FP")

    def test_reference_detection_frequency(self):
        freq = {(r["iou_threshold"], r["reference_id"]): r["trials_matched"] for r in self.res.reference_rows}
        self.assertEqual([freq[("0.50", h)] for h in ("H01", "H02", "H03")], [2, 1, 0])
        self.assertEqual([freq[("0.75", h)] for h in ("H01", "H02", "H03")], [2, 0, 0])

    def test_summary_mean_sd_and_pooled(self):
        s = self.ea.summary_value(self.res, HALF, "precision")
        self.assertAlmostEqual(s["mean"], (2 / 3 + 1 / 2) / 2)
        self.assertAlmostEqual(s["sd"], statistics.stdev([2 / 3, 1 / 2]))
        self.assertEqual((s["min"], s["max"]), (0.5, 2 / 3))
        self.assertAlmostEqual(s["pooled_over_trials"], 3 / 5)
        self.assertAlmostEqual(self.ea.summary_value(self.res, HALF, "recall")["pooled_over_trials"], 3 / 6)

    def test_containment_measured(self):
        # exact A twice (fully inside), shifted B: 8 of its 10 units inside H02
        self.assertEqual(sorted(self.res.containment["0.50"]), [0.8, 1.0, 1.0])

    def test_refuses_incomplete_reference(self):
        with self.assertRaises(ec.ReferenceNotAvailableError):
            self.ea.evaluate_accuracy(self.u, self.ref, "PARTIAL", ec.THRESHOLDS)


class ReviewValidation(MockBenchmark):
    """manual_review.csv is hand-edited (often in Excel); bad edits must stop the evaluation, not be rewritten."""

    def edit_review(self, change):
        path = self.review / "manual_review.csv"
        with path.open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        change(rows)
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        return path.read_bytes()

    def assert_rejected(self, change, text):
        self.evaluate()
        report_before = (self.tmp / "eval" / "evaluation_report.md").read_bytes()
        content = self.edit_review(change)
        with self.assertRaises(ev.InputError) as cm:
            self.evaluate()
        self.assertIn(text, str(cm.exception))
        self.assertEqual((self.review / "manual_review.csv").read_bytes(), content)        # not rewritten
        self.assertEqual((self.tmp / "eval" / "evaluation_report.md").read_bytes(), report_before)  # outputs kept

    def test_spreadsheet_date_conversion_detected(self):
        def to_date(rows):
            r = next(r for r in rows if r["persistence_bin"] == "2-3")
            r["persistence_bin"] = "3-Feb"
        self.assert_rejected(to_date, "converted it to a date")

    def test_category_must_agree_with_n_trials(self):
        def wrong(rows):
            r = next(r for r in rows if r["n_trials"] == "1")
            r["persistence_bin"] = "all 4"
        self.assert_rejected(wrong, "should be '1-1' for n_trials=1")

    def test_trials_detected_must_agree_with_count(self):
        def wrong(rows):
            r = next(r for r in rows if r["n_trials"] == "1")
            r["trials_detected"] = r["trials_detected"] + ";4" if r["trials_detected"] != "4" else "3;4"
        self.assert_rejected(wrong, "trials_detected lists 2 distinct trials but n_trials is 1")

    def test_generated_columns_must_not_change(self):
        def wrong(rows):
            rows[0]["crop_path"] = "elsewhere.jpg"
        self.assert_rejected(wrong, "generated column(s) differ from the evaluation: crop_path")

    def test_human_fields_may_change(self):
        self.evaluate()
        def label(rows):
            rows[0].update(human_label="TREE", human_note="note, with comma", reviewer="R", review_date="10/30/2026")
        content = self.edit_review(label)
        res = ev.run(self.results, None, self.tmp / "unused", self.review, self.manifest, self.tmp,
                     figures=False, validate_only=True, log=quiet)
        self.assertEqual(res["review_errors"], [])
        self.assertFalse((self.tmp / "unused").exists())   # validation alone writes nothing
        self.evaluate()
        self.assertEqual((self.review / "manual_review.csv").read_bytes(), content)


@unittest.skipUnless((REPO / "benchmark_results" / "lowalt_pilot_pt02_64m_v1").exists(), "pilot results not present")
class TestPilotResults(unittest.TestCase):
    def test_evaluation_of_saved_pilot_trials(self):
        tmp = Path(tempfile.mkdtemp())
        try:
            res = ev.run(REPO / "benchmark_results", ["lowalt_pilot_ptr1_27m_v1", "lowalt_pilot_pt02_64m_v1"],
                         tmp / "eval", tmp / "review", REPO / "manifests" / "manifest.csv", REPO, figures=False, log=quiet)
            self.assertEqual(len(res["units"]), 2)
            for u in res["units"]:
                self.assertEqual(u.n_trials, 10)
                for t in ec.THRESHOLDS:
                    self.assertEqual(res["results"][u.key][t].summary["n_pairs"], 45)
                self.assertTrue(ev.verify_counts(u, res["stats"][u.key]))
            self.assertEqual(res["accuracy_rows"], [])
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
