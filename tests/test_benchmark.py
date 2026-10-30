"""
Offline tests for the benchmark runner. No network access and no API charges:
runner tests use MockAdapter, and ClaudeAdapter tests use a fake client object.

    python -m pytest tests -q
"""

import csv
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import httpx2
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import anthropic  # noqa: E402
import benchmark_runner as br  # noqa: E402
import box_generation as bg  # noqa: E402
from model_adapters import ClaudeAdapter, FatalAdapterError, MockAdapter  # noqa: E402

REPO = Path(__file__).resolve().parents[1]


def quiet(*_args, **_kwargs):
    pass


class BenchmarkTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.out = self.tmp / "results"
        rows = []
        for i, (image_id, site) in enumerate([("IMG_A", "PT-02"), ("IMG_B", "PT-R1"), ("IMG_C", "PT-02")]):
            path = self.tmp / "imgs" / f"{image_id}.jpg"
            path.parent.mkdir(exist_ok=True)
            Image.new("RGB", (320, 240), (40 * i, 120, 60)).save(path, quality=90)
            rows.append({"image_id": image_id, "image_path": f"imgs/{image_id}.jpg", "site": site,
                         "week": "WEEK 3", "condition": "BEFORE", "altitude_m": "95.0",
                         "coverage_scope": "UNKNOWN", "collection": "standard"})
        self.manifest = self.tmp / "manifest.csv"
        self._write_manifest(rows)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _write_manifest(self, rows):
        with self.manifest.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)

    def rows(self, ids=("IMG_A", "IMG_B")):
        return br.load_manifest(self.manifest, self.tmp, list(ids) if ids else None)

    def run_mock(self, adapter, ids=("IMG_A", "IMG_B"), **kw):
        kw.setdefault("runs", 10)
        kw.setdefault("max_usd", 100.0)
        kw.setdefault("dry_run", False)
        return br.run_benchmark(self.rows(ids), adapter, self.out, "TEST", manifest_path=self.manifest,
                                log=quiet, **kw)

    def exp_dir(self):
        return self.out / "TEST"

    def run_dir(self, adapter):
        return self.exp_dir() / br.config_id(adapter.config())

    def read_csv(self, name):
        with (self.exp_dir() / name).open(newline="") as f:
            return list(csv.DictReader(f))


# 1. Image-manifest loading
class TestManifest(BenchmarkTestCase):
    def test_loads_rows_in_manifest_order(self):
        rows = br.load_manifest(self.manifest, self.tmp, ["IMG_B", "IMG_A"])
        self.assertEqual([r["image_id"] for r in rows], ["IMG_A", "IMG_B"])
        self.assertTrue(rows[0]["_abs_path"].is_file())

    def test_rejects_missing_column_duplicate_id_missing_file_unknown_id(self):
        base = {"image_id": "X", "image_path": "imgs/IMG_A.jpg", "site": "PT-02", "week": "WEEK 3",
                "condition": "BEFORE", "altitude_m": "95", "coverage_scope": "UNKNOWN"}
        self._write_manifest([{k: v for k, v in base.items() if k != "altitude_m"}])
        with self.assertRaisesRegex(ValueError, "missing columns"):
            br.load_manifest(self.manifest, self.tmp)
        self._write_manifest([base, base])
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            br.load_manifest(self.manifest, self.tmp)
        self._write_manifest([{**base, "image_path": "imgs/nope.jpg"}])
        with self.assertRaisesRegex(ValueError, "not found"):
            br.load_manifest(self.manifest, self.tmp)
        self._write_manifest([base])
        with self.assertRaisesRegex(ValueError, "not in manifest"):
            br.load_manifest(self.manifest, self.tmp, ["Y"])

    def test_real_project_manifest(self):
        rows = br.load_manifest(REPO / "manifests" / "manifest.csv", REPO)
        self.assertEqual(len(rows), 30)
        self.assertEqual(len({r["image_id"] for r in rows}), 30)
        collections = {r["collection"] for r in rows}
        self.assertEqual(collections, {"standard", "low_altitude"})
        for r in rows:
            if r["collection"] == "low_altitude":
                self.assertTrue(r["image_id"].startswith("LA_"))
                self.assertEqual(r["week"], "NOT_APPLICABLE")  # not UNKNOWN: week does not apply

    def test_unknown_prompt_metadata_is_skipped_not_guessed(self):
        rows = self.rows(("IMG_A",))
        rows[0]["week"] = "UNKNOWN"
        adapter = MockAdapter()
        report = br.run_benchmark(rows, adapter, self.out, "TEST", runs=2, max_usd=10, dry_run=False, log=quiet)
        self.assertEqual(adapter.calls, [])
        self.assertEqual(report["skipped_images"][0][0], "IMG_A")


# 2. Ten-trial iteration  and  3. progression to the next image
class TestIteration(BenchmarkTestCase):
    def test_ten_independent_trials_then_next_image(self):
        adapter = MockAdapter()
        report = self.run_mock(adapter)
        order = [(c["image_id"], c["trial"]) for c in adapter.calls]
        expected = [("IMG_A", t) for t in range(1, 11)] + [("IMG_B", t) for t in range(1, 11)]
        self.assertEqual(order, expected)
        self.assertEqual(report["trials_run"], 20)
        self.assertEqual(report["incomplete"], {})

    def test_every_trial_sends_identical_bytes_and_prompt(self):
        seen = []

        class Recording(MockAdapter):
            def run(self, jpeg_bytes, prompt, context):
                seen.append((br.sha256_bytes(jpeg_bytes), br.sha256_text(prompt)))
                return super().run(jpeg_bytes, prompt, context)

        self.run_mock(Recording(), ids=("IMG_A",))
        self.assertEqual(len(seen), 10)
        self.assertEqual(len(set(seen)), 1)


# 4. Unique output filenames
class TestOutputs(BenchmarkTestCase):
    def test_unique_paths_and_required_artifacts(self):
        adapter = MockAdapter()
        self.run_mock(adapter)
        records = sorted(self.run_dir(adapter).glob("*/trial_*/attempt_*/record.json"))
        self.assertEqual(len(records), 20)
        self.assertEqual(len({p.parent for p in records}), 20)
        rec = json.loads(records[0].read_text())
        for key in ("model", "inference_settings", "image", "prompt_sha256", "trial", "started_utc",
                    "finished_utc", "status", "usage", "estimated_cost_usd", "validation", "tree_count"):
            self.assertIn(key, rec)
        for key in ("original_width", "upload_width", "file_sha256", "upload_sha256"):
            self.assertIn(key, rec["image"])
        for name in ("request.json", "response_raw.json", "detections.json", "annotated.jpg", "prompt.txt"):
            self.assertTrue((records[0].parent / name).exists(), name)

    def test_api_key_never_written(self):
        import os
        os.environ["ANTHROPIC_API_KEY"] = "sk-ant-test-SECRET-123"
        try:
            self.run_mock(MockAdapter(), ids=("IMG_A",), runs=2)
        finally:
            del os.environ["ANTHROPIC_API_KEY"]
        for p in self.out.rglob("*"):
            if p.is_file() and p.suffix in (".json", ".csv", ".txt"):
                self.assertNotIn("SECRET", p.read_text(encoding="utf-8"))


# 5. JSON validation
class TestValidation(BenchmarkTestCase):
    row = {"site": "PT-02", "week": "WEEK 3", "condition": "BEFORE"}

    def good(self):
        return {"site": "{PT-02}", "week": "WEEK 3", "condition": "BEFORE",
                "trees": [{"tree_id": "T001", "bbox": [10, 10, 50, 60]},
                          {"tree_id": "T002", "bbox": [100, 100, 200, 220]}]}

    def test_valid(self):
        errors, warnings = br.validate_detections(self.good(), self.row)
        self.assertEqual(errors, [])
        self.assertEqual(warnings, [])

    def test_each_defect_is_reported(self):
        cases = {
            "outside 0..1000": [0, 0, 1001, 10],
            "x_min < x_max": [50, 0, 50, 10],
            "4 values": [1, 2, 3],
            "integers": [1.5, 2, 30, 40],
        }
        for msg, bbox in cases.items():
            d = self.good()
            d["trees"][0]["bbox"] = bbox
            errors, _ = br.validate_detections(d, self.row)
            self.assertTrue(any(msg in e for e in errors), (msg, errors))
        d = self.good()
        d["trees"][1]["tree_id"] = "T001"
        self.assertTrue(any("duplicate" in e for e in br.validate_detections(d, self.row)[0]))
        self.assertEqual(br.validate_detections([], self.row)[0], ["response is not a JSON object"])

    def test_warnings_do_not_fail(self):
        d = self.good()
        d["trees"][1]["tree_id"] = "T007"
        d["site"] = "PT-R1"
        errors, warnings = br.validate_detections(d, self.row)
        self.assertEqual(errors, [])
        self.assertEqual(len(warnings), 2)

    def test_invalid_response_preserved_uncorrected(self):
        adapter = MockAdapter(behaviors={("IMG_A", 2): "invalid", ("IMG_A", 3): "bad_json"})
        self.run_mock(adapter, ids=("IMG_A",), runs=3)
        t2 = self.run_dir(adapter) / "IMG_A" / "trial_02" / "attempt_01"
        rec = json.loads((t2 / "record.json").read_text())
        self.assertEqual(rec["status"], "invalid")
        self.assertIsNone(rec["tree_count"])
        self.assertFalse((t2 / "detections.json").exists())
        saved = json.loads((t2 / "invalid_response.json").read_text())
        self.assertEqual(saved["parsed"]["trees"][0]["bbox"], [10, 10, 1200, 50])  # not clamped
        self.assertTrue((t2 / "annotated_INVALID.jpg").exists())
        t3 = self.run_dir(adapter) / "IMG_A" / "trial_03" / "attempt_01"
        self.assertEqual(json.loads((t3 / "record.json").read_text())["status"], "invalid")
        invalid = self.read_csv("invalid_trials.csv")
        self.assertEqual(sorted(r["trial"] for r in invalid), ["2", "3"])
        # Invalid is a model outcome: it completes the trial and is not re-run.
        self.run_mock(adapter, ids=("IMG_A",), runs=3, resume=True)
        self.assertEqual(len(adapter.calls), 3)


# 6. CSV aggregation
class TestAggregation(BenchmarkTestCase):
    def test_long_and_summary(self):
        adapter = MockAdapter(behaviors={("IMG_A", 4): "refusal", ("IMG_A", 5): "invalid"})
        self.run_mock(adapter)
        long_rows = self.read_csv("counts_long.csv")
        self.assertEqual(len(long_rows), 20)
        for col in ("experiment_id", "image_id", "site", "week", "condition", "altitude_m", "model",
                    "trial", "tree_count", "input_tokens", "output_tokens", "estimated_cost_usd", "status"):
            self.assertIn(col, long_rows[0])
        a_rows = [r for r in long_rows if r["image_id"] == "IMG_A"]
        self.assertEqual(sorted(int(r["trial"]) for r in a_rows), list(range(1, 11)))

        counts = [int(r["tree_count"]) for r in a_rows if r["status"] == "success"]
        summary = {r["image_id"]: r for r in self.read_csv("counts_summary.csv")}
        s = summary["IMG_A"]
        import statistics
        self.assertEqual(int(s["successful_trials"]), 8)
        self.assertAlmostEqual(float(s["mean_tree_count"]), statistics.fmean(counts), places=3)
        self.assertAlmostEqual(float(s["standard_deviation"]), statistics.stdev(counts), places=3)
        self.assertEqual(int(s["minimum_count"]), min(counts))
        self.assertEqual(int(s["maximum_count"]), max(counts))
        self.assertAlmostEqual(float(s["coefficient_of_variation"]),
                               statistics.stdev(counts) / statistics.fmean(counts), places=4)
        self.assertEqual(int(s["refusal_trials"]), 1)
        self.assertEqual(int(s["invalid_trials"]), 1)
        self.assertEqual(int(summary["IMG_B"]["successful_trials"]), 10)

        boxes = self.read_csv("boxes_long.csv")
        self.assertEqual(len(boxes), sum(int(r["tree_count"]) for r in long_rows if r["status"] == "success"))

    def test_single_success_has_blank_sd(self):
        self.run_mock(MockAdapter(), ids=("IMG_A",), runs=1)
        s = self.read_csv("counts_summary.csv")[0]
        self.assertEqual(s["successful_trials"], "1")
        self.assertEqual(s["standard_deviation"], "")
        self.assertEqual(s["coefficient_of_variation"], "")


# 7. Budget stopping
class TestBudget(BenchmarkTestCase):
    def test_stops_before_a_request_that_could_exceed_budget(self):
        adapter = MockAdapter()
        actual = adapter.cost_usd({"input_tokens": 6500, "output_tokens": 9000})       # 0.206
        worst = adapter.cost_usd({"input_tokens": 6500, "output_tokens": bg.MAX_TOKENS})  # 0.666
        budget = 1.50
        report = self.run_mock(adapter, max_usd=budget)
        n = len(adapter.calls)
        # Each request started only if spent + worst <= budget.
        self.assertLessEqual((n - 1) * actual + worst, budget + 1e-9)
        self.assertGreater(n * actual + worst, budget)
        self.assertIn("budget", report["stopped"])
        self.assertLessEqual(report["spent_usd"], budget)
        self.assertIn("IMG_A", report["incomplete"])
        self.assertIn("IMG_B", report["incomplete"])

    def test_budget_is_cumulative_across_sessions(self):
        adapter = MockAdapter()
        self.run_mock(adapter, ids=("IMG_A",), max_usd=1.5)
        first = len(adapter.calls)
        self.run_mock(adapter, ids=("IMG_A",), max_usd=1.5, resume=True)
        self.assertEqual(len(adapter.calls), first)  # nothing more fits

    def test_dry_run_makes_no_calls_and_writes_nothing(self):
        adapter = MockAdapter()
        report = self.run_mock(adapter, dry_run=True)
        self.assertEqual(adapter.calls, [])
        self.assertFalse(self.out.exists())
        self.assertEqual(report["planned_trials"], 20)


# 8. Error recovery
class TestErrorRecovery(BenchmarkTestCase):
    def test_error_then_retry_within_session_keeps_both_attempts(self):
        adapter = MockAdapter(behaviors={("IMG_A", 2, 1): "error"})
        self.run_mock(adapter, ids=("IMG_A",), runs=3)
        trial = self.run_dir(adapter) / "IMG_A" / "trial_02"
        recs = [json.loads((a / "record.json").read_text()) for a in sorted(trial.glob("attempt_*"))]
        self.assertEqual([r["status"] for r in recs], ["error", "success"])
        self.assertEqual(recs[0]["estimated_cost_usd"], 0.0)

    def test_failed_trial_retried_on_resume_without_destroying_record(self):
        adapter = MockAdapter(behaviors={("IMG_A", 2, 1): "midstream_error", ("IMG_A", 2, 2): "error"})
        self.run_mock(adapter, ids=("IMG_A",), runs=3)
        trial = self.run_dir(adapter) / "IMG_A" / "trial_02"
        first = (trial / "attempt_01" / "record.json").read_bytes()
        rec1 = json.loads(first)
        self.assertTrue(rec1["possibly_billed"])
        self.assertGreater(rec1["estimated_cost_usd"], 0.6)  # charged at worst case
        self.assertEqual(br.trial_state(trial)[0], "retry")

        self.run_mock(adapter, ids=("IMG_A",), runs=3, resume=True)
        self.assertEqual((trial / "attempt_01" / "record.json").read_bytes(), first)
        self.assertEqual(json.loads((trial / "attempt_03" / "record.json").read_text())["status"], "success")
        row = [r for r in self.read_csv("counts_long.csv") if r["trial"] == "2"][0]
        self.assertEqual((row["status"], row["attempts_made"]), ("success", "3"))

    def test_fatal_error_stops_run(self):
        adapter = MockAdapter(behaviors={("IMG_A", 1): "fatal"})
        report = self.run_mock(adapter)
        self.assertEqual(len(adapter.calls), 1)
        self.assertIn("error", report["stopped"])

    def test_repeated_failures_trip_circuit_breaker(self):
        adapter = MockAdapter(behaviors={("IMG_A", t): "error" for t in range(1, 11)})
        report = self.run_mock(adapter)
        self.assertEqual(len(adapter.calls), br.MAX_CONSECUTIVE_FAILED_TRIALS * 2)
        self.assertIn("consecutive", report["stopped"])


# 9. Resuming an interrupted experiment  and  10. no duplicate paid trials
class TestResume(BenchmarkTestCase):
    def test_interrupt_then_resume(self):
        adapter = MockAdapter(interrupt_after=13)  # Ctrl+C during the 14th request (IMG_B trial 4)
        report = self.run_mock(adapter)
        self.assertEqual(report["stopped"], "interrupted by user")
        self.assertEqual(report["incomplete"], {"IMG_B": list(range(4, 11))})
        rec = json.loads((self.run_dir(adapter) / "IMG_B" / "trial_04" / "attempt_01" / "record.json").read_text())
        self.assertEqual(rec["status"], "interrupted")

        resumed = MockAdapter()
        report = self.run_mock(resumed, resume=True)
        self.assertEqual([(c["image_id"], c["trial"]) for c in resumed.calls],
                         [("IMG_B", t) for t in range(4, 11)])
        self.assertEqual(resumed.calls[0]["attempt"], 2)
        self.assertEqual(report["incomplete"], {})
        self.assertEqual(len(self.read_csv("counts_long.csv")), 20)

    def test_crash_without_record_is_charged_and_retried(self):
        adapter = MockAdapter()
        self.run_mock(adapter, ids=("IMG_A",), runs=2)
        # Simulate a hard crash: attempt folder with request.json but no record.json.
        orphan = self.run_dir(adapter) / "IMG_A" / "trial_03" / "attempt_01"
        orphan.mkdir(parents=True)
        (orphan / "request.json").write_text(json.dumps({"image_id": "IMG_A", "trial": 3, "attempt": 1,
                                                          "pre_request_worst_case_usd": 0.7}))
        self.assertAlmostEqual(br.spent_usd(self.run_dir(adapter)),
                               2 * adapter.cost_usd({"input_tokens": 6500, "output_tokens": 9000}) + 0.7)
        self.run_mock(adapter, ids=("IMG_A",), runs=3, resume=True)
        self.assertEqual(adapter.calls[-1]["trial"], 3)
        self.assertEqual(adapter.calls[-1]["attempt"], 2)

    def test_completed_trials_never_rerun(self):
        adapter = MockAdapter()
        self.run_mock(adapter)
        stamp = {p: p.stat().st_mtime_ns for p in self.run_dir(adapter).rglob("record.json")}
        again = MockAdapter()
        self.run_mock(again, resume=True)
        self.assertEqual(again.calls, [])
        self.assertEqual(stamp, {p: p.stat().st_mtime_ns for p in self.run_dir(adapter).rglob("record.json")})

    def test_existing_results_require_resume_flag(self):
        self.run_mock(MockAdapter(), ids=("IMG_A",), runs=1)
        with self.assertRaises(SystemExit):
            self.run_mock(MockAdapter(), ids=("IMG_A",), runs=1)

    def test_extending_runs_adds_only_new_trials(self):
        adapter = MockAdapter()
        self.run_mock(adapter, ids=("IMG_A",), runs=3)
        self.run_mock(adapter, ids=("IMG_A",), runs=5, resume=True)
        self.assertEqual([c["trial"] for c in adapter.calls], [1, 2, 3, 4, 5])

    def test_changed_prompt_metadata_blocks_mixing(self):
        adapter = MockAdapter()
        self.run_mock(adapter, ids=("IMG_A",), runs=2)
        rows = self.rows(("IMG_A",))
        rows[0]["week"] = "WEEK 4"
        report = br.run_benchmark(rows, adapter, self.out, "TEST", runs=3, max_usd=10, resume=True,
                                  dry_run=False, log=quiet)
        self.assertIn("prompt differs", report["stopped"])
        self.assertEqual(len(adapter.calls), 2)

    def test_config_change_goes_to_a_new_directory(self):
        a = MockAdapter()
        cfg = a.config()
        changed = {**cfg, "inference": {**cfg["inference"], "effort": "medium"}}
        self.assertNotEqual(br.config_id(cfg), br.config_id(changed))


# ClaudeAdapter with a fake client: request parameters, retries, error classes. No network.
def _status_error(cls, code, retry_after="0"):
    req = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    resp = httpx2.Response(code, request=req, headers={"retry-after": retry_after})
    return cls(f"status {code}", response=resp, body=None)


def _message(text='{"site":"a","week":"b","condition":"c","trees":[]}', stop="end_turn"):
    usage = SimpleNamespace(input_tokens=6000, output_tokens=7000,
                            cache_creation_input_tokens=None, cache_read_input_tokens=None)
    msg = SimpleNamespace(content=[SimpleNamespace(type="thinking", thinking=""),
                                   SimpleNamespace(type="text", text=text)],
                          stop_reason=stop, usage=usage, _request_id="req_fake")
    msg.to_json = lambda: json.dumps({"id": "msg_fake", "stop_reason": stop})
    return msg


class FakeClient:
    """Mimics client.messages.stream(...) as a context manager."""

    def __init__(self, script):
        self.script = list(script)
        self.params = []
        self.messages = self

    def stream(self, **params):
        self.params.append(params)
        outcome = self.script.pop(0)
        outer = self

        class Ctx:
            def __enter__(self):
                if isinstance(outcome, Exception):
                    raise outcome
                return self

            def get_final_message(self):
                if isinstance(outcome, tuple):  # ("midstream", exc)
                    raise outcome[1]
                return outcome

            def __exit__(self, *a):
                return False

        return Ctx()


class TestClaudeAdapter(unittest.TestCase):
    def test_request_uses_fixed_experimental_configuration(self):
        client = FakeClient([_message()])
        result = ClaudeAdapter(client=client, sleep=lambda s: None).run(b"jpeg", "PROMPT", {})
        p = client.params[0]
        self.assertEqual(p["model"], "claude-opus-5-5")
        self.assertEqual(p["max_tokens"], 32000)
        self.assertEqual(p["thinking"], {"type": "adaptive"})
        self.assertEqual(p["output_config"]["effort"], "high")
        self.assertEqual(p["output_config"]["format"]["schema"], bg.OUTPUT_SCHEMA)
        self.assertEqual(len(p["messages"]), 1)  # no history carried between trials
        self.assertEqual(p["messages"][0]["content"][1]["text"], "PROMPT")
        self.assertEqual(result.status, "ok")
        self.assertEqual(result.usage["output_tokens"], 7000)
        self.assertEqual(result.text, '{"site":"a","week":"b","condition":"c","trees":[]}')

    def test_transient_errors_retried_with_backoff(self):
        waits = []
        conn = anthropic.APIConnectionError(request=httpx2.Request("POST", "https://x"))
        client = FakeClient([_status_error(anthropic.RateLimitError, 429, "7"),
                             _status_error(anthropic.InternalServerError, 529, ""),
                             conn, _message()])
        result = ClaudeAdapter(client=client, sleep=waits.append).run(b"j", "p", {})
        self.assertEqual(result.status, "ok")
        self.assertEqual(len(result.retries), 3)
        self.assertEqual(waits[0], 7.0)       # honours retry-after
        self.assertGreaterEqual(waits[1], 10.0)  # exponential: 5 * 2**1

    def test_retries_are_limited(self):
        errs = [_status_error(anthropic.InternalServerError, 500) for _ in range(10)]
        client = FakeClient(errs)
        result = ClaudeAdapter(client=client, sleep=lambda s: None).run(b"j", "p", {})
        self.assertEqual(result.status, "error")
        self.assertFalse(result.possibly_billed)
        self.assertEqual(len(client.params), ClaudeAdapter.MAX_RETRIES + 1)

    def test_midstream_failure_is_flagged_possibly_billed_and_not_retried(self):
        conn = anthropic.APIConnectionError(request=httpx2.Request("POST", "https://x"))
        client = FakeClient([("midstream", conn), _message()])
        result = ClaudeAdapter(client=client, sleep=lambda s: None).run(b"j", "p", {})
        self.assertEqual(result.status, "error")
        self.assertTrue(result.possibly_billed)
        self.assertEqual(len(client.params), 1)

    def test_auth_and_bad_request_are_fatal(self):
        for cls, code in ((anthropic.AuthenticationError, 401), (anthropic.BadRequestError, 400)):
            client = FakeClient([_status_error(cls, code)])
            with self.assertRaises(FatalAdapterError):
                ClaudeAdapter(client=client, sleep=lambda s: None).run(b"j", "p", {})

    def test_cost(self):
        a = ClaudeAdapter(client=object())
        self.assertAlmostEqual(a.cost_usd({"input_tokens": 1_000_000, "output_tokens": 1_000_000}), 24.0)


class TestSingleImageWorkflowUnchanged(unittest.TestCase):
    def test_encode_and_request_shape(self):
        img = Image.new("RGB", (5000, 3000), (10, 120, 30))
        jpeg, info = bg.prepare_upload(img)
        # Preprocessing v2: the documented API target size, not just a 2576 px long edge.
        self.assertEqual((info["width"], info["height"]), bg.api_resized_size(5000, 3000))
        self.assertLessEqual(info["visual_tokens"], bg.MAX_VISUAL_TOKENS)
        self.assertEqual(bg.encode_for_api(img), __import__("base64").standard_b64encode(jpeg).decode())
        req = bg.build_request("B64", "PROMPT")
        self.assertEqual(req["messages"][0]["content"][0]["source"]["data"], "B64")

    def test_cli_from_json_renders_without_api(self):
        tmp = Path(tempfile.mkdtemp())
        try:
            img = tmp / "X.jpg"
            Image.new("RGB", (400, 300), (0, 100, 0)).save(img)
            det = tmp / "det.json"
            det.write_text(json.dumps({"site": "S", "week": "W", "condition": "C",
                                       "trees": [{"tree_id": "T001", "bbox": [10, 10, 200, 200]}]}))
            import subprocess
            out = subprocess.run([sys.executable, str(REPO / "box_generation.py"), str(img),
                                  "--from-json", str(det), "--output-dir", str(tmp / "o")],
                                 capture_output=True, text=True)
            self.assertEqual(out.returncode, 0, out.stderr)
            self.assertIn("1 trees", out.stdout)
            self.assertTrue((tmp / "o" / "X_annotated.jpg").exists())
            self.assertTrue((tmp / "o" / "tree_counts.csv").exists())
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


LA_IDS = ["LA_PT-02_64m", "LA_PT-02_66m", "LA_PT-02_67m", "LA_PT-02_71m", "LA_PT-02_86m",
          "LA_PT-R1_27m", "LA_PT-R1_28m", "LA_PT-R1_30m", "LA_PT-R1_30m_angled", "LA_PT-R1_31m"]


# Image preprocessing (v2)
class TestPreprocessing(unittest.TestCase):
    def test_matches_documented_reference_examples(self):
        # Examples from Anthropic's vision / vision-coordinates documentation.
        self.assertEqual(bg.api_resized_size(1075, 1520, 1568, 1568), (924, 1307))
        self.assertEqual(bg.api_resized_size(3840, 2160), (2576, 1449))
        self.assertEqual(bg.api_resized_size(1920, 1080), (1920, 1080))
        self.assertEqual(bg.api_resized_size(2000, 1500), (2000, 1500))
        self.assertEqual(bg.count_image_tokens(2000, 1500), 3888)

    def test_drone_resolution_needs_no_server_resize(self):
        w, h = bg.api_resized_size(8192, 6144)
        self.assertEqual((w, h), (2212, 1659))
        self.assertLessEqual(bg.count_image_tokens(w, h), bg.MAX_VISUAL_TOKENS)
        # The v1 upload (2576x1932) exceeded the token limit, so the API resized it again.
        self.assertGreater(bg.count_image_tokens(2576, 1932), bg.MAX_VISUAL_TOKENS)
        self.assertEqual(bg.api_resized_size(2212, 1659), (2212, 1659))  # idempotent: API leaves it alone

    def test_request_rejects_rather_than_downsizes(self):
        block = bg.build_request("B64", "P")["messages"][0]["content"][0]
        self.assertEqual(block["transformations"], {"oversized_image": "error"})

    def test_identical_bytes_across_ten_repetitions_of_a_real_image(self):
        path = REPO / "drone_images" / "height" / "PT_R1" / "PT_R1_30m_angled.JPG"
        digests, sizes = set(), set()
        for i in range(10):
            img = bg.load_image(path) if i % 5 == 0 else img  # reload from disk twice
            jpeg, info = bg.prepare_upload(img)
            digests.add(br.sha256_bytes(jpeg))
            sizes.add((info["width"], info["height"], info["jpeg_quality"]))
        self.assertEqual(len(digests), 1)
        self.assertEqual(sizes, {(2212, 1659, 95)})
        self.assertEqual(img.size, (8192, 6144))  # original kept full-resolution for rendering

    def test_preprocessing_change_gives_new_config_id(self):
        cfg = MockAdapter().config()
        self.assertEqual(cfg["preprocessing"]["version"], 2)
        v1 = {**cfg, "preprocessing": {**cfg["preprocessing"], "version": 1}}
        self.assertNotEqual(br.config_id(cfg), br.config_id(v1))

    def test_record_documents_three_resolutions(self):
        tmp = Path(tempfile.mkdtemp())
        try:
            p = tmp / "x.jpg"
            Image.new("RGB", (4000, 3000), (0, 90, 0)).save(p)
            image = br.prepare_image({"_abs_path": p, "image_id": "X", "site": "PT-02",
                                      "week": "WEEK 1", "condition": "BEFORE"})
            m = image["meta"]
            self.assertEqual((m["original_width"], m["original_height"]), (4000, 3000))
            self.assertEqual((m["upload_width"], m["upload_height"]), bg.api_resized_size(4000, 3000))
            self.assertEqual((m["api_effective_width"], m["api_effective_height"]),
                             (m["upload_width"], m["upload_height"]))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


# Fixed experimental settings
class TestFixedConfiguration(unittest.TestCase):
    def test_settings_unchanged(self):
        cfg = ClaudeAdapter(client=object()).config()
        self.assertEqual(cfg["model"], "claude-opus-5-5")
        self.assertEqual(cfg["inference"]["effort"], "high")
        self.assertEqual(cfg["inference"]["thinking"], {"type": "adaptive"})
        self.assertEqual(cfg["inference"]["max_tokens"], 32000)
        self.assertEqual(cfg["prompt_template_sha256"],
                         "e53504b32a5931a597ebd8499aba89f602d90b720d8a0bb9d9ea92cfc96c100f")
        self.assertEqual(cfg["output_schema_sha256"],
                         "963a771dd008a68fbca9cb2b42e762d9109fb40cd52423d8b706b7b8e2bc8e38")
        self.assertIn("0-1000", cfg["coordinates"])

    def test_budget_never_alters_settings(self):
        adapter = MockAdapter()
        before = json.dumps(adapter.config(), sort_keys=True)
        tmp = Path(tempfile.mkdtemp())
        try:
            img = tmp / "a.jpg"
            Image.new("RGB", (300, 200)).save(img)
            rows = [{"image_id": "A", "image_path": "a.jpg", "site": "PT-02", "week": "WEEK 3",
                     "condition": "BEFORE", "altitude_m": "95", "coverage_scope": "UNKNOWN", "_abs_path": img}]
            br.run_benchmark(rows, adapter, tmp / "o", "E", runs=10, max_usd=0.9, dry_run=False, log=quiet)
            self.assertEqual(before, json.dumps(adapter.config(), sort_keys=True))
            recs = [json.loads(p.read_text()) for p in (tmp / "o").rglob("record.json")]
            self.assertTrue(recs)
            for r in recs:
                self.assertEqual((r["model"], r["inference_settings"]["effort"],
                                  r["inference_settings"]["max_tokens"]), ("claude-opus-5-5", "high", 32000))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


# Manifest metadata: vocabulary, NOT_APPLICABLE vs UNKNOWN, confirmation gate
class TestManifestMetadata(BenchmarkTestCase):
    def base(self, **kw):
        row = {"image_id": "IMG_A", "image_path": "imgs/IMG_A.jpg", "site": "PT-02", "week": "WEEK 3",
               "condition": "BEFORE", "altitude_m": "95.0", "coverage_scope": "UNKNOWN",
               "collection": "standard", "needs_review": ""}
        row.update(kw)
        return row

    def test_vocabulary_enforced(self):
        for bad in ({"coverage_scope": "Full"}, {"condition": "before"}, {"week": "week 3"},
                    {"site": "PT 02"}, {"altitude_m": "high"}, {"week": ""}):
            self._write_manifest([self.base(**bad)])
            with self.assertRaises(ValueError, msg=bad):
                br.load_manifest(self.manifest, self.tmp)
        for ok in ({"coverage_scope": "FULL_SITE"}, {"coverage_scope": "PARTIAL_SITE"},
                   {"week": "NOT_APPLICABLE", "condition": "ALTITUDE_TEST"}, {"condition": "MORNING_START"}):
            self._write_manifest([self.base(**ok)])
            br.load_manifest(self.manifest, self.tmp)

    def test_not_applicable_runs_without_override_but_unknown_does_not(self):
        na = self.base(week="NOT_APPLICABLE", condition="ALTITUDE_TEST", needs_review="coverage_scope")
        self.assertEqual(br.prompt_metadata_blockers(na), [])
        self.assertTrue(br.prompt_metadata_blockers(self.base(week="UNKNOWN")))
        self._write_manifest([na])
        adapter = MockAdapter()
        report = br.run_benchmark(br.load_manifest(self.manifest, self.tmp), adapter, self.out, "TEST",
                                  runs=2, max_usd=10, dry_run=False, log=quiet)
        self.assertEqual(len(adapter.calls), 2)
        self.assertEqual(report["skipped_images"], [])
        prompt = (next(self.out.rglob("prompt.txt"))).read_text()
        self.assertIn("NOT_APPLICABLE", prompt)
        self.assertIn("ALTITUDE_TEST", prompt)

    def test_unconfirmed_prompt_fields_block_execution(self):
        self._write_manifest([self.base(week="NOT_APPLICABLE", condition="ALTITUDE_TEST",
                                        needs_review="week;condition;coverage_scope")])
        adapter = MockAdapter()
        report = br.run_benchmark(br.load_manifest(self.manifest, self.tmp), adapter, self.out, "TEST",
                                  runs=2, max_usd=10, dry_run=False, log=quiet)
        self.assertEqual(adapter.calls, [])
        self.assertIn("awaiting confirmation", report["skipped_images"][0][1])
        self.assertEqual(report["blocked_trials"], 2)

    def test_coverage_review_alone_does_not_block(self):
        self.assertEqual(br.prompt_metadata_blockers(self.base(needs_review="coverage_scope")), [])

    def test_real_manifest_uses_explicit_labels(self):
        rows = br.load_manifest(REPO / "manifests" / "manifest.csv", REPO, collection="low_altitude")
        for r in rows:
            self.assertEqual((r["week"], r["condition"]), ("NOT_APPLICABLE", "ALTITUDE_TEST"))
            self.assertNotIn(r["condition"], ("BEFORE", "AFTER"))
            self.assertFalse({"week", "condition"} & br.review_fields(r))  # confirmed 2026-10-30
            self.assertTrue(r["capture_datetime"].startswith("2026:10:24"))
            float(r["altitude_m"])
        w1 = br.load_manifest(REPO / "manifests" / "manifest.csv", REPO, image_ids=["W01_MORNING-START_PT-02"])[0]
        self.assertEqual(w1["condition"], "MORNING_START")  # original label kept, mapping unconfirmed
        self.assertIn("condition", br.review_fields(w1))


class TestManifestUpdatePreservesEdits(unittest.TestCase):
    def test_update_never_overwrites_manual_corrections(self):
        import make_manifest as mm
        tmp = Path(tempfile.mkdtemp())
        try:
            path = tmp / "manifest.csv"
            shutil.copyfile(REPO / "manifests" / "manifest.csv", path)
            with path.open(newline="") as f:
                rows = list(csv.DictReader(f))
                fields = list(rows[0])
            edits = {"LA_PT-02_64m": {"coverage_scope": "FULL_SITE", "needs_review": "condition"},
                     "W01_MORNING-START_PT-02": {"condition": "BEFORE", "needs_review": "coverage_scope"},
                     "LA_PT-R1_27m": {"week": "UNKNOWN"}}
            for r in rows:
                r.update(edits.get(r["image_id"], {}))
            with path.open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fields)
                w.writeheader()
                w.writerows(rows)
            mm.update(path)
            with path.open(newline="") as f:
                after = {r["image_id"]: r for r in csv.DictReader(f)}
            self.assertEqual(after["LA_PT-02_64m"]["coverage_scope"], "FULL_SITE")
            self.assertEqual(after["LA_PT-02_64m"]["needs_review"], "condition")
            self.assertEqual(after["W01_MORNING-START_PT-02"]["condition"], "BEFORE")
            self.assertEqual(after["W01_MORNING-START_PT-02"]["needs_review"], "coverage_scope")
            # An UNKNOWN cell may be filled with the proposal, which is then flagged for confirmation.
            self.assertEqual(after["LA_PT-R1_27m"]["week"], "NOT_APPLICABLE")
            self.assertIn("week", after["LA_PT-R1_27m"]["needs_review"].split(";"))
            self.assertTrue(list(tmp.glob("manifest.csv.bak-*")))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


# Cross-experiment budget accounting
class TestProjectLedger(BenchmarkTestCase):
    def test_new_experiment_id_does_not_reset_budget(self):
        ledger = self.tmp / "project_ledger.jsonl"
        a = MockAdapter()
        br.run_benchmark(self.rows(("IMG_A",)), a, self.tmp / "out1", "EXP_ONE", runs=10, max_usd=1.5,
                         dry_run=False, ledger_path=ledger, log=quiet)
        first = len(a.calls)
        self.assertGreater(first, 0)
        b = MockAdapter()
        report = br.run_benchmark(self.rows(("IMG_B",)), b, self.tmp / "out2", "EXP_TWO", runs=10,
                                  max_usd=1.5, dry_run=False, ledger_path=ledger, log=quiet)
        self.assertEqual(b.calls, [])  # same credits already spent by EXP_ONE
        self.assertIn("budget", report["stopped"])

    def test_confirmed_estimated_and_unresolved_are_separated(self):
        ledger_path = self.tmp / "l.jsonl"
        adapter = MockAdapter(behaviors={("IMG_A", 2, 1): "midstream_error"})
        br.run_benchmark(self.rows(("IMG_A",)), adapter, self.out, "TEST", runs=2, max_usd=50,
                         dry_run=False, ledger_path=ledger_path, max_attempts=1, log=quiet)
        s = br.SpendLedger(ledger_path).summary()
        one = adapter.cost_usd({"input_tokens": 6500, "output_tokens": 9000})
        self.assertAlmostEqual(s["confirmed_usd"], one, places=6)
        self.assertGreater(s["estimated_usd"], 0.6)  # worst-case bound for the failed request
        self.assertEqual(s["unresolved_usd"], 0)
        # A crash between ledger start and final leaves the attempt unresolved at its worst case.
        br.SpendLedger(ledger_path).start("crash-uid", {}, 0.7)
        self.assertAlmostEqual(br.SpendLedger(ledger_path).summary()["unresolved_usd"], 0.7)

    def test_no_double_counting(self):
        ledger_path = self.tmp / "l.jsonl"
        adapter = MockAdapter()
        br.run_benchmark(self.rows(("IMG_A",)), adapter, self.out, "TEST", runs=3, max_usd=50,
                         dry_run=False, ledger_path=ledger_path, log=quiet)
        ledger = br.SpendLedger(ledger_path)
        before = ledger.summary()
        run_dir = self.out / "TEST" / br.config_id(adapter.config())
        self.assertEqual(ledger.reconcile(br.all_attempts(run_dir)), 0)
        uid = json.loads(next(run_dir.rglob("record.json")).read_text())["attempt_uid"]
        ledger.finish(uid, 99.0, "actual_usage", "success")  # duplicate final entry
        self.assertEqual(ledger.summary(), before)
        self.assertEqual(before["attempts"], 3)
        # Resuming with nothing left to do adds nothing.
        br.run_benchmark(self.rows(("IMG_A",)), adapter, self.out, "TEST", runs=3, max_usd=50, resume=True,
                         dry_run=False, ledger_path=ledger_path, log=quiet)
        self.assertEqual(br.SpendLedger(ledger_path).summary()["total_usd"], before["total_usd"])

    def test_legacy_records_backfilled_once(self):
        ledger_path = self.tmp / "l.jsonl"
        adapter = MockAdapter()
        br.run_benchmark(self.rows(("IMG_A",)), adapter, self.out, "TEST", runs=2, max_usd=50,
                         dry_run=False, ledger_path=ledger_path, log=quiet)
        total = br.SpendLedger(ledger_path).summary()["total_usd"]
        ledger_path.unlink()  # results that predate the ledger
        fresh = br.SpendLedger(ledger_path)
        run_dir = self.out / "TEST" / br.config_id(adapter.config())
        fresh.reconcile(br.all_attempts(run_dir))
        fresh.reconcile(br.all_attempts(run_dir))
        self.assertAlmostEqual(fresh.summary()["total_usd"], total)

    def test_dry_run_reports_project_spend_and_writes_nothing(self):
        ledger_path = self.tmp / "l.jsonl"
        br.run_benchmark(self.rows(("IMG_A",)), MockAdapter(), self.tmp / "o1", "E1", runs=2, max_usd=5,
                         dry_run=False, ledger_path=ledger_path, log=quiet)
        size = ledger_path.stat().st_size
        lines = []
        br.run_benchmark(self.rows(("IMG_B",)), MockAdapter(), self.tmp / "o2", "E2", runs=10, max_usd=5,
                         dry_run=True, ledger_path=ledger_path, log=lines.append)
        text = "\n".join(lines)
        self.assertIn("confirmed from API usage:          $0.4120", text)
        self.assertIn("remaining $4.5880", text)
        self.assertIn("not visible here", text)
        self.assertEqual(ledger_path.stat().st_size, size)
        self.assertFalse((self.tmp / "o2").exists())

    def test_mock_cannot_write_into_production_results(self):
        with self.assertRaises(SystemExit):
            br.main(["--manifest", str(self.manifest), "--image-root", str(self.tmp), "--mock", "--dry-run",
                     "--output-dir", str(br.PRODUCTION_OUTPUT_DIR / "x")])
        self.assertNotEqual(br.MOCK_OUTPUT_DIR.resolve(), br.PRODUCTION_OUTPUT_DIR.resolve())


# Repeated-invalid-output safeguard
class TestPauseSafeguard(BenchmarkTestCase):
    def test_pauses_after_three_consecutive_and_keeps_records(self):
        adapter = MockAdapter(behaviors={("IMG_A", 2): "invalid", ("IMG_A", 3): "refusal",
                                         ("IMG_A", 4): "max_tokens"})
        report = self.run_mock(adapter)
        self.assertEqual([c["trial"] for c in adapter.calls], [1, 2, 3, 4])
        self.assertTrue(report["stopped"].startswith("paused"))
        run_dir = self.run_dir(adapter)
        pause = json.loads((run_dir / br.PAUSE_FILE).read_text())
        self.assertEqual([t["status"] for t in pause["trials"]], ["invalid", "refusal", "max_tokens"])
        statuses = [r["status"] for r in self.read_csv("counts_long.csv")]
        self.assertEqual(statuses, ["success", "invalid", "refusal", "max_tokens"])

    def test_resume_requires_acknowledgment_then_continues_without_rerunning(self):
        adapter = MockAdapter(behaviors={("IMG_A", t): "invalid" for t in (1, 2, 3)})
        self.run_mock(adapter)
        blocked = MockAdapter()
        report = self.run_mock(blocked, resume=True)
        self.assertEqual(blocked.calls, [])
        self.assertIn("acknowledge-pause", report["stopped"])

        resumed = MockAdapter()
        report = self.run_mock(resumed, resume=True, acknowledge_paused=True)
        self.assertEqual(resumed.calls[0]["trial"], 4)  # invalid trials 1-3 are not retried
        self.assertEqual(report["incomplete"], {})
        run_dir = self.run_dir(resumed)
        self.assertFalse((run_dir / br.PAUSE_FILE).exists())
        self.assertEqual(len(list(run_dir.glob("PAUSED_acknowledged_*.json"))), 1)
        summary = {r["image_id"]: r for r in self.read_csv("counts_summary.csv")}
        self.assertEqual(summary["IMG_A"]["invalid_trials"], "3")
        self.assertEqual(summary["IMG_A"]["successful_trials"], "7")

    def test_non_consecutive_outcomes_do_not_pause(self):
        adapter = MockAdapter(behaviors={("IMG_A", 2): "invalid", ("IMG_A", 3): "invalid",
                                         ("IMG_A", 5): "refusal", ("IMG_A", 6): "invalid"})
        report = self.run_mock(adapter, ids=("IMG_A",))
        self.assertIsNone(report["stopped"])
        self.assertEqual(len(adapter.calls), 10)

    def test_errors_do_not_count_or_reset(self):
        adapter = MockAdapter(behaviors={("IMG_A", 1): "invalid", ("IMG_A", 2, 1): "error",
                                         ("IMG_A", 2, 2): "invalid", ("IMG_A", 3): "invalid"})
        report = self.run_mock(adapter, ids=("IMG_A",))
        self.assertTrue(report["stopped"].startswith("paused"))

    def test_streak_carries_across_sessions(self):
        adapter = MockAdapter(behaviors={("IMG_A", 1): "invalid", ("IMG_A", 2): "invalid"}, interrupt_after=2)
        self.run_mock(adapter, ids=("IMG_A",))
        later = MockAdapter(behaviors={("IMG_A", 3): "invalid"})
        report = self.run_mock(later, ids=("IMG_A",), resume=True)
        self.assertTrue(report["stopped"].startswith("paused"))
        self.assertEqual([c["trial"] for c in later.calls], [3])

    def test_threshold_configurable_and_disableable(self):
        bad = {("IMG_A", t): "invalid" for t in range(1, 11)}
        adapter = MockAdapter(behaviors=bad)
        self.run_mock(adapter, ids=("IMG_A",), pause_after=0)
        self.assertEqual(len(adapter.calls), 10)
        shutil.rmtree(self.out)
        adapter = MockAdapter(behaviors=bad)
        self.run_mock(adapter, ids=("IMG_A",), pause_after=5)
        self.assertEqual(len(adapter.calls), 5)


# Ten-image October 24 selection
class TestOctober24Selection(unittest.TestCase):
    def test_collection_filter_selects_exactly_the_ten(self):
        rows = br.load_manifest(REPO / "manifests" / "manifest.csv", REPO, collection="low_altitude")
        self.assertEqual(sorted(r["image_id"] for r in rows), sorted(LA_IDS))
        by_ids = br.load_manifest(REPO / "manifests" / "manifest.csv", REPO, image_ids=LA_IDS)
        self.assertEqual([r["image_id"] for r in rows], [r["image_id"] for r in by_ids])

    def test_paths_and_hashes_verified(self):
        for r in br.load_manifest(REPO / "manifests" / "manifest.csv", REPO, image_ids=LA_IDS):
            self.assertEqual(br.sha256_bytes(r["_abs_path"].read_bytes()), r["sha256"], r["image_id"])

    def test_dry_run_plans_one_hundred_trials(self):
        tmp = Path(tempfile.mkdtemp())
        try:
            rows = br.load_manifest(REPO / "manifests" / "manifest.csv", REPO, image_ids=LA_IDS)
            report = br.run_benchmark(rows, MockAdapter(), tmp / "o", "E", runs=10, max_usd=5, dry_run=True,
                                      ledger_path=tmp / "l.jsonl", log=quiet)
            self.assertEqual(report["planned_trials"] + report["blocked_trials"], 100)
            self.assertFalse((tmp / "o").exists())
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


def _order(adapter):
    return [(c["image_id"], c["trial"]) for c in adapter.calls]


class TestRoundRobin(BenchmarkTestCase):
    def test_round_robin_alternates_images_by_trial(self):
        adapter = MockAdapter()
        report = self.run_mock(adapter, round_robin=True)
        expected = [(i, t) for t in range(1, 11) for i in ("IMG_A", "IMG_B")]
        self.assertEqual(_order(adapter), expected)
        self.assertEqual(report["order"], expected)
        self.assertEqual(report["incomplete"], {})
        self.assertEqual(len(self.read_csv("counts_long.csv")), 20)

    def test_default_order_is_still_sequential(self):
        adapter = MockAdapter()
        report = self.run_mock(adapter)
        expected = [("IMG_A", t) for t in range(1, 11)] + [("IMG_B", t) for t in range(1, 11)]
        self.assertEqual(_order(adapter), expected)
        self.assertEqual(report["order"], expected)

    def test_same_output_structure_either_way(self):
        seq, rr = MockAdapter(), MockAdapter()
        self.run_mock(seq, runs=3)
        layout = lambda: sorted(str(p.relative_to(self.exp_dir())) for p in self.exp_dir().rglob("*")
                                if p.is_file() and "spend_ledger" not in p.name)
        first = layout()
        shutil.rmtree(self.out)
        self.run_mock(rr, runs=3, round_robin=True)
        self.assertEqual(first, layout())

    def test_round_robin_resume_continues_the_cycle(self):
        adapter = MockAdapter(interrupt_after=5)  # Ctrl+C during the 6th request: IMG_B trial 3
        report = self.run_mock(adapter, round_robin=True)
        self.assertEqual(report["stopped"], "interrupted by user")
        self.assertEqual(_order(adapter), [("IMG_A", 1), ("IMG_B", 1), ("IMG_A", 2), ("IMG_B", 2),
                                           ("IMG_A", 3), ("IMG_B", 3)])
        self.assertEqual(report["incomplete"], {"IMG_A": list(range(4, 11)), "IMG_B": list(range(3, 11))})

        resumed = MockAdapter()
        report = self.run_mock(resumed, round_robin=True, resume=True)
        self.assertEqual(_order(resumed)[:4], [("IMG_B", 3), ("IMG_A", 4), ("IMG_B", 4), ("IMG_A", 5)])
        self.assertEqual(resumed.calls[0]["attempt"], 2)  # the interrupted attempt is kept, not overwritten
        self.assertEqual(len(resumed.calls), 15)
        self.assertEqual(report["incomplete"], {})
        self.assertEqual(len(self.read_csv("counts_long.csv")), 20)

    def test_round_robin_budget_stop_keeps_completed_trials(self):
        adapter = MockAdapter()
        worst = br.worst_case_usd(adapter, bg.build_prompt("PT-02", "WEEK 3", "BEFORE"), None)
        per_trial = adapter.cost_usd({"input_tokens": 6500, "output_tokens": 9000})
        report = self.run_mock(adapter, round_robin=True, max_usd=3 * per_trial + worst + 1e-6)
        self.assertTrue(report["stopped"].startswith("budget"))
        self.assertEqual(_order(adapter), [("IMG_A", 1), ("IMG_B", 1), ("IMG_A", 2), ("IMG_B", 2)])
        self.assertEqual(len(self.read_csv("counts_long.csv")), 4)
        self.assertLessEqual(report["spent_usd"], 3 * per_trial + worst + 1e-6)

    def test_round_robin_pause_counts_across_images(self):
        adapter = MockAdapter(behaviors={("IMG_A", 2): "invalid", ("IMG_B", 2): "refusal",
                                         ("IMG_A", 3): "max_tokens"})
        report = self.run_mock(adapter, round_robin=True)
        self.assertTrue(report["stopped"].startswith("paused"))
        self.assertEqual(_order(adapter)[-1], ("IMG_A", 3))
        self.assertTrue((self.run_dir(adapter) / br.PAUSE_FILE).exists())

    def test_cli_flag(self):
        br.main(["--manifest", str(self.manifest), "--image-root", str(self.tmp), "--image-ids", "IMG_A,IMG_B",
                 "--runs", "2", "--max-usd", "100", "--execute", "--mock", "--round-robin",
                 "--output-dir", str(self.out), "--experiment-id", "CLI"])
        records = [json.loads(p.read_text()) for p in self.out.rglob("record.json")]
        records.sort(key=lambda r: r["finished_unix_ns"])
        self.assertEqual([(r["image_id"], r["trial"]) for r in records],
                         [("IMG_A", 1), ("IMG_B", 1), ("IMG_A", 2), ("IMG_B", 2)])


class TestCoverageDoesNotBlock(BenchmarkTestCase):
    def test_unknown_coverage_under_review_still_runs(self):
        rows = [{"image_id": i, "image_path": f"imgs/{i}.jpg", "site": s, "week": "NOT_APPLICABLE",
                 "condition": "ALTITUDE_TEST", "altitude_m": "27.0", "coverage_scope": "UNKNOWN",
                 "collection": "low_altitude", "needs_review": "coverage_scope;capture_geometry"}
                for i, s in (("IMG_A", "PT-02"), ("IMG_B", "PT-R1"))]
        self._write_manifest(rows)
        for r in self.rows():
            self.assertEqual(br.prompt_metadata_blockers(r), [])
        adapter = MockAdapter()
        report = self.run_mock(adapter, runs=2)
        self.assertEqual(report["skipped_images"], [])
        self.assertEqual(report["blocked_trials"], 0)
        self.assertEqual(len(adapter.calls), 4)

    def test_prompt_fields_still_guarded(self):
        row = {"site": "PT-02", "week": "UNKNOWN", "condition": "ALTITUDE_TEST", "coverage_scope": "UNKNOWN",
               "needs_review": "coverage_scope"}
        self.assertTrue(br.prompt_metadata_blockers(row))
        row.update(week="NOT_APPLICABLE", needs_review="condition;coverage_scope")
        self.assertTrue(br.prompt_metadata_blockers(row))


PILOT_ONE = ["LA_PT-R1_27m"]
PILOT_TWO = ["LA_PT-R1_27m", "LA_PT-02_64m"]


class TestLowAltitudePilot(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def load(self, ids):
        return br.load_manifest(REPO / "manifests" / "manifest.csv", REPO, image_ids=ids)

    def dry_run(self, ids, **kw):
        return br.run_benchmark(self.load(ids), MockAdapter(), self.tmp / "o", "E", runs=10, max_usd=5,
                                dry_run=True, ledger_path=self.tmp / "l.jsonl", log=quiet, **kw)

    def test_pilot_images_exist_with_valid_hashes_and_confirmed_labels(self):
        for r in self.load(PILOT_TWO):
            self.assertEqual(br.sha256_bytes(r["_abs_path"].read_bytes()), r["sha256"], r["image_id"])
            self.assertEqual((r["week"], r["condition"]), ("NOT_APPLICABLE", "ALTITUDE_TEST"))
            self.assertEqual(r["coverage_scope"], "UNKNOWN")
            self.assertEqual(br.prompt_metadata_blockers(r), [], r["image_id"])

    def test_select_only_pt_r1_27m(self):
        rows = self.load(PILOT_ONE)
        self.assertEqual([r["image_id"] for r in rows], PILOT_ONE)
        report = self.dry_run(PILOT_ONE)
        self.assertEqual((report["planned_trials"], report["blocked_trials"]), (10, 0))
        self.assertEqual(report["order"], [("LA_PT-R1_27m", t) for t in range(1, 11)])
        self.assertFalse((self.tmp / "o").exists())

    def test_select_two_lowest_round_robin(self):
        rows = self.load(PILOT_TWO)
        self.assertEqual(sorted(r["image_id"] for r in rows), sorted(PILOT_TWO))
        report = self.dry_run(PILOT_TWO, round_robin=True)
        self.assertEqual((report["planned_trials"], report["blocked_trials"]), (20, 0))
        first = rows[0]["image_id"]
        self.assertEqual([i for i, _ in report["order"][:4]],
                         [first, [i for i in PILOT_TWO if i != first][0]] * 2)
        self.assertEqual([t for _, t in report["order"]], [t for t in range(1, 11) for _ in range(2)])
        self.assertFalse((self.tmp / "o").exists())


if __name__ == "__main__":
    unittest.main()
