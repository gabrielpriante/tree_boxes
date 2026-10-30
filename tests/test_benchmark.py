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
                self.assertEqual(r["week"], "UNKNOWN")

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
        self.assertEqual((info["width"], info["height"]), (2576, 1546))
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


if __name__ == "__main__":
    unittest.main()
