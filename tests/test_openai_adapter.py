"""
Offline tests for M09 (OpenAI GPT-6.1 Sol). A fake client built on the SDK's own response types stands in
for the API: no network access, no API charges, and the real .env is never read.

    python -m pytest tests -q
"""

import base64
import csv
import hashlib
import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import httpx
import openai
from openai.types.responses import Response
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import benchmark_runner as br  # noqa: E402
import box_generation as bg  # noqa: E402
import evaluate_consistency as ev  # noqa: E402
import model_adapters as ma  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
PINNED_M08_CONFIG_ID = "M08_high_a28fe01ac5"
PINNED_M09_CONFIG_ID = "M09_high_90be168b98"
FAKE_KEY = "sk-proj-SECRETtestkey0123456789"
PROTECTED = ["benchmark_results", "spend_ledger", "human_review", "manifests"]


def _snapshot():
    out = {}
    for d in PROTECTED:
        for p in sorted((REPO / d).rglob("*")):
            if p.is_file():
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


# ------------------------------------------------------------------ fake OpenAI API

def trees_json(n=3, site="PT-02", week="WEEK 3", condition="BEFORE", bbox=None):
    trees = [{"tree_id": f"T{i + 1:03d}", "bbox": bbox or [100 * i + 10, 100, 100 * i + 90, 190]} for i in range(n)]
    return json.dumps({"site": site, "week": week, "condition": condition, "trees": trees})


def make_response(text=None, status="completed", reason=None, refusal=None, model="gpt-6.1-sol",
                  compression="disabled", input_tokens=6000, output_tokens=2500, reasoning_tokens=2000,
                  error=None):
    content = [{"type": "refusal", "refusal": refusal}] if refusal else \
        [{"type": "output_text", "text": text if text is not None else trees_json(), "annotations": []}]
    body = {
        "id": "resp_fake", "object": "response", "created_at": 0, "model": model, "status": status,
        "output": [{"type": "reasoning", "id": "rs_1", "summary": []},
                   {"type": "message", "id": "msg_1", "role": "assistant",
                    "status": "completed" if status == "completed" else "incomplete", "content": content}],
        "parallel_tool_calls": True, "tool_choice": "auto", "tools": [], "store": False,
        "service_tier": "default", "reasoning": {"effort": "high", "summary": None},
        "incomplete_details": {"reason": reason} if reason else None, "error": error,
        "usage": {"input_tokens": input_tokens,
                  "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
                  "output_tokens": output_tokens, "output_tokens_details": {"reasoning_tokens": reasoning_tokens},
                  "total_tokens": input_tokens + output_tokens},
    }
    if compression is not None:
        body["image"] = {"context_compression": compression}
    return Response.model_validate(body)


def status_error(cls, code, err_code=None, message="error"):
    req = httpx.Request("POST", "https://api.openai.com/v1/responses")
    resp = httpx.Response(code, request=req, headers={"retry-after": "0"})
    return cls(message, response=resp, body={"error": {"code": err_code, "message": message}})


def connection_error():
    return openai.APIConnectionError(request=httpx.Request("POST", "https://api.openai.com/v1/responses"))


class _Stream:
    def __init__(self, item):
        self.item = item

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def get_final_response(self):
        if isinstance(self.item, tuple) and self.item[0] == "midstream":
            raise self.item[1]
        if isinstance(self.item, BaseException):
            raise self.item
        return self.item


class FakeOpenAI:
    """client.responses.stream(**params) -> context manager. Queue items: Response, exception raised
    when the request is made, or ('midstream', exc) raised after the stream opened."""

    def __init__(self, items=None, default=None):
        self.items = list(items or [])
        self.default = default
        self.calls = []
        self.responses = self

    def stream(self, **params):
        self.calls.append(params)
        item = self.items.pop(0) if self.items else (self.default or make_response())
        if callable(item) and not isinstance(item, (Response, BaseException)):
            item = item(params)
        if isinstance(item, (BaseException, type)) and not isinstance(item, tuple):
            if isinstance(item, type):
                item = item()
            raise item
        return _Stream(item)


def adapter(items=None, default=None):
    client = FakeOpenAI(items, default)
    a = ma.OpenAIAdapter(client=client, sleep=lambda s: None)
    return a, client


def run_once(a, prompt="p", jpeg=b"\xff\xd8jpeg"):
    return a.run(jpeg, prompt, {"image_id": "X", "trial": 1, "attempt": 1})


# ------------------------------------------------------------------ configuration and request

class TestFixedConfiguration(unittest.TestCase):
    def test_model_and_reasoning_settings(self):
        self.assertEqual(ma.OPENAI_MODEL, "gpt-6.1-sol")
        self.assertEqual(ma.OPENAI_REASONING_EFFORT, "high")
        self.assertEqual(ma.OPENAI_MAX_OUTPUT_TOKENS, 32000)
        self.assertEqual(ma.OPENAI_IMAGE_DETAIL, "original")
        cfg = ma.OpenAIAdapter().config()
        self.assertEqual((cfg["model_code"], cfg["model"], cfg["inference"]["effort"]), ("M09", "gpt-6.1-sol", "high"))
        self.assertEqual(br.config_id(cfg), PINNED_M09_CONFIG_ID)

    def test_m08_configuration_unchanged(self):
        self.assertEqual(br.config_id(ma.ClaudeAdapter().config()), PINNED_M08_CONFIG_ID)

    def test_registered_with_separate_billing(self):
        self.assertIsInstance(ma.get_adapter("M09"), ma.OpenAIAdapter)
        self.assertIsInstance(ma.get_adapter("M09", mock=True), ma.MockOpenAIAdapter)
        self.assertEqual(ma.get_adapter("M09").billing_account, "openai_api")
        self.assertNotEqual(ma.get_adapter("M09").billing_account, ma.get_adapter("M08").billing_account)
        self.assertNotIn("M09", ma.PLANNED_ADAPTERS)


class TestRequestConstruction(unittest.TestCase):
    def setUp(self):
        self.jpeg = b"\xff\xd8\xff\xe0 fake jpeg bytes"
        self.prompt = bg.build_prompt("PT-R1", "NOT_APPLICABLE", "ALTITUDE_TEST")
        self.req = ma.openai_request(base64.standard_b64encode(self.jpeg).decode(), self.prompt)

    def test_exact_parameters(self):
        r = self.req
        self.assertEqual(set(r), {"model", "input", "reasoning", "max_output_tokens", "text", "store", "truncation",
                                  "service_tier", "prompt_cache_options", "extra_body"})
        self.assertEqual(r["model"], "gpt-6.1-sol")
        self.assertEqual(r["reasoning"], {"effort": "high"})
        self.assertEqual(r["max_output_tokens"], 32000)
        self.assertIs(r["store"], False)
        self.assertEqual(r["prompt_cache_options"], {"mode": "explicit"})
        self.assertEqual(r["extra_body"], {"image": {"context_compression": "disabled"}})
        self.assertEqual((r["service_tier"], r["truncation"]), ("default", "disabled"))

    def test_no_tools_chaining_or_extra_instructions(self):
        for key in ("tools", "tool_choice", "instructions", "previous_response_id", "conversation", "include",
                    "temperature", "top_p", "metadata", "background", "prompt_cache_key"):
            self.assertNotIn(key, self.req)

    def test_image_then_prompt_like_m08(self):
        content = self.req["input"][0]["content"]
        self.assertEqual([c["type"] for c in content], ["input_image", "input_text"])
        self.assertEqual(content[0]["detail"], "original")
        self.assertTrue(content[0]["image_url"].startswith("data:image/jpeg;base64,"))
        self.assertEqual(base64.standard_b64decode(content[0]["image_url"].split(",", 1)[1]), self.jpeg)
        self.assertEqual(content[1]["text"], self.prompt)
        m08 = bg.build_request("x", self.prompt)["messages"][0]["content"]
        self.assertEqual([c["type"] for c in m08], ["image", "text"])

    def test_schema_is_the_m08_schema_enforced_strictly(self):
        fmt = self.req["text"]["format"]
        self.assertEqual((fmt["type"], fmt["strict"]), ("json_schema", True))
        self.assertIs(fmt["schema"], bg.OUTPUT_SCHEMA)
        self.assertEqual(bg.build_request("x", "p")["output_config"]["format"]["schema"], fmt["schema"])


class TestPromptAndPreprocessingEquivalence(unittest.TestCase):
    def test_prompt_template_and_schema_hashes_match_m08(self):
        m08, m09 = ma.ClaudeAdapter().config(), ma.OpenAIAdapter().config()
        self.assertEqual(m08["prompt_template_sha256"], m09["prompt_template_sha256"])
        self.assertEqual(m08["output_schema_sha256"], m09["output_schema_sha256"])
        self.assertEqual(m08["coordinates"], m09["coordinates"])
        shared = {k: v for k, v in m08["preprocessing"].items() if k != "server_resize_guard"}
        self.assertEqual({k: m09["preprocessing"][k] for k in shared}, shared)

    @unittest.skipUnless((REPO / "benchmark_results" / "lowalt_pilot_ptr1_27m_v1").exists(), "M08 results not present")
    def test_upload_bytes_and_prompts_identical_to_frozen_m08_trials(self):
        rows = br.load_manifest(REPO / "manifests" / "manifest.csv", REPO, ["LA_PT-R1_27m", "LA_PT-02_64m"])
        exps = {"LA_PT-R1_27m": "lowalt_pilot_ptr1_27m_v1", "LA_PT-02_64m": "lowalt_pilot_pt02_64m_v1"}
        for row in rows:
            img = br.prepare_image(row, ma.OpenAIAdapter())
            records = sorted((REPO / "benchmark_results" / exps[row["image_id"]]).rglob("record.json"))
            self.assertEqual(len(records), 10)
            for p in records:
                rec = json.loads(p.read_text())
                self.assertEqual(rec["image"]["upload_sha256"], img["meta"]["upload_sha256"], p)
                self.assertEqual((p.parent / "prompt.txt").read_text(encoding="utf-8"), img["prompt"], p)
            self.assertEqual((img["meta"]["upload_width"], img["meta"]["upload_height"]), (2212, 1659))
            self.assertEqual(img["meta"]["api_detail"], "original")
            self.assertFalse(img["meta"]["api_internal_size_observable"])

    def test_image_token_estimate(self):
        self.assertEqual(ma.openai_image_tokens(2212, 1659), 4368)   # 70 x 52 patches x 1.2


class TestCost(unittest.TestCase):
    def setUp(self):
        self.a = ma.OpenAIAdapter()

    def test_list_prices(self):
        self.assertAlmostEqual(self.a.cost_usd({"input_tokens": 6000, "output_tokens": 2000}), 0.032)
        self.assertAlmostEqual(self.a.cost_usd({"input_tokens": 100_000, "output_tokens": 100_000}), 1.2)

    def test_cached_and_cache_write_tokens(self):
        usage = {"input_tokens": 10_000, "cached_input_tokens": 4_000, "cache_write_tokens": 2_000, "output_tokens": 0}
        self.assertAlmostEqual(self.a.cost_usd(usage), (4_000 * 2 + 4_000 * 0.10 + 2_000 * 2.50) / 1e6)

    def test_long_context_surcharge(self):
        self.assertAlmostEqual(self.a.cost_usd({"input_tokens": 300_000, "output_tokens": 1000}),
                               (300_000 * 2 * 2 + 1000 * 10 * 1.5) / 1e6)

    def test_reasoning_tokens_are_part_of_output(self):
        a, _ = adapter([make_response(output_tokens=2500, reasoning_tokens=2000)])
        res = run_once(a)
        self.assertEqual((res.usage["output_tokens"], res.usage["reasoning_tokens"]), (2500, 2000))
        self.assertAlmostEqual(a.cost_usd(res.usage), (6000 * 2 + 2500 * 10) / 1e6)

    def test_worst_case_uses_full_output_budget(self):
        a = ma.OpenAIAdapter()
        prompt = bg.build_prompt("PT-R1", "NOT_APPLICABLE", "ALTITUDE_TEST")
        worst = br.worst_case_usd(a, prompt, None, a.image_tokens(2212, 1659))
        self.assertAlmostEqual(worst, a.cost_usd({"input_tokens": a.estimate_input_tokens(prompt, 4368),
                                                  "output_tokens": 32000}))
        self.assertGreater(worst, 0.32)


# ------------------------------------------------------------------ response handling and retries

class TestResponses(unittest.TestCase):
    def test_success(self):
        a, client = adapter()
        res = run_once(a)
        self.assertEqual((res.status, res.stop_reason), ("ok", "completed"))
        self.assertEqual(json.loads(res.text)["trees"][0]["tree_id"], "T001")
        self.assertEqual(res.provider_response["model"], "gpt-6.1-sol")
        self.assertIs(res.provider_response["store"], False)
        self.assertEqual(res.provider_response["image_context_compression"], "disabled")
        self.assertEqual(res.raw["image"], {"context_compression": "disabled"})
        self.assertEqual(len(client.calls), 1)

    def test_refusal(self):
        a, _ = adapter([make_response(refusal="I can't help with that.")])
        res = run_once(a)
        self.assertEqual((res.status, res.stop_reason, res.text), ("ok", "refusal", ""))
        self.assertIn("can't help", res.stop_detail)

    def test_incomplete_max_output_tokens(self):
        a, _ = adapter([make_response(text='{"site": "PT', status="incomplete", reason="max_output_tokens")])
        res = run_once(a)
        self.assertEqual((res.stop_reason, res.stop_detail), ("max_tokens", "max_output_tokens"))

    def test_content_filter_is_a_refusal(self):
        a, _ = adapter([make_response(text="", status="incomplete", reason="content_filter")])
        self.assertEqual(run_once(a).stop_reason, "refusal")

    def test_failed_response_keeps_usage(self):
        a, _ = adapter([make_response(status="failed", error={"code": "server_error", "message": "boom"})])
        res = run_once(a)
        self.assertEqual(res.status, "error")
        self.assertTrue(res.possibly_billed)
        self.assertEqual(res.usage["input_tokens"], 6000)

    def test_substituted_model_stops_the_run(self):
        a, _ = adapter([make_response(model="gpt-6-astra")])
        res = run_once(a)
        self.assertEqual(res.status, "error")
        self.assertIn("served model", res.error)
        with self.assertRaises(ma.FatalAdapterError):
            run_once(a)

    def test_image_compression_applied_stops_the_run(self):
        a, _ = adapter([make_response(compression="enabled")])
        self.assertIn("context compression", run_once(a).error)
        with self.assertRaises(ma.FatalAdapterError):
            run_once(a)

    def test_rate_limit_retried_then_success(self):
        a, client = adapter([status_error(openai.RateLimitError, 429, "rate_limit_exceeded"), make_response()])
        res = run_once(a)
        self.assertEqual(res.status, "ok")
        self.assertEqual(len(res.retries), 1)
        self.assertEqual(len(client.calls), 2)

    def test_insufficient_quota_is_fatal(self):
        a, client = adapter([status_error(openai.RateLimitError, 429, "insufficient_quota", "You exceeded your quota")])
        with self.assertRaises(ma.FatalAdapterError):
            run_once(a)
        self.assertEqual(len(client.calls), 1)

    def test_bad_request_is_fatal_not_retried(self):
        a, client = adapter([status_error(openai.BadRequestError, 400, "unsupported_value", "detail 'original' unsupported")])
        with self.assertRaises(ma.FatalAdapterError):
            run_once(a)
        self.assertEqual(len(client.calls), 1)

    def test_server_errors_exhaust_retries(self):
        a, client = adapter([status_error(openai.InternalServerError, 500)] * 5)
        res = run_once(a)
        self.assertEqual((res.status, len(res.retries), len(client.calls)), ("error", 4, 5))
        self.assertFalse(res.possibly_billed)

    def test_connection_error_retried(self):
        a, _ = adapter([connection_error(), make_response()])
        self.assertEqual(run_once(a).status, "ok")

    def test_midstream_failure_not_retried_and_possibly_billed(self):
        a, client = adapter([("midstream", connection_error())])
        res = run_once(a)
        self.assertEqual((res.status, res.possibly_billed, len(client.calls)), ("error", True, 1))

    def test_each_trial_is_an_independent_identical_request(self):
        a, client = adapter()
        run_once(a)
        run_once(a)
        self.assertEqual(client.calls[0], client.calls[1])
        self.assertNotIn("previous_response_id", client.calls[1])


class TestSecrets(unittest.TestCase):
    def test_redaction(self):
        self.assertNotIn("SECRET", ma.redact_secrets(f"Incorrect API key provided: {FAKE_KEY}."))
        self.assertNotIn("abcd", ma.redact_secrets("Incorrect API key provided: sk-proj-****abcd"))
        self.assertNotIn("tok", ma.redact_secrets("Authorization: Bearer tok123"))
        self.assertEqual(ma.redact_secrets("plain message"), "plain message")

    def test_authentication_error_is_fatal_and_redacted(self):
        a, _ = adapter([status_error(openai.AuthenticationError, 401, "invalid_api_key",
                                     f"Incorrect API key provided: {FAKE_KEY}")])
        with self.assertRaises(ma.FatalAdapterError) as cm:
            run_once(a)
        self.assertNotIn("SECRET", str(cm.exception))

    @staticmethod
    def env(**values):
        """Patch only the OpenAI variables; the rest of the environment (needed for SSL on Windows) stays."""
        patcher = mock.patch.dict(os.environ, {})
        patcher.start()
        for k in ("OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_ORG_ID", "OPENAI_PROJECT_ID"):
            os.environ.pop(k, None)
        os.environ.update(values)
        return patcher

    def test_missing_key_is_fatal(self):
        patcher = self.env()
        try:
            with mock.patch.object(bg, "load_env", lambda: None), self.assertRaises(ma.FatalAdapterError):
                ma.OpenAIAdapter().prepare()
        finally:
            patcher.stop()

    def test_redirected_base_url_refused(self):
        patcher = self.env(OPENAI_API_KEY=FAKE_KEY, OPENAI_BASE_URL="https://proxy.example.com/v1")
        try:
            with mock.patch.object(bg, "load_env", lambda: None), self.assertRaises(ma.FatalAdapterError):
                ma.OpenAIAdapter().prepare()
        finally:
            patcher.stop()

    def test_client_reads_key_from_environment_only(self):
        patcher = self.env(OPENAI_API_KEY=FAKE_KEY)
        self.addCleanup(patcher.stop)
        with mock.patch.object(bg, "load_env", lambda: None):
            a = ma.OpenAIAdapter()
            a.prepare()                      # builds a client; makes no request
            self.assertEqual(str(a._client.base_url).rstrip("/"), ma.OPENAI_BASE_URL)
            self.assertEqual(a._client.max_retries, 0)
            self.assertNotIn(FAKE_KEY, json.dumps(a.config()))

    def test_gitignore_excludes_env(self):
        lines = (REPO / ".gitignore").read_text().splitlines()
        self.assertIn(".env", lines)
        self.assertIn("OPENAI_API_KEY=", (REPO / ".env.example").read_text())


# ------------------------------------------------------------------ runner integration (temporary folders only)

class RunnerTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        rows = []
        for i, (image_id, site) in enumerate([("IMG_A", "PT-02"), ("IMG_B", "PT-R1")]):
            path = self.tmp / "imgs" / f"{image_id}.jpg"
            path.parent.mkdir(exist_ok=True)
            Image.new("RGB", (320, 240), (40 * i, 120, 60)).save(path, quality=90)
            rows.append({"image_id": image_id, "image_path": f"imgs/{image_id}.jpg", "site": site, "week": "WEEK 3",
                         "condition": "BEFORE", "altitude_m": "95.0", "coverage_scope": "UNKNOWN",
                         "collection": "standard", "width": "320", "height": "240"})
        self.manifest = self.tmp / "manifest.csv"
        with self.manifest.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        self.out = self.tmp / "results"
        self.ledger = self.tmp / "openai_ledger.jsonl"

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def rows(self, ids=("IMG_A", "IMG_B")):
        return br.load_manifest(self.manifest, self.tmp, list(ids))

    def run_m09(self, a, ids=("IMG_A",), runs=3, exp="EXP_M09", **kw):
        kw.setdefault("max_usd", 50.0)
        kw.setdefault("dry_run", False)
        return br.run_benchmark(self.rows(ids), a, self.out, exp, runs=runs, manifest_path=self.manifest,
                                ledger_path=self.ledger, log=quiet, **kw)

    def records(self, exp="EXP_M09"):
        return [json.loads(p.read_text()) for p in sorted((self.out / exp).rglob("record.json"))]


class TestRunnerWithOpenAI(RunnerTest):
    def test_trials_recorded_with_provenance(self):
        a, client = adapter()
        report = self.run_m09(a)
        self.assertEqual((report["trials_run"], report["model_config_id"]), (3, PINNED_M09_CONFIG_ID))
        recs = self.records()
        self.assertEqual([r["status"] for r in recs], ["success"] * 3)
        r = recs[0]
        self.assertEqual((r["model"], r["model_code"], r["adapter"]), ("gpt-6.1-sol", "M09", "openai-responses"))
        self.assertEqual(r["inference_settings"]["effort"], "high")
        self.assertEqual(r["pricing"]["source"], ma.GPT_6_1_SOL_PRICING["source"])
        self.assertEqual(r["provider_response"]["image_context_compression"], "disabled")
        self.assertEqual(r["image"]["api_detail"], "original")
        self.assertEqual(r["usage"]["reasoning_tokens"], 2000)
        self.assertEqual(r["cost_basis"], "actual_usage")
        self.assertAlmostEqual(r["estimated_cost_usd"], a.cost_usd(r["usage"]), places=6)
        self.assertEqual(r["prompt_template_sha256"], ma.ClaudeAdapter().config()["prompt_template_sha256"])
        adir = self.out / "EXP_M09" / PINNED_M09_CONFIG_ID / "IMG_A" / "trial_01" / "attempt_01"
        for name in ("request.json", "prompt.txt", "response_raw.json", "detections.json", "annotated.jpg", "record.json"):
            self.assertTrue((adir / name).exists(), name)
        with Image.open(adir / "annotated.jpg") as im:
            self.assertEqual(im.size, (320, 240))     # drawn on the original image
        self.assertEqual(len(client.calls), 3)
        self.assertEqual(len({json.dumps(c, sort_keys=True) for c in client.calls}), 1)
        long = list(csv.DictReader((self.out / "EXP_M09" / "counts_long.csv").open()))
        self.assertEqual([row["tree_count"] for row in long], ["3", "3", "3"])

    def test_ledger_actual_costs(self):
        a, _ = adapter()
        self.run_m09(a)
        s = br.SpendLedger(self.ledger).summary()
        self.assertEqual(s["attempts"], 3)
        self.assertAlmostEqual(s["confirmed_usd"], 3 * (6000 * 2 + 2500 * 10) / 1e6, places=6)
        self.assertEqual((s["estimated_usd"], s["unresolved_usd"]), (0, 0))

    def test_invalid_coordinates_and_empty_detections(self):
        a, _ = adapter([make_response(text=trees_json(bbox=[10, 10, 1200, 50])),
                        make_response(text=trees_json(n=0))])
        self.run_m09(a, runs=2)
        recs = self.records()
        self.assertEqual([r["status"] for r in recs], ["invalid", "success"])
        self.assertIn("outside 0..1000", " ".join(recs[0]["validation"]["errors"]))
        self.assertEqual(recs[1]["tree_count"], 0)

    def test_refusal_and_truncation_statuses(self):
        a, _ = adapter([make_response(refusal="no"),
                        make_response(text='{"si', status="incomplete", reason="max_output_tokens")])
        self.run_m09(a, runs=2, pause_after=0)
        recs = self.records()
        self.assertEqual([r["status"] for r in recs], ["refusal", "max_tokens"])
        self.assertEqual(recs[1]["stop_detail"], "max_output_tokens")
        self.assertTrue(all(r["cost_basis"] == "actual_usage" for r in recs))

    def test_failed_response_charged_from_reported_usage(self):
        a, _ = adapter([make_response(status="failed", error={"code": "server_error", "message": "x"}), make_response()])
        self.run_m09(a, runs=1)
        recs = self.records()
        self.assertEqual([r["status"] for r in recs], ["error", "success"])
        self.assertEqual(recs[0]["cost_basis"], "actual_usage (error response)")
        self.assertTrue((self.out / "EXP_M09" / PINNED_M09_CONFIG_ID / "IMG_A" / "trial_01" / "attempt_01"
                         / "response_raw.json").exists())

    def test_interrupt_then_resume(self):
        a, _ = adapter([make_response(), KeyboardInterrupt()])
        report = self.run_m09(a)
        self.assertEqual(report["stopped"], "interrupted by user")
        interrupted = [r for r in self.records() if r["status"] == "interrupted"]
        self.assertEqual(len(interrupted), 1)
        self.assertTrue(interrupted[0]["possibly_billed"])
        b, client = adapter()
        report = self.run_m09(b, resume=True)
        self.assertEqual(report["incomplete"], {})
        self.assertEqual(len(client.calls), 2)        # trial 2 (new attempt) and trial 3; trial 1 kept
        self.assertEqual(len([r for r in self.records() if r["status"] == "success"]), 3)

    def test_budget_guard_uses_openai_worst_case(self):
        a, client = adapter()
        prompt = bg.build_prompt("PT-02", "WEEK 3", "BEFORE")
        worst = br.worst_case_usd(a, prompt, None, a.image_tokens(320, 240))
        report = self.run_m09(a, runs=3, max_usd=worst * 0.99)
        self.assertTrue(report["stopped"].startswith("budget"))
        self.assertEqual(client.calls, [])

    def test_secret_never_written(self):
        a, _ = adapter([status_error(openai.AuthenticationError, 401, "invalid_api_key",
                                     f"Incorrect API key provided: {FAKE_KEY}")])
        report = self.run_m09(a, runs=1)
        self.assertTrue(report["stopped"].startswith("error"))
        for p in list(self.tmp.rglob("*")):
            if p.is_file() and p.suffix in (".json", ".jsonl", ".csv", ".txt"):
                self.assertNotIn("SECRET", p.read_text(encoding="utf-8", errors="ignore"), p)

    def test_m09_cannot_write_into_another_models_experiment(self):
        br.run_benchmark(self.rows(("IMG_A",)), ma.MockAdapter(), self.out, "SHARED", runs=1, max_usd=50,
                         dry_run=False, ledger_path=self.tmp / "m.jsonl", log=quiet)
        a, client = adapter()
        with self.assertRaises(SystemExit):
            self.run_m09(a, exp="SHARED")
        self.assertEqual(client.calls, [])

    def test_dry_run_contacts_nobody_and_writes_nothing(self):
        with mock.patch.object(openai, "OpenAI", side_effect=AssertionError("client must not be created")):
            br.main(["--manifest", str(self.manifest), "--image-root", str(self.tmp), "--model", "M09",
                     "--image-ids", "IMG_A,IMG_B", "--runs", "10", "--max-usd", "5", "--experiment-id", "DRY",
                     "--output-dir", str(self.out), "--dry-run"])
        self.assertFalse(self.out.exists())
        report = br.run_benchmark(self.rows(), ma.OpenAIAdapter(), self.out, "DRY", runs=10, max_usd=5,
                                  dry_run=True, ledger_path=self.ledger, log=quiet)
        self.assertEqual((report["planned_trials"], report["model_config_id"]), (20, PINNED_M09_CONFIG_ID))
        self.assertFalse(self.out.exists() or self.ledger.exists())

    def test_mock_cli_rehearsal(self):
        br.main(["--manifest", str(self.manifest), "--image-root", str(self.tmp), "--model", "M09", "--mock",
                 "--image-ids", "IMG_A", "--runs", "2", "--max-usd", "5", "--experiment-id", "REH",
                 "--output-dir", str(self.out), "--execute"])
        cfg = json.loads(next(self.out.rglob("model_config.json")).read_text())
        self.assertEqual((cfg["model_code"], cfg["model"], cfg["adapter"]), ("M09", "gpt-6.1-sol", "mock"))
        self.assertTrue((self.out / "mock_spend_ledger.jsonl").exists())


class TestPilotProtocol(RunnerTest):
    """The planned pilot: one paid trial first, then resume the same experiment to 10 trials."""

    def files(self, trial):
        d = self.out / "EXP_M09" / PINNED_M09_CONFIG_ID / "IMG_A" / f"trial_{trial:02d}"
        return {p.relative_to(d).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(d.rglob("*")) if p.is_file()}

    def test_one_trial_then_resume_to_ten(self):
        a, client = adapter()
        report = self.run_m09(a, runs=1)
        self.assertEqual((report["trials_run"], report["incomplete"]), (1, {}))
        first = self.files(1)
        with self.assertRaises(SystemExit):                 # repeating without --resume is refused
            self.run_m09(adapter()[0], runs=10)
        b, client2 = adapter()
        report = self.run_m09(b, runs=10, resume=True)
        self.assertEqual((report["trials_run"], report["incomplete"]), (9, {}))
        self.assertEqual(len(client2.calls), 9)
        self.assertEqual(self.files(1), first)               # trial 1 untouched
        recs = self.records()
        self.assertEqual(sorted(r["trial"] for r in recs), list(range(1, 11)))
        self.assertTrue(all(r["attempt"] == 1 for r in recs))
        self.assertEqual(len({r["attempt_uid"] for r in recs}), 10)
        self.assertEqual(br.SpendLedger(self.ledger).summary()["attempts"], 10)
        long = list(csv.DictReader((self.out / "EXP_M09" / "counts_long.csv").open()))
        self.assertEqual(len(long), 10)
        again, client3 = adapter()
        self.assertEqual(self.run_m09(again, runs=10, resume=True)["trials_run"], 0)
        self.assertEqual(client3.calls, [])

    def test_request_record_holds_hashes_not_data(self):
        a, _ = adapter()
        self.run_m09(a, runs=1)
        req = json.loads(next((self.out / "EXP_M09").rglob("request.json")).read_text())["api_request"]
        img = req["input"][0]["content"][0]
        rec = self.records()[0]
        self.assertIn(rec["image"]["upload_sha256"], img["image_url"])
        self.assertNotIn(";base64,/9j", img["image_url"])
        self.assertIn(rec["prompt_sha256"], req["input"][0]["content"][1]["text"])
        self.assertEqual({k: req[k] for k in ("model", "reasoning", "max_output_tokens", "store", "image", "stream")},
                         {"model": "gpt-6.1-sol", "reasoning": {"effort": "high"}, "max_output_tokens": 32000,
                          "store": False, "image": {"context_compression": "disabled"}, "stream": True})
        self.assertEqual(img["detail"], "original")
        self.assertNotIn("summary", req["reasoning"])

    def test_pause_after_one_does_not_stop_successful_trials_on_resume(self):
        a, _ = adapter()
        self.run_m09(a, runs=1, pause_after=1)
        b, client = adapter()
        report = self.run_m09(b, runs=10, resume=True, pause_after=1)
        self.assertEqual((report["stopped"], report["trials_run"], len(client.calls)), (None, 9, 9))
        self.assertEqual(report["incomplete"], {})

    def test_pause_after_one_stops_on_first_incomplete_response(self):
        a, client = adapter([make_response(text='{"si', status="incomplete", reason="max_output_tokens")])
        report = self.run_m09(a, runs=10, pause_after=1)
        self.assertTrue(report["stopped"].startswith("paused"))
        self.assertEqual(len(client.calls), 1)

    def test_unsupported_setting_stops_without_changing_it(self):
        a, client = adapter([status_error(openai.BadRequestError, 400, "unsupported_parameter",
                                          "image.context_compression is not supported for this model")])
        report = self.run_m09(a, runs=10)
        self.assertTrue(report["stopped"].startswith("error"))
        self.assertEqual(len(client.calls), 1)
        self.assertEqual(self.records()[0]["status"], "error")
        self.assertEqual(self.records()[0]["estimated_cost_usd"], 0.0)


class TestWireFormat(unittest.TestCase):
    """Capture the HTTP request the SDK actually builds (mock transport; nothing leaves the machine)."""

    def test_serialized_body_matches_responses_api_reference(self):
        import httpx2
        captured = {}

        def handler(request):
            captured["url"], captured["body"] = str(request.url), json.loads(request.content)
            return httpx2.Response(400, json={"error": {"code": "captured", "message": "offline capture"}})

        client = openai.OpenAI(api_key=FAKE_KEY, base_url=ma.OPENAI_BASE_URL, max_retries=0,
                               http_client=httpx2.Client(transport=httpx2.MockTransport(handler)))
        a = ma.OpenAIAdapter(client=client)
        with self.assertRaises(ma.FatalAdapterError):       # a rejected request stops the run
            run_once(a, prompt="PROMPT", jpeg=b"\xff\xd8bytes")
        b = captured["body"]
        self.assertEqual(captured["url"], "https://api.openai.com/v1/responses")
        self.assertEqual(sorted(b), ["image", "input", "max_output_tokens", "model", "prompt_cache_options",
                                     "reasoning", "service_tier", "store", "stream", "text", "truncation"])
        self.assertEqual(b["image"], {"context_compression": "disabled"})
        self.assertEqual(b["reasoning"], {"effort": "high"})
        self.assertEqual(b["input"][0]["content"][0]["detail"], "original")
        self.assertEqual(b["text"]["format"]["schema"], bg.OUTPUT_SCHEMA)
        self.assertNotIn("SECRET", json.dumps(b))
        self.assertEqual(b, {**ma.OpenAIAdapter().request_record(b"\xff\xd8bytes", "PROMPT"),
                             "input": b["input"]})


class TestEvaluationCompatibility(RunnerTest):
    def test_m08_and_m09_compared_on_the_same_image(self):
        br.run_benchmark(self.rows(), ma.MockAdapter(), self.out, "EXP_M08", runs=3, max_usd=50, dry_run=False,
                         ledger_path=self.tmp / "m08.jsonl", log=quiet)
        a, _ = adapter(default=None)
        self.run_m09(a, ids=("IMG_A", "IMG_B"), runs=2)
        res = ev.run(self.out, None, self.tmp / "eval", self.tmp / "review", self.manifest, self.tmp,
                     figures=False, log=quiet)
        self.assertEqual(len(res["units"]), 4)
        with (self.tmp / "eval" / "model_comparison.csv").open() as f:
            comp = list(csv.DictReader(f))
        self.assertEqual(sorted({r["model_config_id"][:3] for r in comp}), ["M08", "M09"])
        self.assertEqual(len(comp), 4)
        with (self.tmp / "eval" / "cross_model_agreement.csv").open() as f:
            cross = list(csv.DictReader(f))
        self.assertEqual(len(cross), 2 * 3)                          # 2 images x 3 thresholds
        self.assertTrue(all(r["trial_pairs"] == "6" for r in cross))  # 3 M08 trials x 2 M09 trials
        report = (self.tmp / "eval" / "evaluation_report.md").read_text(encoding="utf-8")
        self.assertIn("Comparison between model configurations", report)
        self.assertIn("not computationally equivalent", report)

    def test_comparison_figures_with_and_without_reference(self):
        br.run_benchmark(self.rows(), ma.MockAdapter(), self.out, "EXP_M08", runs=2, max_usd=50, dry_run=False,
                         ledger_path=self.tmp / "m08.jsonl", log=quiet)
        a, _ = adapter()
        self.run_m09(a, ids=("IMG_A", "IMG_B"), runs=2)
        review = self.tmp / "review"
        ev.run(self.out, None, self.tmp / "eval", review, self.manifest, self.tmp, figures=False, log=quiet)
        with (review / "reference_boxes.csv").open("a", newline="") as f:          # IMG_A gets a complete reference
            csv.writer(f).writerows([[f"H{i}", "IMG_A", 10 + 60 * i, 20, 50 + 60 * i, 80, "TREE", "", "", ""]
                                     for i in range(1, 4)])
        rows = list(csv.DictReader((review / "reference_status.csv").open()))
        for r in rows:
            r["reference_status"] = "COMPLETE" if r["image_id"] == "IMG_A" else "NONE"
        with (review / "reference_status.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        res = ev.run(self.out, None, self.tmp / "eval", review, self.manifest, self.tmp, figures=True,
                     full_res_maps=False, log=quiet)
        for image_id in ("IMG_A", "IMG_B"):
            paths = res["figures"][f"comparison/{image_id}"]
            self.assertEqual([p.name for p in paths], ["K_model_comparison.png", "L_models_side_by_side_iou050.png"])
            self.assertTrue(all(p.exists() for p in paths))
        acc_units = {u.key for u, _ in res["accuracy"]}
        self.assertEqual({k.split("/")[1] for k in acc_units}, {"IMG_A"})        # no accuracy for IMG_B

    def test_single_model_produces_no_comparison(self):
        a, _ = adapter()
        self.run_m09(a)
        ev.run(self.out, None, self.tmp / "eval", self.tmp / "review", self.manifest, self.tmp,
               figures=False, log=quiet)
        self.assertFalse((self.tmp / "eval" / "model_comparison.csv").exists())


if __name__ == "__main__":
    unittest.main()
