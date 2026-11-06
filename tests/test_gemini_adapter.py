"""
Offline tests for M10 (Google Gemini 3.1 Pro Preview). The real google-genai SDK runs on top of an in-memory
httpx.MockTransport: requests are serialized and responses parsed by the SDK itself, but nothing leaves the
machine, no API key is read from .env, and no charges are possible.

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
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import benchmark_runner as br  # noqa: E402
import box_generation as bg  # noqa: E402
import evaluate_consistency as ev  # noqa: E402
import import_deepforest as idf  # noqa: E402
import model_adapters as ma  # noqa: E402
from test_deepforest_import import DeepForestFixture, make_df_run  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
PINNED_M10_CONFIG_ID = "M10_high_177c41fb69"
FAKE_KEY = "AIzaFAKEtestKEY0123456789abcdefghijklmnop"
PROTECTED = ["benchmark_results", "spend_ledger", "human_review", "manifests", "evaluation_results",
             "evaluation_results_lowalt_v1_deepforest"]
PILOT = {"LA_PT-R1_27m": "lowalt_pilot_ptr1_27m_m10_v1", "LA_PT-02_64m": "lowalt_pilot_pt02_64m_m10_v1"}


def _snapshot():
    out = {}
    for d in PROTECTED:
        results_only = d.startswith("evaluation_results")       # figures are large; hash the result tables
        for p in sorted((REPO / d).rglob("*")):
            if p.is_file() and (not results_only or p.suffix in (".csv", ".json", ".md")):
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


def b64(data: str) -> bytes:
    """The SDK sends inline data as URL-safe base64 (accepted by the API); decode either alphabet."""
    return base64.urlsafe_b64decode(data.replace("+", "-").replace("/", "_") + "=" * (-len(data) % 4))


def trees_json(n=3, site="PT-02", week="WEEK 3", condition="BEFORE", bbox=None):
    return json.dumps({"site": site, "week": week, "condition": condition,
                       "trees": [{"tree_id": f"T{i + 1:03d}", "bbox": bbox or [100 * i, 100, 100 * i + 80, 180]}
                                 for i in range(n)]})


def gresp(text=None, finish="STOP", prompt=3300, candidates=400, thoughts=6000, image_tokens=2240, cached=None,
          model="gemini-3.1-pro-preview", block=None, no_candidates=False, usage=True, thought_part=False,
          total=None):
    """A generateContent response body in the REST (camelCase) form the API returns."""
    body = {"modelVersion": model, "responseId": "resp-gem-001"}
    if not no_candidates:
        parts = []
        if thought_part:
            parts.append({"text": "internal reasoning text", "thought": True})
        parts.append({"text": trees_json() if text is None else text, "thoughtSignature": "c2lnbmF0dXJl"})
        body["candidates"] = [{"content": {"role": "model", "parts": parts}, "finishReason": finish, "index": 0}]
    if block:
        body["promptFeedback"] = {"blockReason": block}
    if usage:
        u = {"promptTokenCount": prompt, "candidatesTokenCount": candidates,
             "totalTokenCount": total if total is not None else prompt + candidates + (thoughts or 0)}
        if thoughts is not None:
            u["thoughtsTokenCount"] = thoughts
        if cached is not None:
            u["cachedContentTokenCount"] = cached
        details = [{"modality": "TEXT", "tokenCount": prompt - (image_tokens or 0)}]
        if image_tokens is not None:
            details.append({"modality": "IMAGE", "tokenCount": image_tokens})
        u["promptTokensDetails"] = details
        body["usageMetadata"] = u
    return (200, body)


def gerror(code, status, message="error"):
    return (code, {"error": {"code": code, "message": message, "status": status}})


class FakeGemini:
    """httpx transport handler: records every request and replays queued responses (default: a valid one)."""

    def __init__(self, items=None, default=None):
        self.items = list(items or [])
        self.default = default or gresp()
        self.requests = []

    def __call__(self, request: httpx.Request):
        self.requests.append({"url": str(request.url), "headers": dict(request.headers),
                              "body": json.loads(request.content)})
        item = self.items.pop(0) if self.items else self.default
        if isinstance(item, Exception):
            raise item
        if callable(item):
            item = item(request)
        status, body = item
        return httpx.Response(status, json=body)


def adapter(items=None, default=None, sleeps=None):
    fake = FakeGemini(items, default)
    a = ma.GeminiAdapter(transport=httpx.MockTransport(fake), api_key=FAKE_KEY,
                         sleep=(sleeps.append if sleeps is not None else (lambda s: None)))
    return a, fake


def run_once(a, prompt="p", jpeg=b"\xff\xd8jpeg"):
    return a.run(jpeg, prompt, {"image_id": "IMG", "trial": 1, "attempt": 1})


# ------------------------------------------------------------------ configuration

class TestFixedConfiguration(unittest.TestCase):
    def test_model_thinking_resolution_and_limits(self):
        cfg = ma.GeminiAdapter().config()
        inf = cfg["inference"]
        self.assertEqual((cfg["model"], cfg["model_code"]), ("gemini-3.1-pro-preview", "M10"))
        self.assertEqual(inf["thinking"], {"thinking_level": "HIGH", "include_thoughts": False, "thinking_budget": None})
        self.assertEqual(inf["effort"], "high")
        self.assertEqual(inf["media_resolution"]["level"], "MEDIA_RESOLUTION_ULTRA_HIGH")
        self.assertEqual(inf["max_output_tokens"], bg.MAX_TOKENS)
        self.assertEqual(inf["max_output_tokens"], 32000)
        self.assertIn("none", inf["tools"])
        self.assertEqual(inf["system_instruction"], "none")

    def test_config_id_pinned_and_stable(self):
        a, b = ma.GeminiAdapter(), ma.get_adapter("M10")
        self.assertEqual(br.config_id(a.config()), PINNED_M10_CONFIG_ID)
        self.assertEqual(br.config_id(b.config()), PINNED_M10_CONFIG_ID)
        self.assertNotEqual(br.config_id(ma.MockGeminiAdapter().config()), PINNED_M10_CONFIG_ID)  # adapter name differs

    def test_frozen_prompt_schema_and_preprocessing_shared_with_m08_m09(self):
        g, c, o = ma.GeminiAdapter().config(), ma.ClaudeAdapter().config(), ma.OpenAIAdapter().config()
        for k in ("prompt_template_sha256", "output_schema_sha256", "coordinates"):
            self.assertEqual(g[k], c[k], k)
            self.assertEqual(g[k], o[k], k)
        self.assertEqual(g["prompt_template_sha256"], "e53504b32a5931a597ebd8499aba89f602d90b720d8a0bb9d9ea92cfc96c100f")
        self.assertEqual(g["output_schema_sha256"], "963a771dd008a68fbca9cb2b42e762d9109fb40cd52423d8b706b7b8e2bc8e38")
        transport_only = {"provenance", "server_resize_guard"}
        self.assertEqual({k: v for k, v in g["preprocessing"].items() if k not in transport_only},
                         {k: v for k, v in c["preprocessing"].items() if k not in transport_only})

    def test_adapter_registry_and_ledger_account(self):
        self.assertIsInstance(ma.get_adapter("M10"), ma.GeminiAdapter)
        self.assertIsInstance(ma.get_adapter("M10", mock=True), ma.MockGeminiAdapter)
        self.assertNotIn("M10", ma.PLANNED_ADAPTERS)
        self.assertEqual(ma.GeminiAdapter.billing_account, "gemini_api")
        self.assertNotIn(ma.GeminiAdapter.billing_account, {ma.ClaudeAdapter.billing_account,
                                                            ma.OpenAIAdapter.billing_account})

    def test_pricing_recorded(self):
        p = ma.GEMINI_3_1_PRO_PREVIEW_PRICING
        self.assertEqual((p["input_per_mtok"], p["output_per_mtok"], p["cached_input_per_mtok"]), (2.0, 12.0, 0.2))
        self.assertIn("2026-11-05", p["source"])


# ------------------------------------------------------------------ wire format (real SDK, captured offline)

class TestWireFormat(unittest.TestCase):
    def test_serialized_request_matches_record_and_settings(self):
        a, fake = adapter()
        jpeg = b"\xff\xd8\xff\xe0exact-jpeg-bytes"
        r = run_once(a, prompt="THE PROMPT", jpeg=jpeg)
        self.assertEqual(r.status, "ok")
        (req,) = fake.requests
        self.assertEqual(req["url"], ma.GEMINI_ENDPOINT)
        self.assertEqual(req["url"], "https://generativelanguage.googleapis.com/v1beta/models/"
                                     "gemini-3.1-pro-preview:generateContent")
        body = req["body"]
        self.assertEqual(sorted(body), ["contents", "generationConfig"])        # no tools, system instruction, cache
        img, txt = body["contents"][0]["parts"]
        self.assertEqual(b64(img["inlineData"]["data"]), jpeg)     # exact bytes
        self.assertEqual(img["inlineData"]["mime_type"], "image/jpeg")
        self.assertEqual(img["mediaResolution"], {"level": "MEDIA_RESOLUTION_ULTRA_HIGH"})
        self.assertEqual(txt, {"text": "THE PROMPT"})
        gc = body["generationConfig"]
        self.assertEqual(gc["thinkingConfig"], {"include_thoughts": False, "thinking_level": "HIGH"})
        self.assertEqual(gc["responseJsonSchema"], bg.OUTPUT_SCHEMA)
        self.assertEqual((gc["responseMimeType"], gc["maxOutputTokens"], gc["candidateCount"]),
                         ("application/json", 32000, 1))
        # The recorded request equals the real one, apart from the image data and prompt text (hashed).
        rec = a.request_record(jpeg, "THE PROMPT")
        expected = json.loads(json.dumps(rec["body"]))
        expected["contents"][0]["parts"][0]["inlineData"]["data"] = img["inlineData"]["data"]
        expected["contents"][0]["parts"][1]["text"] = "THE PROMPT"
        self.assertEqual(body, expected)
        self.assertIn(hashlib.sha256(jpeg).hexdigest(), json.dumps(rec))
        self.assertNotIn(FAKE_KEY, json.dumps(rec))
        self.assertEqual(req["headers"]["x-goog-api-key"], FAKE_KEY)            # sent only as a header

    def test_environment_cannot_redirect_or_swap_credentials(self):
        fake = FakeGemini()
        env = {"GEMINI_API_KEY": FAKE_KEY, "GOOGLE_API_KEY": "AIzaOTHERkeyThatMustNotBeUsed0000000000",
               "GOOGLE_GEMINI_BASE_URL": "https://example.invalid/", "GOOGLE_GENAI_USE_VERTEXAI": "true"}
        with mock.patch.dict(os.environ, env), mock.patch.object(bg, "load_env", lambda: None):
            a = ma.GeminiAdapter(transport=httpx.MockTransport(fake))
            run_once(a)
        self.assertTrue(fake.requests[0]["url"].startswith("https://generativelanguage.googleapis.com/v1beta/"))
        self.assertEqual(fake.requests[0]["headers"]["x-goog-api-key"], FAKE_KEY)

    def test_missing_key_is_fatal_without_network(self):
        fake = FakeGemini()
        with mock.patch.dict(os.environ, {}, clear=True), mock.patch.object(bg, "load_env", lambda: None):
            with self.assertRaises(ma.FatalAdapterError):
                run_once(ma.GeminiAdapter(transport=httpx.MockTransport(fake)))
        self.assertEqual(fake.requests, [])


# ------------------------------------------------------------------ responses

class TestResponses(unittest.TestCase):
    def test_valid_response_usage_and_raw_body_preserved(self):
        a, fake = adapter(default=gresp(cached=0))
        r = run_once(a)
        self.assertEqual((r.status, r.stop_reason, r.request_id), ("ok", "completed", "resp-gem-001"))
        self.assertEqual(json.loads(r.text)["trees"][0]["tree_id"], "T001")
        u = r.usage
        self.assertEqual((u["input_tokens"], u["output_tokens"], u["thoughts_tokens"], u["image_input_tokens"]),
                         (3300, 6400, 6000, 2240))
        self.assertEqual(u["output_tokens_basis"], "candidates + thoughts")
        self.assertTrue(r.usage_complete)
        self.assertEqual(r.raw["http"]["status_code"], 200)
        self.assertEqual(r.raw["http"]["body"], gresp(cached=0)[1])          # exact server body
        self.assertEqual(r.provider_response["media_resolution_check"], "consistent with ULTRA_HIGH")
        self.assertEqual(r.provider_response["model_version"], "gemini-3.1-pro-preview")

    def test_thought_parts_never_enter_the_answer(self):
        a, _ = adapter(default=gresp(thought_part=True))
        self.assertNotIn("internal reasoning", run_once(a).text)

    def test_max_tokens(self):
        a, _ = adapter(default=gresp(text='{"site": "PT', finish="MAX_TOKENS"))
        r = run_once(a)
        self.assertEqual((r.status, r.stop_reason), ("ok", "max_tokens"))

    def test_safety_and_prompt_block_are_refusals(self):
        for item in (gresp(text="", finish="SAFETY"), gresp(no_candidates=True, block="PROHIBITED_CONTENT")):
            a, _ = adapter(default=item)
            r = run_once(a)
            self.assertEqual((r.status, r.stop_reason), ("ok", "refusal"))
            self.assertTrue(r.stop_detail)

    def test_no_candidates_or_unusual_finish_is_an_error(self):
        for item in (gresp(no_candidates=True), gresp(finish="MALFORMED_RESPONSE"), gresp(finish="OTHER")):
            a, _ = adapter(default=item)
            r = run_once(a)
            self.assertEqual(r.status, "error")
            self.assertTrue(r.possibly_billed)
            self.assertTrue(r.usage_complete)                                # priced from reported usage

    def test_missing_usage_is_recorded_not_invented(self):
        a, _ = adapter(default=gresp(usage=False))
        r = run_once(a)
        self.assertFalse(r.usage_complete)
        self.assertEqual(r.usage, {"usage_reported": False})
        self.assertEqual(r.provider_response["media_resolution_check"], "not verifiable: image token count not reported")

    def test_thoughts_missing_uses_total(self):
        a, _ = adapter(default=gresp(thoughts=None, total=3300 + 400 + 5000))
        r = run_once(a)
        self.assertEqual((r.usage["output_tokens"], r.usage["output_tokens_basis"]),
                         (5400, "total - prompt - tool_use_prompt"))

    def test_lower_media_resolution_halts_the_run(self):
        a, fake = adapter(default=gresp(image_tokens=1120))
        r = run_once(a)
        self.assertEqual(r.status, "error")
        self.assertIn("MEDIA_RESOLUTION_ULTRA_HIGH was not applied", r.error)
        with self.assertRaises(ma.FatalAdapterError):
            run_once(a)
        self.assertEqual(len(fake.requests), 1)

    def test_other_served_model_halts_the_run(self):
        a, _ = adapter(default=gresp(model="gemini-3.6-flash"))
        self.assertIn("served model", run_once(a).error)
        with self.assertRaises(ma.FatalAdapterError):
            run_once(a)


class TestErrors(unittest.TestCase):
    def test_unsupported_setting_is_fatal_and_not_retried(self):
        for code, status in ((400, "INVALID_ARGUMENT"), (401, "UNAUTHENTICATED"), (403, "PERMISSION_DENIED"),
                             (404, "NOT_FOUND")):
            a, fake = adapter([gerror(code, status, "media_resolution ultra_high is not supported")])
            with self.assertRaises(ma.FatalAdapterError):
                run_once(a)
            self.assertEqual(len(fake.requests), 1)

    def test_rate_limit_and_overload_retried_with_backoff(self):
        sleeps = []
        a, fake = adapter([gerror(429, "RESOURCE_EXHAUSTED"), gerror(503, "UNAVAILABLE")], sleeps=sleeps)
        r = run_once(a)
        self.assertEqual(r.status, "ok")
        self.assertEqual(len(fake.requests), 3)
        self.assertEqual([x["retry"] for x in r.retries], [1, 2])
        self.assertEqual(len(sleeps), 2)

    def test_retries_exhausted_not_billed(self):
        a, fake = adapter(default=gerror(503, "UNAVAILABLE"))
        r = run_once(a)
        self.assertEqual((r.status, r.possibly_billed), ("error", False))
        self.assertEqual(len(fake.requests), ma.GeminiAdapter.MAX_RETRIES + 1)

    def test_server_error_possibly_billed_and_not_retried(self):
        for item in (gerror(500, "INTERNAL"), gerror(504, "DEADLINE_EXCEEDED")):
            a, fake = adapter([item])
            r = run_once(a)
            self.assertEqual((r.status, r.possibly_billed, len(fake.requests)), ("error", True, 1))

    def test_connection_failure_retried_read_timeout_not(self):
        req = httpx.Request("POST", ma.GEMINI_ENDPOINT)
        a, fake = adapter([httpx.ConnectError("refused", request=req)])
        self.assertEqual(run_once(a).status, "ok")
        self.assertEqual(len(fake.requests), 2)
        b, fake2 = adapter([httpx.ReadTimeout("timed out", request=req)])
        r = run_once(b)
        self.assertEqual((r.status, r.possibly_billed, len(fake2.requests)), ("error", True, 1))

    def test_secrets_redacted_from_errors(self):
        a, _ = adapter([gerror(500, "INTERNAL", f"bad key {FAKE_KEY} and x-goog-api-key: {FAKE_KEY}")])
        r = run_once(a)
        self.assertNotIn(FAKE_KEY, r.error)
        self.assertIn("[REDACTED]", r.error)
        self.assertEqual(ma.redact_secrets(f"url?key={FAKE_KEY}&alt=json"), "url[REDACTED]&alt=json")


class TestCost(unittest.TestCase):
    def test_cost_formula(self):
        a = ma.GeminiAdapter()
        self.assertAlmostEqual(a.cost_usd({"input_tokens": 3300, "output_tokens": 6400}),
                               (3300 * 2 + 6400 * 12) / 1e6)
        self.assertAlmostEqual(a.cost_usd({"input_tokens": 3300, "cached_input_tokens": 3000, "output_tokens": 6400}),
                               (300 * 2 + 3000 * 0.2 + 6400 * 12) / 1e6)
        self.assertAlmostEqual(a.cost_usd({"input_tokens": 250_000, "output_tokens": 1000}),
                               (250_000 * 4 + 1000 * 18) / 1e6)

    def test_worst_case_reservation(self):
        a = ma.GeminiAdapter()
        prompt = bg.build_prompt("PT-R1", "NOT_APPLICABLE", "ALTITUDE_TEST")
        worst = br.worst_case_usd(a, prompt, None, a.image_tokens(2212, 1659))
        self.assertEqual(a.image_tokens(2212, 1659), 2240)
        self.assertAlmostEqual(worst, ((2240 + -(-len(prompt) // 3) + 2000) * 2 + 32000 * 12) / 1e6)
        self.assertAlmostEqual(worst, 0.394826, places=6)


# ------------------------------------------------------------------ runner

class GeminiRunnerTest(DeepForestFixture):
    """Temp project with a mock M08 experiment (EXP_LLM, 4 trials per image) as the input reference."""

    def run_m10(self, a, ids=("IMG_B",), runs=3, exp="EXP_M10", **kw):
        kw.setdefault("max_usd", 50.0)
        kw.setdefault("dry_run", False)
        return br.run_benchmark(br.load_manifest(self.manifest, self.tmp, list(ids)), a, self.results, exp,
                                runs=runs, manifest_path=self.manifest, ledger_path=self.tmp / "gemini_ledger.jsonl",
                                log=quiet, **kw)

    def records(self, exp="EXP_M10"):
        return [json.loads(p.read_text()) for p in sorted((self.results / exp).rglob("record.json"))]

    def tree_files(self, exp="EXP_M10"):
        return {p.relative_to(self.results).as_posix(): p.read_bytes() for p in (self.results / exp).rglob("*")
                if p.is_file()}


class TestRunnerWithGemini(GeminiRunnerTest):
    def body(self, n):
        return gresp(text=trees_json(n, site="PT-R1", week="NOT_APPLICABLE", condition="ALTITUDE_TEST"))

    def test_dry_run_makes_no_request_and_writes_nothing(self):
        a, fake = adapter()
        before = sorted(p.as_posix() for p in self.results.rglob("*"))
        report = self.run_m10(a, dry_run=True)
        self.assertEqual(fake.requests, [])
        self.assertEqual(sorted(p.as_posix() for p in self.results.rglob("*")), before)
        self.assertFalse((self.tmp / "gemini_ledger.jsonl").exists())
        self.assertEqual(report["input_checks"]["IMG_B"]["records_by_model"], {"M08": 4})

    def test_trials_recorded_with_provenance_and_identical_input(self):
        a, fake = adapter(default=self.body(3))
        report = self.run_m10(a)
        self.assertEqual((report["trials_run"], report["model_config_id"]), (3, PINNED_M10_CONFIG_ID))
        recs = self.records()
        self.assertEqual([r["status"] for r in recs], ["success"] * 3)
        r = recs[0]
        self.assertEqual((r["model"], r["model_code"], r["adapter"]), ("gemini-3.1-pro-preview", "M10",
                                                                       "google-genai-generate-content"))
        self.assertEqual(r["image"]["api_media_resolution"], "MEDIA_RESOLUTION_ULTRA_HIGH")
        chk = r["image"]["reference_input_check"]
        self.assertEqual((chk["records_compared"], chk["records_by_model"]), (4, {"M08": 4}))
        m08 = next(json.loads(p.read_text()) for p in (self.results / "EXP_LLM").rglob("record.json")
                   if "IMG_B" in p.as_posix())
        self.assertEqual(r["image"]["upload_sha256"], m08["image"]["upload_sha256"])
        self.assertEqual(r["prompt_sha256"], m08["prompt_sha256"])
        sent = b64(fake.requests[0]["body"]["contents"][0]["parts"][0]["inlineData"]["data"])
        self.assertEqual(hashlib.sha256(sent).hexdigest(), m08["image"]["upload_sha256"])
        self.assertEqual(r["cost_basis"], "actual_usage")
        self.assertAlmostEqual(r["estimated_cost_usd"], a.cost_usd(r["usage"]), places=6)
        self.assertEqual(r["usage"]["thoughts_tokens"], 6000)
        adir = self.results / "EXP_M10" / PINNED_M10_CONFIG_ID / "IMG_B" / "trial_01" / "attempt_01"
        for name in ("request.json", "prompt.txt", "response_raw.json", "detections.json", "annotated.jpg", "record.json"):
            self.assertTrue((adir / name).exists(), name)
        raw = json.loads((adir / "response_raw.json").read_text())
        self.assertEqual(raw["http"]["body"]["responseId"], "resp-gem-001")
        req = json.loads((adir / "request.json").read_text())["api_request"]
        self.assertEqual(req["body"]["generationConfig"]["thinkingConfig"]["thinking_level"], "HIGH")
        long = list(csv.DictReader((self.results / "EXP_M10" / "counts_long.csv").open()))
        self.assertEqual([row["tree_count"] for row in long], ["3", "3", "3"])

    def test_trials_are_independent_requests(self):
        a, fake = adapter(default=self.body(3))
        self.run_m10(a)
        bodies = [json.dumps(r["body"], sort_keys=True) for r in fake.requests]
        self.assertEqual(len(bodies), 3)
        self.assertEqual(len(set(bodies)), 1)                     # nothing from earlier trials is sent
        self.assertEqual(len(fake.requests[0]["body"]["contents"]), 1)
        self.assertEqual([c["role"] for c in fake.requests[2]["body"]["contents"]], ["user"])   # no model turns
        self.assertNotIn("[0, 100, 80, 180]", bodies[2])         # earlier trials' boxes never sent back

    def test_input_mismatch_or_missing_reference_refused(self):
        a, fake = adapter()
        rec = next((self.results / "EXP_LLM").rglob("IMG_B/trial_01/attempt_01/record.json"))
        r = json.loads(rec.read_text())
        r["image"]["upload_sha256"] = "0" * 64
        rec.write_text(json.dumps(r))
        with self.assertRaises(SystemExit):
            self.run_m10(a)
        self.assertEqual(fake.requests, [])
        shutil.rmtree(self.results / "EXP_LLM")
        report = self.run_m10(adapter()[0], dry_run=True)
        self.assertIn("error", report["input_checks"]["IMG_B"])

    def test_one_trial_then_resume_to_ten(self):
        a, _ = adapter(default=self.body(3))
        self.assertEqual(self.run_m10(a, runs=1)["trials_run"], 1)
        first = self.tree_files()
        with self.assertRaises(SystemExit):                     # repeating without --resume is refused
            self.run_m10(adapter()[0], runs=10)
        b, fake = adapter(default=self.body(4))
        report = self.run_m10(b, runs=10, resume=True)
        self.assertEqual((report["trials_run"], report["incomplete"], len(fake.requests)), (9, {}, 9))
        after = self.tree_files()
        self.assertTrue(all(after[k] == v for k, v in first.items() if "/trial_01/" in k))   # trial 1 untouched
        self.assertEqual(sorted(r["trial"] for r in self.records()), list(range(1, 11)))
        again, fake3 = adapter()
        self.assertEqual(self.run_m10(again, runs=10, resume=True)["trials_run"], 0)
        self.assertEqual(fake3.requests, [])

    def test_budget_guard_reserves_worst_case(self):
        a, fake = adapter()
        report = self.run_m10(a, max_usd=0.30)                  # below one worst-case reservation (~$0.39)
        self.assertTrue(report["stopped"].startswith("budget"))
        self.assertEqual(fake.requests, [])

    def test_incomplete_usage_books_worst_case(self):
        a, _ = adapter(default=gresp(text=trees_json(2, site="PT-R1", week="NOT_APPLICABLE",
                                                     condition="ALTITUDE_TEST"), usage=False))
        self.run_m10(a, runs=1)
        r = self.records()[0]
        self.assertEqual(r["status"], "success")
        self.assertEqual(r["cost_basis"], "worst_case_bound (provider usage incomplete)")
        self.assertEqual(r["estimated_cost_usd"], r["pre_request_worst_case_usd"])
        self.assertTrue(r["possibly_billed"])

    def test_unsupported_setting_stops_run_at_no_cost(self):
        a, fake = adapter([gerror(400, "INVALID_ARGUMENT", "ultra_high media resolution is not supported")])
        report = self.run_m10(a, runs=10)
        self.assertTrue(report["stopped"].startswith("error"))
        self.assertEqual(len(fake.requests), 1)
        r = self.records()[0]
        self.assertEqual((r["status"], r["estimated_cost_usd"]), ("error", 0.0))

    def test_media_resolution_fallback_stops_run(self):
        a, fake = adapter(default=gresp(image_tokens=1120))
        report = self.run_m10(a, runs=10)
        self.assertTrue(report["stopped"].startswith("error"))
        self.assertEqual(len(fake.requests), 1)
        self.assertIn("not honoured", self.records()[0]["error"])

    def test_pause_on_consecutive_truncation(self):
        a, fake = adapter(default=gresp(text='{"si', finish="MAX_TOKENS"))
        report = self.run_m10(a, runs=10, pause_after=3)
        self.assertTrue(report["stopped"].startswith("paused"))
        self.assertEqual(len(fake.requests), 3)

    def test_cannot_write_into_another_models_experiment(self):
        a, fake = adapter()
        with self.assertRaises(SystemExit):
            self.run_m10(a, exp="EXP_LLM")
        self.do_import(self.df_run())                           # DeepForest experiment EXP_DF (M01)
        with self.assertRaises(SystemExit):
            self.run_m10(adapter()[0], exp="EXP_DF")
        self.assertEqual(fake.requests, [])

    def test_no_secret_in_any_written_file(self):
        a, _ = adapter([gerror(500, "INTERNAL", f"echo {FAKE_KEY}")], default=self.body(3))
        self.run_m10(a, runs=2)
        for p in list(self.results.rglob("*")) + [self.tmp / "gemini_ledger.jsonl"]:
            if p.is_file() and p.suffix in (".json", ".jsonl", ".csv", ".txt"):
                self.assertNotIn(FAKE_KEY, p.read_text(encoding="utf-8", errors="ignore"), p)


class TestPilotInputs(unittest.TestCase):
    """The real approved pilot images: M10 would send exactly the bytes and prompt M08 and M09 received.
    Reads the frozen records only; nothing is written."""

    @unittest.skipUnless((REPO / "benchmark_results" / "lowalt_pilot_ptr1_27m_v1").exists(), "pilot records absent")
    def test_pilot_upload_and_prompt_identical_to_frozen_m08_m09(self):
        rows = br.load_manifest(REPO / "manifests" / "manifest.csv", REPO, list(PILOT))
        for row in rows:
            image = br.prepare_image(row, ma.GeminiAdapter())
            chk = br.reference_input_check(image, row["image_id"], REPO / "benchmark_results", ("M08", "M09"))
            self.assertEqual(chk["records_by_model"], {"M08": 10, "M09": 10})
            self.assertEqual((image["meta"]["upload_width"], image["meta"]["upload_height"]), (2212, 1659))
            self.assertEqual(image["meta"]["api_mime_type"], "image/jpeg")

    def test_experiment_ids_free(self):
        for exp in PILOT.values():
            self.assertFalse((REPO / "benchmark_results" / exp).exists(), exp)


# ------------------------------------------------------------------ five-configuration evaluation

class TestFiveModelEvaluation(GeminiRunnerTest):
    def test_m10_joins_llm_and_deepforest_comparison(self):
        br.run_benchmark(br.load_manifest(self.manifest, self.tmp), ma.MockOpenAIAdapter(), self.results, "EXP_M09",
                         runs=3, max_usd=50, dry_run=False, manifest_path=self.manifest,
                         ledger_path=self.tmp / "o.jsonl", log=quiet)
        a, _ = adapter(default=gresp(text=trees_json(3, site="PT-R1", week="NOT_APPLICABLE",
                                                     condition="ALTITUDE_TEST", bbox=[30, 80, 190, 340])))
        self.run_m10(a, runs=3)
        self.do_import(self.df_run(exp="EXP_DF_M01"))
        self.do_import(self.df_run(exp="EXP_DF_M02", code="M02v2"))
        self.evaluate("eval_setup")                                        # creates human_review/ files
        with (self.review / "reference_boxes.csv").open("a", newline="") as f:   # IMG_B gets a COMPLETE reference
            csv.writer(f).writerows([["H01", "IMG_B", 10, 20, 60, 80, "TREE", "", "", ""],
                                     ["H02", "IMG_B", 9, 19, 61, 82, "TREE", "", "", ""]])
        rows = list(csv.DictReader((self.review / "reference_status.csv").open()))
        for r in rows:
            r["reference_status"] = "COMPLETE" if r["image_id"] == "IMG_B" else "NONE"
        with (self.review / "reference_status.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        res = self.evaluate(figures=True, full_res_maps=False)
        acc = [r for r in self.read("accuracy_vs_reference.csv") if r["experiment_id"] == "EXP_M10"]
        self.assertEqual(len(acc), 3 * 3)                                  # 3 trials x 3 thresholds
        freq = [r for r in self.read("reference_detection_frequency.csv") if r["experiment_id"] == "EXP_M10"]
        self.assertEqual({r["n_trials"] for r in freq}, {"3"})
        units = [u for u in res["units"] if u.image_id == "IMG_B"]
        self.assertEqual(sorted(u.model_config_id[:3] for u in units), ["M01", "M02", "M08", "M09", "M10"])
        cross = [r for r in self.read("cross_model_agreement.csv") if r["image_id"] == "IMG_B"
                 and r["iou_threshold"] == "0.50"]
        pairs = {frozenset((r["model_config_a"][:3], r["model_config_b"][:3])): int(r["trial_pairs"]) for r in cross}
        self.assertEqual(len(pairs), 10)                                   # 5 choose 2
        self.assertEqual(pairs[frozenset(("M10", "M08"))], 3 * 4)
        self.assertEqual(pairs[frozenset(("M10", "M09"))], 3 * 3)
        self.assertEqual(pairs[frozenset(("M10", "M01"))], 3)              # DeepForest never replicated
        self.assertEqual(pairs[frozenset(("M01", "M02"))], 1)
        m10 = next(r for r in self.read("count_consistency.csv") if r["experiment_id"] == "EXP_M10")
        self.assertEqual((m10["successful_trials"], m10["matches_counts_summary_csv"]), ("3", "True"))
        names = [p.name for p in res["figures"]["comparison/IMG_B"]]
        self.assertEqual(names, ["K_model_comparison.png", "L_models_side_by_side_iou050.png",
                                 "M_agreement_matrices.png", "N_geometry_coverage.png", "O_coverage_maps.png"])
        self.assertTrue(all(p.exists() for p in res["figures"]["comparison/IMG_B"]))
        self.assertIn("EXP_M10/IMG_B", res["figures"])                    # repeatability figures for M10
        geo = [r for r in self.read("geometry_summary.csv") if r["experiment_id"] == "EXP_M10"]
        self.assertEqual(len(geo), 3)


if __name__ == "__main__":
    unittest.main()
