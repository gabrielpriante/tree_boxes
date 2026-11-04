"""
Model adapters for the tree-count benchmark.

An adapter turns (preprocessed image bytes, prompt) into one raw model response.
The batch runner owns everything else: trial bookkeeping, budget, validation,
rendering, and CSV output, so every model shares the same result format.

Implemented:  M08 Claude Opus 5.5 (ClaudeAdapter), M09 OpenAI GPT-6.1 Sol (OpenAIAdapter),
              plus MockAdapter for offline tests.
Planned:      see PLANNED_ADAPTERS.
"""

import base64
import hashlib
import json
import math
import platform
import random
import re
import time
from dataclasses import dataclass, field

import box_generation as bg

_SECRET_PATTERNS = [re.compile(r"sk-[A-Za-z0-9_\-*.]{4,}"), re.compile(r"(?i)bearer\s+\S+")]


def redact_secrets(text):
    """Remove anything that looks like an API key (including masked fragments some APIs echo back)."""
    if not isinstance(text, str):
        return text
    for pattern in _SECRET_PATTERNS:
        text = pattern.sub("[REDACTED]", text)
    return text

# First-party Anthropic API list prices in USD per million tokens for claude-opus-5-5.
# Thinking tokens are billed as output tokens and are included in usage.output_tokens.
CLAUDE_OPUS_5_5_PRICING = {
    "currency": "USD",
    "input_per_mtok": 4.00,
    "output_per_mtok": 20.00,
    "cache_write_per_mtok": 5.00,
    "cache_read_per_mtok": 0.20,
    "source": "Anthropic API list pricing for claude-opus-5-5 (verified 2026-10-30)",
}

# Upper bound on image input tokens for high-resolution vision models.
IMAGE_TOKEN_UPPER_BOUND = bg.MAX_VISUAL_TOKENS
# Allowance for the structured-output system prompt and message framing.
REQUEST_OVERHEAD_TOKENS = 2000

PLANNED_ADAPTERS = {
    "M01": "DeepForest (existing implementation; results are frozen, adapter not implemented)",
    "M02": "DeepForest (existing implementation; results are frozen, adapter not implemented)",
    "M10": "Gemini (adapter not implemented)",
    "COPILOT": "Microsoft Copilot (programmatic endpoint not yet identified)",
}


class FatalAdapterError(Exception):
    """Configuration or authorization problem that would affect every request. Stops the run."""


@dataclass
class AdapterResult:
    status: str                     # "ok" (a response came back) or "error"
    raw: dict | None = None         # full JSON-serialisable response
    text: str | None = None         # concatenated text output
    stop_reason: str | None = None
    usage: dict = field(default_factory=dict)
    request_id: str | None = None
    possibly_billed: bool = False   # error happened after generation may have started
    error: str | None = None
    retries: list = field(default_factory=list)
    stop_detail: str | None = None  # provider-specific reason behind stop_reason, e.g. "content_filter"
    provider_response: dict = field(default_factory=dict)  # effective settings echoed by the provider


def _sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


class ModelAdapter:
    model_code = ""
    adapter_name = ""
    billing_account = ""  # adapters that share credits share a spending ledger
    pricing = CLAUDE_OPUS_5_5_PRICING

    def config(self) -> dict:
        """Everything that defines the experimental condition. Hashed into model_config_id."""
        raise NotImplementedError

    def software(self) -> dict:
        import PIL
        return {"python": platform.python_version(), "pillow": PIL.__version__}

    def prepare(self) -> None:
        """Called once before the first live request (credentials, clients)."""

    def max_output_tokens(self) -> int:
        raise NotImplementedError

    def image_tokens(self, width: int, height: int) -> int:
        """Input tokens this provider charges for an uploaded image of this size (estimate)."""
        return bg.count_image_tokens(width, height)

    def image_meta(self, upload: dict) -> dict:
        """Provider-specific facts about the upload, stored in every record's `image` block."""
        return {
            # Equal to the upload size: the upload is pre-resized to the API's documented target,
            # and the request sets oversized_image=error, so the API cannot downsize it further.
            "api_effective_width": upload["width"],
            "api_effective_height": upload["height"],
            "upload_visual_tokens": upload["visual_tokens"],
        }

    def request_record(self, jpeg_bytes: bytes, prompt: str) -> dict | None:
        """The exact request parameters sent, with the image and prompt replaced by their hashes."""
        return None

    def describe(self, cfg: dict) -> str:
        inf = cfg["inference"]
        return (f"{cfg['adapter']}: {cfg['model']}, effort={inf['effort']}, thinking={inf['thinking']['type']}, "
                f"max_tokens={inf['max_tokens']}, preprocessing v{cfg['preprocessing']['version']}")

    def estimate_input_tokens(self, prompt: str, image_tokens: int | None = None) -> int:
        """Conservative upper bound used before any request for this image has completed."""
        image = image_tokens if image_tokens is not None else IMAGE_TOKEN_UPPER_BOUND
        return image + math.ceil(len(prompt) / 3) + REQUEST_OVERHEAD_TOKENS

    def cost_usd(self, usage: dict) -> float:
        p = self.pricing
        return (
            usage.get("input_tokens", 0) * p["input_per_mtok"]
            + usage.get("output_tokens", 0) * p["output_per_mtok"]
            + usage.get("cache_creation_input_tokens", 0) * p["cache_write_per_mtok"]
            + usage.get("cache_read_input_tokens", 0) * p["cache_read_per_mtok"]
        ) / 1_000_000

    def run(self, jpeg_bytes: bytes, prompt: str, context: dict) -> AdapterResult:
        raise NotImplementedError


def claude_config(model_code: str, adapter_name: str) -> dict:
    return {
        "adapter": adapter_name,
        "model_code": model_code,
        "model": bg.MODEL,
        "inference": {
            "effort": bg.EFFORT,
            "thinking": {"type": "adaptive"},
            "max_tokens": bg.MAX_TOKENS,
            "streaming": True,
            "structured_output": "output_config.format json_schema",
        },
        "prompt_template_sha256": _sha256_text(bg.PROMPT_TEMPLATE),
        "output_schema_sha256": _sha256_text(json.dumps(bg.OUTPUT_SCHEMA, sort_keys=True)),
        "coordinates": "normalized 0-1000 [x_min, y_min, x_max, y_max], origin top-left",
        "preprocessing": {
            "version": bg.PREPROCESSING_VERSION,
            "exif_transpose": True,
            "color_mode": "RGB",
            "max_upload_edge_px": bg.MAX_UPLOAD_EDGE,
            "max_visual_tokens": bg.MAX_VISUAL_TOKENS,
            "resize": "PIL Image.resize, LANCZOS, to the documented API target size "
                      "(largest aspect-preserving size within both limits)",
            "server_resize_guard": "image transformations.oversized_image = error",
            "format": "JPEG",
            "jpeg_qualities_tried": list(bg.JPEG_QUALITIES),
            "max_image_bytes": bg.MAX_IMAGE_BYTES,
        },
    }


class ClaudeAdapter(ModelAdapter):
    """M08: Claude Opus 5.5 through the Anthropic Messages API, one independent request per call."""

    model_code = "M08"
    adapter_name = "anthropic-messages"
    billing_account = "anthropic_api"

    # Retries for failures that occur before generation starts (not billed).
    MAX_RETRIES = 4
    BACKOFF_BASE_S = 5.0
    BACKOFF_MAX_S = 120.0
    FATAL_STATUS = {400, 401, 403, 404, 413, 422}

    def __init__(self, client=None, sleep=time.sleep):
        self._client = client
        self._sleep = sleep

    def config(self) -> dict:
        return claude_config(self.model_code, self.adapter_name)

    def software(self) -> dict:
        import anthropic
        return {**super().software(), "anthropic": anthropic.__version__}

    def max_output_tokens(self) -> int:
        return bg.MAX_TOKENS

    def prepare(self) -> None:
        if self._client is not None:
            return
        import os
        import anthropic

        bg.load_env()
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise FatalAdapterError("ANTHROPIC_API_KEY is not set (expected in .env).")
        # SDK retries off: retries are handled here so each one is logged.
        self._client = anthropic.Anthropic(max_retries=0, timeout=900.0)

    def _backoff(self, n: int, exc) -> float:
        response = getattr(exc, "response", None)
        retry_after = response.headers.get("retry-after") if response is not None else None
        if retry_after:
            try:
                return min(float(retry_after), 300.0)
            except ValueError:
                pass
        return min(self.BACKOFF_BASE_S * 2 ** n, self.BACKOFF_MAX_S) + random.uniform(0, 1)

    def run(self, jpeg_bytes: bytes, prompt: str, context: dict) -> AdapterResult:
        import anthropic

        self.prepare()
        image_b64 = base64.standard_b64encode(jpeg_bytes).decode("ascii")
        params = bg.build_request(image_b64, prompt)
        retries = []

        for n in range(self.MAX_RETRIES + 1):
            started = False
            try:
                with self._client.messages.stream(**params) as stream:
                    started = True
                    message = stream.get_final_message()
            except anthropic.APIStatusError as e:
                if e.status_code in self.FATAL_STATUS:
                    raise FatalAdapterError(f"{type(e).__name__} ({e.status_code}): {e.message}") from e
                retryable = e.status_code in (408, 409, 429) or e.status_code >= 500
                error = f"{type(e).__name__} ({e.status_code}): {e.message}"
                exc = e
            except anthropic.APIConnectionError as e:
                retryable = True
                error = f"{type(e).__name__}: {e}"
                exc = e
            else:
                return AdapterResult(
                    status="ok",
                    raw=json.loads(message.to_json()),
                    text="".join(b.text for b in message.content if b.type == "text"),
                    stop_reason=message.stop_reason,
                    usage={
                        "input_tokens": message.usage.input_tokens or 0,
                        "output_tokens": message.usage.output_tokens or 0,
                        "cache_creation_input_tokens": message.usage.cache_creation_input_tokens or 0,
                        "cache_read_input_tokens": message.usage.cache_read_input_tokens or 0,
                    },
                    request_id=getattr(message, "_request_id", None),
                    retries=retries,
                )

            if started:
                # The stream opened, so tokens may have been generated and billed.
                # Not retried here; the runner re-checks the budget before another attempt.
                return AdapterResult(status="error", error=error, possibly_billed=True, retries=retries)
            if not retryable or n == self.MAX_RETRIES:
                return AdapterResult(status="error", error=error, retries=retries)
            wait = self._backoff(n, exc)
            retries.append({"retry": n + 1, "error": error, "wait_s": round(wait, 1)})
            self._sleep(wait)


# ---------------------------------------------------------------- M09: OpenAI GPT-6.1 Sol

OPENAI_MODEL = "gpt-6.1-sol"
OPENAI_REASONING_EFFORT = "high"
OPENAI_MAX_OUTPUT_TOKENS = bg.MAX_TOKENS          # 32,000, the M08 limit (model maximum: 128,000)
OPENAI_IMAGE_DETAIL = "original"                   # keep the 2212 x 1659 upload as is (see BENCHMARK.md)
OPENAI_IMAGE_CONTEXT_COMPRESSION = "disabled"      # request-level `image.context_compression`
OPENAI_SCHEMA_NAME = "tree_detections"             # name required by the API; transport only
OPENAI_BASE_URL = "https://api.openai.com/v1"
OPENAI_PATCH_PX = 32                               # GPT-5.6 / GPT-6 family patch size (images-vision guide)
OPENAI_IMAGE_TOKEN_MULTIPLIER = 1.2                # GPT-5.6 / GPT-6 family; gpt-6.1-sol not yet listed
OPENAI_MAX_PATCHES = 30_000                        # requests above this are rejected, not resized

# OpenAI API Standard-tier list prices for gpt-6.1-sol, USD per million tokens. Reasoning tokens are
# reported inside usage.output_tokens and billed as output. Prompts over 272K input tokens are billed
# at 2x input/cache rates and 1.5x output for the whole request (not reachable here).
GPT_6_1_SOL_PRICING = {
    "currency": "USD",
    "input_per_mtok": 2.00,
    "cached_input_per_mtok": 0.10,
    "cache_write_per_mtok": 2.50,
    "output_per_mtok": 10.00,
    "long_context_threshold_input_tokens": 272_000,
    "long_context_input_multiplier": 2.0,
    "long_context_output_multiplier": 1.5,
    "service_tier": "default (Standard)",
    "source": "developers.openai.com/api/docs/models/gpt-6.1-sol (verified 2026-11-04)",
}


def openai_image_tokens(width: int, height: int) -> int:
    """Documented patch formula for the GPT-5.6 / GPT-6 family at detail=original: 32 px patches x 1.2.
    Not yet documented for gpt-6.1-sol, so this is an estimate until the first trial reports usage."""
    patches = math.ceil(width / OPENAI_PATCH_PX) * math.ceil(height / OPENAI_PATCH_PX)
    return math.ceil(patches * OPENAI_IMAGE_TOKEN_MULTIPLIER)


def openai_request(image_b64: str, prompt: str) -> dict:
    """Responses API parameters for one independent M09 trial.

    Scientific content is identical to M08: the same preprocessed JPEG (sent first, as in M08), the same
    prompt text and the same JSON schema object. Everything else is transport or a protective setting:
    no instructions/system prompt, no tools, no conversation or previous_response_id, nothing stored,
    no prompt caching (explicit mode without breakpoints), no image context compression.
    """
    return {
        "model": OPENAI_MODEL,
        "input": [{
            "role": "user",
            "content": [
                {"type": "input_image", "image_url": f"data:image/jpeg;base64,{image_b64}",
                 "detail": OPENAI_IMAGE_DETAIL},
                {"type": "input_text", "text": prompt},
            ],
        }],
        "reasoning": {"effort": OPENAI_REASONING_EFFORT},
        "max_output_tokens": OPENAI_MAX_OUTPUT_TOKENS,
        "text": {"format": {"type": "json_schema", "name": OPENAI_SCHEMA_NAME,
                            "schema": bg.OUTPUT_SCHEMA, "strict": True}},
        "store": False,
        "truncation": "disabled",
        "service_tier": "default",
        "prompt_cache_options": {"mode": "explicit"},
        # Not yet a typed argument in the Python SDK; sent as documented in the REST reference.
        "extra_body": {"image": {"context_compression": OPENAI_IMAGE_CONTEXT_COMPRESSION}},
    }


def openai_config(model_code: str, adapter_name: str) -> dict:
    preprocessing = claude_config(model_code, adapter_name)["preprocessing"]
    preprocessing = {**preprocessing,
                     "provenance": "identical v2 pipeline and bytes as M08 (target size from Anthropic's limits)",
                     "server_resize_guard": f"input_image.detail = {OPENAI_IMAGE_DETAIL}; "
                                            f"image.context_compression = {OPENAI_IMAGE_CONTEXT_COMPRESSION}"}
    return {
        "adapter": adapter_name,
        "model_code": model_code,
        "model": OPENAI_MODEL,
        "inference": {
            "api": "OpenAI Responses API (POST /v1/responses)",
            "effort": OPENAI_REASONING_EFFORT,
            "reasoning": {"effort": OPENAI_REASONING_EFFORT, "summary": None},
            "max_output_tokens": OPENAI_MAX_OUTPUT_TOKENS,
            "streaming": True,
            "structured_output": f"text.format json_schema strict=true (name {OPENAI_SCHEMA_NAME!r})",
            "image_detail": OPENAI_IMAGE_DETAIL,
            "image_context_compression": OPENAI_IMAGE_CONTEXT_COMPRESSION,
            "store": False,
            "prompt_cache": "explicit mode, no breakpoints (no cache reads or writes)",
            "service_tier": "default",
            "truncation": "disabled",
            "tools": "none",
            "instructions": "none",
            "sampling": "provider defaults (temperature/top_p unsupported with reasoning)",
        },
        "prompt_template_sha256": _sha256_text(bg.PROMPT_TEMPLATE),
        "output_schema_sha256": _sha256_text(json.dumps(bg.OUTPUT_SCHEMA, sort_keys=True)),
        "coordinates": "normalized 0-1000 [x_min, y_min, x_max, y_max], origin top-left",
        "preprocessing": preprocessing,
    }


class _OpenAIBehaviour:
    """Pieces shared by the real M09 adapter and its mock (config, pricing, image accounting)."""

    model_code = "M09"
    pricing = GPT_6_1_SOL_PRICING

    def max_output_tokens(self) -> int:
        return OPENAI_MAX_OUTPUT_TOKENS

    def image_tokens(self, width: int, height: int) -> int:
        return openai_image_tokens(width, height)

    def image_meta(self, upload: dict) -> dict:
        w, h = upload["width"], upload["height"]
        patches = math.ceil(w / OPENAI_PATCH_PX) * math.ceil(h / OPENAI_PATCH_PX)
        return {
            "api_detail": OPENAI_IMAGE_DETAIL,
            # The API does not report the resolution it processes internally; token usage cannot prove it.
            "api_internal_size_observable": False,
            "api_patches_32px": patches,                                   # if 32 px patches apply (unverified)
            "api_estimated_image_tokens": openai_image_tokens(w, h),       # budget estimate only
        }

    def request_record(self, jpeg_bytes: bytes, prompt: str) -> dict:
        req = openai_request("", prompt)
        req.update(req.pop("extra_body"))   # the SDK merges extra_body into the top level of the JSON body
        content = req["input"][0]["content"]
        content[0]["image_url"] = (f"data:image/jpeg;base64,<omitted: {len(jpeg_bytes)} bytes, "
                                   f"sha256 {hashlib.sha256(jpeg_bytes).hexdigest()}>")
        content[1]["text"] = f"<prompt: {len(prompt)} characters, sha256 {_sha256_text(prompt)}>"
        req["stream"] = True   # added by client.responses.stream()
        return req

    def describe(self, cfg: dict) -> str:
        inf = cfg["inference"]
        return (f"{cfg['adapter']}: {cfg['model']}, reasoning.effort={inf['effort']}, "
                f"max_output_tokens={inf['max_output_tokens']}, detail={inf['image_detail']}, "
                f"store={inf['store']}, preprocessing v{cfg['preprocessing']['version']}")

    def cost_usd(self, usage: dict) -> float:
        p = self.pricing
        inp = usage.get("input_tokens", 0) or 0
        cached = usage.get("cached_input_tokens", 0) or 0
        written = usage.get("cache_write_tokens", 0) or 0
        out = usage.get("output_tokens", 0) or 0
        in_mult = out_mult = 1.0
        if inp > p["long_context_threshold_input_tokens"]:
            in_mult, out_mult = p["long_context_input_multiplier"], p["long_context_output_multiplier"]
        uncached = max(inp - cached - written, 0)
        return (in_mult * (uncached * p["input_per_mtok"] + cached * p["cached_input_per_mtok"]
                           + written * p["cache_write_per_mtok"])
                + out_mult * out * p["output_per_mtok"]) / 1_000_000


def _openai_usage(usage) -> dict:
    if usage is None:
        return {}
    u = usage.model_dump() if hasattr(usage, "model_dump") else dict(usage)
    inp = u.get("input_tokens_details") or {}
    out = u.get("output_tokens_details") or {}
    return {
        "input_tokens": u.get("input_tokens") or 0,
        "output_tokens": u.get("output_tokens") or 0,
        "reasoning_tokens": out.get("reasoning_tokens") or 0,
        "cached_input_tokens": inp.get("cached_tokens") or 0,
        "cache_write_tokens": inp.get("cache_write_tokens") or 0,
        "total_tokens": u.get("total_tokens") or 0,
    }


class OpenAIAdapter(_OpenAIBehaviour, ModelAdapter):
    """M09: GPT-6.1 Sol through the OpenAI Responses API, one independent request per call."""

    adapter_name = "openai-responses"
    billing_account = "openai_api"

    MAX_RETRIES = 4
    BACKOFF_BASE_S = 5.0
    BACKOFF_MAX_S = 120.0
    FATAL_STATUS = {400, 401, 403, 404, 413, 422}
    FATAL_ERROR_CODES = {"insufficient_quota", "billing_hard_limit_reached", "invalid_api_key"}

    def __init__(self, client=None, sleep=time.sleep):
        self._client = client
        self._sleep = sleep
        self._halt = None   # set when a response shows the experiment conditions were not honoured

    def config(self) -> dict:
        return openai_config(self.model_code, self.adapter_name)

    def software(self) -> dict:
        import openai
        return {**super().software(), "openai": openai.__version__}

    def prepare(self) -> None:
        if self._client is not None:
            return
        import os
        import openai

        bg.load_env()
        if not os.environ.get("OPENAI_API_KEY"):
            raise FatalAdapterError("OPENAI_API_KEY is not set (expected in .env).")
        base_url = os.environ.get("OPENAI_BASE_URL")
        if base_url and base_url.rstrip("/") != OPENAI_BASE_URL:
            raise FatalAdapterError("OPENAI_BASE_URL points away from api.openai.com; unset it for this benchmark.")
        # SDK retries off: retries are handled here so each one is logged. Long timeout for high effort.
        self._client = openai.OpenAI(max_retries=0, timeout=1800.0, base_url=OPENAI_BASE_URL)

    def _backoff(self, n: int, exc) -> float:
        response = getattr(exc, "response", None)
        retry_after = response.headers.get("retry-after") if response is not None else None
        if retry_after:
            try:
                return min(float(retry_after), 300.0)
            except ValueError:
                pass
        return min(self.BACKOFF_BASE_S * 2 ** n, self.BACKOFF_MAX_S) + random.uniform(0, 1)

    @staticmethod
    def _error_code(exc) -> str | None:
        body = getattr(exc, "body", None)
        if isinstance(body, dict):
            err = body.get("error", body)
            if isinstance(err, dict):
                return err.get("code") or err.get("type")
        return getattr(exc, "code", None)

    def _result(self, response, retries) -> AdapterResult:
        raw = json.loads(response.to_json())
        usage = _openai_usage(response.usage)
        served = response.model or ""
        compression = (raw.get("image") or {}).get("context_compression")
        provider = {"response_id": response.id, "model": served, "status": response.status,
                    "service_tier": raw.get("service_tier"), "store": raw.get("store"),
                    "reasoning": raw.get("reasoning"), "image_context_compression": compression,
                    "incomplete_details": raw.get("incomplete_details")}
        common = dict(raw=raw, usage=usage, request_id=getattr(response, "_request_id", None) or response.id,
                      retries=retries, provider_response=provider)

        problems = []
        if not served.startswith(OPENAI_MODEL):
            problems.append(f"served model {served!r} is not {OPENAI_MODEL}")
        if compression is not None and compression != OPENAI_IMAGE_CONTEXT_COMPRESSION:
            problems.append(f"image context compression was {compression!r}, requested {OPENAI_IMAGE_CONTEXT_COMPRESSION!r}")
        if problems:
            self._halt = "; ".join(problems)
            return AdapterResult(status="error", error=f"experiment condition not honoured: {self._halt}",
                                 possibly_billed=True, **common)

        if response.status == "completed":
            refusals = [c.refusal for item in response.output if getattr(item, "type", "") == "message"
                        for c in item.content if getattr(c, "type", "") == "refusal"]
            if refusals:
                return AdapterResult(status="ok", text="", stop_reason="refusal",
                                     stop_detail=redact_secrets(" ".join(refusals))[:500], **common)
            return AdapterResult(status="ok", text=response.output_text, stop_reason="completed", **common)
        if response.status == "incomplete":
            reason = getattr(response.incomplete_details, "reason", None)
            if reason == "max_output_tokens":
                return AdapterResult(status="ok", text=response.output_text, stop_reason="max_tokens",
                                     stop_detail=reason, **common)
            if reason == "content_filter":
                return AdapterResult(status="ok", text="", stop_reason="refusal", stop_detail=reason, **common)
            return AdapterResult(status="error", error=f"response incomplete: {reason}", possibly_billed=True,
                                 stop_detail=reason, **common)
        err = raw.get("error") or {}
        return AdapterResult(status="error", possibly_billed=True, stop_detail=response.status,
                             error=redact_secrets(f"response {response.status}: {err.get('code')} {err.get('message')}"),
                             **common)

    def run(self, jpeg_bytes: bytes, prompt: str, context: dict) -> AdapterResult:
        import openai

        if self._halt:
            raise FatalAdapterError(f"stopped: {self._halt}")
        self.prepare()
        params = openai_request(base64.standard_b64encode(jpeg_bytes).decode("ascii"), prompt)
        retries = []
        for n in range(self.MAX_RETRIES + 1):
            started = False
            try:
                with self._client.responses.stream(**params) as stream:
                    started = True
                    response = stream.get_final_response()
            except openai.APIStatusError as e:
                code = self._error_code(e)
                error = redact_secrets(f"{type(e).__name__} ({e.status_code}, {code}): {e.message}")
                if e.status_code in self.FATAL_STATUS or code in self.FATAL_ERROR_CODES:
                    raise FatalAdapterError(error) from None
                retryable = e.status_code in (408, 409, 429) or e.status_code >= 500
                exc = e
            except openai.APIConnectionError as e:
                retryable = True
                error = redact_secrets(f"{type(e).__name__}: {e}")
                exc = e
            except openai.OpenAIError as e:
                retryable = False
                error = redact_secrets(f"{type(e).__name__}: {e}")
                exc = e
            else:
                return self._result(response, retries)

            if started:
                # The stream opened, so tokens may have been generated and billed. Not retried here.
                return AdapterResult(status="error", error=error, possibly_billed=True, retries=retries)
            if not retryable or n == self.MAX_RETRIES:
                return AdapterResult(status="error", error=error, retries=retries)
            wait = self._backoff(n, exc)
            retries.append({"retry": n + 1, "error": error, "wait_s": round(wait, 1)})
            self._sleep(wait)


class MockAdapter(ModelAdapter):
    """Offline stand-in with Claude's config and pricing. Never touches the network.

    behaviors maps (image_id, trial, attempt) or (image_id, trial) to one of:
      ok, invalid, bad_json, refusal, max_tokens, error, midstream_error, interrupt, fatal
    """

    model_code = "M08"
    adapter_name = "mock"
    billing_account = "mock"

    def __init__(self, behaviors=None, input_tokens=6500, output_tokens=9000, interrupt_after=None):
        self.behaviors = behaviors or {}
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.interrupt_after = interrupt_after
        self.calls = []

    def config(self) -> dict:
        return claude_config(self.model_code, self.adapter_name)

    def max_output_tokens(self) -> int:
        return bg.MAX_TOKENS

    def _boxes(self, seed: str) -> list:
        rng = random.Random(seed)
        n = 25 + rng.randint(-3, 3)
        trees = []
        for i in range(n):
            x, y = rng.randint(0, 900), rng.randint(0, 900)
            trees.append({"tree_id": f"T{i + 1:03d}", "bbox": [x, y, x + rng.randint(20, 99), y + rng.randint(20, 99)]})
        return trees

    def run(self, jpeg_bytes: bytes, prompt: str, context: dict) -> AdapterResult:
        key3 = (context["image_id"], context["trial"], context["attempt"])
        behavior = self.behaviors.get(key3, self.behaviors.get(key3[:2], "ok"))
        self.calls.append({**context, "behavior": behavior})
        if self.interrupt_after is not None and len(self.calls) > self.interrupt_after:
            behavior = "interrupt"

        usage = {"input_tokens": self.input_tokens, "output_tokens": self.output_tokens}
        if behavior == "interrupt":
            raise KeyboardInterrupt
        if behavior == "fatal":
            raise FatalAdapterError("mock authentication failure")
        if behavior == "error":
            return AdapterResult(status="error", error="mock 529 overloaded (retries exhausted)",
                                 retries=[{"retry": i, "error": "mock 529", "wait_s": 0} for i in (1, 2)])
        if behavior == "midstream_error":
            return AdapterResult(status="error", error="mock connection dropped mid-stream", possibly_billed=True)

        seed = f"{context['image_id']}|{context['trial']}"
        body = {"site": "PT-02", "week": "WEEK 1", "condition": "BEFORE", "trees": self._boxes(seed)}
        stop_reason = "end_turn"
        if behavior == "invalid":
            body["trees"][0]["bbox"] = [10, 10, 1200, 50]   # out of range
            body["trees"][1]["bbox"] = [500, 500, 400, 600]  # x_min > x_max
        text = json.dumps(body)
        if behavior == "bad_json":
            text = text[:40]
        elif behavior == "max_tokens":
            text, stop_reason = text[:40], "max_tokens"
        elif behavior == "refusal":
            text, stop_reason = "", "refusal"

        raw = {"id": f"mock_{len(self.calls)}", "type": "message", "model": "mock",
               "stop_reason": stop_reason, "content": [{"type": "text", "text": text}], "usage": usage}
        return AdapterResult(status="ok", raw=raw, text=text, stop_reason=stop_reason,
                             usage=usage, request_id=raw["id"])


class MockOpenAIAdapter(_OpenAIBehaviour, MockAdapter):
    """Offline stand-in with the M09 config, pricing and image accounting. Never touches the network."""

    adapter_name = "mock"

    def config(self) -> dict:
        return openai_config(self.model_code, self.adapter_name)


def get_adapter(model_code: str, mock: bool = False, **kwargs) -> ModelAdapter:
    if model_code == "M08":
        return MockAdapter(**kwargs) if mock else ClaudeAdapter(**kwargs)
    if model_code == "M09":
        return MockOpenAIAdapter(**kwargs) if mock else OpenAIAdapter(**kwargs)
    if model_code in PLANNED_ADAPTERS:
        raise NotImplementedError(f"{model_code}: {PLANNED_ADAPTERS[model_code]}")
    raise ValueError(f"Unknown model code {model_code!r}")
