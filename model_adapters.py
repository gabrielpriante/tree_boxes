"""
Model adapters for the tree-count benchmark.

An adapter turns (preprocessed image bytes, prompt) into one raw model response.
The batch runner owns everything else: trial bookkeeping, budget, validation,
rendering, and CSV output, so every model shares the same result format.

Implemented:  M08 Claude Opus 5.5 (ClaudeAdapter), M09 OpenAI GPT-6.1 Sol (OpenAIAdapter),
              M10 Google Gemini 3.1 Pro Preview (GeminiAdapter),
              plus MockAdapter / MockOpenAIAdapter / MockGeminiAdapter for offline tests.
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

_SECRET_PATTERNS = [re.compile(r"sk-[A-Za-z0-9_\-*.]{4,}"), re.compile(r"(?i)bearer\s+\S+"),
                    re.compile(r"AIza[0-9A-Za-z_\-]{10,}"),                       # Google API keys
                    re.compile(r"(?i)(x-goog-api-key|[?&]key)\s*[:=]\s*[^\s&\"',}]+")]


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
    "M01": "DeepForest: local inference in urban-tree-vision-baselines, imported with import_deepforest.py "
           "(no API adapter)",
    "M02": "DeepForest: local inference in urban-tree-vision-baselines, imported with import_deepforest.py "
           "(no API adapter)",
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
    # False when the provider did not report enough usage to price the response. The runner then books the
    # pre-request worst-case bound instead of inventing a cost.
    usage_complete: bool = True


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


# ---------------------------------------------------------------- M10: Google Gemini 3.1 Pro Preview

GEMINI_MODEL = "gemini-3.1-pro-preview"            # preview endpoint; Google lists no dated snapshot
GEMINI_THINKING_LEVEL = "HIGH"                     # explicit; supported levels for this model: LOW, MEDIUM, HIGH
GEMINI_INCLUDE_THOUGHTS = False                    # no thought summaries requested
GEMINI_MEDIA_RESOLUTION = "MEDIA_RESOLUTION_ULTRA_HIGH"   # per-part only; not settable globally
GEMINI_MAX_OUTPUT_TOKENS = bg.MAX_TOKENS           # 32,000, the M08/M09 limit (model maximum: 65,536)
GEMINI_API_VERSION = "v1beta"
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/"
GEMINI_ENDPOINT = f"{GEMINI_BASE_URL}{GEMINI_API_VERSION}/models/{GEMINI_MODEL}:generateContent"
GEMINI_IMAGE_MIME = "image/jpeg"
GEMINI_TIMEOUT_S = 1800.0
# Documented per-image token budgets for Gemini 3 media resolution levels (media-resolution guide).
GEMINI_IMAGE_TOKENS = {"MEDIA_RESOLUTION_LOW": 280, "MEDIA_RESOLUTION_MEDIUM": 560,
                       "MEDIA_RESOLUTION_HIGH": 1120, "MEDIA_RESOLUTION_ULTRA_HIGH": 2240}
# A reported image token count at or below the HIGH budget means the image was not processed at ULTRA_HIGH.
GEMINI_FALLBACK_IMAGE_TOKENS = GEMINI_IMAGE_TOKENS["MEDIA_RESOLUTION_HIGH"]

# Gemini Developer API paid tier, Standard, USD per million tokens, for gemini-3.1-pro-preview.
# Output includes thinking tokens. Prompts over 200K tokens use the higher rates (not reachable here).
# No free tier for this model. Implicit-cache hits are billed at the cached rate.
GEMINI_3_1_PRO_PREVIEW_PRICING = {
    "currency": "USD",
    "input_per_mtok": 2.00,
    "output_per_mtok": 12.00,
    "cached_input_per_mtok": 0.20,
    "long_context_threshold_input_tokens": 200_000,
    "long_context_input_per_mtok": 4.00,
    "long_context_output_per_mtok": 18.00,
    "long_context_cached_input_per_mtok": 0.40,
    "service_tier": "standard (request leaves service_tier unset; SDK default is standard)",
    "free_tier": "not available for this model",
    "source": "aistudio.google.com/docs/pricing (Gemini 3.1 Pro Preview, Standard; verified 2026-11-05)",
}


def gemini_request_record(jpeg_bytes: bytes, prompt: str) -> dict:
    """The REST request google-genai sends for one M10 trial, with the image and prompt replaced by hashes.
    A test captures the SDK's real HTTP body offline and checks it equals this (apart from those two values)."""
    return {
        "method": "POST",
        "url": GEMINI_ENDPOINT,
        "body": {
            "contents": [{
                "parts": [
                    {"mediaResolution": {"level": GEMINI_MEDIA_RESOLUTION},
                     "inlineData": {"data": f"<omitted: {len(jpeg_bytes)} bytes, sha256 "
                                            f"{hashlib.sha256(jpeg_bytes).hexdigest()}>",
                                    "mime_type": GEMINI_IMAGE_MIME}},
                    {"text": f"<prompt: {len(prompt)} characters, sha256 {_sha256_text(prompt)}>"},
                ],
                "role": "user",
            }],
            "generationConfig": {
                "candidateCount": 1,
                "maxOutputTokens": GEMINI_MAX_OUTPUT_TOKENS,
                "responseMimeType": "application/json",
                "responseJsonSchema": bg.OUTPUT_SCHEMA,
                "thinkingConfig": {"include_thoughts": GEMINI_INCLUDE_THOUGHTS,
                                   "thinking_level": GEMINI_THINKING_LEVEL},
            },
        },
        "headers": "x-goog-api-key (never recorded), content-type application/json, google-genai SDK version headers",
    }


def gemini_config(model_code: str, adapter_name: str) -> dict:
    preprocessing = claude_config(model_code, adapter_name)["preprocessing"]
    preprocessing = {**preprocessing,
                     "provenance": "identical v2 pipeline and bytes as M08 and M09 (target size from Anthropic's "
                                   "limits); checked against their saved records before execution",
                     "server_resize_guard": f"image part mediaResolution = {GEMINI_MEDIA_RESOLUTION} "
                                            "(provider-internal encoding; not observable)"}
    return {
        "adapter": adapter_name,
        "model_code": model_code,
        "model": GEMINI_MODEL,
        "inference": {
            "api": f"Gemini Developer API generateContent ({GEMINI_API_VERSION}) via the official google-genai SDK",
            "effort": GEMINI_THINKING_LEVEL.lower(),
            "thinking": {"thinking_level": GEMINI_THINKING_LEVEL, "include_thoughts": GEMINI_INCLUDE_THOUGHTS,
                         "thinking_budget": None},
            "max_output_tokens": GEMINI_MAX_OUTPUT_TOKENS,
            "candidate_count": 1,
            "streaming": False,
            "structured_output": "responseMimeType application/json + responseJsonSchema (the frozen OUTPUT_SCHEMA, "
                                 "passed unchanged)",
            "media_resolution": {"scope": "image part (per-part; ultra_high is not available globally)",
                                 "level": GEMINI_MEDIA_RESOLUTION},
            "image_mime_type": GEMINI_IMAGE_MIME,
            "content_order": "image part, then prompt text part, one user turn (as M08/M09)",
            "system_instruction": "none",
            "tools": "none (no function calling, Google Search grounding, code execution or URL context)",
            "conversation_state": "none (stateless generateContent; no chat history, no cached_content)",
            "implicit_caching": "provider default (cannot be disabled); cache hits recorded from usage",
            "service_tier": "unset (SDK default: standard)",
            "safety_settings": "provider defaults",
            "sampling": "provider defaults (temperature, top_p, top_k, seed not set)",
            "sdk_retries": "disabled (HttpRetryOptions attempts=1); retries are logged by the adapter",
            "timeout_s": GEMINI_TIMEOUT_S,
        },
        "prompt_template_sha256": _sha256_text(bg.PROMPT_TEMPLATE),
        "output_schema_sha256": _sha256_text(json.dumps(bg.OUTPUT_SCHEMA, sort_keys=True)),
        "coordinates": "normalized 0-1000 [x_min, y_min, x_max, y_max], origin top-left",
        "preprocessing": preprocessing,
    }


def _gemini_usage(meta) -> tuple:
    """Normalise usageMetadata. Returns (usage, complete). Missing fields are recorded as missing, never invented.

    promptTokenCount includes implicit-cache hits (cachedContentTokenCount). Billed output = candidates +
    thoughts; if either is missing but totalTokenCount is present, output = total - prompt - tool-use prompt."""
    if meta is None:
        return {"usage_reported": False}, False
    u = meta.model_dump(mode="json", exclude_none=True) if hasattr(meta, "model_dump") else dict(meta)
    prompt, cand = u.get("prompt_token_count"), u.get("candidates_token_count")
    thoughts, total = u.get("thoughts_token_count"), u.get("total_token_count")
    tool = u.get("tool_use_prompt_token_count") or 0
    by_modality = {d.get("modality"): d.get("token_count") for d in u.get("prompt_tokens_details") or []}
    out = {"usage_reported": True, "raw_usage_metadata": u,
           "input_tokens": prompt, "cached_input_tokens": u.get("cached_content_token_count") or 0,
           "candidates_tokens": cand, "thoughts_tokens": thoughts, "reasoning_tokens": thoughts,
           "tool_use_prompt_tokens": tool, "total_tokens": total,
           "image_input_tokens": by_modality.get("IMAGE"), "text_input_tokens": by_modality.get("TEXT")}
    if cand is not None and thoughts is not None:
        out["output_tokens"], out["output_tokens_basis"] = cand + thoughts, "candidates + thoughts"
    elif total is not None and prompt is not None:
        out["output_tokens"], out["output_tokens_basis"] = total - prompt - tool, "total - prompt - tool_use_prompt"
    elif cand is not None and thoughts is None:
        out["output_tokens"], out["output_tokens_basis"] = None, "thoughts_token_count missing; cannot price thinking"
    complete = prompt is not None and out.get("output_tokens") is not None
    return out, complete


class _GeminiBehaviour:
    """Pieces shared by the real M10 adapter and its mock (config, pricing, image accounting)."""

    model_code = "M10"
    pricing = GEMINI_3_1_PRO_PREVIEW_PRICING
    # M10 must submit exactly the bytes and prompt M08/M09 received; the runner checks their saved records.
    input_reference_models = ("M08", "M09")

    def max_output_tokens(self) -> int:
        return GEMINI_MAX_OUTPUT_TOKENS

    def image_tokens(self, width: int, height: int) -> int:
        # Gemini 3 bills a fixed per-image budget per media-resolution level, not by pixel count.
        return GEMINI_IMAGE_TOKENS[GEMINI_MEDIA_RESOLUTION]

    def image_meta(self, upload: dict) -> dict:
        return {
            "api_media_resolution": GEMINI_MEDIA_RESOLUTION,
            "api_media_resolution_scope": "image part",
            "api_mime_type": GEMINI_IMAGE_MIME,
            # Google does not report the pixel size it processes; only the image token count is observable.
            "api_internal_size_observable": False,
            "api_documented_image_tokens": GEMINI_IMAGE_TOKENS[GEMINI_MEDIA_RESOLUTION],
        }

    def request_record(self, jpeg_bytes: bytes, prompt: str) -> dict:
        return gemini_request_record(jpeg_bytes, prompt)

    def describe(self, cfg: dict) -> str:
        inf = cfg["inference"]
        return (f"{cfg['adapter']}: {cfg['model']}, thinking_level={inf['thinking']['thinking_level']}, "
                f"max_output_tokens={inf['max_output_tokens']}, media_resolution={inf['media_resolution']['level']} "
                f"(image part), preprocessing v{cfg['preprocessing']['version']}")

    def cost_usd(self, usage: dict) -> float:
        p = self.pricing
        inp = usage.get("input_tokens") or 0
        cached = min(usage.get("cached_input_tokens") or 0, inp)
        out = usage.get("output_tokens") or 0
        long = inp > p["long_context_threshold_input_tokens"]
        rate_in = p["long_context_input_per_mtok"] if long else p["input_per_mtok"]
        rate_cached = p["long_context_cached_input_per_mtok"] if long else p["cached_input_per_mtok"]
        rate_out = p["long_context_output_per_mtok"] if long else p["output_per_mtok"]
        tool = usage.get("tool_use_prompt_tokens") or 0
        return ((inp - cached + tool) * rate_in + cached * rate_cached + out * rate_out) / 1_000_000


class GeminiAdapter(_GeminiBehaviour, ModelAdapter):
    """M10: Gemini 3.1 Pro Preview through generateContent (google-genai), one independent request per call."""

    adapter_name = "google-genai-generate-content"
    billing_account = "gemini_api"

    MAX_RETRIES = 4
    BACKOFF_BASE_S = 5.0
    BACKOFF_MAX_S = 120.0
    FATAL_STATUS = {400, 401, 403, 404, 413}
    RETRY_UNBILLED = {429, 503}         # rejected or overloaded before generation
    POSSIBLY_BILLED = {500, 504}        # failure may have occurred after generation started
    BLOCK_REASONS = {"SAFETY", "RECITATION", "BLOCKLIST", "PROHIBITED_CONTENT", "SPII", "LANGUAGE",
                     "IMAGE_SAFETY", "IMAGE_PROHIBITED_CONTENT", "IMAGE_RECITATION", "IMAGE_OTHER"}

    def __init__(self, client=None, sleep=time.sleep, transport=None, api_key=None):
        """client: a prepared genai.Client (tests). transport: an httpx transport used under the real SDK
        (tests use httpx.MockTransport, so nothing leaves the machine). api_key: tests only."""
        self._client = client
        self._sleep = sleep
        self._transport = transport
        self._api_key = api_key
        self._halt = None
        self._last_http = None

    def config(self) -> dict:
        return gemini_config(self.model_code, self.adapter_name)

    def software(self) -> dict:
        import httpx
        from google import genai
        return {**super().software(), "google-genai": genai.__version__, "httpx": httpx.__version__}

    def _capture(self, response) -> None:
        """httpx response hook: keep the exact response body (the SDK object keeps only parsed fields)."""
        response.read()
        try:
            body = response.json()
            text = json.dumps(body)
            redacted = redact_secrets(text)
            if redacted != text:            # e.g. an error message echoing a key: store the redacted form
                try:
                    body = json.loads(redacted)
                except ValueError:
                    body = {"redacted_body": redacted}
        except ValueError:
            body = {"non_json_body": redact_secrets(response.text[:5000])}
        keep = ("content-type", "date", "server-timing", "x-goog-request-id", "x-request-id")
        self._last_http = {"status_code": response.status_code,
                           "headers": {k: v for k, v in response.headers.items() if k.lower() in keep},
                           "body": body}

    def prepare(self) -> None:
        if self._client is not None:
            return
        import os

        import httpx
        from google import genai
        from google.genai import types

        key = self._api_key
        if key is None:
            bg.load_env()
            key = os.environ.get("GEMINI_API_KEY")
        if not key:
            raise FatalAdapterError("GEMINI_API_KEY is not set (expected in .env).")
        http = httpx.Client(transport=self._transport, timeout=httpx.Timeout(GEMINI_TIMEOUT_S, connect=60.0),
                            event_hooks={"response": [self._capture]})
        # Explicit key, base URL and Developer-API mode, so GOOGLE_API_KEY, GOOGLE_GEMINI_BASE_URL or
        # GOOGLE_GENAI_USE_VERTEXAI in the environment cannot redirect the request or change credentials.
        self._client = genai.Client(
            api_key=key, vertexai=False,
            http_options=types.HttpOptions(base_url=GEMINI_BASE_URL, api_version=GEMINI_API_VERSION,
                                           timeout=int(GEMINI_TIMEOUT_S * 1000),
                                           retry_options=types.HttpRetryOptions(attempts=1), httpx_client=http))

    @staticmethod
    def build_request(jpeg_bytes: bytes, prompt: str) -> dict:
        from google.genai import types
        return {
            "model": GEMINI_MODEL,
            "contents": [types.Content(role="user", parts=[
                types.Part(inline_data=types.Blob(data=jpeg_bytes, mime_type=GEMINI_IMAGE_MIME),
                           media_resolution=types.PartMediaResolution(
                               level=types.PartMediaResolutionLevel(GEMINI_MEDIA_RESOLUTION))),
                types.Part(text=prompt)])],
            "config": types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(thinking_level=types.ThinkingLevel(GEMINI_THINKING_LEVEL),
                                                     include_thoughts=GEMINI_INCLUDE_THOUGHTS),
                response_mime_type="application/json",
                response_json_schema=bg.OUTPUT_SCHEMA,
                max_output_tokens=GEMINI_MAX_OUTPUT_TOKENS,
                candidate_count=1),
        }

    def _backoff(self, n: int, exc) -> float:
        response = getattr(exc, "response", None)
        headers = getattr(response, "headers", None) or {}
        retry_after = headers.get("retry-after") if hasattr(headers, "get") else None
        if retry_after:
            try:
                return min(float(retry_after), 300.0)
            except ValueError:
                pass
        return min(self.BACKOFF_BASE_S * 2 ** n, self.BACKOFF_MAX_S) + random.uniform(0, 1)

    def _raw(self, response) -> dict:
        parsed = response.model_dump(mode="json", exclude_none=True)
        parsed.pop("sdk_http_response", None)
        return {"http": self._last_http, "sdk_parsed": parsed}

    def _result(self, response, retries) -> AdapterResult:
        raw = self._raw(response)
        usage, complete = _gemini_usage(response.usage_metadata)
        served = response.model_version or ""
        cands = response.candidates or []
        cand = cands[0] if cands else None
        finish = getattr(getattr(cand, "finish_reason", None), "value", None) or (
            str(cand.finish_reason) if cand is not None and cand.finish_reason is not None else None)
        feedback = response.prompt_feedback
        block = getattr(getattr(feedback, "block_reason", None), "value", None) if feedback else None
        image_tokens = usage.get("image_input_tokens")
        provider = {"response_id": response.response_id, "model_version": served or None,
                    "finish_reason": finish, "block_reason": block, "candidates": len(cands),
                    "requested_media_resolution": GEMINI_MEDIA_RESOLUTION,
                    "reported_image_tokens": image_tokens,
                    "media_resolution_check": ("not verifiable: image token count not reported" if image_tokens is None
                                               else "consistent with ULTRA_HIGH" if image_tokens > GEMINI_FALLBACK_IMAGE_TOKENS
                                               else "LOWER THAN REQUESTED"),
                    "thinking_level_requested": GEMINI_THINKING_LEVEL,
                    "thoughts_tokens": usage.get("thoughts_tokens"),
                    "cached_input_tokens": usage.get("cached_input_tokens"),
                    "usage_complete": complete}
        common = dict(raw=raw, usage=usage, request_id=response.response_id, retries=retries,
                      provider_response=provider, usage_complete=complete)

        problems = []
        if served and not served.startswith(GEMINI_MODEL):
            problems.append(f"served model {served!r} is not {GEMINI_MODEL}")
        if image_tokens is not None and image_tokens <= GEMINI_FALLBACK_IMAGE_TOKENS:
            problems.append(f"reported image tokens {image_tokens} <= {GEMINI_FALLBACK_IMAGE_TOKENS} (HIGH budget): "
                            f"{GEMINI_MEDIA_RESOLUTION} was not applied")
        if problems:
            self._halt = "; ".join(problems)
            return AdapterResult(status="error", error=f"experiment condition not honoured: {self._halt}",
                                 possibly_billed=True, **common)

        if block:
            return AdapterResult(status="ok", text="", stop_reason="refusal", stop_detail=f"prompt blocked: {block}",
                                 **common)
        if cand is None:
            return AdapterResult(status="error", error="response contained no candidates", possibly_billed=True,
                                 **common)
        parts = (cand.content.parts if cand.content and cand.content.parts else []) or []
        text = "".join(p.text for p in parts if p.text and not p.thought)
        if finish == "STOP":
            return AdapterResult(status="ok", text=text, stop_reason="completed", **common)
        if finish == "MAX_TOKENS":
            return AdapterResult(status="ok", text=text, stop_reason="max_tokens", stop_detail=finish, **common)
        if finish in self.BLOCK_REASONS:
            return AdapterResult(status="ok", text="", stop_reason="refusal", stop_detail=finish, **common)
        return AdapterResult(status="error", error=f"response finished with {finish}", stop_detail=finish,
                             possibly_billed=True, **common)

    def run(self, jpeg_bytes: bytes, prompt: str, context: dict) -> AdapterResult:
        import httpx
        from google.genai import errors

        if self._halt:
            raise FatalAdapterError(f"stopped: {self._halt}")
        self.prepare()
        params = self.build_request(jpeg_bytes, prompt)
        retries = []
        for n in range(self.MAX_RETRIES + 1):
            self._last_http = None
            billed = False
            try:
                response = self._client.models.generate_content(**params)
            except errors.APIError as e:
                error = redact_secrets(f"{type(e).__name__} ({e.code} {e.status}): {e.message}")
                if e.code in self.FATAL_STATUS:
                    raise FatalAdapterError(error) from None
                retryable = e.code in self.RETRY_UNBILLED
                billed = e.code in self.POSSIBLY_BILLED or not retryable
                exc = e
            except (httpx.ConnectError, httpx.ConnectTimeout) as e:     # request never reached the server
                retryable, error, exc = True, redact_secrets(f"{type(e).__name__}: {e}"), e
            except httpx.HTTPError as e:                                # sent; may have been processed
                retryable, billed, error, exc = False, True, redact_secrets(f"{type(e).__name__}: {e}"), e
            except (ValueError, TypeError) as e:
                error = redact_secrets(f"{type(e).__name__}: {e}")
                if self._last_http is None:          # rejected by the SDK before sending: a configuration fault
                    raise FatalAdapterError(f"request could not be sent: {error}") from None
                return AdapterResult(status="error", error=f"unparseable response: {error}", possibly_billed=True,
                                     retries=retries, raw={"http": self._last_http})
            else:
                return self._result(response, retries)

            raw = {"http": self._last_http} if self._last_http else None
            if billed or not retryable or n == self.MAX_RETRIES:
                return AdapterResult(status="error", error=error, possibly_billed=billed, retries=retries, raw=raw)
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


class MockGeminiAdapter(_GeminiBehaviour, MockAdapter):
    """Offline stand-in with the M10 config, pricing and image accounting. Never touches the network."""

    adapter_name = "mock"

    def config(self) -> dict:
        return gemini_config(self.model_code, self.adapter_name)


def get_adapter(model_code: str, mock: bool = False, **kwargs) -> ModelAdapter:
    if model_code == "M08":
        return MockAdapter(**kwargs) if mock else ClaudeAdapter(**kwargs)
    if model_code == "M09":
        return MockOpenAIAdapter(**kwargs) if mock else OpenAIAdapter(**kwargs)
    if model_code == "M10":
        return MockGeminiAdapter(**kwargs) if mock else GeminiAdapter(**kwargs)
    if model_code in PLANNED_ADAPTERS:
        raise NotImplementedError(f"{model_code}: {PLANNED_ADAPTERS[model_code]}")
    raise ValueError(f"Unknown model code {model_code!r}")
