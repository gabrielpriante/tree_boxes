"""
Model adapters for the tree-count benchmark.

An adapter turns (preprocessed image bytes, prompt) into one raw model response.
The batch runner owns everything else: trial bookkeeping, budget, validation,
rendering, and CSV output, so every model shares the same result format.

Implemented:  M08 Claude Opus 5.5 (ClaudeAdapter), plus MockAdapter for offline tests.
Planned:      see PLANNED_ADAPTERS.
"""

import base64
import hashlib
import json
import math
import platform
import random
import time
from dataclasses import dataclass, field

import box_generation as bg

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

# Upper bound on image input tokens for high-resolution vision models
# (2576 px long edge / 3.75 MP; the API downsizes anything larger).
IMAGE_TOKEN_UPPER_BOUND = 4784
# Allowance for the structured-output system prompt and message framing.
REQUEST_OVERHEAD_TOKENS = 2000

PLANNED_ADAPTERS = {
    "M01": "DeepForest (existing implementation; results are frozen, adapter not implemented)",
    "M02": "DeepForest (existing implementation; results are frozen, adapter not implemented)",
    "M09": "OpenAI / ChatGPT (adapter not implemented)",
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


def _sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


class ModelAdapter:
    model_code = ""
    adapter_name = ""
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

    def estimate_input_tokens(self, prompt: str) -> int:
        """Conservative upper bound used before any request for this image has completed."""
        return IMAGE_TOKEN_UPPER_BOUND + math.ceil(len(prompt) / 3) + REQUEST_OVERHEAD_TOKENS

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
            "exif_transpose": True,
            "color_mode": "RGB",
            "max_upload_edge_px": bg.MAX_UPLOAD_EDGE,
            "resize": "PIL Image.thumbnail, LANCZOS, only when larger than max edge",
            "format": "JPEG",
            "jpeg_qualities_tried": list(bg.JPEG_QUALITIES),
            "max_image_bytes": bg.MAX_IMAGE_BYTES,
        },
    }


class ClaudeAdapter(ModelAdapter):
    """M08: Claude Opus 5.5 through the Anthropic Messages API, one independent request per call."""

    model_code = "M08"
    adapter_name = "anthropic-messages"

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


class MockAdapter(ModelAdapter):
    """Offline stand-in with Claude's config and pricing. Never touches the network.

    behaviors maps (image_id, trial, attempt) or (image_id, trial) to one of:
      ok, invalid, bad_json, refusal, max_tokens, error, midstream_error, interrupt, fatal
    """

    model_code = "M08"
    adapter_name = "mock"

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


def get_adapter(model_code: str, mock: bool = False, **kwargs) -> ModelAdapter:
    if model_code == "M08":
        return MockAdapter(**kwargs) if mock else ClaudeAdapter(**kwargs)
    if model_code in PLANNED_ADAPTERS:
        raise NotImplementedError(f"{model_code}: {PLANNED_ADAPTERS[model_code]}")
    raise ValueError(f"Unknown model code {model_code!r}")
