# Tree-count consistency benchmark

Measures how consistently a model counts individual trees when the same standardized
drone image is processed independently 10 times. The statistics describe count
consistency, not detection accuracy.

## Fixed configuration (M08)

| Setting | Value |
|---|---|
| Model | `claude-opus-5-5` |
| Effort | `high` |
| Thinking | adaptive |
| `max_tokens` | 32,000 |
| Output | structured JSON (`OUTPUT_SCHEMA` in `box_generation.py`) |
| Prompt | `PROMPT_TEMPLATE` in `box_generation.py`, unchanged |
| Coordinates | normalized 0–1000 `[x_min, y_min, x_max, y_max]` |
| Preprocessing | EXIF transpose → RGB → longest edge 2576 px (LANCZOS) → JPEG q95 (lower only if > 5 MB) |

The runner never changes these settings, including when the budget runs low; it stops instead.
Refusal fallbacks to other models are deliberately not enabled. Any change to model, settings, prompt
template, schema or preprocessing produces a different `model_config_id`, so results from
different configurations are written to different folders and can't be mixed.

## Files

| File | Purpose |
|---|---|
| `box_generation.py` | Original single-image CLI; also supplies prompt, schema, preprocessing and request parameters to the runner |
| `make_manifest.py` | Builds `manifests/manifest.csv` from `drone_images/` |
| `manifests/manifest.csv` | Image list and metadata (edit by hand to complete `UNKNOWN` fields) |
| `model_adapters.py` | `ClaudeAdapter` (M08), `MockAdapter` (offline), placeholders for M01/M02/M09/M10/Copilot |
| `benchmark_runner.py` | Batch runner |
| `tests/test_benchmark.py` | Offline tests (`python -m pytest tests -q`) |

## Manifest

Columns required by the runner: `image_id, image_path, site, week, condition, altitude_m, coverage_scope`.
`make_manifest.py` also writes `collection` (`standard` or `low_altitude`), capture time, gimbal pitch,
altitude source, dimensions, `sha256`, `needs_review` and `notes`.

* `altitude_m` is the DJI XMP `RelativeAltitude` (metres above the takeoff point, not above the canopy).
* `coverage_scope` is `UNKNOWN` for every image; it has to be confirmed by looking at the images.
* Week 1 condition is the folder label (`MORNING_START` / `AFTERNOON_END`); confirm whether it maps to BEFORE/AFTER.
* Low-altitude images (`LA_*`) have `UNKNOWN` week and condition.
* `site`, `week` and `condition` are inserted into the prompt. The runner skips images where any of them
  is `UNKNOWN` (override: `--allow-unknown-metadata`). Editing these fields after trials have run changes the
  prompt, and the runner refuses to mix the two.
* `make_manifest.py` will not overwrite an existing manifest without `--force`.
* `PT_02.JPG` in the repository root is byte-identical to `drone_images/week1/afternoon_end/PT_02.JPG` and is not in the manifest.

## Running

```powershell
# Plan and cost bounds; no API calls, nothing written
python benchmark_runner.py --manifest manifests/manifest.csv --image-ids W03_BEFORE_PT-02,W03_BEFORE_PT-R1 --runs 10 --max-usd 5 --dry-run

# Live (paid)
python benchmark_runner.py --manifest manifests/manifest.csv --image-ids W03_BEFORE_PT-02,W03_BEFORE_PT-R1 --runs 10 --max-usd 5 --execute

# Continue after an interruption, budget stop or failure
python benchmark_runner.py ... --execute --resume
```

Images run in manifest order; trials 1–10 of one image complete before the next image starts.
Requests are strictly sequential. Each trial is a new single-message request: no conversation
history or earlier detections are sent.

## Output layout

```
benchmark_results/<experiment_id>/
  experiment.json
  counts_long.csv        one row per image × model config × trial
  counts_summary.csv     one row per image × model config
  boxes_long.csv         every box from every successful trial (for IoU analysis)
  attempts_log.csv       every attempt, including failed ones
  invalid_trials.csv     index of responses that failed validation
  <model_config_id>/
    model_config.json
    manifest_snapshots/<sha>.csv
    <image_id>/trial_NN/attempt_NN/
      request.json         settings, hashes, dimensions, pre-request worst-case cost
      prompt.txt
      response_raw.json    full API response
      detections.json      validated detections (successful trials only)
      annotated.jpg        boxes on the original full-resolution image
      invalid_response.json + annotated_INVALID.jpg   (invalid trials only)
      record.json          status, tokens, cost, validation, timestamps (written last)
```

The CSVs are always rebuilt from the `record.json` files, so they can be regenerated without any API call.

## Trial statuses

| Status | Meaning | Re-run on `--resume`? |
|---|---|---|
| `success` | Valid JSON, every box valid | No |
| `invalid` | Response returned but failed validation; preserved exactly as returned, not corrected | No |
| `refusal` | Model declined | No |
| `max_tokens` | Truncated at 32,000 tokens | No |
| `error` | API/network failure with no usable response | Yes, as a new attempt |
| `interrupted` | Stopped during the request (Ctrl+C or crash) | Yes, as a new attempt |

The first four are model outcomes and complete the trial. Re-running them until a valid answer appeared
would bias the consistency statistics. Earlier attempts are never deleted.

`counts_summary.csv` statistics use successful trials only. The standard deviation is the sample SD (n − 1).
It is blank when fewer than 2 trials succeeded. CV = SD / mean.

## Budget guard and its limits

* Cost = actual `usage` tokens × list price ($4 input / $20 output per million tokens; thinking is billed as output).
* Before each request, the runner computes that request's worst case: input tokens (the actual figure from an
  earlier trial of the same image, otherwise a conservative bound) plus the full 32,000 output tokens,
  about $0.67. The request is sent only if recorded spend + worst case ≤ `--max-usd`.
* `--max-usd` applies to cumulative recorded spend for this model config across all sessions.
* A request that fails after the response stream opened, or is interrupted, is charged at its worst case
  because its real usage is unknown.

The guard is an estimate, not a hard ceiling. List prices are hard-coded and may change. Server-side billing
can differ from local estimates, and spend outside this runner isn't visible to it. Because it reserves the
worst case, the guard can stop with some budget unused: with $5 and about $0.20–0.30 per trial, expect
roughly 15–20 trials before it stops, not the full 20. Set a spending limit in the Anthropic Console as the
real hard cap.

## Reproducibility checks

* Image file hash must match the manifest `sha256`.
* Preprocessed upload bytes and prompt hash must match all earlier trials of that image (a Pillow upgrade that
  changes JPEG bytes is detected).
* `model_config.json` must match the current configuration.
* Records store Python, Pillow and SDK versions, and the request ID.

## Future adapters

`model_adapters.PLANNED_ADAPTERS` lists M01/M02 (DeepForest), M09 (OpenAI), M10 (Gemini) and Copilot.
A new adapter implements `config()`, `max_output_tokens()`, `cost_usd()` and `run()`, and inherits the same manifest,
trial numbering, folder layout and CSVs. Deterministic models such as DeepForest may produce identical results
across trials; that's expected and should not be "fixed" by adding randomness. No DeepForest outputs exist in
this repository (`deepforest_images/` is empty), so none were touched.
