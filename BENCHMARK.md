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
| Preprocessing (v2) | EXIF transpose → RGB → resize (LANCZOS) to the API's documented target size → JPEG q95 (lower only if > 5 MB) |

The runner never changes these settings, including when the budget runs low; it stops instead.
Refusal fallbacks to other models are deliberately not enabled. Any change to model, settings, prompt
template, schema or preprocessing produces a different `model_config_id`, so results from
different configurations are written to different folders and can't be mixed.

## Image resolution (preprocessing change v1 → v2)

This is a **preprocessing** change. Model settings are unchanged.

Anthropic's documentation (`build-with-claude/vision` and `vision-coordinates`, checked 2026-10-30)
says Opus 5.5 is in the high-resolution tier. Images are limited to **2,576 px on the long edge
and 4,784 visual tokens**, where tokens = ⌈w/28⌉ × ⌈h/28⌉. Any image over either limit is
silently downsized by the API to the largest aspect-preserving size that fits both. It is then
padded (not resized) to a multiple of 28 px.

| Stage | v1 (before) | v2 (now) |
|---|---|---|
| Original file | 8192 × 6144 | 8192 × 6144 (kept for rendering boxes) |
| Locally preprocessed upload | 2576 × 1932 (6,348 tokens: over the limit) | 2212 × 1659 (4,740 tokens) |
| Effective resolution the model sees | 2212 × 1659, resized on the server with an undocumented method | 2212 × 1659, exactly the uploaded bytes |

The v1 upload satisfied the edge limit but not the token limit, so every v1 image was downsized again
on the server. v2 resizes locally to the documented target, using Anthropic's reference algorithm
(`box_generation.api_resized_size`). The resolution the model sees is unchanged, but the resampling
is now ours, deterministic and recorded. The image block also sets `transformations.oversized_image = "error"`,
so the API rejects (HTTP 400, unbilled) any image that would need resizing instead of resizing it silently.
Each record stores the original, upload and effective dimensions plus the upload's visual tokens.

The v2 configuration has a new `model_config_id`, so it can't mix with any v1 results.
Single-image results already in `output/` were produced with v1 preprocessing.

Anthropic's guide notes that Claude handles pixel coordinates better than normalized 0–1000
coordinates. The 0–1000 convention is a fixed design decision here and was not changed, but it is
worth stating as a methodological caveat.

## Files

| File | Purpose |
|---|---|
| `box_generation.py` | Original single-image CLI; also supplies prompt, schema, preprocessing and request parameters to the runner |
| `make_manifest.py` | Builds `manifests/manifest.csv` (`--update` fills gaps without overwriting edits; `--review` lists open questions) |
| `manifests/manifest.csv` | Image list and metadata (edit by hand to complete or confirm fields) |
| `manifests/METADATA_REVIEW.md` | Fields awaiting your confirmation, October 24 images first |
| `spend_ledger/anthropic_api_ledger.jsonl` | Project-wide spending ledger (created on the first paid request) |
| `model_adapters.py` | `ClaudeAdapter` (M08), `MockAdapter` (offline), placeholders for M01/M02/M09/M10/Copilot |
| `benchmark_runner.py` | Batch runner |
| `tests/test_benchmark.py` | Offline tests (`python -m pytest tests -q`) |

## Manifest

Columns required by the runner: `image_id, image_path, site, week, condition, altitude_m, coverage_scope`.
`make_manifest.py` also writes `collection` (`standard` or `low_altitude`), capture time, gimbal pitch and
yaw, `capture_geometry` (NADIR/OBLIQUE from gimbal pitch), altitude source, dimensions, `sha256`,
`needs_review` and `notes`.

Controlled vocabulary, checked on every load (any other value is an error):

| Field | Allowed values |
|---|---|
| site | `PT-02`, `PT-R1`, `UNKNOWN` |
| week | `WEEK <n>`, `NOT_APPLICABLE`, `UNKNOWN` |
| condition | `BEFORE`, `AFTER`, `ALTITUDE_TEST`, `MORNING_START`, `AFTERNOON_END`, `UNKNOWN` |
| coverage_scope | `FULL_SITE`, `PARTIAL_SITE`, `UNKNOWN` |
| altitude_m | number or `UNKNOWN` |

`UNKNOWN` means not yet established. `NOT_APPLICABLE` means the field does not apply.

* `altitude_m` is the DJI XMP `RelativeAltitude` (metres above the takeoff point, not above the canopy).
* The October 24 altitude series has *proposed* labels `week = NOT_APPLICABLE` and
  `condition = ALTITUDE_TEST`, still listed in `needs_review` for confirmation. They do not imply
  intervention BEFORE/AFTER status.
* Week 1 keeps its folder labels (`MORNING_START` / `AFTERNOON_END`) until the mapping is confirmed.
* `coverage_scope` is `UNKNOWN` everywhere; it needs a visual check and is never inferred from altitude.
* **Run gate:** `site`, `week` and `condition` are inserted into the prompt. The runner won't send an image
  while any of them is `UNKNOWN` or still listed in `needs_review`. `NOT_APPLICABLE` is a valid value and
  needs no override. `--allow-unknown-metadata` bypasses the gate. Editing these fields after trials have
  run changes the prompt, and the runner refuses to mix the two.
* `make_manifest.py --update` only fills cells that are `UNKNOWN` or blank, adds new columns and new
  images, keeps a timestamped backup, and never overwrites a value or `needs_review` entry someone has set.
  Without `--update` or `--force` it refuses to touch an existing manifest.
* `PT_02.JPG` in the repository root is byte-identical to `drone_images/week1/afternoon_end/PT_02.JPG` and is not in the manifest.

## Running

```powershell
# Plan and cost bounds; no API calls, nothing written
python benchmark_runner.py --manifest manifests/manifest.csv --image-ids W03_BEFORE_PT-02,W03_BEFORE_PT-R1 --runs 10 --max-usd 5 --dry-run

# Live (paid)
python benchmark_runner.py --manifest manifests/manifest.csv --image-ids W03_BEFORE_PT-02,W03_BEFORE_PT-R1 --runs 10 --max-usd 5 --execute

# Continue after an interruption, budget stop or failure
python benchmark_runner.py ... --execute --resume

# October 24 altitude series (all ten low-altitude images)
python benchmark_runner.py --manifest manifests/manifest.csv --collection low_altitude --runs 10 --max-usd 5 --experiment-id altitude_oct24_v1 --dry-run
```

`--mock` runs write to `benchmark_results_mock/` (git-ignored) with their own ledger. The runner refuses
to write mock results inside `benchmark_results/`.

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

## Repeated-outcome pause

After 3 consecutive trials (any image) ending `invalid`, `refusal` or `max_tokens`, the runner pauses and writes
`<model_config_id>/PAUSED.json` listing those trials and their validation errors. Nothing is retried or
discarded: the trials remain complete experimental outcomes in every CSV. `error`/`interrupted` attempts
neither count toward nor reset the streak. The streak carries across sessions.

To continue after reviewing the trials, run `--execute --resume --acknowledge-pause`. The pause file is
archived as `PAUSED_acknowledged_<time>.json` and the count restarts. `--pause-after N` changes the
threshold; `0` disables the pause.

## Spending ledger and budget guard

* **One ledger for the project.** Every paid attempt is written to `spend_ledger/anthropic_api_ledger.jsonl`,
  whatever `--experiment-id` or `--output-dir` is used. A new experiment ID therefore can't reset the budget:
  `--max-usd` is checked against the ledger total for all Claude experiments in this project.
* **Two entries per attempt**, keyed by a unique `attempt_uid`: `start` (with its worst-case cost) before the
  request and `final` after it. Reading or reconciling the ledger never counts an attempt twice, and a
  duplicate entry is ignored.
* **Confirmed vs estimated.** The ledger reports three categories separately:
  * *confirmed*: actual `usage` tokens × list price ($4 input / $20 output per million; thinking is
    billed as output);
  * *estimated*: worst-case cost for requests that failed after the stream opened or were interrupted;
  * *unresolved*: a `start` with no `final` (the process died mid-request), counted at its worst case.
* **Reconciliation.** Before running, attempts found on disk but missing from the ledger (for example from a
  crash between writing `record.json` and the ledger) are added once.
* **Pre-request check (unchanged).** A request is sent only if ledger total + that request's worst case
  (full 32,000 output tokens, about $0.67) ≤ `--max-usd`. Otherwise the run stops and lists the outstanding trials.
* The dry run prints the ledger totals, the remaining budget, and an illustrative projection of how many
  trials the guard would admit at several output-token levels.
* Original per-trial costs (`record.json`) and per-experiment CSV costs are kept as before.

Limits: this is local accounting, not an account-level ceiling.
* It can't see spending elsewhere on the Anthropic account (Console, other scripts, other machines, or
  other copies of this folder).
* List prices are hard-coded, and server-side billing may differ.
* Interrupted requests are estimates.

A guaranteed $5 cap can only come from a spend limit configured on the Anthropic account itself.
Because the guard reserves the worst case, it can stop with some budget unused.

## Reproducibility checks

* Image file hash must match the manifest `sha256`.
* Preprocessing is deterministic: tests confirm identical upload bytes across ten repetitions of a real image.
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
