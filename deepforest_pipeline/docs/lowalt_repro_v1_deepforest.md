# DeepForest repeated-execution experiment (lowalt_repro_v1)

**Status: completed (9 November 2026).** All 40 runs succeeded between 22:38 and 22:50 UTC. They were mirrored to
Windows (276 files, SHA-256 verified) and analysed in `tree_boxes` (`evaluation_results_deepforest_repro_v1/`).

*This copy in `tree_boxes/deepforest_pipeline/docs/` differs from the original in
`urban-tree-vision-baselines/docs/` only in this status paragraph and the "Results" section below. The original
was written before execution; its hash is in `../provenance/SOURCE.md`.*

## Results (verified from `evaluation_results_deepforest_repro_v1/repro_summary.csv`)

| Experiment | Runs | Detections per run | Distinct output hashes | Pairwise agreement at 0.50 / 0.75 / 1.00 | Identical to pilot runs |
|---|---|---|---|---|---|
| `lowalt_repro_ptr1_27m_m01_v1` | 10 | 4,836 | 1 | 1.000 / 1.000 / 1.000 (45 pairs) | yes |
| `lowalt_repro_pt02_64m_m01_v1` | 10 | 6,320 | 1 | 1.000 / 1.000 / 1.000 (45 pairs) | yes |
| `lowalt_repro_ptr1_27m_m02v2_v1` | 10 | 20 | 1 | 1.000 / 1.000 / 1.000 (45 pairs) | yes |
| `lowalt_repro_pt02_64m_m02v2_v1` | 10 | 183 | 1 | 1.000 / 1.000 / 1.000 (45 pairs) | yes |

* **Identity.** Raw prediction files, canonical detection files, preprocessed images and annotated images were
  byte-identical within every combination. The canonical boxes and scores were exactly equal, and no discrepancy
  was recorded.
* **Pilot runs.** Including the pilot's primary and verification runs, each configuration produced the same output
  in 12 executions.
* **Interpretation.** This is computational reproducibility on one machine and software stack, not predictive
  accuracy, and not one observation per run.

## Question

Does each deterministic DeepForest configuration from the matched-image pilot produce identical detections when it
is executed 10 times on the same approved image?

The pilot (`docs/lowalt_pilot_v1_deepforest.md`) already ran each configuration twice (primary + verification), and
the two runs were bit-identical. Those runs are preserved unchanged. This experiment checks the same behaviour over
10 fresh executions per combination and cross-references the pilot runs.

## Design

| Experiment id | Configuration | Image | Runs | Config id (must equal the pilot's) |
|---|---|---|---|---|
| `lowalt_repro_ptr1_27m_m01_v1` | M01 DeepForest Native | `LA_PT-R1_27m` | 10 | `M01_9c192941f5` |
| `lowalt_repro_pt02_64m_m01_v1` | M01 DeepForest Native | `LA_PT-02_64m` | 10 | `M01_9c192941f5` |
| `lowalt_repro_ptr1_27m_m02v2_v1` | M02v2, nominal 10 cm/px (0.472 → 10.0) | `LA_PT-R1_27m` | 10 | `M02v2_b25eccbcc7` |
| `lowalt_repro_pt02_64m_m02v2_v1` | M02v2, nominal 10 cm/px (1.132 → 10.0) | `LA_PT-02_64m` | 10 | `M02v2_1e05a4037f` |

That is 40 runs in total. M02v2 is the 10 cm/px pilot configuration, **not** the historical M02 at 7.89 cm/px.

* **Runner:** `scripts/run_deepforest_lowalt_v1.py`. The inference path, checkpoint pin (revision `cc21436b…`,
  weights SHA-256 `d37a7af0…`), settings and preprocessing are unchanged. Its only change is that it accepts run
  labels `repeat_01` … `repeat_99` (folders `run_NN_repeat`), and it refuses to write those into the pilot output
  root.
* **Driver:** `scripts/run_deepforest_repeatability_v1.py`.
  * It refuses to start unless all four configuration ids equal the pilot's.
  * Each run is a **separate process**: fresh model load from the pinned checkpoint (hash and tensor-equality
    verified per run) and a fresh CUDA context. M02v2 re-resamples the image each run.
  * Order is **interleaved by repetition**: repeat 1 of all four combinations, then repeat 2, and so on. Any drift
    over the session is therefore spread evenly.
  * Completed runs are skipped on a re-run. The first failed run stops the driver and is never overwritten.
  * Every execution is appended to `driver_log.jsonl`: exit code, status and wall time.
* **Output:** `outputs/lowalt_repro_v1/<m01_native|m02v2_gsd>/<experiment>/<config_id>/<image>/run_NN_repeat/`,
  with the same files as the pilot runs:
  * `run_record.json`, `detections.json`, `predictions_raw.csv`, `annotated_original.jpg`, `run.log`;
  * for M02v2: `preprocessed/` and `annotated_processed.jpg`.

  `experiment_design.json` in the output root records the design and the pilot cross-reference.
  `outputs/lowalt_pilot_v1/` and `outputs/experiment_001/` are never written.

## Analysis (in `tree_boxes`, offline)

`evaluate_deepforest_repeatability.py` reads the mirrored runs and writes `evaluation_results_deepforest_repro_v1/`.
For each combination it reports:
* **Raw-file identity:** SHA-256 of `predictions_raw.csv`, in DeepForest's own order. This is kept separate from
  `detections.json` (canonical order) and from exact float equality of the canonically sorted boxes and scores.
  A difference in raw order only is reported as such.
* **Counts:** detections per run, with mean and SD.
* **Pairwise agreement:** all 45 run pairs, with the existing exact-IoU one-to-one matching at 0.50, 0.75 and 1.00.
* **Differences:** for any non-identical pair, the count difference, unmatched boxes, and the largest coordinate
  and score differences.
* **Pilot cross-reference:** identity with the pilot primary and verification runs.
* **LLM context:** the frozen M08/M09 within-model statistics from `evaluation_results/`, placed beside the
  DeepForest results (read-only).

Figures:
* P, counts per run;
* Q, 10 × 10 pairwise agreement at IoU 0.50 and 1.00;
* R, deterministic repeated executions against stochastic LLM trials, labelled as different kinds of repetition.

## Interpretation

These are repeated executions of deterministic programs on one machine and software stack. They are not 10
independent statistical observations. They do not increase any sample size and are not comparable in kind to the
10 independent stochastic LLM trials. If all outputs are identical, the result is: *reproducibility verified for
this environment and configuration*. It is not a statement about accuracy, and not about other hardware or software
versions.

## Commands

From the WSL working copy (`/home/gabpe/src/urban-tree-vision-baselines`):

```bash
.venv/bin/python -m unittest discover -s tests                       # offline tests
.venv/bin/python scripts/run_deepforest_repeatability_v1.py --dry-run   # plan and status; runs nothing
.venv/bin/python scripts/run_deepforest_repeatability_v1.py --execute   # the 40 runs (inference; needs authorization)
.venv/bin/python scripts/run_deepforest_repeatability_v1.py --summary   # quick count/hash summary
bash scripts/mirror_outputs_to_windows.sh lowalt_repro_v1             # copy to Windows, SHA-256 verified
```
