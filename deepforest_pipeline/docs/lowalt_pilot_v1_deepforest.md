# DeepForest M01 / M02 v2 on the low-altitude pilot images (lowalt_pilot_v1)

Registry entry for the DeepForest runs on the two approved site-focused pilot images used in the
`tree_boxes` cross-model benchmark (M08 Claude Opus 5.5, M09 GPT-6.1 Sol). Executed 2026-11-05 in WSL.
Historical outputs under `outputs/experiment_001/` were not read, modified or reused.

## Images (unchanged; selection decision closed)

| Image ID | File (in `tree_boxes`) | Size | SHA-256 | Relative altitude |
|---|---|---|---|---|
| LA_PT-R1_27m | `drone_images/height/PT_R1/PT_R1_27m.JPG` | 8192 × 6144 | `f3ba7ede…` (full hash in `tree_boxes/manifests/manifest.csv`) | 26.8 m |
| LA_PT-02_64m | `drone_images/height/PT_02/PT_02_64m.JPG` | 8192 × 6144 | `25f3c331…` | 64.3 m |

Each run verifies the image against the `tree_boxes` manifest before inference. The importer also checks that
every M08/M09 record on the same image recorded the same hash.

**Inputs differ between models.** M08 and M09 received a 2212 × 1659 JPEG (q95, Lanczos; `tree_boxes`
preprocessing v2). M01 processes the original 8192 × 6144 pixels. M02 v2 processes a Lanczos resample to a nominal
10 cm/px. All boxes are compared in original-image pixels.

## Checkpoint (pinned, offline)

- `weecology/deepforest-tree`, revision `cc21436bc5d572dde8ff5f93c1e71a32f563cace`, loaded with `HF_HUB_OFFLINE=1`.
- `model.safetensors` SHA-256 `d37a7af0b5ba2754282a80d41b4c7b66e8c7149234df5bf3f41bdaf57d329fc8` (129,049,980 bytes).
- `config.json` SHA-256 `521e0149f0b060237c6994a69bffdff53ce5deb3beb854dad2752a3f8dd49bc4`.
- Both hashes are checked before loading. After loading, all 301 tensors of the model's `state_dict()` are compared
  with the safetensors file and must be equal. The result is recorded in each `run_record.json` (`loaded_weights`).
- The WSL cache holds only this snapshot, and `refs/main` also points to it.
- **`lightning_logs/*/hparams.yaml` shows `model.revision: main`.** That is DeepForest's config default, written by
  Lightning on every predict call. `load_model(revision=…)` takes the revision as an argument and does not write it
  back into the config. The weights actually used are established by the file hash and tensor-equality checks
  above, not by that log.

## Configurations (frozen; identical inference settings)

Settings: DeepForest 2.1.0, patch 400 px, overlap 0.05, cross-window IoU 0.15, `score_thresh` 0.10, `nms_thresh`
0.05, `dataloader_strategy="single"`. These equal the constants in `run_deepforest_single.py` and
`run_deepforest_gsd_7_89.py`, which a unit test checks. No tuning.

| Config ID | Experiment | Input to the detector |
|---|---|---|
| `M01_9c192941f5` | `lowalt_pilot_ptr1_27m_m01_v1`, `lowalt_pilot_pt02_64m_m01_v1` | original image path, native resolution |
| `M02v2_b25eccbcc7` | `lowalt_pilot_ptr1_27m_m02_v2` | 387 × 290 resample (nominal 0.472 → 10.0 cm/px), padded to 400 × 400 (see below) |
| `M02v2_1e05a4037f` | `lowalt_pilot_pt02_64m_m02_v2` | 927 × 696 resample (nominal 1.132 → 10.0 cm/px) |

The config ID is the code plus the first 10 hex digits of the SHA-256 of the canonical configuration JSON. It
includes the native GSD, so the two M02 v2 images have different IDs.

### Why 10.0 cm/px, and why not 7.89

10 cm/px is close to the approximate resolution of DeepForest's training imagery (NEON ~10 cm orthophotos). At that
scale a 400 px window spans about 40 m, the setting the released model was built around. This is a principled
**alignment** choice, **not** an optimised or validated one. The historical M02 target of 7.89 cm/px is not reused
because its derivation is not documented anywhere in this repository or the project notes.

A 7.89 cm/px sensitivity run is possible with the same script (`--target-gsd-cm 7.89`). It was **not** executed.

### Native GSD estimate (nominal)

GSD = altitude × sensor width / (focal length × image width). The inputs:
- EXIF focal length 8.67 mm, 35 mm-equivalent 24 mm, giving a crop factor of 2.768.
- Sensor width 12.50 mm.
- Altitude = DJI `RelativeAltitude`, metres above the **takeoff point**.

| Image | Relative altitude | Nominal GSD | Exact resize factor | Output size | Applied x / y factors |
|---|---|---|---|---|---|
| LA_PT-R1_27m | 26.8 m | 0.472 cm/px | 0.0472 | 387 × 290 | 0.04724121 / 0.04720052 |
| LA_PT-02_64m | 64.3 m | 1.132 cm/px | 0.1132 | 927 × 696 | 0.11315918 / 0.11328125 |

The same formula reproduces the historical 1.67 cm/px at 95 m.

Assumptions and uncertainty:
- **Terrain.** Altitude is measured above takeoff, not above the ground under the image. Both images share a takeoff
  elevation of about 152 m (AbsoluteAltitude − RelativeAltitude), and terrain differences are not modelled.
- **Canopy height.** Crown tops are closer to the camera than the ground. With 5–10 m crowns, the crown-top GSD at
  26.8 m is about 19–37 % finer than nominal.
- **Calibration.** No calibrated focal length or distortion data is available.

The effective target is therefore "about 10 cm/px at ground level", not a measured value.

Output sizes are `round(dimension × factor)`. The x and y factors actually applied come from the integer sizes and
are used for the inverse mapping (`x_original = x_resampled / applied_scale_x`). Resampling uses PIL Lanczos, JPEG
quality 95, and EXIF orientation must be 1 (checked). Script: `scripts/resample_to_target_gsd_v2.py`; a
`resample_record.json` sits beside every resampled image.

### PT-R1 small image: padding (`pad_to_patch_v1`)

At 10 cm/px, PT-R1 becomes 387 × 290, smaller than one 400 px patch.
- **Without padding:** the sliding window shrinks to 387 × 290, and the RetinaNet transform (`min_size` 800) then
  upsamples it ×2.76. Every standard 400 × 400 window is upsampled ×2.0. The detector would see the trees about 38 %
  larger than in any other run, an unrecorded scale change.
- **With padding:** the image is padded with black pixels on the right (13 px) and bottom (110 px) to 400 × 400 and
  passed as an in-memory RGB array. Nothing is rescaled or moved, so coordinates are unchanged (top-left origin).
  Boxes wholly in the padding would be dropped and boxes crossing into it clipped; both are counted. In this run,
  0 were dropped and 0 clipped.
- **Not applied** to PT-02 M02 v2 (927 × 696) or to M01. Their windows are already 400 × 400.

Engineering preflight, run on a scratch copy and not part of the experiment:
- Passing the decoded image as an array gives bit-identical detections to passing the file path. This was checked
  on the PT-02 resample: 183 detections either way.
- The RetinaNet transform resized the padded input 400 → 800 (×2.0) and the unpadded input 290 → 800 (×2.76).
- The unpadded PT-R1 input produced 25 boxes, the padded one 20. That comparison is a diagnostic only. Padding was
  chosen for the scale reason above, before any experimental output existed, and the unpadded variant is not an
  experimental result.

## Execution and determinism

Runner: `scripts/run_deepforest_lowalt_v1.py` (WSL `.venv`, Python 3.12.14, torch 2.14.0+cu130, cuDNN 92400,
NVIDIA GeForce RTX 5070 Laptop GPU; `cudnn.deterministic=False`, `cudnn.benchmark=False`). Each combination ran
twice: `run_01_primary` and `run_02_verification`. `scripts/compare_deepforest_runs_v1.py` compares them and
writes `determinism.json`. Timestamps and paths are excluded; boxes and scores are compared at full float precision.

| Experiment | Detections (primary = verification) | Inference s | Windows | Max boxes centred in one window | Verdict |
|---|---|---|---|---|---|
| lowalt_pilot_ptr1_27m_m01_v1 | 4836 | 19.2 | 374 | 89 | bitwise identical |
| lowalt_pilot_pt02_64m_m01_v1 | 6320 | 18.4 | 374 | 87 | bitwise identical |
| lowalt_pilot_ptr1_27m_m02_v2 | 20 | 1.1 | 1 (padded) | 20 | bitwise identical |
| lowalt_pilot_pt02_64m_m02_v2 | 183 | 1.4 | 6 | 62 | bitwise identical |

No window approached the per-window cap of 300 detections. The verification run is a reproducibility check, **not**
an independent observation; only the primary run is imported into `tree_boxes` (as trial 1).

## Outputs

`outputs/lowalt_pilot_v1/{m01_native|m02v2_gsd}/<experiment>/<config_id>/` contains `model_config.json`. Below it,
`<image_id>/` contains `determinism.json` and two run folders, `run_01_primary/` and `run_02_verification/`. Each
run folder holds:
- `run_record.json`: config, hashes, checkpoint, weights check, environment, durations, window diagnostics,
  padding and preprocessing.
- `detections.json`: original-pixel float boxes, processed-image boxes and scores.
- `predictions_raw.csv`: unfiltered DeepForest output, full precision.
- `annotated_original.jpg`, plus `annotated_processed.jpg` for M02 v2.
- `run.log`, and `error.txt` on failure.
- M02 v2 only: `preprocessed/` with the resample and its record.

The runner refuses to overwrite an existing run folder.

The outputs are mirrored from WSL `/home/gabpe/src/urban-tree-vision-baselines/outputs/lowalt_pilot_v1` to the
Windows repository; 62 files were SHA-256-verified identical.

## Tests

`tests/test_lowalt_pilot_v1_deepforest.py` has 29 `unittest` tests. A fake model stands in for inference. They
cover:
- the pinned revision and hashes, and refusal of a tampered or missing checkpoint;
- equality with the historical constants, and the M01 call arguments;
- M02 dimensions and factors, and the inverse round trip;
- padding: array content, dropping and clipping, and the padded call;
- float boxes and scores, the no-detection case, overwrite refusal, experiment isolation, source-hash refusal,
  error recording and canonical ordering;
- determinism comparison;
- the pilot image hashes.

Run with `.venv/bin/python -m unittest discover -s tests -v`.

Results and the cross-model comparison are reported in `tree_boxes` (`RESEARCH_PROGRESS.md`, `EVALUATION.md`,
`evaluation_results_lowalt_v1_deepforest/`).
