# DeepForest pipeline (M01, M02v2): code and provenance

This folder holds the exact code that produced the DeepForest results in this repository, plus the provenance
needed to check them. The code was copied unchanged from the working copy where it ran (see
`provenance/SOURCE.md`). Its algorithms and frozen settings are unchanged.

| Path | Content |
|---|---|
| `scripts/run_deepforest_lowalt_v1.py` | Runs one inference. It verifies the pinned checkpoint (`weecology/deepforest-tree` at revision `cc21436b…`, weights SHA-256 `d37a7af0…`, every tensor compared) and the manifest hash of the source image, applies the frozen settings (400 px patches, overlap 0.05, IoU 0.15, score 0.10, NMS 0.05), and records everything in `run_record.json`. |
| `scripts/resample_to_target_gsd_v2.py` | M02v2 preprocessing: Lanczos resample to a nominal 10 cm/px, with the exact forward and inverse coordinate transforms. |
| `scripts/compare_deepforest_runs_v1.py` | Pilot determinism check (primary against verification run). |
| `scripts/run_deepforest_repeatability_v1.py` | Driver for the 40 repeated executions (10 per configuration and image, interleaved, one process per run). |
| `scripts/mirror_outputs_to_windows.sh` | Copies an output folder from WSL to the Windows mirror and verifies it by SHA-256. |
| `scripts/run_deepforest_single.py`, `scripts/run_deepforest_gsd_7_89.py` | Historical 95 m runners, **not used for this paper**. Kept as the documented source of the frozen inference settings, which `tests/` checks M01/M02v2 against. |
| `tests/` | Offline unit tests of the above. A fake model stands in for inference. |
| `docs/lowalt_pilot_v1_deepforest.md` | Pilot methods: checkpoint, settings, GSD estimate, why 10 cm/px, PT-R1 padding, determinism. |
| `docs/lowalt_repro_v1_deepforest.md` | Repeated-execution experiment: design, results, interpretation. |
| `provenance/` | Origin and hashes of these files, all run records and configurations, the checksums of the externally archived raw outputs, and the environment. |
| `LICENSE` | MIT license of the original code. |

**How the results connect to the rest of the repository.**
* The pilot's primary run for each configuration and image was imported by `../import_deepforest.py` into
  `../benchmark_results/lowalt_pilot_*_{m01_v1,m02_v2}/`. That holds detections, raw predictions, run record,
  determinism check and annotated image.
* `../evaluate_consistency.py` evaluated them with the LLM trials (`../evaluation_results_lowalt_v1_deepforest/`).
* The 40 repeated runs were evaluated by `../evaluate_deepforest_repeatability.py`
  (`../evaluation_results_deepforest_repro_v1/`).

**What can and cannot be reproduced from this repository alone.**
* **Without external files:** every accuracy, consistency and cross-model metric. That evaluation reads only files
  stored in `tree_boxes`: the imported primary runs, the LLM records and the reference.
* **Re-running the repeatability analysis** needs the externally archived repeat outputs (see below). Its result
  tables, figures, run records and checksums are stored in `tree_boxes`.
* **Re-running DeepForest inference** needs:
  * the Linux/WSL environment listed in `provenance/environment/wsl_venv_packages.txt` (Python 3.12, DeepForest
    2.1.0, torch 2.14.0 with CUDA; tested only on an NVIDIA RTX 5070);
  * the pinned model weights in the Hugging Face cache (the scripts run with `HF_HUB_OFFLINE=1`);
  * the two source images in `../drone_images/height/`.

  The scripts' defaults point to that machine's paths: the manifest at `/mnt/c/.../tree_boxes/manifests/manifest.csv`
  and outputs under `outputs/` relative to the working directory. Pass `--manifest` and `--output-root` explicitly
  elsewhere.
* **Bit-identical outputs** are verified only for that environment. Other GPUs or library versions may differ.

**Raw outputs (not in Git).** The full outputs (`lowalt_pilot_v1`: 62 files, 157 MB; `lowalt_repro_v1`: 276 files,
784 MB) are kept in the WSL working copy and its Windows OneDrive mirror, verified identical file by file. Their
SHA-256 lists are in `provenance/checksums/`. They are not public. Archive them in a citable data repository
before publication if reviewers need them.

```bash
# Tests (in the DeepForest environment; no inference)
cd deepforest_pipeline && python -m unittest discover -s tests

# Re-running inference (GPU; writes new run folders, refuses to overwrite)
HF_HUB_OFFLINE=1 python scripts/run_deepforest_repeatability_v1.py --dry-run \
    --manifest ../manifests/manifest.csv --output-root outputs/lowalt_repro_v1
```

The main test suite (`python -m pytest tests -q` in the repository root) does not include these tests, because they
need the DeepForest environment.
