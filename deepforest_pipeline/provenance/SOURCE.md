# Provenance of the DeepForest pipeline copies

**Origin.** The DeepForest experiments (M01, M02v2) were run in the working copy of the sibling repository
`urban-tree-vision-baselines` on 5 November (pilot) and 9 November 2026 (repeated executions):
* WSL: `/home/gabpe/src/urban-tree-vision-baselines`;
* Windows mirror: `C:\Users\gabpe\OneDrive\Documents\src\urban-tree-vision-baselines`.

That repository's last commit is `ec714a1` (15 October 2026). Except for the two historical scripts at the end of
the table, the files below were never committed there. They existed as untracked working-copy files, identical in
WSL and Windows. On 9 November 2026 they were copied here
unchanged so that `tree_boxes` holds the exact scientific code.

| File (origin path = path in this folder) | SHA-256 of the original | Copy |
|---|---|---|
| `scripts/resample_to_target_gsd_v2.py` | `1128d5c94f81e79321bea91999f06c123cf20c6b29530a86431775ffc422f3d3` | identical |
| `scripts/run_deepforest_lowalt_v1.py` | `02660139dbced0e7278dfee365b5064a8540c7674dab4089fb77ba7f46209020` | identical |
| `scripts/compare_deepforest_runs_v1.py` | `5e2c55576ad52ceca150e563a4c53fe897008c16c079292579978b7cd866bbd6` | identical |
| `scripts/run_deepforest_repeatability_v1.py` | `593cca799bac6c8338fcfbfd0342d2a83a2bfd4a5ff8acbcf87d89b8a9e06af7` | identical |
| `scripts/mirror_outputs_to_windows.sh` | `f1feaae898660aa5897301fc016416021d62d1a3c6b723b37d55d8935d6a5b83` | identical |
| `tests/test_lowalt_pilot_v1_deepforest.py` | `88db1399dba144e7d37efedaeffa792c80811305b1a4808f7c26623b3abbea29` | identical |
| `tests/test_lowalt_repro_v1_deepforest.py` | `73c6a34796d46d9e4fe0c3fb66fbcf042bbcfe2df2d1bb9867e0215638106d5d` | identical |
| `docs/lowalt_pilot_v1_deepforest.md` | `48c989279dacd3ebfe0657e37cfa73dc5d3018825d007d2e844348bac330934c` | identical |
| `docs/lowalt_repro_v1_deepforest.md` | `fd1336c0614634e45705ff5047643cdab2b0f762a18cc206a3d3a413c30fd368` | status paragraph and results section updated after execution (see the note in the file) |
| `LICENSE` (MIT, the baselines repository's own) | `712947832d7271f430b9e4c49b46c7555a26c87527ae8abaadb456bc48aafc4a` | identical (Windows copy, CRLF) |
| `scripts/run_deepforest_single.py` (historical; committed in `urban-tree-vision-baselines` at `8e01216`, 25 September 2026) | `dcbe9d6ab2de57a9506cf7f8cfa589a55be44db8f1ba01700398531d2440a253` | identical |
| `scripts/run_deepforest_gsd_7_89.py` (historical; same commit) | `1bc4135239fa803ac4f72a0dbf18fb3f5685653148de3d3d6e84d7996a772013` | identical |

The two historical scripts are the earlier 95 m DeepForest runners. They are included only because they define the
inference settings M01/M02v2 reuse unchanged. A unit test checks that equality. They were not run for this paper.

`checksums/copied_files.sha256` lists the hashes of the files as stored here.

**Run metadata** (`run_metadata_lowalt_pilot_v1/`, `run_metadata_lowalt_repro_v1/`). These are copied unchanged
from the Windows mirror of `outputs/`, keeping their folder structure:
* every `run_record.json` (48: 8 pilot runs, 40 repeated runs), each recording configuration, input and checkpoint
  hashes, weight verification, environment, durations and output hashes;
* every `model_config.json` (8), `determinism.json` (4, pilot) and `resample_record.json` (24, M02v2 runs);
* the repeat experiment's `experiment_design.json` and `driver_log.jsonl`.

Images, prediction tables, logs and annotated images are not copied (see below). The records contain absolute
paths of the original machine.

**Raw outputs, archived outside Git.** The complete outputs are kept in two places, verified identical file by file
on 9 November 2026: `outputs/lowalt_pilot_v1/` (62 files, 157 MB) and `outputs/lowalt_repro_v1/` (276 files,
784 MB) in the WSL working copy and in the Windows OneDrive mirror.
* `checksums/outputs_lowalt_pilot_v1.sha256` and `checksums/outputs_lowalt_repro_v1.sha256` list every file with
  its SHA-256 (paths relative to `outputs/<name>/`). Verify a copy with
  `cd outputs/lowalt_repro_v1 && sha256sum -c <this folder>/checksums/outputs_lowalt_repro_v1.sha256`.
* The `outputs/` folder is git-ignored in `urban-tree-vision-baselines` and is not public.
* The pilot's **primary** runs (detections, raw predictions, run records, annotated images) are also imported into
  `tree_boxes/benchmark_results/lowalt_pilot_*_{m01_v1,m02_v2}/`. The evaluation uses those.

**Environment** (`environment/wsl_venv_packages.txt`, read from the virtual environment; nothing installed):
* Python 3.12.14, DeepForest 2.1.0, torch 2.14.0 (CUDA 13.0, cuDNN 92400);
* NVIDIA GeForce RTX 5070 Laptop GPU, WSL2 (Ubuntu).

**Model weights.** These are not stored here. They are the Hugging Face model `weecology/deepforest-tree`, revision
`cc21436bc5d572dde8ff5f93c1e71a32f563cace`, `model.safetensors` SHA-256
`d37a7af0b5ba2754282a80d41b4c7b66e8c7149234df5bf3f41bdaf57d329fc8`, verified in every run. The DeepForest package
is MIT-licensed (package metadata). For the model weights' terms, see the model card on Hugging Face.
