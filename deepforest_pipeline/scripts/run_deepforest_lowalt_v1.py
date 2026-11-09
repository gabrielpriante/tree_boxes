"""
DeepForest M01 / M02 v2 runner for the low-altitude pilot images (lowalt_pilot_v1).

Runs one inference of the pinned DeepForest checkpoint on one approved pilot image and records
everything needed to audit and import the result. It never writes into outputs/experiment_001/.

    HF_HUB_OFFLINE=1 .venv/bin/python scripts/run_deepforest_lowalt_v1.py --model M01 \
        --manifest /mnt/c/Users/gabpe/OneDrive/Documents/src/tree_boxes/manifests/manifest.csv \
        --image-id LA_PT-R1_27m --experiment-id lowalt_pilot_ptr1_27m_m01_v1 --run primary

    ... --model M02v2 --native-gsd-cm 0.472 --target-gsd-cm 10.0 ...

Configurations
  M01   historical "DeepForest Native" settings, original image.
  M02v2 the same inference settings on the image resampled to a target GSD with
        resample_to_target_gsd_v2.py. If the resampled image is smaller than one patch, it is padded
        with black pixels on the bottom/right to the patch size (no rescaling), so the detector sees a
        standard 400x400 window. See PADDING below.

Frozen inference settings (identical to run_deepforest_single.py / run_deepforest_gsd_7_89.py):
patch 400 px, overlap 0.05, cross-tile IoU 0.15, score 0.10, NMS 0.05, dataloader "single".
"""

from __future__ import annotations

import os

os.environ["HF_HUB_OFFLINE"] = "1"  # before any huggingface_hub import: never download or move revisions

import argparse
import csv
import hashlib
import json
import platform
import re
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
import resample_to_target_gsd_v2 as rs  # noqa: E402

RUNNER_VERSION = "lowalt_pilot_v1"
MODEL_NAME = "weecology/deepforest-tree"
MODEL_REVISION = "cc21436bc5d572dde8ff5f93c1e71a32f563cace"
CHECKPOINT_SHA256 = "d37a7af0b5ba2754282a80d41b4c7b66e8c7149234df5bf3f41bdaf57d329fc8"
CONFIG_JSON_SHA256 = "521e0149f0b060237c6994a69bffdff53ce5deb3beb854dad2752a3f8dd49bc4"
DEEPFOREST_VERSION = "2.1.0"

# Frozen historical inference settings (see run_deepforest_single.py).
PATCH_SIZE = 400
PATCH_OVERLAP = 0.05
IOU_THRESHOLD = 0.15
SCORE_THRESHOLD = 0.10
NMS_THRESHOLD = 0.05
DATALOADER_STRATEGY = "single"

# PADDING: DeepForest's sliding window shrinks to the image when the image is smaller than the patch,
# and the RetinaNet transform then upsamples every window to min side 800 px. A 387x290 image would be
# enlarged x2.76 instead of the x2.0 applied to every standard 400x400 window, silently changing the
# effective scale. Padding to the patch size keeps the standard window and scale; boxes keep their
# coordinates (top-left origin). Boxes lying wholly in the padding are dropped, boxes crossing into it are
# clipped, and both are counted.
PADDING_VERSION = "pad_to_patch_v1"
PAD_VALUE = 0

OUTPUT_ROOT = Path("outputs/lowalt_pilot_v1")
MODEL_DIRS = {"M01": "m01_native", "M02v2": "m02v2_gsd"}


def sha256_file(path: Path) -> str:
    return rs.sha256_file(path)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ------------------------------------------------------------------ inputs

def manifest_row(manifest: Path, image_id: str) -> dict:
    with open(manifest, newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["image_id"] == image_id]
    if len(rows) != 1:
        raise ValueError(f"{image_id} not found exactly once in {manifest}")
    return rows[0]


def resolve_source(manifest: Path, image_id: str) -> tuple:
    """Source image path (manifest paths are relative to the repository holding the manifest) and its
    verified SHA-256."""
    row = manifest_row(manifest, image_id)
    path = manifest.resolve().parent.parent / row["image_path"]
    actual = sha256_file(path)
    if actual != row["sha256"]:
        raise ValueError(f"{path}: SHA-256 {actual} does not match the manifest ({row['sha256']})")
    return path, actual, row


def hub_cache() -> Path:
    if os.environ.get("HF_HUB_CACHE"):
        return Path(os.environ["HF_HUB_CACHE"])
    return Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"


def verify_checkpoint(cache: Path | None = None) -> dict:
    """Locate the pinned snapshot and check its file hashes before anything is loaded."""
    snap = (cache or hub_cache()) / "models--weecology--deepforest-tree" / "snapshots" / MODEL_REVISION
    weights, cfg = snap / "model.safetensors", snap / "config.json"
    if not weights.exists():
        raise FileNotFoundError(f"pinned checkpoint revision {MODEL_REVISION} not in the local cache ({snap})")
    got = sha256_file(weights.resolve())
    if got != CHECKPOINT_SHA256:
        raise ValueError(f"model.safetensors SHA-256 {got} != expected {CHECKPOINT_SHA256}")
    got_cfg = sha256_file(cfg.resolve())
    if got_cfg != CONFIG_JSON_SHA256:
        raise ValueError(f"config.json SHA-256 {got_cfg} != expected {CONFIG_JSON_SHA256}")
    return {"model_name": MODEL_NAME, "revision": MODEL_REVISION, "weights_path": str(weights.resolve()),
            "model_safetensors_sha256": got, "config_json_sha256": got_cfg,
            "weights_bytes": weights.resolve().stat().st_size}


# ------------------------------------------------------------------ configuration

def build_config(model_code: str, native_gsd_cm: float | None = None, target_gsd_cm: float | None = None) -> dict:
    if model_code not in MODEL_DIRS:
        raise ValueError(f"unknown model code {model_code}")
    inference = {"patch_size": PATCH_SIZE, "patch_overlap": PATCH_OVERLAP, "iou_threshold": IOU_THRESHOLD,
                 "score_thresh": SCORE_THRESHOLD, "nms_thresh": NMS_THRESHOLD,
                 "dataloader_strategy": DATALOADER_STRATEGY}
    if model_code == "M01":
        if native_gsd_cm is not None or target_gsd_cm is not None:
            raise ValueError("M01 runs at native resolution; GSD arguments are not allowed")
        preprocessing = {"version": "none", "input": "original image, native resolution"}
    else:
        if not native_gsd_cm or not target_gsd_cm:
            raise ValueError("M02v2 needs --native-gsd-cm and --target-gsd-cm")
        preprocessing = {"version": rs.PREPROCESSING_VERSION, "resampling": rs.RESAMPLING,
                         "jpeg_quality": rs.JPEG_QUALITY, "native_gsd_cm": native_gsd_cm,
                         "native_gsd_basis": "nominal estimate from camera geometry (EXIF focal length and "
                                             "35 mm equivalent) and DJI relative altitude above the takeoff point; "
                                             "not a measured GSD",
                         "target_gsd_cm": target_gsd_cm,
                         "padding": {"version": PADDING_VERSION, "applies_when": "resampled side < patch_size",
                                     "pad_to": PATCH_SIZE, "anchor": "top-left (pad bottom/right)",
                                     "pad_value": PAD_VALUE}}
    return {"runner_version": RUNNER_VERSION, "model_code": model_code,
            "method": "DeepForest Native" if model_code == "M01" else "DeepForest GSD-Corrected v2",
            "deepforest_version": DEEPFOREST_VERSION, "model_name": MODEL_NAME, "model_revision": MODEL_REVISION,
            "model_safetensors_sha256": CHECKPOINT_SHA256, "inference": inference, "preprocessing": preprocessing,
            "output_coordinates": "original-image pixels, float"}


def config_id(cfg: dict) -> str:
    return f"{cfg['model_code']}_{sha256_text(json.dumps(cfg, sort_keys=True))[:10]}"


REPEAT_RE = re.compile(r"repeat_(0[1-9]|[1-9][0-9])")   # repeated-execution runs: repeat_01 .. repeat_99


def run_label(value: str) -> str:
    """argparse type: 'primary', 'verification' (pilot) or 'repeat_NN' (repeatability experiment)."""
    if value in ("primary", "verification") or REPEAT_RE.fullmatch(value):
        return value
    raise argparse.ArgumentTypeError(f"{value!r}: expected primary, verification or repeat_01..repeat_99")


def run_dir_for(root: Path, cfg: dict, experiment_id: str, image_id: str, run: str) -> Path:
    m = REPEAT_RE.fullmatch(run)
    if m:
        name = f"run_{m.group(1)}_repeat"
    else:
        index = {"primary": 1, "verification": 2}[run]
        name = f"run_{index:02d}_{run}"
    return root / MODEL_DIRS[cfg["model_code"]] / experiment_id / config_id(cfg) / image_id / name


# ------------------------------------------------------------------ preprocessing helpers

def pad_to_patch(array: np.ndarray, patch: int = PATCH_SIZE) -> tuple:
    """HxWx3 uint8 -> padded array and record. Pads only the short sides, bottom/right."""
    h, w = array.shape[:2]
    ph, pw = max(h, patch), max(w, patch)
    if (ph, pw) == (h, w):
        return array, None
    out = np.full((ph, pw, array.shape[2]), PAD_VALUE, dtype=array.dtype)
    out[:h, :w] = array
    return out, {"version": PADDING_VERSION, "image_size": [w, h], "padded_size": [pw, ph],
                 "pad_right": pw - w, "pad_bottom": ph - h, "pad_value": PAD_VALUE}


def remove_padding_artifacts(rows: list, width: int, height: int) -> tuple:
    """Drop boxes wholly inside the padding, clip boxes crossing into it. Returns (kept, dropped, clipped)."""
    kept, dropped, clipped = [], 0, 0
    for r in rows:
        x0, y0, x1, y1 = r["xmin"], r["ymin"], r["xmax"], r["ymax"]
        if x0 >= width or y0 >= height:
            dropped += 1
            continue
        if x1 > width or y1 > height:
            r = {**r, "xmax": min(x1, float(width)), "ymax": min(y1, float(height)), "clipped_at_padding": True}
            clipped += 1
        kept.append(r)
    return kept, dropped, clipped


def canonical_rows(rows: list) -> list:
    """Deterministic order for ids and comparisons: by coordinates, then score (full precision)."""
    return sorted(rows, key=lambda r: (r["xmin"], r["ymin"], r["xmax"], r["ymax"], -r["score"]))


def predictions_hash(rows: list) -> str:
    """Hash of processed-image boxes and scores at full float precision (no timestamps or paths)."""
    payload = [[repr(float(r[k])) for k in ("xmin", "ymin", "xmax", "ymax", "score")] + [str(r["label"])]
               for r in canonical_rows(rows)]
    return sha256_text(json.dumps(payload))


def window_diagnostics(width: int, height: int, rows: list) -> dict:
    import slidingwindow
    windows = slidingwindow.generateForSize(width, height, slidingwindow.DimOrder.ChannelHeightWidth,
                                            PATCH_SIZE, PATCH_OVERLAP)
    centers = np.array([[(r["xmin"] + r["xmax"]) / 2, (r["ymin"] + r["ymax"]) / 2] for r in rows]) if rows else np.zeros((0, 2))
    per = [int(((centers[:, 0] >= w.x) & (centers[:, 0] < w.x + w.w) & (centers[:, 1] >= w.y) & (centers[:, 1] < w.y + w.h)).sum())
           for w in windows] if len(centers) else [0] * len(windows)
    return {"n_windows": len(windows), "window_sizes": sorted({(w.w, w.h) for w in windows}),
            "max_final_boxes_centred_in_one_window": max(per) if per else 0,
            "model_detections_per_img_limit": 300,
            "note": "counts are after cross-window mosaic/NMS; a value near 300 would suggest the per-window cap was reached"}


def draw(image: Image.Image, boxes: list, path: Path, color=(255, 0, 255)) -> None:
    img = image.convert("RGB").copy()
    d = ImageDraw.Draw(img)
    lw = max(1, round(min(img.size) / 1500))
    for b in boxes:
        d.rectangle(b, outline=color, width=lw)
    img.save(path, quality=90)


def environment() -> dict:
    import deepforest
    import torch
    env = {"python": platform.python_version(), "platform": platform.platform(), "deepforest": deepforest.__version__,
           "torch": torch.__version__, "torch_cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
           "cuda_available": torch.cuda.is_available(),
           "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
           "cudnn_deterministic": torch.backends.cudnn.deterministic, "cudnn_benchmark": torch.backends.cudnn.benchmark,
           "hf_hub_offline": os.environ.get("HF_HUB_OFFLINE")}
    for pkg in ("numpy", "pillow", "pytorch-lightning", "huggingface-hub", "slidingwindow", "torchvision"):
        try:
            from importlib.metadata import version
            env[pkg] = version(pkg)
        except Exception:
            env[pkg] = None
    return env


# ------------------------------------------------------------------ model

def load_model():
    from deepforest import main as df_main
    model = df_main.deepforest()
    model.load_model(model_name=MODEL_NAME, revision=MODEL_REVISION)
    model.config.score_thresh = SCORE_THRESHOLD
    model.config.nms_thresh = NMS_THRESHOLD
    return model


def verify_loaded_weights(model, weights_path: str) -> dict:
    import torch
    from safetensors.torch import load_file
    file_sd, sd = load_file(weights_path), model.state_dict()
    common = set(file_sd) & set(sd)
    equal = len(common) == len(file_sd) and all(torch.equal(sd[k].cpu(), file_sd[k].cpu()) for k in common)
    if not equal:
        raise ValueError("loaded model weights differ from the pinned checkpoint file")
    t = model.model.transform
    return {"tensors_compared": len(common), "all_equal": True,
            "rcnn_transform": {"min_size": list(t.min_size), "max_size": t.max_size},
            "model_score_thresh": float(model.model.score_thresh), "model_nms_thresh": float(model.model.nms_thresh),
            "detections_per_img": int(model.model.detections_per_img)}


def predict(model, path: Path | None, array: np.ndarray | None) -> list:
    kwargs = dict(patch_size=PATCH_SIZE, patch_overlap=PATCH_OVERLAP, iou_threshold=IOU_THRESHOLD,
                  dataloader_strategy=DATALOADER_STRATEGY)
    df = model.predict_tile(path=str(path), **kwargs) if array is None else model.predict_tile(image=array, **kwargs)
    if df is None or len(df) == 0:  # DeepForest returns None when nothing is detected
        return []
    return [{"xmin": float(r.xmin), "ymin": float(r.ymin), "xmax": float(r.xmax), "ymax": float(r.ymax),
             "score": float(r.score), "label": str(r.label)} for r in df.itertuples()]


# ------------------------------------------------------------------ run

class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            st.write(s)
            st.flush()

    def flush(self):
        for st in self.streams:
            st.flush()


def run(model_code: str, manifest: Path, image_id: str, experiment_id: str, run_label: str,
        native_gsd_cm=None, target_gsd_cm=None, output_root: Path = OUTPUT_ROOT, model_loader=load_model,
        cache: Path | None = None, verify_weights: bool = True) -> Path:
    cfg = build_config(model_code, native_gsd_cm, target_gsd_cm)
    cid = config_id(cfg)
    if REPEAT_RE.fullmatch(run_label) and Path(output_root).resolve() == OUTPUT_ROOT.resolve():
        raise ValueError(f"repeat runs may not be written into the pilot output root {OUTPUT_ROOT}")
    run_dir = run_dir_for(output_root, cfg, experiment_id, image_id, run_label)
    if run_dir.exists():
        raise FileExistsError(f"{run_dir} already exists; refusing to overwrite an inference record")
    source, source_sha, row = resolve_source(manifest, image_id)
    checkpoint = verify_checkpoint(cache)
    run_dir.mkdir(parents=True)
    exp_dir = run_dir.parent.parent
    cfg_path = exp_dir / "model_config.json"
    if cfg_path.exists():
        if json.loads(cfg_path.read_text()) != cfg:
            raise ValueError(f"{cfg_path} differs from the current configuration")
    else:
        cfg_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    log = open(run_dir / "run.log", "w", encoding="utf-8")
    old_out, old_err = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = Tee(old_out, log), Tee(old_err, log)
    record = {"runner_version": RUNNER_VERSION, "experiment_id": experiment_id, "model_code": model_code,
              "model_config_id": cid, "run": run_label, "image_id": image_id, "site": row["site"],
              "altitude_m": row["altitude_m"], "source_path": str(source), "source_sha256": source_sha,
              "manifest_path": str(manifest), "manifest_sha256": sha256_file(manifest), "checkpoint": checkpoint,
              "config": cfg, "started_utc": now_utc(), "status": "started"}
    try:
        with Image.open(source) as im:
            original = im.convert("RGB")
        W, H = original.size
        record["original_size"] = [W, H]
        plan, padding, array, processed_path = None, None, None, source
        if model_code == "M02v2":
            rec = rs.resample(source, native_gsd_cm, target_gsd_cm, run_dir / "preprocessed")
            plan = {k: rec[k] for k in rec}
            processed_path = Path(rec["output_path"])
            with Image.open(processed_path) as pim:
                processed = pim.convert("RGB")
            arr, padding = pad_to_patch(np.array(processed))
            if padding:
                array = arr
                Image.fromarray(arr).save(run_dir / "preprocessed" / "padded_view_not_used_as_input.png")
            processed_size = list(processed.size)
        else:
            processed = original
            processed_size = [W, H]
        model_input_size = padding["padded_size"] if padding else processed_size
        record.update(preprocessing=plan, padding=padding, processed_size=processed_size,
                      model_input_size=model_input_size,
                      model_input="in-memory padded array (decoded resampled JPEG, RGB uint8)" if padding else
                      f"image path {processed_path}")
        record["environment"] = environment()

        t_load = time.perf_counter()
        model = model_loader()
        record["model_load_seconds"] = round(time.perf_counter() - t_load, 3)
        if verify_weights:
            record["loaded_weights"] = verify_loaded_weights(model, checkpoint["weights_path"])
        t0 = time.perf_counter()
        rows = predict(model, None if padding else processed_path, array)
        record["inference_seconds"] = round(time.perf_counter() - t0, 3)

        with open(run_dir / "predictions_raw.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["xmin", "ymin", "xmax", "ymax", "score", "label"])
            w.writeheader()
            w.writerows([{k: (repr(v) if isinstance(v, float) else v) for k, v in r.items()} for r in rows])
        record["raw_detections"] = len(rows)
        record["raw_predictions_hash"] = predictions_hash(rows)

        if padding:
            rows, dropped, clipped = remove_padding_artifacts(rows, *processed_size)
            record["padding"] = {**padding, "boxes_dropped_in_padding": dropped, "boxes_clipped_at_padding": clipped}
        rows = canonical_rows(rows)
        trees = []
        for i, r in enumerate(rows, start=1):
            pbox = [r["xmin"], r["ymin"], r["xmax"], r["ymax"]]
            obox = rs.to_original(pbox, plan) if plan else pbox
            trees.append({"tree_id": f"D{i:05d}", "bbox_px": obox, "bbox_processed_px": pbox,
                          "score": r["score"], "label": r["label"],
                          **({"clipped_at_padding": True} if r.get("clipped_at_padding") else {})})
        detections = {"coordinate_system": "original_pixels", "image_id": image_id, "image_size": [W, H],
                      "processed_size": processed_size, "model_config_id": cid, "trees": trees}
        (run_dir / "detections.json").write_text(json.dumps(detections, indent=1), encoding="utf-8")
        record["detections"] = len(trees)
        record["predictions_hash"] = predictions_hash(rows)
        record["window_diagnostics"] = window_diagnostics(*model_input_size, rows)
        draw(original, [t["bbox_px"] for t in trees], run_dir / "annotated_original.jpg")
        if plan:
            draw(processed, [t["bbox_processed_px"] for t in trees], run_dir / "annotated_processed.jpg")
        record["status"] = "success"
    except BaseException as e:  # record every failure, including interrupts
        record["status"] = "error"
        record["error"] = f"{type(e).__name__}: {e}"
        (run_dir / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise
    finally:
        record["finished_utc"] = now_utc()
        (run_dir / "run_record.json").write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")
        sys.stdout, sys.stderr = old_out, old_err
        log.close()
    return run_dir


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=sorted(MODEL_DIRS))
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--image-id", required=True)
    ap.add_argument("--experiment-id", required=True)
    ap.add_argument("--run", required=True, type=run_label,
                    help="primary | verification (pilot) | repeat_01..repeat_99 (repeatability experiment)")
    ap.add_argument("--native-gsd-cm", type=float)
    ap.add_argument("--target-gsd-cm", type=float)
    ap.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = ap.parse_args(argv)
    out = run(args.model, args.manifest, args.image_id, args.experiment_id, args.run,
              args.native_gsd_cm, args.target_gsd_cm, args.output_root)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
