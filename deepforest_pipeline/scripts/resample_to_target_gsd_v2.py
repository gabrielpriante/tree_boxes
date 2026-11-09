"""
Versioned, parameterized GSD resampling for the low-altitude pilot (M02 v2).

Unlike the historical `resample_to_target_gsd.py` (frozen: native 1.67 cm/px for the 95 m imagery,
target 7.89 cm/px), this version takes the nominal native GSD and the target GSD as explicit
parameters and records every number needed to map boxes between the resampled and original images.

Preserved from the historical method: PIL Lanczos resampling, JPEG quality 95, aspect ratio,
output dimensions = round(original dimension * resize factor).

    python scripts/resample_to_target_gsd_v2.py INPUT.JPG --native-gsd-cm 0.472 --target-gsd-cm 10.0 \
        --output-dir outputs/lowalt_pilot_v1/preprocessed/lowalt_pilot_ptr1_27m_m02_v2

The historical script and its outputs are not modified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image

PREPROCESSING_VERSION = "gsd_resample_v2"
RESAMPLING = "PIL.Image.Resampling.LANCZOS"
JPEG_QUALITY = 95


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def plan_resample(width: int, height: int, native_gsd_cm: float, target_gsd_cm: float) -> dict:
    """Exact resize factor, integer output size and the per-axis factors actually applied."""
    if native_gsd_cm <= 0 or target_gsd_cm <= 0:
        raise ValueError("GSD values must be positive")
    factor = native_gsd_cm / target_gsd_cm
    out_w, out_h = round(width * factor), round(height * factor)
    if out_w < 1 or out_h < 1:
        raise ValueError(f"resampled image would be empty ({out_w}x{out_h})")
    sx, sy = out_w / width, out_h / height
    return {
        "native_gsd_cm": native_gsd_cm,
        "target_gsd_cm": target_gsd_cm,
        "resize_factor_exact": factor,
        "original_size": [width, height],
        "resampled_size": [out_w, out_h],
        # Applied factors come from the integer output size and may differ slightly per axis.
        "applied_scale_x": sx,
        "applied_scale_y": sy,
        "applied_vs_exact_relative_difference": [sx / factor - 1, sy / factor - 1],
        "forward_transform": "x_resampled = x_original * applied_scale_x; y_resampled = y_original * applied_scale_y",
        "inverse_transform": "x_original = x_resampled / applied_scale_x; y_original = y_resampled / applied_scale_y",
    }


def to_original(box, plan: dict) -> list:
    """Resampled-image pixel box [xmin, ymin, xmax, ymax] -> original-image pixels."""
    sx, sy = plan["applied_scale_x"], plan["applied_scale_y"]
    return [box[0] / sx, box[1] / sy, box[2] / sx, box[3] / sy]


def to_resampled(box, plan: dict) -> list:
    sx, sy = plan["applied_scale_x"], plan["applied_scale_y"]
    return [box[0] * sx, box[1] * sy, box[2] * sx, box[3] * sy]


def resample(input_path: Path, native_gsd_cm: float, target_gsd_cm: float, output_dir: Path) -> dict:
    """Write the resampled JPEG and a JSON provenance record into output_dir (which must not exist)."""
    input_path, output_dir = Path(input_path), Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"{output_dir} already exists; refusing to overwrite preprocessing outputs")
    with Image.open(input_path) as im:
        orientation = im.getexif().get(274, 1)
        if orientation not in (None, 1):
            raise ValueError(f"{input_path}: EXIF orientation {orientation}; not supported by this version")
        mode = im.mode
        plan = plan_resample(im.width, im.height, native_gsd_cm, target_gsd_cm)
        resized = im.resize(tuple(plan["resampled_size"]), Image.Resampling.LANCZOS)
    output_dir.mkdir(parents=True)
    out_path = output_dir / f"{input_path.stem}_gsd{target_gsd_cm:g}cm.jpg"
    resized.save(out_path, quality=JPEG_QUALITY)
    record = {
        "preprocessing_version": PREPROCESSING_VERSION,
        "resampling": RESAMPLING,
        "jpeg_quality": JPEG_QUALITY,
        "source_path": str(input_path),
        "source_sha256": sha256_file(input_path),
        "source_mode": mode,
        "output_path": str(out_path),
        "output_sha256": sha256_file(out_path),
        **plan,
    }
    (output_dir / "resample_record.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    return record


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", type=Path)
    ap.add_argument("--native-gsd-cm", type=float, required=True, help="Nominal native GSD (an estimate)")
    ap.add_argument("--target-gsd-cm", type=float, required=True)
    ap.add_argument("--output-dir", type=Path, required=True, help="New directory for this experiment's outputs")
    args = ap.parse_args(argv)
    rec = resample(args.input, args.native_gsd_cm, args.target_gsd_cm, args.output_dir)
    print(json.dumps({k: rec[k] for k in ("resampled_size", "resize_factor_exact", "applied_scale_x",
                                          "applied_scale_y", "output_sha256")}, indent=2))


if __name__ == "__main__":
    main()
