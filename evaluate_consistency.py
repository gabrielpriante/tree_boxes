#!/usr/bin/env python3
"""
Offline evaluation of repeated detections: count consistency, spatial (IoU) consistency,
detection persistence, human-review materials and, only once a complete human reference exists,
accuracy metrics. Makes no API requests and never writes inside benchmark_results/.

    python evaluate_consistency.py                     # every experiment in benchmark_results/
    python evaluate_consistency.py --experiments lowalt_pilot_ptr1_27m_v1 lowalt_pilot_pt02_64m_v1

Outputs (regenerated on every run): evaluation_results/
Human input (created once, never overwritten): human_review/
Methods and limitations: EVALUATION.md
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import re
import shutil
import statistics
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from fractions import Fraction
from itertools import combinations
from pathlib import Path

from PIL import Image

import box_generation as bg
import evaluation_accuracy as ea
import evaluation_core as ec
import evaluation_geometry as eg

ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "benchmark_results"
OUTPUT_DIR = ROOT / "evaluation_results"
REVIEW_DIR = ROOT / "human_review"
MANIFEST = ROOT / "manifests" / "manifest.csv"
MARKER = "evaluation_config.json"

REVIEW_LABELS = ("TREE", "NON_TREE", "UNCERTAIN")
REFERENCE_STATUSES = ("NONE", "PARTIAL", ec.REFERENCE_COMPLETE)
REVIEW_THRESHOLD = Fraction(1, 2)  # review units are the persistence groups at IoU >= 0.50
SENSOR_DIAGONAL_35MM = math.hypot(36, 24)


class InputError(RuntimeError):
    pass


# ------------------------------------------------------------------ small helpers

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_csv(path: Path) -> list:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, fields: list, rows: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="raise")
        w.writeheader()
        w.writerows([{k: _cell(r.get(k)) for k in fields} for r in rows])


def _cell(v):
    if v is None:
        return ""
    if isinstance(v, Fraction):
        v = float(v)
    if isinstance(v, float):
        text = f"{v:.6f}".rstrip("0").rstrip(".")
        return "0" if text in ("", "-0") else text
    if isinstance(v, (list, tuple)):
        return ";".join(str(_cell(x)) for x in v)
    return v


def fnum(v, nd=3):
    return "–" if v is None else f"{float(v):.{nd}f}"


def rel(path: Path, base: Path) -> str:
    try:
        return Path(path).resolve().relative_to(base.resolve()).as_posix()
    except ValueError:
        return Path(path).as_posix()


def box_str(box) -> str:
    return ";".join(f"{float(v):.3f}".rstrip("0").rstrip(".") for v in box)


# ------------------------------------------------------------------ loading saved benchmark results

@dataclass
class ImageTrials:
    """Successful trials of one model configuration on one image, verified against the saved CSVs."""
    experiment_id: str
    model_config_id: str
    model: str
    image_id: str
    site: str
    altitude_m: str
    image_path: Path
    image_sha256: str
    width: int
    height: int
    upload_width: int
    upload_height: int
    sets: list
    excluded: list = field(default_factory=list)   # (trial, status) of non-success trials
    saved_summary: dict | None = None
    inputs: list = field(default_factory=list)       # (path, sha256)
    coordinate_system: str = "normalized_0_1000"     # or "original_pixels" (DeepForest, float boxes)
    scores: dict = field(default_factory=dict)       # detection uid -> model confidence (DeepForest only)

    @property
    def key(self) -> str:
        return f"{self.experiment_id}/{self.image_id}"

    @property
    def n_trials(self) -> int:
        return len(self.sets)

    @property
    def single_output(self) -> bool:
        """One deterministic output (DeepForest): no between-trial repeatability can be measured."""
        return self.n_trials < 2

    @property
    def counts(self) -> list:
        return [len(s.detections) for s in sorted(self.sets, key=lambda s: s.trial)]


def detection_uid(experiment_id: str, image_id: str, trial: int, tree_id: str) -> str:
    return f"{experiment_id}:{image_id}:t{trial:02d}:{tree_id}"


def load_experiment(exp_dir: Path, image_root: Path, base: Path) -> list:
    """Read per-trial records and detections.json, cross-checking counts_long.csv, boxes_long.csv,
    counts_summary.csv and the source image hash. Any disagreement stops the evaluation."""
    experiment_id = exp_dir.name
    long_rows = read_csv(exp_dir / "counts_long.csv")
    box_rows = read_csv(exp_dir / "boxes_long.csv")
    summary = {(r["image_id"], r["model_config_id"]): r for r in read_csv(exp_dir / "counts_summary.csv")}
    inputs = [(rel(exp_dir / n, base), sha256_file(exp_dir / n))
              for n in ("counts_long.csv", "boxes_long.csv", "counts_summary.csv")]

    saved_boxes = {}
    for r in box_rows:
        k = (r["model_config_id"], r["image_id"], int(r["trial"]))
        if "x_min_px" in r:   # DeepForest import: float original-image pixels, never the 0-1000 grid
            box = [float(r[c]) for c in ("x_min_px", "y_min_px", "x_max_px", "y_max_px")]
            saved_boxes.setdefault(k, []).append((r["tree_id"], box, float(r["score"])))
        else:
            saved_boxes.setdefault(k, []).append((r["tree_id"], [int(r[c]) for c in ("x_min", "y_min", "x_max", "y_max")]))

    units = {}
    for r in sorted(long_rows, key=lambda r: (r["model_config_id"], r["image_id"], int(r["trial"]))):
        cfg_id, image_id, trial = r["model_config_id"], r["image_id"], int(r["trial"])
        record_path = exp_dir / r["record_path"]
        record = json.loads(record_path.read_text(encoding="utf-8"))
        if (record["experiment_id"], record["image_id"], record["trial"]) != (experiment_id, image_id, trial):
            raise InputError(f"{record_path}: record does not match counts_long.csv row")
        img = record["image"]
        u = units.get((cfg_id, image_id))
        if u is None:
            image_path = image_root / record["image_path"]
            if not image_path.exists():
                raise InputError(f"source image not found: {image_path}")
            src_sha = sha256_file(image_path)
            if src_sha != img["file_sha256"]:
                raise InputError(f"{image_path}: SHA-256 differs from the hash recorded at inference time")
            with Image.open(image_path) as im:
                size = im.size
            if size != (img["original_width"], img["original_height"]):
                raise InputError(f"{image_path}: size {size} differs from the recorded original size")
            u = units[(cfg_id, image_id)] = ImageTrials(
                experiment_id, cfg_id, record["model"], image_id, record["site"], str(record["altitude_m"]),
                image_path, src_sha, size[0], size[1], img["upload_width"], img["upload_height"], [],
                saved_summary=summary.get((image_id, cfg_id)),
                inputs=[(rel(image_path, base), src_sha)],
                coordinate_system=record.get("coordinate_system", "normalized_0_1000"))
        elif img["file_sha256"] != u.image_sha256:
            raise InputError(f"{record_path}: trial used a different source image hash")
        if record.get("coordinate_system", "normalized_0_1000") != u.coordinate_system:
            raise InputError(f"{record_path}: coordinate system differs between trials")

        if r["status"] != "success" or record["status"] != "success":
            u.excluded.append((trial, r["status"]))
            continue
        det_path = record_path.parent / record["files"]["detections"]
        det = json.loads(det_path.read_text(encoding="utf-8"))
        u.inputs += [(rel(record_path, base), sha256_file(record_path)), (rel(det_path, base), sha256_file(det_path))]
        trees = det["trees"]
        if len(trees) != int(r["tree_count"]) or len(trees) != record["tree_count"]:
            raise InputError(f"{det_path}: {len(trees)} boxes but tree_count is {r['tree_count']}")
        if u.coordinate_system == "original_pixels":
            if det.get("coordinate_system") != "original_pixels":
                raise InputError(f"{det_path}: record says original_pixels but detections.json does not")
            if [(t["tree_id"], list(t["bbox_px"]), t["score"]) for t in trees] != saved_boxes.get((cfg_id, image_id, trial), []):
                raise InputError(f"{det_path}: boxes or scores differ from boxes_long.csv")
            dets = tuple(ec.Detection(detection_uid(experiment_id, image_id, trial, t["tree_id"]), t["tree_id"],
                                      ec.box_from_pixels(t["bbox_px"])) for t in trees)
            for d, t in zip(dets, trees):
                u.scores[d.uid] = float(t["score"])
        else:
            if [(t["tree_id"], list(t["bbox"])) for t in trees] != saved_boxes.get((cfg_id, image_id, trial), []):
                raise InputError(f"{det_path}: boxes differ from boxes_long.csv")
            dets = tuple(ec.Detection(detection_uid(experiment_id, image_id, trial, t["tree_id"]), t["tree_id"],
                                      ec.box_from_normalized(t["bbox"], u.width, u.height)) for t in trees)
        u.sets.append(ec.DetectionSet(f"{experiment_id}/{cfg_id}", image_id, u.image_sha256, u.width, u.height,
                                      trial, dets))

    for u in units.values():
        u.inputs = inputs + u.inputs
        s = u.saved_summary
        if s is None or int(s["successful_trials"]) != u.n_trials:
            raise InputError(f"{u.key}: counts_summary.csv disagrees with the number of successful records")
    return [units[k] for k in sorted(units)]


def discover_experiments(results_dir: Path, names=None) -> list:
    if names:
        dirs = [results_dir / n for n in names]
        missing = [d for d in dirs if not (d / "counts_long.csv").exists()]
        if missing:
            raise InputError(f"no counts_long.csv in: {', '.join(map(str, missing))}")
        return dirs
    return sorted(d for d in results_dir.iterdir() if (d / "counts_long.csv").exists())


# ------------------------------------------------------------------ evaluation

@dataclass
class ThresholdResult:
    threshold: Fraction
    pairs: list
    summary: dict
    groups: list
    geometry: dict          # group_id -> geometry dict
    frequency: dict         # uid -> number of other trials matched


def evaluate_unit(u: ImageTrials, thresholds) -> dict:
    out = {}
    if u.single_output:
        # A single deterministic output has no trial pairs: repeatability and persistence are not applicable.
        return {t: ThresholdResult(t, [], ec.summarize_pairs([]), [], {}, {}) for t in thresholds}
    for t in thresholds:
        pairs = ec.pairwise_trials(u.sets, t)
        groups = ec.persistence_groups(u.sets, pairs)
        pair_iou = {frozenset((a, b)): v for r in pairs for a, b, v in r["pairs"]}
        geometry = {g.group_id: ec.group_geometry(g, pair_iou) for g in groups}
        out[t] = ThresholdResult(t, pairs, ec.summarize_pairs(pairs), groups, geometry,
                                 ec.detection_match_frequency(u.sets, pairs))
    return out


def count_stats(counts: list) -> dict:
    n = len(counts)
    mean = statistics.fmean(counts) if n else None
    sd = statistics.stdev(counts) if n >= 2 else None
    return {"n": n, "mean": mean, "sd": sd, "min": min(counts) if n else None, "max": max(counts) if n else None,
            "cv": sd / mean if sd is not None and mean else None}


def verify_counts(u: ImageTrials, stats: dict) -> bool:
    s = u.saved_summary
    return (abs(float(s["mean_tree_count"]) - stats["mean"]) < 5e-4
            and (stats["sd"] is None or abs(float(s["standard_deviation"]) - stats["sd"]) < 5e-4)
            and int(s["minimum_count"]) == stats["min"] and int(s["maximum_count"]) == stats["max"])


# ------------------------------------------------------------------ human review input

REVIEW_FIELDS = ["review_unit_id", "experiment_id", "model_config_id", "model", "site", "image_id", "unit_type",
                 "group_id_iou050", "group_status_iou050", "n_trials", "trials_detected", "persistence_bin",
                 "n_detections", "detection_uids", "median_box_px", "crop_path",
                 "human_label", "human_note", "reviewer", "review_date"]
REFERENCE_FIELDS = ["reference_id", "image_id", "x_min_px", "y_min_px", "x_max_px", "y_max_px", "label",
                    "annotator", "annotation_date", "notes"]
STATUS_FIELDS = ["image_id", "image_sha256", "reference_status", "annotator", "status_date", "notes"]
INVENTORY_FIELDS = ["site", "inventory_tree_count", "scope", "source", "recorded", "notes"]
INITIAL_INVENTORY = [
    {"site": "PT-02", "inventory_tree_count": 24, "scope": "SITE", "source": "field inventory (reported by project lead)",
     "recorded": "2026-10-30", "notes": "Site-level count. Not an image-level reference: image coverage_scope is UNKNOWN."},
    {"site": "PT-R1", "inventory_tree_count": 11, "scope": "SITE", "source": "field inventory (reported by project lead)",
     "recorded": "2026-10-30", "notes": "Site-level count. Not an image-level reference: image coverage_scope is UNKNOWN."},
]

REVIEW_README = """# Human review files

These files hold human judgements. `evaluate_consistency.py` creates each one only if it is
missing and never overwrites it. Everything in `evaluation_results/` is regenerated; nothing here is.

| File | Purpose |
|---|---|
| `manual_review.csv` | One row per review unit (a recurring detection group, or a single detection inside an ambiguous group). Fill `human_label` with `TREE`, `NON_TREE` or `UNCERTAIN`; `human_note`, `reviewer`, `review_date` are optional. Look at the image in `crop_path` (and the contact sheet) to decide. Leave a row blank if not reviewed. |
| `reference_boxes.csv` | Human reference boxes, in **pixels of the original image** (8192×6144 for the October 24 images), e.g. for trees the model missed. One row per tree; `label` is `TREE`. |
| `reference_status.csv` | Per image: `NONE`, `PARTIAL` or `COMPLETE`. Precision, recall and F1 are computed only for images marked `COMPLETE` (every visible tree in the image has a reference box). |
| `field_inventory.csv` | Site-level field counts. Reported next to the model's counts, never used as an image-level reference or to filter predictions. |

`manual_review.csv` is a copy of `evaluation_results/manual_review_template.csv` at the time it was
created. If the evaluation algorithm or inputs change, review unit ids may change; the evaluation
report lists any labelled id it no longer recognises. New units are not added automatically:
copy rows over from the template if needed.

## Which threshold the labels refer to

Review units are the persistence groups at **IoU >= 0.50** (`group_id_iou050`). A label applies to that
unit's detections (`detection_uids`), not to any group id at IoU 0.75 or 1.00: the same id at another
threshold is generally a different set of detections. Labels are never transferred between thresholds;
any such transfer would need an explicit detection-level mapping (the `detection_uids` column and
`evaluation_results/detections.csv` provide the ids for that).

## Editing in Excel without corrupting the file

Opening the CSV by double-clicking lets Excel convert values: `8-9`, `5-7`, `1-4` in `persistence_bin`
become dates (`9-Aug`, `7-May`, `4-Jan`), and the change is written back when the file is saved.
Only the `human_label`, `human_note`, `reviewer` and `review_date` columns should ever change.

1. Do not double-click the CSV. In Excel: **Data > From Text/CSV**, choose the file, click
   **Transform Data**, select all columns, set **Data Type > Text**, then **Close & Load**.
   (Or **Data > Get Data > Legacy Wizards > From Text (Legacy)**: Delimited, Comma, then mark at least
   `persistence_bin`, `trials_detected`, `n_trials` and `review_date` as **Text** in the last step.)
2. Edit only the four human columns. Save as **CSV UTF-8 (Comma delimited)** to `manual_review.csv`.
3. Close the file in Excel (Excel locks open files, so scripts cannot read or write them).
4. Run `python evaluate_consistency.py --validate-review`. It writes nothing; it reports any
   `persistence_bin` that does not match `n_trials` (flagging date-like values), any `trials_detected` that
   disagrees with `n_trials`, duplicate ids, and any generated column that no longer matches the evaluation.
   A full evaluation run performs the same check first and stops without writing outputs if it fails.
   The check never rewrites your file.

A text editor (VS Code) or LibreOffice Calc with the columns imported as Text avoids the problem entirely.
"""


def ensure_review_files(review_dir: Path, template_rows: list, units: list) -> list:
    created = []
    review_dir.mkdir(parents=True, exist_ok=True)
    targets = [
        ("README.md", lambda p: p.write_text(REVIEW_README, encoding="utf-8")),
        ("manual_review.csv", lambda p: write_csv(p, REVIEW_FIELDS, template_rows)),
        ("reference_boxes.csv", lambda p: write_csv(p, REFERENCE_FIELDS, [])),
        ("reference_status.csv", lambda p: write_csv(p, STATUS_FIELDS, [
            {"image_id": i, "image_sha256": s, "reference_status": "NONE"}
            for i, s in sorted({(u.image_id, u.image_sha256) for u in units})])),
        ("field_inventory.csv", lambda p: write_csv(p, INVENTORY_FIELDS, INITIAL_INVENTORY)),
    ]
    for name, make in targets:
        p = review_dir / name
        if not p.exists():
            make(p)
            created.append(name)
    return created


EXCEL_DATE_RE = re.compile(r"^(\d{1,2}-[A-Za-z]{3}|[A-Za-z]{3}-\d{1,2})$")  # e.g. 9-Aug, Aug-9
HUMAN_FIELDS = {"human_label", "human_note", "reviewer", "review_date"}


def validate_review_file(path: Path, template: list, n_trials_by_image: dict) -> list:
    """Problems that make a hand-edited manual_review.csv unsafe to use. Reports; never rewrites.

    Checks every row's persistence_bin is a valid category that agrees with n_trials (catching spreadsheet
    date conversion such as '8-9' -> '9-Aug'), that trials_detected lists n_trials distinct trials, that ids
    are unique, and that the generated (non-human) columns still equal the evaluation's template."""
    if not path.exists():
        return []
    errors = []
    by_id = {r["review_unit_id"]: {k: str(_cell(v)) for k, v in r.items()} for r in template}
    rows = read_csv(path)
    with open(path, newline="", encoding="utf-8") as f:
        header = next(csv.reader(f), [])
    missing = [c for c in REVIEW_FIELDS if c not in header]
    if missing:
        return [f"missing columns: {', '.join(missing)}"]
    seen = set()
    for n, r in enumerate(rows, start=2):
        uid = r["review_unit_id"]
        where = f"line {n} ({uid})"
        if uid in seen:
            errors.append(f"{where}: duplicate review_unit_id")
        seen.add(uid)
        try:
            k = int(r["n_trials"])
            trials = [int(t) for t in r["trials_detected"].split(";")]
        except ValueError:
            errors.append(f"{where}: unreadable n_trials {r['n_trials']!r} / trials_detected {r['trials_detected']!r}")
            continue
        if len(trials) != k or len(set(trials)) != k:
            errors.append(f"{where}: trials_detected lists {len(set(trials))} distinct trials but n_trials is {k}")
        total = n_trials_by_image.get((r["experiment_id"], r["image_id"]))
        b = r["persistence_bin"]
        if total is not None:
            expected = ec.persistence_bin(k, total) if 1 <= k <= total else None
            if b != expected:
                hint = " (looks like a spreadsheet converted it to a date)" if EXCEL_DATE_RE.match(b.strip()) else ""
                errors.append(f"{where}: persistence_bin {b!r} should be {expected!r} for n_trials={k}{hint}")
        t = by_id.get(uid)
        if t is not None:
            changed = [c for c in REVIEW_FIELDS if c not in HUMAN_FIELDS and c != "persistence_bin" and r[c] != t[c]]
            if changed:
                errors.append(f"{where}: generated column(s) differ from the evaluation: {', '.join(changed)}")
    return errors


def load_review(review_dir: Path) -> dict:
    out = {"labels": {}, "label_errors": [], "references": {}, "reference_errors": [], "status": {}, "inventory": {}}
    p = review_dir / "manual_review.csv"
    if p.exists():
        for r in read_csv(p):
            label = (r.get("human_label") or "").strip().upper()
            if not label:
                continue
            if label not in REVIEW_LABELS:
                out["label_errors"].append(f"{r.get('review_unit_id')}: label {label!r} not in {REVIEW_LABELS}")
                continue
            out["labels"][r["review_unit_id"]] = {"label": label, "note": r.get("human_note", "")}
    p = review_dir / "reference_boxes.csv"
    if p.exists():
        for n, r in enumerate(read_csv(p), start=2):
            try:
                box = ec.box_from_pixels([r[c] for c in ("x_min_px", "y_min_px", "x_max_px", "y_max_px")])
            except (ValueError, KeyError) as e:
                out["reference_errors"].append(f"reference_boxes.csv line {n}: {e}")
                continue
            if (r.get("label") or "TREE").strip().upper() != "TREE":
                continue
            out["references"].setdefault(r["image_id"], []).append((r.get("reference_id") or f"line{n}", box))
    p = review_dir / "reference_status.csv"
    if p.exists():
        for r in read_csv(p):
            status = (r.get("reference_status") or "NONE").strip().upper()
            if status not in REFERENCE_STATUSES:
                out["reference_errors"].append(f"reference_status.csv {r['image_id']}: {status!r} not in {REFERENCE_STATUSES}")
                status = "NONE"
            out["status"][r["image_id"]] = (status, r.get("image_sha256", ""))
    p = review_dir / "field_inventory.csv"
    if p.exists():
        for r in read_csv(p):
            out["inventory"][r["site"]] = r
    return out


def reference_set(u: ImageTrials, review: dict) -> ec.DetectionSet | None:
    refs = review["references"].get(u.image_id)
    if not refs:
        return None
    for _, box in refs:
        if box[2] > u.width or box[3] > u.height or box[0] < 0 or box[1] < 0:
            raise InputError(f"reference box {box_str(box)} lies outside {u.image_id} ({u.width}x{u.height})")
    dets = tuple(ec.Detection(f"REF:{u.image_id}:{rid}", rid, box) for rid, box in refs)
    return ec.DetectionSet("human_reference", u.image_id, u.image_sha256, u.width, u.height, None, dets)


# ------------------------------------------------------------------ altitude-series organisation

def xmp_values(path: Path) -> dict:
    with Image.open(path) as im:
        xmp = im.info.get("xmp") or b""
        exif = im.getexif().get_ifd(0x8769)
    s = xmp.decode("utf-8", "ignore") if isinstance(xmp, bytes) else str(xmp)
    vals = dict(re.findall(r'drone-dji:(\w+)="([^"]*)"', s))
    vals["FocalLengthIn35mmFilm"] = exif.get(0xA405)
    return vals


def ground_sampling(alt_m: float, f35: float, width: int, height: int) -> tuple:
    """Approximate nadir footprint (m) and GSD (m/px) from the 35 mm-equivalent focal length
    (diagonal equivalence). Uses height above the TAKEOFF point, not above canopy or ground below."""
    diag_m = alt_m * SENSOR_DIAGONAL_35MM / f35
    w_m = diag_m * width / math.hypot(width, height)
    h_m = diag_m * height / math.hypot(width, height)
    return w_m, h_m, w_m / width


def altitude_series_rows(manifest: Path, image_root: Path, evaluated: list) -> list:
    rows = []
    done = {}
    for u in evaluated:
        done.setdefault(u.image_id, []).append(u.experiment_id)
    for m in read_csv(manifest):
        if m.get("collection") != "low_altitude":
            continue
        path = image_root / m["image_path"]
        x = xmp_values(path) if path.exists() else {}
        w, h = int(m["width"]), int(m["height"])
        uw, uh = bg.api_resized_size(w, h)
        f35 = float(x.get("FocalLengthIn35mmFilm") or 0)
        alt = float(m["altitude_m"])
        fw, fh, gsd = ground_sampling(alt, f35, w, h) if f35 else (None, None, None)
        rows.append({
            "image_id": m["image_id"], "site": m["site"], "relative_altitude_m": alt,
            "capture_datetime": m["capture_datetime"], "gimbal_pitch_deg": m["gimbal_pitch_deg"],
            "gimbal_yaw_deg": m["gimbal_yaw_deg"],
            "heading_flagged": "capture_geometry" in m.get("needs_review", "").split(";") or "rotated" in m.get("notes", ""),
            "gps_lat": x.get("GpsLatitude", "").lstrip("+"), "gps_lon": x.get("GpsLongitude", ""),
            "coverage_scope": m["coverage_scope"], "original_px": f"{w}x{h}", "api_px": f"{uw}x{uh}",
            "api_downscale_factor": w / uw, "focal_35mm_equiv": f35 or None,
            "est_footprint_m": f"{fw:.1f}x{fh:.1f}" if fw else None,
            "est_gsd_original_mm": gsd * 1000 if gsd else None,
            "est_gsd_api_mm": gsd * 1000 * w / uw if gsd else None,
            "model_outputs": ";".join(sorted(done.get(m["image_id"], []))),
        })
    rows.sort(key=lambda r: (r["site"], r["relative_altitude_m"], r["image_id"]))
    return rows


# ------------------------------------------------------------------ outputs

def software_versions() -> dict:
    import matplotlib
    import numpy
    import PIL
    import scipy
    return {"python": platform.python_version(), "numpy": numpy.__version__, "scipy": scipy.__version__,
            "matplotlib": matplotlib.__version__, "pillow": PIL.__version__}


def git_state(root: Path) -> dict:
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--", "evaluation_core.py", "evaluate_consistency.py",
                                "evaluation_plots.py"], cwd=root, capture_output=True, text=True).stdout.strip()
        return {"commit": head, "evaluation_code_uncommitted": bool(dirty)}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "evaluation_code_uncommitted": None}


def prepare_output_dir(out: Path) -> None:
    if out.exists():
        if not (out / MARKER).exists():
            raise InputError(f"{out} exists but was not created by this script; choose another --output-dir")
        shutil.rmtree(out)
    out.mkdir(parents=True)
    # Marker first, so an interrupted run can be regenerated; replaced by the full config at the end.
    (out / MARKER).write_text(json.dumps({"status": "incomplete"}), encoding="utf-8")


def run(results_dir: Path = RESULTS_DIR, experiments=None, output_dir: Path = OUTPUT_DIR,
        review_dir: Path = REVIEW_DIR, manifest: Path = MANIFEST, image_root: Path = ROOT,
        figures: bool = True, full_res_maps: bool = True, validate_only: bool = False, log=print) -> dict:
    thresholds = ec.THRESHOLDS
    if output_dir.resolve() == results_dir.resolve() or results_dir.resolve() in output_dir.resolve().parents:
        raise InputError("the evaluation may not write inside the benchmark results directory")
    base = image_root
    units = [u for d in discover_experiments(results_dir, experiments) for u in load_experiment(d, image_root, base)]
    if not units:
        raise InputError(f"no successful trials found under {results_dir}")
    log(f"Loaded {len(units)} image/experiment unit(s): " + ", ".join(f"{u.key} ({u.n_trials} trials)" for u in units))

    results = {u.key: evaluate_unit(u, thresholds) for u in units}
    stats = {u.key: count_stats(u.counts) for u in units}

    # -- review units (persistence groups at IoU >= 0.50)
    template = []
    unit_crops = {}
    for u in units:
        tr = results[u.key][REVIEW_THRESHOLD]
        for g in tr.groups:
            geo = tr.geometry[g.group_id]
            common = {"experiment_id": u.experiment_id, "model_config_id": u.model_config_id, "model": u.model,
                      "site": u.site, "image_id": u.image_id, "group_id_iou050": g.group_id,
                      "group_status_iou050": g.status, "n_trials": g.n_trials,
                      "trials_detected": g.trials, "persistence_bin": ec.persistence_bin(g.n_trials, u.n_trials)}
            if g.status == "CONSISTENT":
                uid = f"{u.experiment_id}:{u.image_id}:{g.group_id}"
                crop = Path("review") / "crops" / u.experiment_id / u.image_id / f"{g.group_id}.jpg"
                template.append({**common, "review_unit_id": uid, "unit_type": "GROUP", "n_detections": len(g.members),
                                 "detection_uids": [d.uid for _, d in g.members],
                                 "median_box_px": box_str(geo["median_box_px"]), "crop_path": crop.as_posix()})
                unit_crops[uid] = (u, g, None, crop)
            else:
                for trial, d in g.members:
                    uid = f"{u.experiment_id}:{u.image_id}:{g.group_id}:t{trial:02d}:{d.label}"
                    crop = Path("review") / "crops" / u.experiment_id / u.image_id / f"{g.group_id}_t{trial:02d}_{d.label}.jpg"
                    template.append({**common, "review_unit_id": uid, "unit_type": "DETECTION_IN_AMBIGUOUS_GROUP",
                                     "n_detections": 1, "detection_uids": d.uid, "median_box_px": box_str(d.box),
                                     "crop_path": crop.as_posix()})
                    unit_crops[uid] = (u, g, d, crop)

    # Check hand-edited review data before anything is written or used.
    review_errors = validate_review_file(review_dir / "manual_review.csv", template,
                                         {(u.experiment_id, u.image_id): u.n_trials for u in units})
    if review_errors:
        raise InputError("human_review/manual_review.csv failed validation; it was not modified and no outputs "
                         "were written:\n  " + "\n  ".join(review_errors))
    if validate_only:
        log(f"{review_dir / 'manual_review.csv'}: OK")
        return {"units": units, "review_errors": []}

    prepare_output_dir(output_dir)
    write_csv(output_dir / "manual_review_template.csv", REVIEW_FIELDS, template)
    write_csv(output_dir / "reference_boxes_template.csv", REFERENCE_FIELDS, [])
    created = ensure_review_files(review_dir, template, units)
    review = load_review(review_dir)

    # -- CSVs
    write_csv(output_dir / "count_consistency.csv",
              ["experiment_id", "model_config_id", "model", "site", "image_id", "altitude_m", "successful_trials",
               "excluded_trials", "counts_by_trial", "mean_count", "sd_count", "min_count", "max_count", "cv_count",
               "matches_counts_summary_csv"],
              [{"experiment_id": u.experiment_id, "model_config_id": u.model_config_id, "model": u.model,
                "site": u.site, "image_id": u.image_id, "altitude_m": u.altitude_m, "successful_trials": u.n_trials,
                "excluded_trials": [f"t{t:02d}:{s}" for t, s in u.excluded], "counts_by_trial": u.counts,
                "mean_count": stats[u.key]["mean"], "sd_count": stats[u.key]["sd"], "min_count": stats[u.key]["min"],
                "max_count": stats[u.key]["max"], "cv_count": stats[u.key]["cv"],
                "matches_counts_summary_csv": verify_counts(u, stats[u.key])} for u in units])

    inv_rows = []
    for u in units:
        inv = review["inventory"].get(u.site)
        if inv:
            n = float(inv["inventory_tree_count"])
            inv_rows.append({"experiment_id": u.experiment_id, "site": u.site, "image_id": u.image_id,
                             "mean_model_count": stats[u.key]["mean"], "site_field_inventory_count": inv["inventory_tree_count"],
                             "inventory_scope": inv["scope"], "image_coverage_scope": _coverage(manifest, u.image_id),
                             "difference_mean_minus_inventory": stats[u.key]["mean"] - n,
                             "used_as_image_ground_truth": False, "notes": inv.get("notes", "")})
    write_csv(output_dir / "count_vs_field_inventory.csv",
              ["experiment_id", "site", "image_id", "mean_model_count", "site_field_inventory_count", "inventory_scope",
               "image_coverage_scope", "difference_mean_minus_inventory", "used_as_image_ground_truth", "notes"], inv_rows)

    pair_rows, match_rows, summary_rows, group_rows, pers_rows = [], [], [], [], []
    for u in units:
        if u.single_output:
            continue  # no trial pairs or persistence groups for a single deterministic output
        ids = {"experiment_id": u.experiment_id, "model_config_id": u.model_config_id, "site": u.site,
               "image_id": u.image_id}
        for t, tr in results[u.key].items():
            lab = ec.threshold_label(t)
            for r in tr.pairs:
                pair_rows.append({**ids, "iou_threshold": lab, "trial_a": r["trial_a"], "trial_b": r["trial_b"],
                                  "n_a": r["n_a"], "n_b": r["n_b"], "matched": r["matched"],
                                  "unmatched_a": r["unmatched_a"], "unmatched_b": r["unmatched_b"],
                                  "agreement": r["agreement"], "mean_matched_iou": r["mean_iou"],
                                  "median_matched_iou": r["median_iou"]})
                for ua, ub, v in r["pairs"]:
                    match_rows.append({**ids, "iou_threshold": lab, "trial_a": r["trial_a"], "trial_b": r["trial_b"],
                                       "detection_uid_a": ua, "detection_uid_b": ub, "iou": float(v),
                                       "iou_exact": f"{v.numerator}/{v.denominator}"})
            summary_rows.append({**ids, "model": u.model, "iou_threshold": lab, "n_trials": u.n_trials,
                                 **{k: tr.summary[k] for k in tr.summary}})
            bins = ec.persistence_bins(u.n_trials)
            cons = [g for g in tr.groups if g.status == "CONSISTENT"]
            amb = [g for g in tr.groups if g.status == "AMBIGUOUS"]
            n_det = sum(len(s.detections) for s in u.sets)
            bin_counts = Counter(ec.persistence_bin(g.n_trials, u.n_trials) for g in cons)
            in_all = sum(len(g.members) for g in cons if g.n_trials == u.n_trials)
            pers_rows.append({**ids, "iou_threshold": lab, "n_trials": u.n_trials, "detections_total": n_det,
                              **{f"groups_{label}_trials": bin_counts.get(label, 0) for label, _, _ in bins},
                              "ambiguous_groups": len(amb),
                              "detections_in_ambiguous_groups": sum(len(g.members) for g in amb),
                              "detections_in_all_trial_groups_fraction": in_all / n_det if n_det else None})
            for g in tr.groups:
                geo = tr.geometry[g.group_id]
                reason = []
                if g.duplicate_trials:
                    reason.append("several detections from trial(s) " + ",".join(map(str, g.duplicate_trials)))
                if g.missing_links:
                    total = len(g.members) * (len(g.members) - 1) // 2
                    reason.append(f"{len(g.missing_links)} of {total} member pairs not matched to each other")
                group_rows.append({**ids, "iou_threshold": lab, "group_id": g.group_id, "status": g.status,
                                   "n_trials": g.n_trials, "n_detections": len(g.members),
                                   "persistence_bin": ec.persistence_bin(g.n_trials, u.n_trials) if g.status == "CONSISTENT" else "AMBIGUOUS",
                                   "trials_detected": g.trials, "ambiguity_reason": "; ".join(reason),
                                   "median_box_px": box_str(geo["median_box_px"]),
                                   "center_x_sd_px": geo["center_x_sd_px"], "center_y_sd_px": geo["center_y_sd_px"],
                                   "center_max_offset_px": geo["center_max_offset_px"],
                                   "width_mean_px": geo["width_mean_px"], "height_mean_px": geo["height_mean_px"],
                                   "width_cv": geo["width_cv"], "height_cv": geo["height_cv"],
                                   "member_iou_mean": geo["member_iou_mean"], "member_iou_min": geo["member_iou_min"],
                                   "identical_geometry": geo["identical_geometry"],
                                   "detection_uids": [d.uid for _, d in g.members]})
    bin_fields = list(dict.fromkeys(k for r in pers_rows for k in r if k.startswith("groups_")))
    id_fields = ["experiment_id", "model_config_id", "site", "image_id"]
    write_csv(output_dir / "pairwise_iou.csv", id_fields + ["iou_threshold", "trial_a", "trial_b", "n_a", "n_b", "matched",
              "unmatched_a", "unmatched_b", "agreement", "mean_matched_iou", "median_matched_iou"], pair_rows)
    write_csv(output_dir / "pairwise_matches.csv", id_fields + ["iou_threshold", "trial_a", "trial_b", "detection_uid_a",
              "detection_uid_b", "iou", "iou_exact"], match_rows)
    write_csv(output_dir / "iou_summary.csv", id_fields + ["model", "iou_threshold", "n_trials"]
              + list(ec.summarize_pairs([])), summary_rows)
    write_csv(output_dir / "detection_persistence.csv", list(group_rows[0]), group_rows)
    write_csv(output_dir / "persistence_summary.csv", id_fields + ["iou_threshold", "n_trials", "detections_total"]
              + bin_fields + ["ambiguous_groups", "detections_in_ambiguous_groups",
                              "detections_in_all_trial_groups_fraction"], pers_rows)

    group_of = {(u.key, t): {d.uid: g for g in tr.groups for _, d in g.members}
                for u in units for t, tr in results[u.key].items()}
    det_rows = []
    for u in units:
        for s in sorted(u.sets, key=lambda s: s.trial):
            for d in s.detections:
                if u.coordinate_system == "original_pixels":
                    norm = [""] * 4   # pixel-native output: never put on the 0-1000 grid
                    px = [repr(float(v)) for v in d.box]   # full precision (the CSV writer rounds floats)
                else:
                    norm = [round(float(v) * 1000 / (u.width if i % 2 == 0 else u.height)) for i, v in enumerate(d.box)]
                    px = [float(v) for v in d.box]
                row = {"detection_uid": d.uid, "experiment_id": u.experiment_id, "model_config_id": u.model_config_id,
                       "image_id": u.image_id, "trial": s.trial, "tree_id": d.label,
                       "x_min_norm": norm[0], "y_min_norm": norm[1], "x_max_norm": norm[2], "y_max_norm": norm[3],
                       "x_min_px": px[0], "y_min_px": px[1], "x_max_px": px[2], "y_max_px": px[3]}
                for t, tr in results[u.key].items():
                    tag = ec.threshold_tag(t)
                    g = group_of[(u.key, t)].get(d.uid)
                    row[f"group_{tag}"] = g.group_id if g else ""
                    row[f"group_status_{tag}"] = g.status if g else "NOT_APPLICABLE_SINGLE_OUTPUT"
                    row[f"other_trials_matched_{tag}"] = tr.frequency.get(d.uid, "")
                row["score"] = repr(u.scores[d.uid]) if d.uid in u.scores else ""
                det_rows.append(row)
    write_csv(output_dir / "detections.csv", list(det_rows[0]), det_rows)

    # -- human review summary and (gated) accuracy
    review_summary = summarize_review(units, results, review, template)
    accuracy_rows, accuracy_notes, accuracy_results = [], [], []
    unit_of = {}
    for row in template:
        uids = row["detection_uids"] if isinstance(row["detection_uids"], list) else [row["detection_uids"]]
        label = review["labels"].get(row["review_unit_id"], {}).get("label", "")
        for uid in uids:
            unit_of[uid] = (row["review_unit_id"], label)
    for u in units:
        status, ref_sha = review["status"].get(u.image_id, ("NONE", ""))
        ref = reference_set(u, review)
        if status != ec.REFERENCE_COMPLETE:
            accuracy_notes.append(f"{u.image_id}: reference status {status}; precision/recall/F1 not computed")
            continue
        if ref_sha and ref_sha != u.image_sha256:
            raise InputError(f"reference_status.csv: {u.image_id} hash does not match the evaluated image")
        if ref is None:
            ref = ec.DetectionSet("human_reference", u.image_id, u.image_sha256, u.width, u.height, None, ())
        groups050 = {uid: g.group_id for uid, g in group_of[(u.key, REVIEW_THRESHOLD)].items()}
        acc = ea.evaluate_accuracy(u, ref, status, thresholds, groups050, unit_of)
        accuracy_results.append((u, acc))
        accuracy_rows += acc.trial_rows
    if accuracy_results:
        write_csv(output_dir / "accuracy_vs_reference.csv", list(accuracy_rows[0]), accuracy_rows)
        for name, attr in (("accuracy_summary.csv", "summary_rows"), ("reference_detection_frequency.csv", "reference_rows"),
                           ("prediction_outcomes.csv", "outcome_rows")):
            rows = [r for _, acc in accuracy_results for r in getattr(acc, attr)]
            write_csv(output_dir / name, list(rows[0]), rows)

    # -- comparison between model configurations on the same image (only when there are several)
    comparison_rows, cross_rows = model_comparison(units, results, stats, accuracy_results, thresholds)
    if comparison_rows:
        write_csv(output_dir / "model_comparison.csv", list(comparison_rows[0]), comparison_rows)
        write_csv(output_dir / "cross_model_agreement.csv", list(cross_rows[0]), cross_rows)

    # -- box geometry, coverage and (between models) coverage overlap and fragmentation
    geometry_rows = geometry_table(units)
    write_csv(output_dir / "geometry_summary.csv", list(geometry_rows[0]), geometry_rows)
    coverage_rows = cross_model_coverage(units)
    if coverage_rows:
        write_csv(output_dir / "cross_model_coverage.csv", list(coverage_rows[0]), coverage_rows)

    # -- altitude-series organisation
    alt_rows = altitude_series_rows(manifest, image_root, units) if manifest.exists() else []
    if alt_rows:
        write_csv(output_dir / "altitude_series.csv", list(alt_rows[0]), alt_rows)

    # -- figures and review images
    figure_paths = {}
    if figures:
        import evaluation_plots as ep
        for u in units:
            if not u.single_output:  # figures A-E describe between-trial repeatability
                figure_paths[u.key] = ep.make_figures(u, results[u.key], stats[u.key], output_dir, full_res_maps)
        ep.make_review_images(unit_crops, output_dir)
        ref_path = review_dir / "reference_boxes.csv"
        ref_rows = read_csv(ref_path) if ref_path.exists() else []
        for image_id, u in {u.image_id: u for u in units}.items():
            rows = [r for r in ref_rows if r["image_id"] == image_id and (r.get("label") or "TREE").upper() == "TREE"]
            if rows:
                status = review["status"].get(image_id, ("NONE", ""))[0]
                ep.reference_images(u.image_path, image_id, rows, output_dir,
                                    "COMPLETE" if status == ec.REFERENCE_COMPLETE else f"not approved ({status})")
        for u, acc in accuracy_results:
            figure_paths[u.key] = figure_paths.get(u.key, []) + ep.accuracy_figures(u, acc, output_dir)
        acc_by_key = {u.key: acc for u, acc in accuracy_results}
        for image_id in sorted({r["image_id"] for r in comparison_rows}):
            group = [u for u in units if u.image_id == image_id]
            figure_paths[f"comparison/{image_id}"] = ep.comparison_figures(
                image_id, group, results, stats, acc_by_key, [r for r in cross_rows if r["image_id"] == image_id],
                output_dir, [r for r in geometry_rows if r["image_id"] == image_id],
                [r for r in coverage_rows if r["image_id"] == image_id])

    config = {
        "algorithm_version": ec.ALGORITHM_VERSION,
        "iou_thresholds": [f"{t.numerator}/{t.denominator}" if t.denominator != 1 else str(t.numerator) for t in thresholds],
        "iou_definition": "exact rational IoU in original-image pixels; box = 0-1000 coordinate * image size / 1000; "
                          "IoU = 1 only for identical geometry",
        "matching": "one-to-one; maximize number of pairs with IoU >= threshold, ties broken by total IoU "
                    "(scipy.optimize.linear_sum_assignment on K + IoU weights)",
        "agreement": "2 * matched / (n_a + n_b) per trial pair; undefined when both trials are empty",
        "persistence": "connected components of pairwise matches; CONSISTENT only if one detection per trial and "
                       "every member pair directly matched; otherwise AMBIGUOUS (not split)",
        "review_units_threshold": ec.threshold_label(REVIEW_THRESHOLD),
        "inputs": [{"path": p, "sha256": h} for u in units for p, h in u.inputs],
        "software": software_versions(),
        "code": git_state(ROOT),
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "command": " ".join(sys.argv) if sys.argv else None,
        "api_requests_made": 0,
    }
    (output_dir / MARKER).write_text(json.dumps(config, indent=2), encoding="utf-8")

    report = build_report(units, results, stats, inv_rows, review, review_summary, accuracy_rows, accuracy_notes,
                          alt_rows, figure_paths, output_dir, created)
    if comparison_rows:
        report += comparison_section(comparison_rows, cross_rows, thresholds)
    report += geometry_section(units, geometry_rows, coverage_rows)
    (output_dir / "evaluation_report.md").write_text(report, encoding="utf-8")
    if accuracy_results:
        (output_dir / "accuracy_report.md").write_text(
            accuracy_report(accuracy_results, thresholds, review_dir, review, figure_paths, output_dir), encoding="utf-8")
    if created:
        log(f"Created in {review_dir} (never overwritten): {', '.join(created)}")
    log(f"Wrote {output_dir}")
    return {"units": units, "results": results, "stats": stats, "review": review, "figures": figure_paths,
            "accuracy_rows": accuracy_rows, "accuracy": accuracy_results, "created_review_files": created,
            "comparison_rows": comparison_rows, "cross_rows": cross_rows, "geometry_rows": geometry_rows,
            "coverage_rows": coverage_rows}


def image_gsd_m(u: ImageTrials) -> float | None:
    """Nominal GSD (m/px) from relative altitude above takeoff and the 35 mm-equivalent focal length."""
    try:
        f35 = float(xmp_values(u.image_path).get("FocalLengthIn35mmFilm") or 0)
        return ground_sampling(float(u.altitude_m), f35, u.width, u.height)[2] if f35 else None
    except (ValueError, OSError):
        return None


def geometry_table(units) -> list:
    rows = []
    gsd = {}
    for u in units:
        g = gsd.setdefault(u.image_id, image_gsd_m(u))
        for s in sorted(u.sets, key=lambda s: s.trial):
            rows.append({"experiment_id": u.experiment_id, "model_config_id": u.model_config_id, "model": u.model,
                         "site": u.site, "image_id": u.image_id, "trial": s.trial,
                         "output_type": "single deterministic output" if u.single_output else "independent trial",
                         "nominal_gsd_cm": g * 100 if g else None,
                         **eg.set_geometry(s.detections, u.width, u.height, g)})
    return rows


def cross_model_coverage(units) -> list:
    """Every output of one configuration against every output of another on the same image: coverage-mask IoU
    and how many boxes of the configuration with more boxes fall inside each box of the other."""
    rows = []
    by_image = {}
    for u in units:
        by_image.setdefault(u.image_id, []).append(u)
    for image_id, group in sorted(by_image.items()):
        for a, b in combinations(sorted(group, key=lambda u: (u.model_config_id, u.experiment_id)), 2):
            if a.model_config_id == b.model_config_id:
                continue
            fine, coarse = (a, b) if statistics.fmean(a.counts) >= statistics.fmean(b.counts) else (b, a)
            for sf in sorted(fine.sets, key=lambda s: s.trial):
                for sc in sorted(coarse.sets, key=lambda s: s.trial):
                    ec.require_same_image(sf, sc)
                    rows.append({"image_id": image_id, "fine_model": fine.model, "fine_config": fine.model_config_id,
                                 "fine_trial": sf.trial, "coarse_model": coarse.model,
                                 "coarse_config": coarse.model_config_id, "coarse_trial": sc.trial,
                                 "coverage_mask_iou": eg.mask_iou(sf.detections, sc.detections, a.width, a.height),
                                 **eg.fragmentation(sf.detections, sc.detections)})
    return rows


def geometry_section(units, geometry_rows, coverage_rows) -> str:
    L = ["\n## 9. Box geometry, coverage and overlap (descriptive)\n",
         "Per output (one LLM trial, or the single DeepForest output); LLM values are the mean over trials. Box side = "
         "√(width × height). Metres use the nominal GSD from relative altitude above takeoff (an estimate; canopy "
         f"tops are closer to the camera). Coverage = share of the image inside at least one box (raster at 1/"
         f"{eg.COVERAGE_FACTOR} resolution). Overlap ratio = summed box area / covered area (1 = no overlap). "
         f"Near-duplicates: pairs within one output with IoU ≥ {eg.NEAR_DUPLICATE_IOU}. None of this is accuracy.\n",
         "| Image | Model | Outputs | Boxes | Box side median px (m) [p10–p90 m] | Coverage | Overlap ratio | "
         "Boxes in near-duplicate pairs | Boxes per 100 m² |",
         "|---|---|---|---|---|---|---|---|---|"]
    for u in units:
        rs = [r for r in geometry_rows if r["experiment_id"] == u.experiment_id and r["image_id"] == u.image_id]

        def m(k, rs=rs):
            vals = [r[k] for r in rs if r[k] is not None]
            return statistics.fmean(vals) if vals else None
        L.append(f"| {u.image_id} | {u.model} | {len(rs)} | {fnum(m('n_boxes'), 1)} | {fnum(m('box_side_px_median'), 0)} "
                 f"({fnum(m('box_side_m_median'), 2)}) [{fnum(m('box_side_m_p10'), 2)}–{fnum(m('box_side_m_p90'), 2)}] | "
                 f"{fnum(m('coverage_fraction'))} | {fnum(m('overlap_ratio_summed_over_union'), 2)} | "
                 f"{fnum(m('boxes_in_near_duplicate_pairs_fraction'))} | {fnum(m('boxes_per_100_m2'), 2)} |")
    if coverage_rows:
        L += ["\nBetween configurations (every output of one against every output of the other; mean over pairs). "
              "'Fine' is the configuration with more boxes.\n",
              "| Image | Fine model | Coarse model | Pairs | Coverage-mask IoU | Fine centres per coarse box | "
              "Coarse boxes with 0 / ≥2 fine centres | Fine centres inside any coarse box |",
              "|---|---|---|---|---|---|---|---|"]
        keys = sorted({(r["image_id"], r["fine_model"], r["coarse_model"]) for r in coverage_rows})
        for k in keys:
            rs = [r for r in coverage_rows if (r["image_id"], r["fine_model"], r["coarse_model"]) == k]

            def m(f, rs=rs):
                vals = [r[f] for r in rs if r[f] is not None]
                return statistics.fmean(vals) if vals else None
            L.append(f"| {k[0]} | {k[1]} | {k[2]} | {len(rs)} | {fnum(m('coverage_mask_iou'))} | "
                     f"{fnum(m('fine_centres_per_coarse_box_mean'), 2)} | {fnum(m('coarse_boxes_with_0_fine'), 1)} / "
                     f"{fnum(m('coarse_boxes_with_2plus_fine'), 1)} | {fnum(m('fine_centres_inside_any_coarse_fraction'))} |")
    return "\n".join(L) + "\n"


def model_comparison(units, results, stats, accuracy_results, thresholds) -> tuple:
    """Side-by-side summary and between-model box agreement for images evaluated under more than one
    model configuration. Boxes are compared only within the same image (same SHA-256)."""
    by_image = {}
    for u in units:
        by_image.setdefault(u.image_id, []).append(u)
    acc_by_key = {u.key: acc for u, acc in accuracy_results}
    comparison, cross = [], []
    for image_id, group in sorted(by_image.items()):
        if len({u.model_config_id for u in group}) < 2:
            continue
        for u in group:
            s = stats[u.key]
            row = {"image_id": image_id, "site": u.site, "experiment_id": u.experiment_id,
                   "model_config_id": u.model_config_id, "model": u.model, "successful_trials": u.n_trials,
                   "mean_count": s["mean"], "sd_count": s["sd"], "cv_count": s["cv"]}
            for t in thresholds:
                tag = ec.threshold_tag(t)
                row[f"within_model_agreement_{tag}"] = results[u.key][t].summary["agreement_mean"]
            acc = acc_by_key.get(u.key)
            for t in thresholds:
                tag = ec.threshold_tag(t)
                for m in ("precision", "recall", "f1"):
                    row[f"{m}_{tag}"] = ea.summary_value(acc, t, m)["mean"] if acc else None
            row["human_reference"] = "COMPLETE" if acc else "none (consistency only)"
            comparison.append(row)
        for a, b in combinations(sorted(group, key=lambda u: (u.model_config_id, u.experiment_id)), 2):
            if a.model_config_id == b.model_config_id:
                continue
            for t in thresholds:
                pairs = [ec.compare_sets(sa, sb, t) for sa in a.sets for sb in b.sets]
                s = ec.summarize_pairs(pairs)
                cross.append({"image_id": image_id, "model_config_a": a.model_config_id, "model_a": a.model,
                              "experiment_a": a.experiment_id, "model_config_b": b.model_config_id,
                              "model_b": b.model, "experiment_b": b.experiment_id,
                              "iou_threshold": ec.threshold_label(t), "trial_pairs": s["n_pairs"],
                              "agreement_mean": s["agreement_mean"], "agreement_sd": s["agreement_sd"],
                              "agreement_min": s["agreement_min"], "agreement_max": s["agreement_max"],
                              "matched_iou_mean": s["matched_iou_mean"]})
    return comparison, cross


def comparison_section(comparison, cross, thresholds) -> str:
    L = ["\n## 8. Comparison between model configurations (same image)\n",
         "Within-model agreement is repeatability (trial vs trial of one model). Between-model agreement compares "
         "every trial of one model with every trial of the other, using the same one-to-one IoU matching. Neither "
         "is accuracy. Accuracy columns appear only for images with a COMPLETE human reference. Different "
         "providers' reasoning-effort settings are not computationally equivalent.\n",
         "| Image | Model config | Trials | Mean count (CV) | Within-model agreement 0.50 / 0.75 / 1.00 | "
         "F1 0.50 / 0.75 | Reference |",
         "|---|---|---|---|---|---|---|"]
    for r in comparison:
        agree = " / ".join(fnum(r[f"within_model_agreement_{ec.threshold_tag(t)}"]) for t in thresholds)
        f1 = " / ".join(fnum(r[f"f1_{ec.threshold_tag(t)}"]) for t in thresholds[:2])
        cv = "–" if r["cv_count"] is None else f"{r['cv_count']:.1%}"
        L.append(f"| {r['image_id']} | {r['model_config_id']} ({r['model']}) | {r['successful_trials']} | "
                 f"{fnum(r['mean_count'], 1)} ({cv}) | {agree} | {f1} | {r['human_reference']} |")
    L += ["", "| Image | Models | IoU ≥ | Trial pairs | Between-model agreement mean ± SD | Range | Matched IoU |",
          "|---|---|---|---|---|---|---|"]
    for r in cross:
        L.append(f"| {r['image_id']} | {r['model_a']} vs {r['model_b']} | {r['iou_threshold']} | {r['trial_pairs']} | "
                 f"{fnum(r['agreement_mean'])} ± {fnum(r['agreement_sd'])} | {fnum(r['agreement_min'])}–"
                 f"{fnum(r['agreement_max'])} | {fnum(r['matched_iou_mean'])} |")
    return "\n".join(L) + "\n"


def accuracy_report(accuracy_results, thresholds, review_dir: Path, review: dict, figure_paths: dict, out: Path) -> str:
    status_rows = {r["image_id"]: r for r in read_csv(review_dir / "reference_status.csv")}
    labels = {uid: (v["label"], v.get("note", "")) for uid, v in review["labels"].items()}
    L = ["# Detection accuracy against the human reference\n",
         f"Algorithm version {ec.ALGORITHM_VERSION}. Offline; no API requests. Computed only for images whose "
         "human reference is marked COMPLETE in `human_review/reference_status.csv`. Repeatability results are in "
         "`evaluation_report.md`. Inputs, hashes and software versions are in `evaluation_config.json`.\n",
         "Files: `accuracy_vs_reference.csv` (per trial), `accuracy_summary.csv` (across trials), "
         "`reference_detection_frequency.csv` (per reference tree), `prediction_outcomes.csv` (per predicted box, "
         "with its IoU 0.50 group, review unit and human label).\n"]
    for u, acc in accuracy_results:
        s = status_rows.get(u.image_id, {})
        note = f"Approved by {s.get('annotator') or '?'} on {s.get('status_date') or '?'}."
        L.append(ea.report_section(acc, thresholds, note, labels))
        paths = [p for p in figure_paths.get(u.key, []) if p.name[:2] in ("F_", "G_", "H_", "I_", "J_")]
        if paths:
            L.append("### Figures\n")
            L += [f"- `{rel(p, out)}`" for p in paths]
            L.append("")
    L.append(ea.LIMITATIONS)
    return "\n".join(L) + "\n"


def _coverage(manifest: Path, image_id: str) -> str:
    if not manifest.exists():
        return "UNKNOWN"
    for r in read_csv(manifest):
        if r["image_id"] == image_id:
            return r.get("coverage_scope", "UNKNOWN")
    return "UNKNOWN"


def summarize_review(units, results, review, template) -> dict:
    """Cross-tabulate human labels by persistence bin; find reference trees no trial detected."""
    by_id = {r["review_unit_id"]: r for r in template}
    unknown = sorted(set(review["labels"]) - set(by_id))
    table = Counter()
    for uid, lab in review["labels"].items():
        row = by_id.get(uid)
        if row is None:
            continue
        b = row["persistence_bin"] if row["group_status_iou050"] == "CONSISTENT" else "ambiguous"
        table[(row["experiment_id"], row["image_id"], b, lab["label"])] += 1
    missed = {}
    for u in units:
        ref = reference_set(u, review)
        if ref is None:
            continue
        union = [d.box for s in u.sets for d in s.detections]
        not_found = [d.uid for d in ref.detections if not any(ec.iou(d.box, b) >= REVIEW_THRESHOLD for b in union)]
        missed[u.key] = (len(ref.detections), not_found)
    return {"table": table, "unknown_ids": unknown, "missed": missed, "n_labels": len(review["labels"]),
            "n_units": len(template)}


# ------------------------------------------------------------------ report

def build_report(units, results, stats, inv_rows, review, rs, accuracy_rows, accuracy_notes, alt_rows,
                 figure_paths, out: Path, created) -> str:
    L = []
    w = L.append
    w("# Detection consistency evaluation\n")
    w(f"Algorithm version {ec.ALGORITHM_VERSION}. Generated offline from saved benchmark records; no API requests. "
      "Configuration, input hashes and software versions: `evaluation_config.json`. Methods: `EVALUATION.md`.\n")
    w("This report keeps four kinds of evidence apart:\n")
    w("1. **Count consistency**: how much the number of boxes varies between independent trials.")
    w("2. **Spatial consistency**: whether trials place boxes on the same image locations (IoU matching between trials).")
    w("3. **Count vs field inventory**: a site-level field count beside the model's mean count. Not an image-level reference.")
    w("4. **Human-validated correctness**: only from human labels and complete human reference boxes.\n")
    w("Items 1–3 measure repeatability or numeric proximity. They are not precision, recall or accuracy, and "
      "agreement between trials does not show that a box contains a tree.\n")

    w("## Inputs\n")
    w("| Experiment | Image | Site | Altitude (m) | Model config | Successful trials | Excluded | Counts match saved summary |")
    w("|---|---|---|---|---|---|---|---|")
    for u in units:
        w(f"| {u.experiment_id} | {u.image_id} | {u.site} | {u.altitude_m} | {u.model_config_id} ({u.model}) | "
          f"{u.n_trials} | {len(u.excluded) or 'none'} | {'yes' if verify_counts(u, stats[u.key]) else 'NO'} |")
    w("\nEvery trial's `detections.json` was checked against `boxes_long.csv`, `counts_long.csv` and "
      "`record.json`, and each source image against the SHA-256 recorded at inference time.\n")

    w("## 1. Count consistency\n")
    w("| Image | Counts by trial | Mean | SD (n−1) | Range | CV |")
    w("|---|---|---|---|---|---|")
    for u in units:
        s = stats[u.key]
        if u.single_output:
            w(f"| {u.image_id} ({u.model}) | {', '.join(map(str, u.counts))} | {fnum(s['mean'], 2)} | "
              f"single deterministic output; not applicable | – | – |")
            continue
        w(f"| {u.image_id} | {', '.join(map(str, u.counts))} | {fnum(s['mean'], 2)} | {fnum(s['sd'])} | "
          f"{s['min']}–{s['max']} | {fnum(s['cv'] * 100 if s['cv'] is not None else None, 1)}% |")
    if any(u.single_output for u in units):
        w("\nDeepForest outputs are one deterministic run each (a verification rerun was bitwise identical; see "
          "`determinism.json` in each record). Between-trial count, spatial and persistence statistics do not "
          "apply to them and are omitted below.")

    w("\n## 2. Spatial consistency between trials\n")
    w("Each of the n(n−1)/2 trial pairs is matched one-to-one at each threshold. "
      "Agreement = 2 × matched / (boxes in trial A + boxes in trial B). IoU = 1.00 requires identical boxes.\n")
    w("| Image | IoU ≥ | Pairs | Agreement mean | SD | Median | Range | Pairs at 100% | Matched IoU mean | Matched IoU median |")
    w("|---|---|---|---|---|---|---|---|---|---|")
    for u in units:
        if u.single_output:
            continue
        for t, tr in results[u.key].items():
            s = tr.summary
            w(f"| {u.image_id} | {ec.threshold_label(t)} | {s['n_pairs']} | {fnum(s['agreement_mean'])} | "
              f"{fnum(s['agreement_sd'])} | {fnum(s['agreement_median'])} | {fnum(s['agreement_min'])}–{fnum(s['agreement_max'])} | "
              f"{s['pairs_with_full_agreement']} | {fnum(s['matched_iou_mean'])} | {fnum(s['matched_iou_median'])} |")

    w("\n## 3. Detection persistence\n")
    w("Groups are recurring detections: at most one box per trial, and every two members matched directly "
      "(no chaining). Components that fail either rule are **ambiguous** and are listed rather than split.\n")
    for u in units:
        if u.single_output:
            continue
        bins = ec.persistence_bins(u.n_trials)
        w(f"**{u.image_id}** ({u.n_trials} trials, {sum(u.counts)} detections)\n")
        w("| IoU ≥ | " + " | ".join(f"groups in {lab} trials" for lab, _, _ in bins)
          + " | ambiguous groups (detections) | share of detections in all-trial groups |")
        w("|---" * (len(bins) + 3) + "|")
        for t, tr in results[u.key].items():
            cons = [g for g in tr.groups if g.status == "CONSISTENT"]
            amb = [g for g in tr.groups if g.status == "AMBIGUOUS"]
            c = Counter(ec.persistence_bin(g.n_trials, u.n_trials) for g in cons)
            share = sum(len(g.members) for g in cons if g.n_trials == u.n_trials) / max(1, sum(u.counts))
            w(f"| {ec.threshold_label(t)} | " + " | ".join(str(c.get(lab, 0)) for lab, _, _ in bins)
              + f" | {len(amb)} ({sum(len(g.members) for g in amb)}) | {share:.0%} |")
        w("")
    amb_lines = []
    for u in units:
        for t, tr in results[u.key].items():
            for g in tr.groups:
                if g.status == "AMBIGUOUS":
                    n = len(g.members)
                    why = []
                    if g.duplicate_trials:
                        why.append(f"two or more boxes from trial(s) {', '.join(map(str, g.duplicate_trials))}")
                    if g.missing_links:
                        why.append(f"{len(g.missing_links)} of {n * (n - 1) // 2} member pairs not matched")
                    amb_lines.append(f"| {u.image_id} | {ec.threshold_label(t)} | {g.group_id} | {n} | {g.n_trials} | "
                                     f"{'; '.join(why)} |")
    if amb_lines:
        w("Ambiguous groups (full membership in `detection_persistence.csv`):\n")
        w("| Image | IoU ≥ | Group | Detections | Trials | Why ambiguous |")
        w("|---|---|---|---|---|---|")
        L.extend(amb_lines)
        w("")
    else:
        w("No ambiguous groups.\n")

    w("## 4. Count vs field inventory (site level)\n")
    if inv_rows:
        w("| Image | Mean model count | Site field inventory | Difference | Image coverage |")
        w("|---|---|---|---|---|")
        for r in inv_rows:
            w(f"| {r['image_id']} | {fnum(r['mean_model_count'], 1)} | {r['site_field_inventory_count']} | "
              f"{r['difference_mean_minus_inventory']:+.1f} | {r['image_coverage_scope']} |")
        w("\nThe inventory counts the whole site. Image coverage is not established, so this is not an "
          "image-level reference, and a close count does not show that the boxes are the right trees. "
          "The inventory is never used to filter or correct predictions.\n")
    else:
        w("No field inventory supplied.\n")

    w("## 5. Human validation\n")
    w(f"Review units: {rs['n_units']} (persistence groups at IoU ≥ {ec.threshold_label(REVIEW_THRESHOLD)}; "
      "ambiguous groups are split into their individual detections for review). "
      f"Labelled so far: {rs['n_labels']}.\n")
    for e in review["label_errors"] + review["reference_errors"]:
        w(f"- **Input problem:** {e}")
    if rs["unknown_ids"]:
        w(f"- Labelled ids not in the current review units (algorithm or inputs changed?): {', '.join(rs['unknown_ids'])}")
    if rs["table"]:
        w("\n| Image | Persistence | Label | Units |")
        w("|---|---|---|---|")
        for (exp, img, b, lab), n in sorted(rs["table"].items()):
            w(f"| {img} | {b} | {lab} | {n} |")
        w("\nTREE units found in every trial are consistently detected trees; NON_TREE units found in many trials are "
          "repeated false positives; TREE units in fewer than all trials are trees the model detects inconsistently.\n")
    else:
        w("No human labels yet.\n")
    for key, (n_ref, missed) in rs["missed"].items():
        w(f"- {key}: {len(missed)} of {n_ref} human reference trees were not detected in any trial at IoU ≥ 0.50.")
    w("\n**Accuracy against a human reference:** " + ("computed for images with a COMPLETE reference; see "
      "`accuracy_report.md`." if accuracy_rows else "not computed. No image has a COMPLETE human reference annotation."))
    for n in accuracy_notes:
        w(f"- {n}")
    w("\nPreliminary observations reported by the project lead (free text; not machine-readable labels and "
      "not linked to detection ids until confirmed in `manual_review.csv`):\n")
    w("- PT-R1: one object detected repeatedly appears to be a plant rather than a tree.")
    w("- PT-R1: the extra seventh detection in trial 10 appears to be a genuine tree.")
    w("- PT-02: individual detections visibly disagree more between trials.\n")

    w("## 6. Figures\n")
    for key, paths in figure_paths.items():
        w(f"**{key}**\n")
        for p in paths:
            w(f"- `{rel(p, out)}`")
        w("")
    if figure_paths:
        w("Review crops: `review/crops/`; contact sheets: `review/contact_sheet_*.png`.\n")

    w("## 7. Across-altitude organisation (no comparisons made)\n")
    if alt_rows:
        w("| Image | Site | Altitude (m) | Yaw (°) | Heading flagged | Est. GSD original / API (mm/px) | Model outputs |")
        w("|---|---|---|---|---|---|---|")
        for r in alt_rows:
            w(f"| {r['image_id']} | {r['site']} | {r['relative_altitude_m']} | {r['gimbal_yaw_deg']} | "
              f"{'yes' if r['heading_flagged'] else ''} | {fnum(r['est_gsd_original_mm'], 1)} / {fnum(r['est_gsd_api_mm'], 1)} | "
              f"{r['model_outputs'] or '–'} |")
        w("\nGSD is estimated from height above the takeoff point and the 35 mm-equivalent focal length, so it is "
          "approximate. Boxes from different images are not compared: no validated alignment exists. "
          "The evaluation code refuses such comparisons (`CrossImageComparisonError`). See `EVALUATION.md` for the design.\n")
    return "\n".join(L) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    ap.add_argument("--experiments", nargs="*", help="Experiment ids (default: all with counts_long.csv)")
    ap.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    ap.add_argument("--review-dir", type=Path, default=REVIEW_DIR)
    ap.add_argument("--manifest", type=Path, default=MANIFEST)
    ap.add_argument("--image-root", type=Path, default=ROOT)
    ap.add_argument("--no-figures", action="store_true", help="Skip figures, crops and contact sheets")
    ap.add_argument("--no-full-res-maps", action="store_true", help="Skip the full-resolution persistence map JPEGs")
    ap.add_argument("--validate-review", action="store_true",
                    help="Only check human_review/manual_review.csv against the evaluation; write nothing")
    args = ap.parse_args(argv)
    try:
        run(args.results_dir, args.experiments, args.output_dir, args.review_dir, args.manifest, args.image_root,
            figures=not args.no_figures, full_res_maps=not args.no_full_res_maps,
            validate_only=args.validate_review)
    except InputError as e:
        sys.exit(f"ERROR: {e}")


if __name__ == "__main__":
    main()
