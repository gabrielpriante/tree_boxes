#!/usr/bin/env python3
"""
Import human reference boxes from a Make Sense (makesense.ai) export into human_review/reference_boxes.csv.

Both export formats of the same annotation session are required and must agree exactly:
    * Pascal VOC ZIP  (one XML per image: xmin, ymin, xmax, ymax)
    * CSV             (label_name, bbox_x, bbox_y, bbox_width, bbox_height, image_name, image_width, image_height)

    python import_reference.py --image-id LA_PT-R1_27m \\
        --voc-zip human_review/ground_truth/labels_..._06-05-31.zip \\
        --csv human_review/ground_truth/labels_..._06-06-49.csv --annotator Gabe --date 2026-10-30 --dry-run

Coordinates are original-image pixels and are imported unchanged (box edges; Make Sense writes
xmax = x + width). Nothing is derived from model predictions or field inventories. The export files
are only read. Existing reference rows for the image are never replaced; rows for other images are kept.
reference_status.csv is not changed: completeness is a separate, explicit human decision.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import sys
import xml.etree.ElementTree as ET
import zipfile
from itertools import combinations
from pathlib import Path

from PIL import Image

import evaluate_consistency as ev
import evaluation_core as ec

ROOT = Path(__file__).resolve().parent
LABEL = "TREE"
CSV_FIELDS = ["label_name", "bbox_x", "bbox_y", "bbox_width", "bbox_height", "image_name", "image_width", "image_height"]
NEAR_DUPLICATE_IOU = 0.5
MIN_SIDE_PX = 20


class ImportError_(RuntimeError):
    pass


def _int(value: str, what: str) -> int:
    try:
        f = float(value)
    except (TypeError, ValueError):
        raise ImportError_(f"{what}: {value!r} is not a number")
    if f != int(f):
        raise ImportError_(f"{what}: {value!r} is not a whole pixel; refusing to round")
    return int(f)


def read_voc(zip_path: Path, image_name: str) -> dict:
    with zipfile.ZipFile(zip_path) as z:
        xmls = [n for n in z.namelist() if n.lower().endswith(".xml")]
        docs = {}
        for n in xmls:
            root = ET.fromstring(z.read(n))
            docs[n] = root
    matches = [(n, r) for n, r in docs.items() if (r.findtext("filename") or "") == image_name]
    if len(matches) != 1:
        raise ImportError_(f"{zip_path.name}: expected one XML for {image_name}, found {len(matches)} "
                           f"(files: {', '.join(docs) or 'none'})")
    name, root = matches[0]
    size = root.find("size")
    if size is None:
        raise ImportError_(f"{name}: no <size>")
    width = _int(size.findtext("width"), f"{name} width")
    height = _int(size.findtext("height"), f"{name} height")
    boxes = []
    for i, obj in enumerate(root.findall("object"), start=1):
        where = f"{name} object {i}"
        if obj.find("polygon") is not None or obj.find("bndbox") is None:
            raise ImportError_(f"{where}: not a rectangle (no <bndbox>)")
        bb = obj.find("bndbox")
        vals = []
        for tag in ("xmin", "ymin", "xmax", "ymax"):
            text = bb.findtext(tag)
            if text is None or not text.strip():
                raise ImportError_(f"{where}: missing <{tag}>")
            vals.append(_int(text, f"{where} {tag}"))
        boxes.append({"label": (obj.findtext("name") or "").strip(), "box": tuple(vals), "source_index": i})
    return {"file": name, "width": width, "height": height, "boxes": boxes}


def read_makesense_csv(csv_path: Path, image_name: str) -> dict:
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        missing = [c for c in CSV_FIELDS if c not in (reader.fieldnames or [])]
        if missing:
            raise ImportError_(f"{csv_path.name}: missing columns {missing}")
        rows = list(reader)
    rows_here = [r for r in rows if r["image_name"] == image_name]
    others = sorted({r["image_name"] for r in rows} - {image_name})
    if others:
        raise ImportError_(f"{csv_path.name}: also contains other images {others}; import one image at a time")
    sizes = {(r["image_width"], r["image_height"]) for r in rows_here}
    if len(sizes) != 1:
        raise ImportError_(f"{csv_path.name}: inconsistent or missing image size {sizes}")
    (w, h), = sizes
    boxes = []
    for i, r in enumerate(rows_here, start=1):
        where = f"{csv_path.name} row {i}"
        for c in CSV_FIELDS:
            if not (r.get(c) or "").strip():
                raise ImportError_(f"{where}: empty {c}")
        x, y = _int(r["bbox_x"], f"{where} bbox_x"), _int(r["bbox_y"], f"{where} bbox_y")
        bw, bh = _int(r["bbox_width"], f"{where} bbox_width"), _int(r["bbox_height"], f"{where} bbox_height")
        boxes.append({"label": r["label_name"].strip(), "box": (x, y, x + bw, y + bh), "source_index": i})
    return {"file": csv_path.name, "width": _int(w, "image_width"), "height": _int(h, "image_height"), "boxes": boxes}


def validate(voc: dict, mcsv: dict, width: int, height: int) -> tuple:
    """Return (errors, warnings, rounding). Errors block the import. `rounding` lists boxes whose two exports
    differ only by at most 1 px on x_max / y_max (same label, order and min corner). These also block the
    import unless the user explicitly chooses which export to use."""
    errors, warnings, rounding = [], [], []
    for src in (voc, mcsv):
        if (src["width"], src["height"]) != (width, height):
            errors.append(f"{src['file']}: image size {src['width']}x{src['height']} != source image {width}x{height}")
        if not src["boxes"]:
            errors.append(f"{src['file']}: no annotations")
        for b in src["boxes"]:
            where = f"{src['file']} #{b['source_index']} {b['box']}"
            if b["label"].upper() != LABEL:
                errors.append(f"{where}: label {b['label']!r} is not {LABEL!r}")
            x0, y0, x1, y1 = b["box"]
            if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
                errors.append(f"{where}: outside the image or min >= max")
            elif min(x1 - x0, y1 - y0) < MIN_SIDE_PX:
                warnings.append(f"{where}: very small box ({x1 - x0}x{y1 - y0} px)")
    if len(voc["boxes"]) != len(mcsv["boxes"]):
        errors.append(f"exports disagree: {len(voc['boxes'])} XML boxes vs {len(mcsv['boxes'])} CSV boxes")
    else:
        for a, b in zip(voc["boxes"], mcsv["boxes"]):
            A, B = a["box"], b["box"]
            if (a["label"].upper(), A) == (b["label"].upper(), B):
                continue
            if a["label"].upper() == b["label"].upper() and A[:2] == B[:2] and abs(A[2] - B[2]) <= 1 and abs(A[3] - B[3]) <= 1:
                rounding.append((a["source_index"], A, B))
            else:
                errors.append(f"exports disagree at object {a['source_index']}: XML {a['label']} {A} "
                              f"vs CSV {b['label']} {B}")
    boxes = [b["box"] for b in voc["boxes"]]
    for (i, a), (j, b) in combinations(enumerate(boxes, start=1), 2):
        if a == b:
            errors.append(f"duplicate boxes: objects {i} and {j} are identical {a}")
        else:
            v = ec.iou(ec.box_from_pixels(a), ec.box_from_pixels(b))
            if v >= NEAR_DUPLICATE_IOU:
                warnings.append(f"objects {i} and {j} overlap strongly (IoU {float(v):.2f}); possible double annotation")
    return errors, warnings, rounding


def manifest_row(manifest: Path, image_id: str) -> dict:
    rows = ev.read_csv(manifest)
    hit = [r for r in rows if r["image_id"] == image_id]
    if len(hit) != 1:
        raise ImportError_(f"{image_id} not found once in {manifest}")
    return hit[0], rows


def import_reference(image_id: str, voc_zip: Path, csv_path: Path, annotator: str, date: str,
                     review_dir: Path = ev.REVIEW_DIR, manifest: Path = ev.MANIFEST, image_root: Path = ROOT,
                     dry_run: bool = False, use: str | None = None, log=print) -> dict:
    row, all_rows = manifest_row(manifest, image_id)
    image_path = image_root / row["image_path"]
    image_name = Path(row["image_path"]).name
    same_name = [r["image_id"] for r in all_rows if Path(r["image_path"]).name.lower() == image_name.lower()]
    if same_name != [image_id]:
        raise ImportError_(f"image name {image_name} is not unique in the manifest ({same_name}); identity is ambiguous")
    with Image.open(image_path) as im:
        width, height = im.size
        orientation = im.getexif().get(274, 1)
    if orientation not in (None, 1):
        raise ImportError_(f"{image_path}: EXIF orientation {orientation}; annotation tools may have rotated the image")
    image_sha = hashlib.sha256(image_path.read_bytes()).hexdigest()
    if row.get("sha256") and image_sha != row["sha256"]:
        raise ImportError_(f"{image_path}: SHA-256 differs from the manifest")

    voc = read_voc(voc_zip, image_name)
    mcsv = read_makesense_csv(csv_path, image_name)
    errors, warnings, rounding = validate(voc, mcsv, width, height)
    if errors:
        raise ImportError_("exports failed validation; nothing imported:\n  " + "\n  ".join(errors))
    diffs = {i: (A, B) for i, A, B in rounding}
    if rounding and use is None:
        raise ImportError_(
            "exports differ by 1 px on x_max/y_max for some boxes (min corners identical); nothing imported. "
            "Choose a source explicitly with --use voc or --use csv:\n  "
            + "\n  ".join(f"object {i}: XML {A} vs CSV {B}" for i, (A, B) in diffs.items()))
    source = mcsv if use == "csv" else voc

    ref_path = review_dir / "reference_boxes.csv"
    existing = ev.read_csv(ref_path) if ref_path.exists() else []
    if any(r["image_id"] == image_id for r in existing):
        raise ImportError_(f"{ref_path} already has reference rows for {image_id}; not replacing them")
    zip_sha = hashlib.sha256(voc_zip.read_bytes()).hexdigest()
    csv_sha = hashlib.sha256(csv_path.read_bytes()).hexdigest()
    new_rows = []
    for n, b in enumerate(source["boxes"], start=1):
        x0, y0, x1, y1 = b["box"]
        if b["source_index"] in diffs:
            A, B = diffs[b["source_index"]]
            agree = f"VOC {A} vs CSV {B} differ by <=1 px; {use.upper()} used by explicit choice"
        else:
            agree = "VOC = CSV"
        new_rows.append({
            "reference_id": f"H{n:02d}", "image_id": image_id,
            "x_min_px": x0, "y_min_px": y0, "x_max_px": x1, "y_max_px": y1, "label": LABEL,
            "annotator": annotator, "annotation_date": date,
            "notes": (f"Make Sense export object {b['source_index']}; VOC {voc_zip.name} (sha256 {zip_sha[:12]}), "
                      f"CSV {csv_path.name} (sha256 {csv_sha[:12]}); {agree}; draft, completeness not yet approved"),
        })
    for w in warnings:
        log(f"WARNING: {w}")
    if not dry_run:
        ev.write_csv(ref_path, ev.REFERENCE_FIELDS, existing + new_rows)
    return {"image_id": image_id, "image_path": image_path, "image_sha256": image_sha, "width": width,
            "height": height, "xml_file": voc["file"], "n_xml": len(voc["boxes"]), "n_csv": len(mcsv["boxes"]),
            "rows": new_rows, "warnings": warnings, "rounding": rounding,
            "source": "csv" if use == "csv" else "voc", "written": not dry_run}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image-id", required=True)
    ap.add_argument("--voc-zip", type=Path, required=True)
    ap.add_argument("--csv", type=Path, required=True)
    ap.add_argument("--annotator", required=True)
    ap.add_argument("--date", required=True, help="Annotation date, e.g. 2026-10-30")
    ap.add_argument("--review-dir", type=Path, default=ev.REVIEW_DIR)
    ap.add_argument("--manifest", type=Path, default=ev.MANIFEST)
    ap.add_argument("--output-dir", type=Path, default=ev.OUTPUT_DIR, help="Where review/ images are written")
    ap.add_argument("--dry-run", action="store_true", help="Validate and show the rows; write nothing")
    ap.add_argument("--use", choices=("voc", "csv"),
                    help="Only if the exports differ by <=1 px on x_max/y_max: which export to import")
    args = ap.parse_args(argv)
    try:
        res = import_reference(args.image_id, args.voc_zip, args.csv, args.annotator, args.date,
                               args.review_dir, args.manifest, dry_run=args.dry_run, use=args.use)
    except ImportError_ as e:
        sys.exit(f"ERROR: {e}")
    print(f"{res['image_id']}: {res['image_path'].name} {res['width']}x{res['height']} sha256 {res['image_sha256']}")
    agree = ("exports agree" if not res["rounding"] else
             f"{len(res['rounding'])} boxes differ by <=1 px; using {res['source'].upper()} by explicit choice")
    print(f"XML ({res['xml_file']}): {res['n_xml']} boxes; CSV: {res['n_csv']} boxes; {agree}")
    w = csv.DictWriter(sys.stdout, fieldnames=ev.REFERENCE_FIELDS, lineterminator="\n")
    w.writeheader()
    w.writerows(res["rows"])
    if res["written"]:
        import evaluation_plots as ep
        paths = ep.reference_images(res["image_path"], res["image_id"], res["rows"], args.output_dir,
                                    "not approved (draft)")
        print("Wrote reference rows to", args.review_dir / "reference_boxes.csv")
        for p in paths:
            print("Visualization:", p)
    else:
        print("Dry run: nothing written.")


if __name__ == "__main__":
    main()
