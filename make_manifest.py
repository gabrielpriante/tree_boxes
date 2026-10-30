#!/usr/bin/env python3
"""
Build the image manifest for the tree-count benchmark from drone_images/.

Metadata comes only from what is recorded in the repository:
  * site            filename (PT_02 / PT-02 / PT_R1 / PT-R1)
  * week            weekN directory name
  * condition       BEFORE / AFTER directory name, or the week 1 folder label verbatim
  * altitude_m      DJI XMP drone-dji:RelativeAltitude embedded in the JPEG
  * capture time    EXIF DateTimeOriginal

Anything that is not recorded is written as UNKNOWN and listed in needs_review
for manual completion. Nothing is filled in from defaults.

Usage:
    python make_manifest.py                    # writes manifests/manifest.csv
    python make_manifest.py --out other.csv
    python make_manifest.py --force            # overwrite an existing manifest
"""

import argparse
import csv
import hashlib
import re
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent
IMAGE_ROOT = ROOT / "drone_images"
UNKNOWN = "UNKNOWN"

FIELDS = [
    "image_id", "image_path", "site", "week", "condition", "altitude_m",
    "coverage_scope", "collection", "capture_datetime", "gimbal_pitch_deg",
    "altitude_source", "width", "height", "sha256", "needs_review", "notes",
]

SITE_RE = re.compile(r"PT[-_](02|R1)", re.IGNORECASE)
WEEK_RE = re.compile(r"^week(\d+)$", re.IGNORECASE)
XMP_RE = re.compile(rb'drone-dji:(RelativeAltitude|GimbalPitchDegree)="([^"]+)"')


def read_metadata(path: Path) -> dict:
    data = path.read_bytes()
    xmp = {k.decode(): v.decode() for k, v in XMP_RE.findall(data[:512_000])}
    with Image.open(path) as im:
        exif = im.getexif()
        taken = exif.get_ifd(0x8769).get(36867) or exif.get(306) or UNKNOWN
        width, height = im.size
    return {
        "sha256": hashlib.sha256(data).hexdigest(),
        "width": width,
        "height": height,
        "capture_datetime": str(taken),
        "relative_altitude": xmp.get("RelativeAltitude"),
        "gimbal_pitch": xmp.get("GimbalPitchDegree"),
    }


def describe(path: Path) -> dict | None:
    rel = path.relative_to(ROOT)
    parts = rel.parts  # drone_images/<group>/<subfolder>/<file>
    if len(parts) != 4:
        print(f"Skipping {rel.as_posix()}: unexpected directory depth", file=sys.stderr)
        return None
    _, group, sub, filename = parts
    stem = Path(filename).stem

    m = SITE_RE.search(stem)
    if not m:
        print(f"Skipping {rel.as_posix()}: no site code in filename", file=sys.stderr)
        return None
    site = f"PT-{m.group(1).upper()}"

    review, notes = [], []
    week_m = WEEK_RE.match(group)
    if week_m:
        collection = "standard"
        week_no = int(week_m.group(1))
        week = f"WEEK {week_no}"
        condition = sub.upper()
        if condition not in ("BEFORE", "AFTER"):
            review.append("condition")
            notes.append(f"condition is the folder label '{sub}'; confirm whether it maps to BEFORE/AFTER")
        image_id = f"W{week_no:02d}_{condition.replace('_', '-')}_{site}"
    elif group.lower() == "height":
        collection = "low_altitude"
        week, condition = UNKNOWN, UNKNOWN
        review += ["week", "condition"]
        suffix = stem[m.end():].lstrip("-_")
        image_id = f"LA_{site}_{suffix}" if suffix else f"LA_{site}"
        if "angled" in stem.lower():
            notes.append("filename says 'angled'; check camera geometry")
    else:
        print(f"Skipping {rel.as_posix()}: unrecognised collection folder", file=sys.stderr)
        return None

    meta = read_metadata(path)
    if meta["relative_altitude"] is not None:
        altitude = f"{float(meta['relative_altitude']):.1f}"
        altitude_source = "XMP drone-dji:RelativeAltitude (m above takeoff point)"
    else:
        altitude, altitude_source = UNKNOWN, "not recorded"
        review.append("altitude_m")

    review.append("coverage_scope")  # cannot be read from metadata; confirm visually

    return {
        "image_id": image_id,
        "image_path": rel.as_posix(),
        "site": site,
        "week": week,
        "condition": condition,
        "altitude_m": altitude,
        "coverage_scope": UNKNOWN,
        "collection": collection,
        "capture_datetime": meta["capture_datetime"],
        "gimbal_pitch_deg": meta["gimbal_pitch"] or UNKNOWN,
        "altitude_source": altitude_source,
        "width": meta["width"],
        "height": meta["height"],
        "sha256": meta["sha256"],
        "needs_review": ";".join(review),
        "notes": "; ".join(notes),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=ROOT / "manifests" / "manifest.csv")
    ap.add_argument("--force", action="store_true", help="Overwrite an existing manifest (loses manual edits)")
    args = ap.parse_args()

    if args.out.exists() and not args.force:
        sys.exit(f"{args.out} already exists and may contain manual edits. Use --force to overwrite.")

    images = sorted(p for p in IMAGE_ROOT.rglob("*") if p.suffix.lower() in (".jpg", ".jpeg"))
    rows = [r for r in (describe(p) for p in images) if r]

    ids = [r["image_id"] for r in rows]
    dupes = {i for i in ids if ids.count(i) > 1}
    if dupes:
        sys.exit(f"Duplicate image_id values: {sorted(dupes)}")

    order = {"standard": 0, "low_altitude": 1}
    rows.sort(key=lambda r: (order[r["collection"]], r["image_id"]))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    flagged = sum(1 for r in rows if r["needs_review"])
    print(f"Wrote {len(rows)} images to {args.out} ({flagged} with fields flagged for review)")


if __name__ == "__main__":
    main()
