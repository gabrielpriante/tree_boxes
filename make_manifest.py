#!/usr/bin/env python3
"""
Build or update the image manifest for the tree-count benchmark from drone_images/.

Metadata comes only from what is recorded in the repository:
  * site              filename (PT_02 / PT-02 / PT_R1 / PT-R1)
  * week              weekN directory name; NOT_APPLICABLE for the altitude series
  * condition         BEFORE / AFTER directory name, the week 1 folder label verbatim,
                      or ALTITUDE_TEST for the altitude series
  * altitude_m        DJI XMP drone-dji:RelativeAltitude embedded in the JPEG
  * capture time      EXIF DateTimeOriginal
  * camera geometry   DJI XMP gimbal pitch and yaw

UNKNOWN means not yet established; NOT_APPLICABLE means the field does not apply.
Proposed labels and anything not recorded are listed in needs_review. A field stays
there until a person confirms it by removing it from needs_review.

Usage:
    python make_manifest.py                  # create manifests/manifest.csv (refuses to overwrite)
    python make_manifest.py --update         # fill only UNKNOWN/blank cells and add new columns/images;
                                             # never overwrites a value someone has set (backup kept)
    python make_manifest.py --review         # list fields awaiting confirmation
"""

import argparse
import csv
import hashlib
import re
import shutil
import statistics
import sys
from datetime import datetime
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent
IMAGE_ROOT = ROOT / "drone_images"
DEFAULT_OUT = ROOT / "manifests" / "manifest.csv"
UNKNOWN = "UNKNOWN"
NOT_APPLICABLE = "NOT_APPLICABLE"

FIELDS = [
    "image_id", "image_path", "site", "week", "condition", "altitude_m",
    "coverage_scope", "collection", "capture_datetime", "gimbal_pitch_deg", "gimbal_yaw_deg",
    "capture_geometry", "altitude_source", "width", "height", "sha256", "needs_review", "notes",
]
# Fields --update may fill when the existing cell is UNKNOWN or blank.
FILLABLE = {"week", "condition", "altitude_m", "capture_datetime", "gimbal_pitch_deg", "gimbal_yaw_deg",
            "capture_geometry", "altitude_source", "width", "height", "sha256"}

SITE_RE = re.compile(r"PT[-_](02|R1)", re.IGNORECASE)
WEEK_RE = re.compile(r"^week(\d+)$", re.IGNORECASE)
XMP_RE = re.compile(rb'drone-dji:(RelativeAltitude|GimbalPitchDegree|GimbalYawDegree)="([^"]+)"')
NADIR_TOLERANCE_DEG = 2.0
YAW_OUTLIER_DEG = 20.0


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
        "gimbal_yaw": xmp.get("GimbalYawDegree"),
    }


def geometry(pitch: str | None) -> str:
    if pitch is None:
        return UNKNOWN
    return "NADIR" if abs(float(pitch) + 90.0) <= NADIR_TOLERANCE_DEG else "OBLIQUE"


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
        week, condition = NOT_APPLICABLE, "ALTITUDE_TEST"
        review += ["week", "condition"]
        notes.append("proposed labels: altitude-series flight outside the weekly intervention sessions; "
                     "not an intervention BEFORE/AFTER image. Confirm week=NOT_APPLICABLE, condition=ALTITUDE_TEST")
        suffix = stem[m.end():].lstrip("-_")
        image_id = f"LA_{site}_{suffix}" if suffix else f"LA_{site}"
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
        "gimbal_yaw_deg": meta["gimbal_yaw"] or UNKNOWN,
        "capture_geometry": geometry(meta["gimbal_pitch"]),
        "altitude_source": altitude_source,
        "width": meta["width"],
        "height": meta["height"],
        "sha256": meta["sha256"],
        "needs_review": ";".join(review),
        "notes": "; ".join(notes),
    }


def flag_heading_outliers(rows: list) -> None:
    """Within each site's altitude series, flag images whose gimbal yaw differs from the rest."""
    for site in {r["site"] for r in rows if r["collection"] == "low_altitude"}:
        series = [r for r in rows if r["collection"] == "low_altitude" and r["site"] == site
                  and r["gimbal_yaw_deg"] != UNKNOWN]
        if len(series) < 3:
            continue
        median = statistics.median(float(r["gimbal_yaw_deg"]) for r in series)
        for r in series:
            diff = abs((float(r["gimbal_yaw_deg"]) - median + 180) % 360 - 180)
            if diff > YAW_OUTLIER_DEG:
                note = (f"gimbal yaw {float(r['gimbal_yaw_deg']):+.1f} deg differs by {diff:.0f} deg from the other "
                        f"{site} altitude images (median {median:+.1f}); gimbal pitch {r['gimbal_pitch_deg']} is "
                        f"nadir, so the frame is rotated, not oblique. Confirm how to classify it in the altitude series")
                if "angled" in r["image_path"].lower():
                    note = "filename says 'angled' but " + note
                r["notes"] = "; ".join(x for x in (r["notes"], note) if x)
                r["needs_review"] = ";".join(x for x in (r["needs_review"], "capture_geometry") if x)


def scan() -> list:
    images = sorted(p for p in IMAGE_ROOT.rglob("*") if p.suffix.lower() in (".jpg", ".jpeg"))
    rows = [r for r in (describe(p) for p in images) if r]
    ids = [r["image_id"] for r in rows]
    dupes = {i for i in ids if ids.count(i) > 1}
    if dupes:
        sys.exit(f"Duplicate image_id values: {sorted(dupes)}")
    flag_heading_outliers(rows)
    order = {"standard": 0, "low_altitude": 1}
    rows.sort(key=lambda r: (order[r["collection"]], r["image_id"]))
    return rows


def write(path: Path, rows: list, fields: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def is_unset(value) -> bool:
    return value is None or str(value).strip() in ("", UNKNOWN)


def update(path: Path) -> None:
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        existing_fields = list(reader.fieldnames or [])
        existing = list(reader)
    fresh = {r["image_path"]: r for r in scan()}
    fields = FIELDS + [c for c in existing_fields if c not in FIELDS]  # keep any added columns

    changes = []
    for row in existing:
        new = fresh.pop(row["image_path"], None)
        if new is None:
            print(f"WARNING {row['image_id']}: {row['image_path']} no longer found; row left unchanged")
            continue
        if row.get("sha256") and row["sha256"] != new["sha256"]:
            print(f"WARNING {row['image_id']}: file hash changed since the manifest was written; row left unchanged")
            continue
        review = [x for x in (row.get("needs_review") or "").split(";") if x]
        notes = [x.strip() for x in (row.get("notes") or "").split("; ") if x.strip()]
        for col in FIELDS:
            if col not in row:  # new column
                row[col] = new[col]
                if col not in ("needs_review", "notes"):
                    changes.append((row["image_id"], col, "(new column)", new[col]))
            elif col in FILLABLE and is_unset(row[col]) and not is_unset(new[col]):
                changes.append((row["image_id"], col, row[col], new[col]))
                row[col] = new[col]
                if col in new["needs_review"].split(";") and col not in review:
                    review.append(col)
        # Propose reviews/notes for fields this update introduced; never re-add confirmed ones.
        touched = {c for i, c, _, _ in changes if i == row["image_id"]}
        for item in new["needs_review"].split(";"):
            if item and item in touched and item not in review:
                review.append(item)
        if touched:
            notes += [n for n in new["notes"].split("; ") if n and n not in notes]
        row["needs_review"] = ";".join(review)
        row["notes"] = "; ".join(notes)

    for new in fresh.values():
        existing.append(new)
        changes.append((new["image_id"], "(row)", "", "new image added"))

    if not changes:
        print("Manifest already up to date; nothing changed.")
        return
    backup = path.with_name(f"{path.name}.bak-{datetime.now().strftime('%Y%m%dT%H%M%S')}")
    shutil.copyfile(path, backup)
    write(path, existing, fields)
    print(f"Updated {path} (backup: {backup.name})")
    for image_id, col, old, new in changes:
        print(f"  {image_id:<26} {col:<18} {old!s:>14} -> {new}")


def review(path: Path) -> None:
    with path.open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    pending = [r for r in rows if r.get("needs_review")]
    pending.sort(key=lambda r: (r.get("collection") != "low_altitude", r["image_id"]))
    for r in pending:
        fields = r["needs_review"].split(";")
        values = ", ".join(f"{c}={r.get(c, '')}" for c in fields)
        print(f"{r['image_id']:<26} {values}")
        if r.get("notes"):
            print(f"{'':<26} note: {r['notes']}")
    print(f"\n{len(pending)} of {len(rows)} images have fields awaiting confirmation.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--update", action="store_true", help="Fill UNKNOWN/blank cells; keep manual edits")
    mode.add_argument("--review", action="store_true", help="List fields awaiting confirmation")
    mode.add_argument("--force", action="store_true", help="Regenerate from scratch (loses manual edits)")
    args = ap.parse_args()

    if args.review:
        return review(args.out)
    if args.update:
        if not args.out.exists():
            sys.exit(f"{args.out} does not exist; run without --update to create it.")
        return update(args.out)
    if args.out.exists() and not args.force:
        sys.exit(f"{args.out} already exists and may contain manual edits. "
                 "Use --update to fill gaps safely, or --force to regenerate from scratch.")
    rows = scan()
    write(args.out, rows, FIELDS)
    flagged = sum(1 for r in rows if r["needs_review"])
    print(f"Wrote {len(rows)} images to {args.out} ({flagged} with fields flagged for review)")


if __name__ == "__main__":
    main()
