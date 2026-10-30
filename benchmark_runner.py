#!/usr/bin/env python3
"""
Batch runner for the tree-count consistency benchmark.

Each selected image is sent to the model --runs times (default 10). Every trial is an
independent API request with identical settings, prompt and preprocessed image bytes.
Results are written under:

    <output-dir>/<experiment-id>/<model-config-id>/<image-id>/trial_NN/attempt_NN/

and the master CSVs (counts_long.csv, counts_summary.csv, ...) are rebuilt from those
records at <output-dir>/<experiment-id>/ after every run.

Examples:
    # Plan only: no API calls, nothing written
    python benchmark_runner.py --manifest manifests/manifest.csv \\
        --image-ids W03_BEFORE_PT-02,W03_BEFORE_PT-R1 --max-usd 5 --dry-run

    # Live run (paid)
    python benchmark_runner.py --manifest manifests/manifest.csv \\
        --image-ids W03_BEFORE_PT-02,W03_BEFORE_PT-R1 --runs 10 --max-usd 5 --execute

    # Continue an interrupted or budget-stopped run without repeating completed trials
    ... --execute --resume

    # Offline rehearsal with the mock adapter (no network, no cost)
    ... --execute --mock
"""

import argparse
import csv
import hashlib
import json
import os
import shutil
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import box_generation as bg
from model_adapters import FatalAdapterError, get_adapter

ROOT = Path(__file__).resolve().parent
RECORD_SCHEMA_VERSION = 1
UNKNOWN = "UNKNOWN"

MANIFEST_REQUIRED = ["image_id", "image_path", "site", "week", "condition", "altitude_m", "coverage_scope"]
PROMPT_FIELDS = ("site", "week", "condition")  # these values are inserted into the prompt
MANIFEST_CARRIED = MANIFEST_REQUIRED + ["collection"]

# A trial is complete once one attempt reaches a terminal status. Terminal statuses are
# model outcomes and are never re-run automatically, so results cannot be cherry-picked.
TERMINAL = {"success", "invalid", "refusal", "max_tokens"}
# Retryable statuses: no usable model outcome (API/transport failure or interruption).
RETRYABLE = {"error", "interrupted"}

MAX_CONSECUTIVE_FAILED_TRIALS = 3

LONG_FIELDS = [
    "experiment_id", "image_id", "site", "week", "condition", "altitude_m", "model", "trial",
    "tree_count", "input_tokens", "output_tokens", "estimated_cost_usd", "status",
    "model_code", "model_config_id", "collection", "attempt", "attempts_made", "timestamp_utc",
    "record_path",
]
SUMMARY_FIELDS = [
    "image_id", "model", "successful_trials", "mean_tree_count", "standard_deviation",
    "minimum_count", "maximum_count", "coefficient_of_variation",
    "experiment_id", "model_code", "model_config_id", "site", "week", "condition", "altitude_m",
    "collection", "trials_recorded", "invalid_trials", "refusal_trials", "max_tokens_trials",
    "unresolved_trials", "total_cost_usd",
]
ATTEMPT_FIELDS = [
    "experiment_id", "model_config_id", "image_id", "trial", "attempt", "status", "stop_reason",
    "tree_count", "raw_box_count", "input_tokens", "output_tokens", "estimated_cost_usd",
    "cost_basis", "possibly_billed", "api_retries", "error", "started_utc", "finished_utc",
    "record_path",
]
BOX_FIELDS = [
    "experiment_id", "model_config_id", "model", "image_id", "trial", "tree_id",
    "x_min", "y_min", "x_max", "y_max",
]
INVALID_FIELDS = ["experiment_id", "model_config_id", "image_id", "trial", "attempt", "status",
                  "errors", "record_path"]


class BudgetStop(Exception):
    pass


class ConsistencyError(Exception):
    """Stored results for an image were produced from different inputs than the current run."""


# ---------------------------------------------------------------- small helpers

def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_json(path: Path, obj) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2), encoding="utf-8")
    tmp.replace(path)


def write_new_json(path: Path, obj) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}")
    write_json(path, obj)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def is_unknown(value) -> bool:
    return value is None or str(value).strip() in ("", UNKNOWN)


def config_id(cfg: dict) -> str:
    """Short, stable id: any change to model, settings, prompt, schema or preprocessing changes it."""
    digest = sha256_text(json.dumps(cfg, sort_keys=True))[:10]
    return f"{cfg['model_code']}_{cfg['inference']['effort']}_{digest}"


WINDOWS_MAX_PATH = 259
LONGEST_FILENAME = "trial_00/attempt_00/invalid_response.json.tmp"


def check_path_lengths(run_dir: Path, image_ids) -> None:
    """Fail before any request if result files would exceed the Windows path limit."""
    if os.name != "nt":
        return
    longest = max(len(str((run_dir / i / LONGEST_FILENAME).resolve())) for i in image_ids)
    if longest > WINDOWS_MAX_PATH:
        raise SystemExit(f"Result paths would be {longest} characters, over the Windows limit of "
                         f"{WINDOWS_MAX_PATH}. Use a shorter --output-dir or --experiment-id.")


def write_csv(path: Path, fields: list, rows: list) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


# ---------------------------------------------------------------- manifest

def load_manifest(path: Path, image_root: Path = ROOT, image_ids=None) -> list:
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        missing = [c for c in MANIFEST_REQUIRED if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"Manifest {path} is missing columns: {missing}")
        rows = [{k: (v or "").strip() for k, v in r.items()} for r in reader]

    ids = [r["image_id"] for r in rows]
    if any(not i for i in ids):
        raise ValueError("Manifest has a row with an empty image_id")
    dupes = sorted({i for i in ids if ids.count(i) > 1})
    if dupes:
        raise ValueError(f"Duplicate image_id values in manifest: {dupes}")

    if image_ids:
        unknown = [i for i in image_ids if i not in ids]
        if unknown:
            raise ValueError(f"image ids not in manifest: {unknown}")
        wanted = set(image_ids)
        rows = [r for r in rows if r["image_id"] in wanted]  # manifest order, not CLI order

    missing_files = []
    for r in rows:
        r["_abs_path"] = (image_root / r["image_path"]).resolve()
        if not r["_abs_path"].is_file():
            missing_files.append(r["image_path"])
    if missing_files:
        raise ValueError(f"Image files not found: {missing_files}")
    return rows


# ---------------------------------------------------------------- validation

def _norm(value) -> str:
    return str(value).strip().strip("{}").strip().upper()


def validate_detections(parsed, row: dict) -> tuple:
    """Strict check of a parsed model response. Reports problems; never corrects them."""
    errors, warnings = [], []
    if not isinstance(parsed, dict):
        return ["response is not a JSON object"], warnings

    for key in ("site", "week", "condition", "trees"):
        if key not in parsed:
            errors.append(f"missing key '{key}'")
    extra = set(parsed) - {"site", "week", "condition", "trees"}
    if extra:
        warnings.append(f"unexpected keys: {sorted(extra)}")

    trees = parsed.get("trees")
    if "trees" in parsed and not isinstance(trees, list):
        errors.append("'trees' is not an array")
        trees = []
    trees = trees or []

    seen = set()
    for i, t in enumerate(trees):
        label = f"trees[{i}]"
        if not isinstance(t, dict):
            errors.append(f"{label} is not an object")
            continue
        tid = t.get("tree_id")
        if not isinstance(tid, str) or not tid:
            errors.append(f"{label} has no tree_id")
        elif tid in seen:
            errors.append(f"duplicate tree_id {tid}")
        else:
            seen.add(tid)
            label = tid
        bbox = t.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            errors.append(f"{label}: bbox must have 4 values, got {bbox!r}")
            continue
        if not all(isinstance(v, int) and not isinstance(v, bool) for v in bbox):
            errors.append(f"{label}: bbox values must be integers, got {bbox!r}")
            continue
        if any(v < 0 or v > 1000 for v in bbox):
            errors.append(f"{label}: bbox outside 0..1000: {bbox}")
        x0, y0, x1, y1 = bbox
        if x0 >= x1 or y0 >= y1:
            errors.append(f"{label}: bbox requires x_min < x_max and y_min < y_max: {bbox}")

    ids = [t.get("tree_id") for t in trees if isinstance(t, dict)]
    if ids != [f"T{n:03d}" for n in range(1, len(ids) + 1)]:
        warnings.append("tree_ids are not sequential from T001")

    for key in PROMPT_FIELDS:
        if key in parsed and _norm(parsed[key]) != _norm(row[key]):
            warnings.append(f"echoed {key} {parsed[key]!r} differs from manifest {row[key]!r}")
    return errors, warnings


def drawable_boxes(parsed) -> list:
    """Boxes that can be drawn as given, for inspecting invalid responses. Nothing is altered."""
    out = []
    for t in (parsed.get("trees") or []) if isinstance(parsed, dict) else []:
        bbox = t.get("bbox") if isinstance(t, dict) else None
        if (isinstance(bbox, list) and len(bbox) == 4 and all(isinstance(v, int) for v in bbox)
                and bbox[0] < bbox[2] and bbox[1] < bbox[3]):
            out.append({"tree_id": str(t.get("tree_id")), "bbox": bbox})
    return out


# ---------------------------------------------------------------- on-disk state

def attempt_dirs(trial_dir: Path) -> list:
    return sorted(p for p in trial_dir.glob("attempt_*") if p.is_dir())


def load_attempt(adir: Path) -> dict:
    """Record for one attempt. An attempt without record.json was cut off mid-request."""
    from_path = {"image_id": adir.parent.parent.name,
                 "trial": int(adir.parent.name.split("_")[1]),
                 "attempt": int(adir.name.split("_")[1])}
    rec_path = adir / "record.json"
    if rec_path.exists():
        rec = read_json(rec_path)
        rec["_path"] = rec_path
        return rec
    req_path = adir / "request.json"
    req = read_json(req_path) if req_path.exists() else {}
    return {
        **from_path,
        **req,
        "status": "interrupted",
        "possibly_billed": True,
        "estimated_cost_usd": req.get("pre_request_worst_case_usd", 0.0),
        "cost_basis": "worst_case_bound (process ended before a record was written)",
        "error": "no record.json; process ended during the request",
        "_path": req_path if req_path.exists() else adir,
    }


def trial_state(trial_dir: Path) -> tuple:
    """(state, attempts) where state is 'done', 'retry' or 'pending'."""
    attempts = [load_attempt(a) for a in attempt_dirs(trial_dir)] if trial_dir.exists() else []
    if any(a.get("status") in TERMINAL for a in attempts):
        return "done", attempts
    return ("retry" if attempts else "pending"), attempts


def all_attempts(run_dir: Path) -> list:
    if not run_dir.exists():
        return []
    return [load_attempt(a) for a in sorted(run_dir.glob("*/trial_*/attempt_*")) if a.is_dir()]


def image_attempts(run_dir: Path, image_id: str) -> list:
    image_dir = run_dir / image_id
    if not image_dir.exists():
        return []
    return [load_attempt(a) for a in sorted(image_dir.glob("trial_*/attempt_*")) if a.is_dir()]


def spent_usd(run_dir: Path) -> float:
    return sum(float(a.get("estimated_cost_usd") or 0.0) for a in all_attempts(run_dir))


def observed_input_tokens(run_dir: Path, image_id: str) -> int | None:
    """Actual input tokens from an earlier request for this image (identical across trials)."""
    seen = [(a.get("usage") or {}).get("input_tokens") or 0 for a in image_attempts(run_dir, image_id)]
    seen = [s for s in seen if s]
    return max(seen) if seen else None


def worst_case_usd(adapter, prompt: str, observed: int | None) -> float:
    input_tokens = observed if observed else adapter.estimate_input_tokens(prompt)
    return adapter.cost_usd({"input_tokens": input_tokens, "output_tokens": adapter.max_output_tokens()})


# ---------------------------------------------------------------- one attempt

def run_attempt(adapter, cfg, cfg_id, experiment_id, row, trial, image, run_dir, worst_case, manifest_sha):
    trial_dir = run_dir / row["image_id"] / f"trial_{trial:02d}"
    attempt = len(attempt_dirs(trial_dir)) + 1 if trial_dir.exists() else 1
    adir = trial_dir / f"attempt_{attempt:02d}"
    adir.mkdir(parents=True, exist_ok=False)

    base = {
        "record_schema_version": RECORD_SCHEMA_VERSION,
        "experiment_id": experiment_id,
        "model_config_id": cfg_id,
        "model_code": cfg["model_code"],
        "adapter": cfg["adapter"],
        "model": cfg["model"],
        "inference_settings": cfg["inference"],
        "pricing": adapter.pricing,
        **{k: row.get(k, "") for k in MANIFEST_CARRIED},
        "image_path": row["image_path"],
        "manifest_sha256": manifest_sha,
        "trial": trial,
        "attempt": attempt,
        "image": image["meta"],
        "prompt_sha256": image["prompt_sha256"],
        "output_schema_sha256": cfg["output_schema_sha256"],
        "prompt_template_sha256": cfg["prompt_template_sha256"],
        "preprocessing": cfg["preprocessing"],
        "pre_request_worst_case_usd": round(worst_case, 6),
        "software": adapter.software(),
        "started_utc": now_utc(),
    }
    write_new_json(adir / "request.json", base)
    (adir / "prompt.txt").write_text(image["prompt"], encoding="utf-8")

    record = {**base, "status": None, "stop_reason": None, "tree_count": None, "raw_box_count": None,
              "usage": {}, "estimated_cost_usd": 0.0, "cost_basis": "not_billed",
              "possibly_billed": False, "request_id": None, "api_retries": [],
              "validation": {"errors": [], "warnings": []}, "error": None, "files": {}}
    t0 = time.monotonic()

    def finish(status):
        record["status"] = status
        record["finished_utc"] = now_utc()
        record["duration_s"] = round(time.monotonic() - t0, 2)
        record["estimated_cost_usd"] = round(record["estimated_cost_usd"], 6)
        write_new_json(adir / "record.json", record)
        return record

    context = {"image_id": row["image_id"], "trial": trial, "attempt": attempt}
    try:
        result = adapter.run(image["jpeg"], image["prompt"], context)
    except KeyboardInterrupt:
        record.update(possibly_billed=True, estimated_cost_usd=worst_case,
                      cost_basis="worst_case_bound (interrupted during request)",
                      error="interrupted by user during request")
        finish("interrupted")
        raise
    except FatalAdapterError as e:
        record["error"] = str(e)
        finish("error")
        raise

    record["api_retries"] = result.retries
    record["request_id"] = result.request_id
    record["usage"] = result.usage
    record["stop_reason"] = result.stop_reason

    if result.status != "ok":
        record["error"] = result.error
        if result.possibly_billed:
            record.update(possibly_billed=True, estimated_cost_usd=worst_case,
                          cost_basis="worst_case_bound (failed after generation started)")
        return finish("error")

    record["estimated_cost_usd"] = adapter.cost_usd(result.usage)
    record["cost_basis"] = "actual_usage"
    write_new_json(adir / "response_raw.json", result.raw)
    record["files"]["response_raw"] = "response_raw.json"

    if result.stop_reason == "refusal":
        return finish("refusal")
    if result.stop_reason == "max_tokens":
        record["error"] = "response truncated at max_tokens"
        return finish("max_tokens")

    try:
        parsed = json.loads(result.text)
    except (json.JSONDecodeError, TypeError) as e:
        record["validation"]["errors"] = [f"response is not valid JSON: {e}"]
        write_new_json(adir / "invalid_response.json",
                       {"text": result.text, "errors": record["validation"]["errors"]})
        record["files"]["invalid_response"] = "invalid_response.json"
        return finish("invalid")

    errors, warnings = validate_detections(parsed, row)
    record["validation"] = {"errors": errors, "warnings": warnings}
    trees = parsed.get("trees") if isinstance(parsed, dict) else None
    record["raw_box_count"] = len(trees) if isinstance(trees, list) else None

    if errors:
        write_new_json(adir / "invalid_response.json", {"parsed": parsed, "errors": errors, "warnings": warnings})
        record["files"]["invalid_response"] = "invalid_response.json"
        boxes = drawable_boxes(parsed)
        bg.draw_boxes(image["original"], boxes).save(adir / "annotated_INVALID.jpg", quality=92)
        record["files"]["annotated"] = "annotated_INVALID.jpg"
        return finish("invalid")

    write_new_json(adir / "detections.json", parsed)
    record["files"]["detections"] = "detections.json"
    bg.draw_boxes(image["original"], parsed["trees"]).save(adir / "annotated.jpg", quality=92)
    record["files"]["annotated"] = "annotated.jpg"
    record["tree_count"] = len(parsed["trees"])
    return finish("success")


# ---------------------------------------------------------------- image preparation

def prepare_image(row: dict) -> dict:
    raw_bytes = row["_abs_path"].read_bytes()
    file_sha = sha256_bytes(raw_bytes)
    if row.get("sha256") and row["sha256"] != file_sha:
        raise ConsistencyError(f"{row['image_id']}: file hash differs from manifest sha256")
    original = bg.load_image(row["_abs_path"])
    jpeg, upload = bg.prepare_upload(original)
    prompt = bg.build_prompt(row["site"], row["week"], row["condition"])
    meta = {
        "file_sha256": file_sha,
        "original_width": original.width,
        "original_height": original.height,
        "upload_width": upload["width"],
        "upload_height": upload["height"],
        "upload_jpeg_quality": upload["jpeg_quality"],
        "upload_bytes": upload["bytes"],
        "upload_sha256": sha256_bytes(jpeg),
    }
    return {"original": original, "jpeg": jpeg, "prompt": prompt,
            "prompt_sha256": sha256_text(prompt), "meta": meta}


def check_consistency(image: dict, attempts: list, image_id: str) -> None:
    for a in attempts:
        prev = a.get("image") or {}
        if prev.get("upload_sha256") and prev["upload_sha256"] != image["meta"]["upload_sha256"]:
            raise ConsistencyError(f"{image_id}: preprocessed image bytes differ from earlier trials "
                                   f"({a['_path']}). Check Pillow version or source file.")
        if a.get("prompt_sha256") and a["prompt_sha256"] != image["prompt_sha256"]:
            raise ConsistencyError(f"{image_id}: prompt differs from earlier trials ({a['_path']}). "
                                   "Manifest site/week/condition probably changed after trials ran.")


# ---------------------------------------------------------------- aggregation

def _fmt(x, nd):
    return "" if x is None else round(x, nd)


def aggregate(exp_dir: Path) -> dict:
    """Rebuild every CSV from the per-attempt records. Safe to run any time."""
    if not exp_dir.exists():
        return {}
    experiment_id = exp_dir.name
    attempts = []
    for run_dir in sorted(p for p in exp_dir.iterdir() if p.is_dir() and (p / "model_config.json").exists()):
        cfg = read_json(run_dir / "model_config.json")
        for a in all_attempts(run_dir):
            a.setdefault("model_config_id", run_dir.name)
            a.setdefault("model", cfg["model"])
            a.setdefault("model_code", cfg["model_code"])
            attempts.append(a)

    def rel(p):
        return Path(p).relative_to(exp_dir).as_posix()

    attempt_rows, invalid_rows = [], []
    by_trial = {}
    for a in attempts:
        usage = a.get("usage") or {}
        attempt_rows.append({
            **{k: a.get(k) for k in ("experiment_id", "model_config_id", "image_id", "trial", "attempt",
                                     "status", "stop_reason", "tree_count", "raw_box_count", "cost_basis",
                                     "possibly_billed", "error", "started_utc", "finished_utc")},
            "experiment_id": experiment_id,
            "input_tokens": usage.get("input_tokens"),
            "output_tokens": usage.get("output_tokens"),
            "estimated_cost_usd": a.get("estimated_cost_usd"),
            "api_retries": len(a.get("api_retries") or []),
            "record_path": rel(a["_path"]),
        })
        if a.get("status") == "invalid":
            invalid_rows.append({"experiment_id": experiment_id, "model_config_id": a["model_config_id"],
                                 "image_id": a.get("image_id"), "trial": a.get("trial"),
                                 "attempt": a.get("attempt"), "status": a.get("status"),
                                 "errors": " | ".join((a.get("validation") or {}).get("errors") or []),
                                 "record_path": rel(a["_path"])})
        by_trial.setdefault((a["model_config_id"], a.get("image_id"), a.get("trial")), []).append(a)

    long_rows, box_rows = [], []
    for (cfg_id, image_id, trial), group in sorted(by_trial.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2])):
        group.sort(key=lambda a: a.get("attempt") or 0)
        terminal = [a for a in group if a.get("status") in TERMINAL]
        chosen = terminal[0] if terminal else group[-1]
        usage = chosen.get("usage") or {}
        long_rows.append({
            "experiment_id": experiment_id,
            **{k: chosen.get(k, "") for k in ("image_id", "site", "week", "condition", "altitude_m",
                                              "model", "trial", "tree_count", "status", "model_code",
                                              "collection", "attempt")},
            "model_config_id": cfg_id,
            "input_tokens": usage.get("input_tokens"),
            "output_tokens": usage.get("output_tokens"),
            "estimated_cost_usd": chosen.get("estimated_cost_usd"),
            "attempts_made": len(group),
            "timestamp_utc": chosen.get("finished_utc") or chosen.get("started_utc"),
            "record_path": rel(chosen["_path"]),
            "_all_cost": sum(float(a.get("estimated_cost_usd") or 0) for a in group),
        })
        if chosen.get("status") == "success":
            det = read_json(Path(chosen["_path"]).parent / "detections.json")
            for t in det["trees"]:
                x0, y0, x1, y1 = t["bbox"]
                box_rows.append({"experiment_id": experiment_id, "model_config_id": cfg_id,
                                 "model": chosen.get("model"), "image_id": image_id, "trial": trial,
                                 "tree_id": t["tree_id"], "x_min": x0, "y_min": y0, "x_max": x1, "y_max": y1})

    summary_rows = []
    groups = {}
    for r in long_rows:
        groups.setdefault((r["image_id"], r["model_config_id"]), []).append(r)
    for (image_id, cfg_id), rows in sorted(groups.items()):
        counts = [int(r["tree_count"]) for r in rows if r["status"] == "success"]
        n = len(counts)
        mean = statistics.fmean(counts) if n else None
        sd = statistics.stdev(counts) if n >= 2 else None
        cv = sd / mean if sd is not None and mean else None
        first = rows[0]
        status_count = lambda s: sum(1 for r in rows if r["status"] == s)
        summary_rows.append({
            "image_id": image_id, "model": first["model"], "successful_trials": n,
            "mean_tree_count": _fmt(mean, 3), "standard_deviation": _fmt(sd, 3),
            "minimum_count": min(counts) if n else "", "maximum_count": max(counts) if n else "",
            "coefficient_of_variation": _fmt(cv, 4),
            "experiment_id": experiment_id, "model_code": first["model_code"], "model_config_id": cfg_id,
            **{k: first[k] for k in ("site", "week", "condition", "altitude_m", "collection")},
            "trials_recorded": len(rows),
            "invalid_trials": status_count("invalid"),
            "refusal_trials": status_count("refusal"),
            "max_tokens_trials": status_count("max_tokens"),
            "unresolved_trials": sum(1 for r in rows if r["status"] in RETRYABLE),
            "total_cost_usd": round(sum(r["_all_cost"] for r in rows), 6),
        })

    write_csv(exp_dir / "counts_long.csv", LONG_FIELDS, long_rows)
    write_csv(exp_dir / "counts_summary.csv", SUMMARY_FIELDS, summary_rows)
    write_csv(exp_dir / "attempts_log.csv", ATTEMPT_FIELDS, attempt_rows)
    write_csv(exp_dir / "boxes_long.csv", BOX_FIELDS, box_rows)
    write_csv(exp_dir / "invalid_trials.csv", INVALID_FIELDS, invalid_rows)
    return {"long": long_rows, "summary": summary_rows, "attempts": attempt_rows}


# ---------------------------------------------------------------- orchestration

def run_benchmark(rows, adapter, output_dir: Path, experiment_id: str, runs: int = 10,
                  max_usd: float | None = None, resume: bool = False, dry_run: bool = True,
                  max_attempts: int = 2, allow_unknown_metadata: bool = False,
                  manifest_path: Path | None = None, log=print) -> dict:
    cfg = adapter.config()
    cfg_id = config_id(cfg)
    exp_dir = output_dir / experiment_id
    run_dir = exp_dir / cfg_id
    report = {"experiment_id": experiment_id, "model_config_id": cfg_id, "run_dir": str(run_dir),
              "trials_run": 0, "stopped": None, "skipped_images": []}

    runnable = []
    for row in rows:
        unknown = [f for f in PROMPT_FIELDS if is_unknown(row[f])]
        if unknown and not allow_unknown_metadata:
            report["skipped_images"].append((row["image_id"], f"prompt metadata unknown: {unknown}"))
            continue
        runnable.append(row)

    states = {}
    for row in runnable:
        states[row["image_id"]] = {t: trial_state(run_dir / row["image_id"] / f"trial_{t:02d}")
                                   for t in range(1, runs + 1)}
    existing = [i for i, s in states.items() if any(st != "pending" for st, _ in s.values())]
    if existing and not resume:
        raise SystemExit(f"Trials already exist in {run_dir} for {existing}. "
                         "Use --resume to continue without repeating completed trials.")

    spent = spent_usd(run_dir)
    log(f"Experiment:   {experiment_id}")
    log(f"Model config: {cfg_id}  ({cfg['adapter']}: {cfg['model']}, effort={cfg['inference']['effort']}, "
        f"thinking={cfg['inference']['thinking']['type']}, max_tokens={cfg['inference']['max_tokens']})")
    log(f"Results dir:  {run_dir}")
    for image_id, why in report["skipped_images"]:
        log(f"  SKIP {image_id}: {why} (complete the manifest, or pass --allow-unknown-metadata)")

    plan, total_worst = [], 0.0
    log(f"\n{'image_id':<28}{'done':>6}{'to run':>8}{'worst-case/trial':>18}")
    for row in runnable:
        prompt = bg.build_prompt(row["site"], row["week"], row["condition"])
        worst = worst_case_usd(adapter, prompt, observed_input_tokens(run_dir, row["image_id"]))
        todo = [t for t, (st, _) in states[row["image_id"]].items() if st != "done"]
        done = runs - len(todo)
        plan.append((row, todo))
        total_worst += worst * len(todo)
        log(f"{row['image_id']:<28}{done:>6}{len(todo):>8}{'$' + format(worst, '.4f'):>18}")
    n_todo = sum(len(t) for _, t in plan)
    log(f"\nTrials to run: {n_todo}. Already spent (recorded): ${spent:.4f}.")
    log(f"Worst-case cost if every remaining trial used all {adapter.max_output_tokens()} output tokens: "
        f"${total_worst:.2f}")
    if max_usd is not None:
        log(f"Budget limit: ${max_usd:.2f} (remaining ${max_usd - spent:.4f}). A trial starts only if "
            "spent + its worst-case cost stays within the limit.")
    report["planned_trials"] = n_todo
    if runnable:
        check_path_lengths(run_dir, [r["image_id"] for r in runnable])

    if dry_run:
        log("\nDry run: no API requests made, nothing written.")
        return report
    if max_usd is None:
        raise SystemExit("--max-usd is required with --execute")
    if n_todo == 0:
        log("\nNothing to run: all selected trials are complete.")
        aggregate(exp_dir)
        return report

    adapter.prepare()
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = run_dir / "model_config.json"
    if cfg_path.exists():
        if read_json(cfg_path) != cfg:
            raise ConsistencyError(f"{cfg_path} does not match the current configuration")
    else:
        write_new_json(cfg_path, cfg)
    exp_json = exp_dir / "experiment.json"
    if not exp_json.exists():
        write_new_json(exp_json, {"experiment_id": experiment_id, "created_utc": now_utc(),
                                  "trial_convention": "trials numbered 1..N per image and model; "
                                                      "each trial is one independent request",
                                  "record_schema_version": RECORD_SCHEMA_VERSION})
    manifest_sha = None
    if manifest_path is not None:
        manifest_bytes = Path(manifest_path).read_bytes()
        manifest_sha = sha256_bytes(manifest_bytes)
        snap = run_dir / "manifest_snapshots" / f"{manifest_sha[:12]}.csv"
        if not snap.exists():
            snap.parent.mkdir(exist_ok=True)
            shutil.copyfile(manifest_path, snap)

    consecutive_failed = 0
    try:
        for row, todo in plan:
            if not todo:
                continue
            image_id = row["image_id"]
            image = prepare_image(row)
            check_consistency(image, image_attempts(run_dir, image_id), image_id)
            log(f"\n{image_id}: upload {image['meta']['upload_width']}x{image['meta']['upload_height']} "
                f"(from {image['meta']['original_width']}x{image['meta']['original_height']})")
            for trial in todo:
                for _ in range(max_attempts):
                    worst = worst_case_usd(adapter, image["prompt"], observed_input_tokens(run_dir, image_id))
                    if spent + worst > max_usd:
                        raise BudgetStop(f"next request worst-case ${worst:.4f} would exceed the remaining "
                                         f"budget ${max_usd - spent:.4f} (limit ${max_usd:.2f})")
                    rec = run_attempt(adapter, cfg, cfg_id, experiment_id, row, trial, image,
                                      run_dir, worst, manifest_sha)
                    spent += rec["estimated_cost_usd"]
                    report["trials_run"] += 1
                    usage = rec["usage"] or {}
                    log(f"  trial {trial:02d} attempt {rec['attempt']:02d}: {rec['status']:<10} "
                        f"trees={rec['tree_count'] if rec['tree_count'] is not None else '-':<4} "
                        f"in={usage.get('input_tokens', 0)} out={usage.get('output_tokens', 0)} "
                        f"cost=${rec['estimated_cost_usd']:.4f} total=${spent:.4f}"
                        + (f"  [{rec['error']}]" if rec.get("error") else ""))
                    if rec["status"] in TERMINAL:
                        break
                if rec["status"] in TERMINAL:
                    consecutive_failed = 0
                else:
                    consecutive_failed += 1
                    if consecutive_failed >= MAX_CONSECUTIVE_FAILED_TRIALS:
                        raise FatalAdapterError(f"{consecutive_failed} consecutive trials failed; "
                                                "stopping. Re-run with --resume later.")
    except BudgetStop as e:
        report["stopped"] = f"budget: {e}"
    except FatalAdapterError as e:
        report["stopped"] = f"error: {e}"
    except ConsistencyError as e:
        report["stopped"] = f"consistency check failed: {e}"
    except KeyboardInterrupt:
        report["stopped"] = "interrupted by user"
    finally:
        aggregate(exp_dir)

    report["spent_usd"] = spent
    remaining = {}
    for row, _ in plan:
        left = [t for t in range(1, runs + 1)
                if trial_state(run_dir / row["image_id"] / f"trial_{t:02d}")[0] != "done"]
        if left:
            remaining[row["image_id"]] = left
    report["incomplete"] = remaining

    log(f"\nRecorded spend for this model config: ${spent:.4f} of ${max_usd:.2f}")
    if report["stopped"]:
        log(f"STOPPED: {report['stopped']}")
    if remaining:
        log("Incomplete images (trials not yet complete):")
        for image_id, left in remaining.items():
            log(f"  {image_id}: {runs - len(left)}/{runs} complete; outstanding trials {left}")
        log("Re-run the same command with --resume to continue.")
    else:
        log(f"All selected images have {runs} completed trials.")
    log(f"CSVs: {exp_dir / 'counts_long.csv'}, {exp_dir / 'counts_summary.csv'}")
    return report


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--image-ids", help="Comma-separated image_ids to run (default: every manifest row)")
    ap.add_argument("--runs", type=int, default=10, help="Trials per image (default 10)")
    ap.add_argument("--max-usd", type=float, help="Cumulative spending limit for this model config")
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true", help="Show the plan and cost bounds; no API calls")
    mode.add_argument("--execute", action="store_true", help="Make the API requests (paid unless --mock)")
    ap.add_argument("--resume", action="store_true", help="Continue; completed trials are skipped")
    ap.add_argument("--output-dir", type=Path, default=ROOT / "benchmark_results")
    ap.add_argument("--experiment-id", default="tree_count_consistency_v1")
    ap.add_argument("--model", default="M08", help="Model code (only M08 is implemented)")
    ap.add_argument("--max-attempts", type=int, default=2,
                    help="Attempts per trial per session after API errors (default 2)")
    ap.add_argument("--allow-unknown-metadata", action="store_true",
                    help="Run images whose site/week/condition are UNKNOWN (they enter the prompt as-is)")
    ap.add_argument("--image-root", type=Path, default=ROOT, help="Base for manifest image_path values")
    ap.add_argument("--mock", action="store_true", help="Use the offline mock adapter (no network, no cost)")
    ap.add_argument("--mock-interrupt-after", type=int,
                    help="Mock only: simulate Ctrl+C during request N+1")
    args = ap.parse_args(argv)

    if args.runs < 1:
        ap.error("--runs must be at least 1")
    if args.max_attempts < 1:
        ap.error("--max-attempts must be at least 1")
    if args.mock_interrupt_after is not None and not args.mock:
        ap.error("--mock-interrupt-after requires --mock")

    image_ids = [s.strip() for s in args.image_ids.split(",") if s.strip()] if args.image_ids else None
    rows = load_manifest(args.manifest, args.image_root, image_ids)

    kwargs = {"interrupt_after": args.mock_interrupt_after} if args.mock else {}
    adapter = get_adapter(args.model, mock=args.mock, **kwargs)
    experiment_id = f"MOCK_{args.experiment_id}" if args.mock else args.experiment_id

    report = run_benchmark(rows, adapter, args.output_dir, experiment_id, runs=args.runs,
                           max_usd=args.max_usd, resume=args.resume, dry_run=args.dry_run,
                           max_attempts=args.max_attempts,
                           allow_unknown_metadata=args.allow_unknown_metadata,
                           manifest_path=args.manifest)
    if report.get("stopped") == "interrupted by user":
        sys.exit(130)


if __name__ == "__main__":
    main()
