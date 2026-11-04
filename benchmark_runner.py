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

    # Alternate between the selected images by trial (A1, B1, A2, B2, ...)
    ... --round-robin

    # Offline rehearsal with the mock adapter (no network, no cost)
    ... --execute --mock
"""

import argparse
import csv
import hashlib
import json
import os
import shutil
import re
import statistics
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import box_generation as bg
from model_adapters import FatalAdapterError, ModelAdapter, get_adapter, redact_secrets

ROOT = Path(__file__).resolve().parent
PRODUCTION_OUTPUT_DIR = ROOT / "benchmark_results"
MOCK_OUTPUT_DIR = ROOT / "benchmark_results_mock"
LEDGER_DIR = ROOT / "spend_ledger"  # project-wide, independent of --output-dir and --experiment-id
RECORD_SCHEMA_VERSION = 2
UNKNOWN = "UNKNOWN"
NOT_APPLICABLE = "NOT_APPLICABLE"

MANIFEST_REQUIRED = ["image_id", "image_path", "site", "week", "condition", "altitude_m", "coverage_scope"]
PROMPT_FIELDS = ("site", "week", "condition")  # these values are inserted into the prompt
MANIFEST_CARRIED = MANIFEST_REQUIRED + ["collection"]

# Controlled vocabulary. UNKNOWN = not yet established; NOT_APPLICABLE = does not apply.
ALLOWED_SITES = {"PT-02", "PT-R1", UNKNOWN}
WEEK_RE = re.compile(r"^WEEK \d+$")
ALLOWED_WEEK_LABELS = {NOT_APPLICABLE, UNKNOWN}
# BEFORE/AFTER: intervention sessions. ALTITUDE_TEST: altitude-series flights outside the
# intervention sessions. MORNING_START/AFTERNOON_END: week 1 folder labels, mapping unconfirmed.
ALLOWED_CONDITIONS = {"BEFORE", "AFTER", "ALTITUDE_TEST", "MORNING_START", "AFTERNOON_END", UNKNOWN}
ALLOWED_COVERAGE = {"FULL_SITE", "PARTIAL_SITE", UNKNOWN}

# A trial is complete once one attempt reaches a terminal status. Terminal statuses are
# model outcomes and are never re-run automatically, so results cannot be cherry-picked.
TERMINAL = {"success", "invalid", "refusal", "max_tokens"}
NONSUCCESS_TERMINAL = TERMINAL - {"success"}
# Retryable statuses: no usable model outcome (API/transport failure or interruption).
RETRYABLE = {"error", "interrupted"}

MAX_CONSECUTIVE_FAILED_TRIALS = 3
DEFAULT_PAUSE_AFTER = 3  # consecutive invalid/refusal/max_tokens trials before pausing
PAUSE_FILE = "PAUSED.json"

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


class PauseStop(Exception):
    """Too many consecutive non-success outcomes; a person should look before spending more."""


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

def metadata_errors(row: dict) -> list:
    """Values outside the controlled vocabulary (typos, free text, blanks)."""
    errors = []
    if row["site"] not in ALLOWED_SITES:
        errors.append(f"site {row['site']!r} not in {sorted(ALLOWED_SITES)}")
    if not (WEEK_RE.match(row["week"]) or row["week"] in ALLOWED_WEEK_LABELS):
        errors.append(f"week {row['week']!r} must be 'WEEK <n>', {NOT_APPLICABLE} or {UNKNOWN}")
    if row["condition"] not in ALLOWED_CONDITIONS:
        errors.append(f"condition {row['condition']!r} not in {sorted(ALLOWED_CONDITIONS)}")
    if row["coverage_scope"] not in ALLOWED_COVERAGE:
        errors.append(f"coverage_scope {row['coverage_scope']!r} not in {sorted(ALLOWED_COVERAGE)}")
    if row["altitude_m"] != UNKNOWN:
        try:
            float(row["altitude_m"])
        except ValueError:
            errors.append(f"altitude_m {row['altitude_m']!r} is not a number or {UNKNOWN}")
    return errors


def review_fields(row: dict) -> set:
    return {f.strip() for f in row.get("needs_review", "").split(";") if f.strip()}


def prompt_metadata_blockers(row: dict) -> list:
    """Reasons the image cannot run yet. Prompt fields must be known (NOT_APPLICABLE is a
    legitimate value) and must not be awaiting confirmation in needs_review."""
    reasons = []
    unknown = [f for f in PROMPT_FIELDS if is_unknown(row[f])]
    if unknown:
        reasons.append(f"unknown: {', '.join(unknown)}")
    unconfirmed = [f for f in PROMPT_FIELDS if f in review_fields(row) and f not in unknown]
    if unconfirmed:
        reasons.append(f"awaiting confirmation (needs_review): {', '.join(unconfirmed)}")
    return reasons


def load_manifest(path: Path, image_root: Path = ROOT, image_ids=None, collection=None) -> list:
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
    bad = [f"{r['image_id']}: {e}" for r in rows for e in metadata_errors(r)]
    if bad:
        raise ValueError("Manifest values outside the allowed vocabulary:\n  " + "\n  ".join(bad))

    if image_ids:
        unknown = [i for i in image_ids if i not in ids]
        if unknown:
            raise ValueError(f"image ids not in manifest: {unknown}")
        wanted = set(image_ids)
        rows = [r for r in rows if r["image_id"] in wanted]  # manifest order, not CLI order
    if collection:
        rows = [r for r in rows if r.get("collection") == collection]
        if not rows:
            raise ValueError(f"No manifest rows with collection {collection!r}")

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
        "_orphan": True,
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


def expected_image_tokens(row: dict, adapter=None) -> int | None:
    """Image input tokens of the upload, from the manifest dimensions (no image decoding)."""
    try:
        w, h = int(row["width"]), int(row["height"])
    except (KeyError, ValueError):
        return None
    size = bg.api_resized_size(w, h)
    return adapter.image_tokens(*size) if adapter is not None else bg.count_image_tokens(*size)


def worst_case_usd(adapter, prompt: str, observed: int | None, image_tokens: int | None = None) -> float:
    input_tokens = observed if observed else adapter.estimate_input_tokens(prompt, image_tokens)
    return adapter.cost_usd({"input_tokens": input_tokens, "output_tokens": adapter.max_output_tokens()})


# ---------------------------------------------------------------- project spending ledger

def attempt_key(a: dict) -> str:
    """Stable identity of one attempt, so it is never counted twice."""
    if a.get("attempt_uid"):
        return a["attempt_uid"]
    return "legacy:{}/{}/{}/trial_{:02d}/attempt_{:02d}".format(
        a.get("experiment_id"), a.get("model_config_id"), a.get("image_id"),
        int(a.get("trial") or 0), int(a.get("attempt") or 0))


class SpendLedger:
    """Append-only JSONL ledger of every paid attempt for one billing account, shared by all
    experiments, experiment ids and output directories in this project.

    Each attempt writes a 'start' entry (with its worst-case cost) before the request and a
    'final' entry afterwards. Entries are keyed by attempt_uid, so re-reading, reconciling or
    a repeated write never counts an attempt twice. An attempt with no 'final' entry (the
    process died mid-request) is counted at its worst case.
    """

    def __init__(self, path: Path):
        self.path = Path(path)

    def _append(self, entry: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, sort_keys=True) + "\n")
            f.flush()
            os.fsync(f.fileno())

    def entries(self) -> list:
        if not self.path.exists():
            return []
        lines = self.path.read_text(encoding="utf-8").splitlines()
        out = []
        for i, line in enumerate(lines):
            if not line.strip():
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                if i == len(lines) - 1:
                    continue  # partial final line from a crash mid-write
                raise ValueError(f"Corrupt ledger line {i + 1} in {self.path}")
        return out

    def by_attempt(self) -> dict:
        starts, finals = {}, {}
        for e in self.entries():
            target = starts if e["event"] == "start" else finals
            target.setdefault(e["attempt_uid"], e)  # first entry wins; duplicates ignored
        return {"starts": starts, "finals": finals}

    def start(self, uid: str, meta: dict, worst_case: float, backfilled: bool = False) -> None:
        self._append({"event": "start", "attempt_uid": uid, "timestamp_utc": now_utc(),
                      "worst_case_usd": round(worst_case, 6), "backfilled": backfilled, **meta})

    def finish(self, uid: str, cost: float, cost_basis: str, status: str, backfilled: bool = False) -> None:
        self._append({"event": "final", "attempt_uid": uid, "timestamp_utc": now_utc(),
                      "cost_usd": round(cost, 6), "cost_basis": cost_basis, "status": status,
                      "backfilled": backfilled})

    def summary(self) -> dict:
        idx = self.by_attempt()
        confirmed = estimated = unresolved = 0.0
        for uid in set(idx["starts"]) | set(idx["finals"]):
            final = idx["finals"].get(uid)
            if final is None:
                unresolved += idx["starts"][uid]["worst_case_usd"]
            elif final["cost_basis"] == "actual_usage":
                confirmed += final["cost_usd"]
            else:
                estimated += final["cost_usd"]
        return {"confirmed_usd": confirmed, "estimated_usd": estimated, "unresolved_usd": unresolved,
                "total_usd": confirmed + estimated + unresolved,
                "attempts": len(set(idx["starts"]) | set(idx["finals"]))}

    def reconcile(self, attempts: list) -> int:
        """Add entries for on-disk attempts the ledger is missing (a crash between record.json
        and the ledger write, or results written before the ledger existed)."""
        idx = self.by_attempt()
        added = 0
        for a in attempts:
            uid = attempt_key(a)
            if uid in idx["finals"]:
                continue
            meta = {k: a.get(k) for k in ("experiment_id", "model_config_id", "image_id", "trial", "attempt")}
            if uid not in idx["starts"]:
                self.start(uid, meta, float(a.get("pre_request_worst_case_usd") or 0.0), backfilled=True)
                added += 1
            if not a.get("_orphan"):
                self.finish(uid, float(a.get("estimated_cost_usd") or 0.0), a.get("cost_basis", ""),
                            a.get("status", ""), backfilled=True)
                added += 1
        return added


def ledger_report(summary: dict, max_usd: float | None) -> list:
    lines = [f"Project spending ledger ({summary['attempts']} recorded attempts):",
             f"  confirmed from API usage:          ${summary['confirmed_usd']:.4f}",
             f"  estimated (interrupted/failed):    ${summary['estimated_usd']:.4f}",
             f"  unresolved (no final entry):       ${summary['unresolved_usd']:.4f}",
             f"  total counted against the budget:  ${summary['total_usd']:.4f}"]
    if max_usd is not None:
        lines.append(f"  budget ${max_usd:.2f}; remaining ${max_usd - summary['total_usd']:.4f}")
    lines.append("  Spending outside this project (Console, other scripts, other machines) is not visible here.")
    return lines


def affordable_trials(remaining: float, worst: float, per_trial: float) -> int:
    """Trials the pre-request guard admits if each actually costs per_trial."""
    if remaining < worst or per_trial <= 0:
        return 0
    return int((remaining - worst) // per_trial) + 1


# ---------------------------------------------------------------- pause safeguard

def pause_state(run_dir: Path) -> tuple:
    """(active pause dict or None, unix-ns time of the most recent acknowledgment)."""
    active = read_json(run_dir / PAUSE_FILE) if (run_dir / PAUSE_FILE).exists() else None
    acks = [read_json(p).get("acknowledged_unix_ns", 0) for p in run_dir.glob("PAUSED_acknowledged_*.json")]
    return active, max(acks, default=0)


def acknowledge_pause(run_dir: Path) -> dict | None:
    active, _ = pause_state(run_dir)
    if not active:
        return None
    active["acknowledged_utc"] = now_utc()
    active["acknowledged_unix_ns"] = time.time_ns()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    write_new_json(run_dir / f"PAUSED_acknowledged_{stamp}.json", active)
    (run_dir / PAUSE_FILE).unlink()
    return active


def write_pause(run_dir: Path, streak: list, threshold: int, exp_dir: Path) -> None:
    write_json(run_dir / PAUSE_FILE, {
        "created_utc": now_utc(),
        "reason": f"{len(streak)} consecutive trials ended invalid, refused or truncated "
                  f"(threshold {threshold})",
        "threshold": threshold,
        "trials": [{"image_id": r.get("image_id"), "trial": r.get("trial"), "attempt": r.get("attempt"),
                    "status": r.get("status"), "stop_reason": r.get("stop_reason"),
                    "errors": (r.get("validation") or {}).get("errors", [])[:5],
                    "record": Path(r["_path"]).relative_to(exp_dir).as_posix() if r.get("_path")
                    else None} for r in streak],
        "note": "These trials are complete, recorded experimental outcomes and will not be re-run. "
                "Continue with --resume --acknowledge-pause after reviewing them.",
    })


def nonsuccess_streak(run_dir: Path, since_ns: int) -> list:
    """Trailing run of consecutive non-success terminal trials, in execution order,
    counting only trials finished after the last acknowledged pause."""
    done = [a for a in all_attempts(run_dir)
            if a.get("status") in TERMINAL and (a.get("finished_unix_ns") or 0) > since_ns]
    done.sort(key=lambda a: a.get("finished_unix_ns") or 0)
    streak = []
    for a in done:
        streak = streak + [a] if a["status"] in NONSUCCESS_TERMINAL else []
    return streak


# ---------------------------------------------------------------- one attempt

def run_attempt(adapter, cfg, cfg_id, experiment_id, row, trial, image, run_dir, worst_case, manifest_sha,
                ledger: SpendLedger):
    trial_dir = run_dir / row["image_id"] / f"trial_{trial:02d}"
    attempt = len(attempt_dirs(trial_dir)) + 1 if trial_dir.exists() else 1
    adir = trial_dir / f"attempt_{attempt:02d}"
    adir.mkdir(parents=True, exist_ok=False)
    uid = uuid.uuid4().hex

    base = {
        "record_schema_version": RECORD_SCHEMA_VERSION,
        "attempt_uid": uid,
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
    ledger.start(uid, {"experiment_id": experiment_id, "model_config_id": cfg_id,
                       "image_id": row["image_id"], "trial": trial, "attempt": attempt,
                       "record_dir": adir.as_posix()}, worst_case)

    record = {**base, "status": None, "stop_reason": None, "tree_count": None, "raw_box_count": None,
              "usage": {}, "estimated_cost_usd": 0.0, "cost_basis": "not_billed",
              "possibly_billed": False, "request_id": None, "api_retries": [],
              "validation": {"errors": [], "warnings": []}, "error": None, "files": {}}
    t0 = time.monotonic()

    def finish(status):
        record["status"] = status
        record["error"] = redact_secrets(record["error"])
        record["finished_utc"] = now_utc()
        record["finished_unix_ns"] = time.time_ns()
        record["duration_s"] = round(time.monotonic() - t0, 2)
        record["estimated_cost_usd"] = round(record["estimated_cost_usd"], 6)
        write_new_json(adir / "record.json", record)
        ledger.finish(uid, record["estimated_cost_usd"], record["cost_basis"], status)
        record["_path"] = adir / "record.json"  # in-memory only
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
        record["error"] = redact_secrets(str(e))
        finish("error")
        raise

    record["api_retries"] = result.retries
    record["request_id"] = result.request_id
    record["usage"] = result.usage
    record["stop_reason"] = result.stop_reason
    if result.stop_detail:
        record["stop_detail"] = result.stop_detail
    if result.provider_response:
        record["provider_response"] = result.provider_response

    if result.status != "ok":
        record["error"] = result.error
        if result.raw is not None:
            write_new_json(adir / "response_raw.json", result.raw)
            record["files"]["response_raw"] = "response_raw.json"
        if result.usage:
            # The provider reported usage for the failed response: charge that, not a bound.
            record.update(possibly_billed=True, estimated_cost_usd=adapter.cost_usd(result.usage),
                          cost_basis="actual_usage (error response)")
        elif result.possibly_billed:
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

def prepare_image(row: dict, adapter=None) -> dict:
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
        # Provider-specific facts (for M08: the API's effective size and visual tokens).
        **(adapter.image_meta(upload) if adapter is not None else ModelAdapter.image_meta(None, upload)),
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

def execution_order(plan, round_robin: bool = False) -> list:
    """(row, trial) pairs for the ready, outstanding trials. Sequential (default): every trial of one
    image before the next image. Round-robin: by trial number, cycling through the images in manifest
    order (A1, B1, A2, B2, ...). Completed trials are already excluded from each image's todo list, so
    a resumed run continues the same order where it stopped."""
    pairs = [(i, row, trial) for i, (row, todo, ready) in enumerate(plan) if ready for trial in todo]
    if round_robin:
        pairs.sort(key=lambda p: (p[2], p[0]))
    return [(row, trial) for _, row, trial in pairs]


def run_benchmark(rows, adapter, output_dir: Path, experiment_id: str, runs: int = 10,
                  max_usd: float | None = None, resume: bool = False, dry_run: bool = True,
                  max_attempts: int = 2, allow_unknown_metadata: bool = False,
                  manifest_path: Path | None = None, ledger_path: Path | None = None,
                  pause_after: int = DEFAULT_PAUSE_AFTER, acknowledge_paused: bool = False,
                  round_robin: bool = False, log=print) -> dict:
    cfg = adapter.config()
    cfg_id = config_id(cfg)
    exp_dir = output_dir / experiment_id
    run_dir = exp_dir / cfg_id
    ledger = SpendLedger(ledger_path or output_dir / f"{adapter.billing_account}_spend_ledger.jsonl")
    report = {"experiment_id": experiment_id, "model_config_id": cfg_id, "run_dir": str(run_dir),
              "trials_run": 0, "stopped": None, "skipped_images": [], "ledger": str(ledger.path)}

    if exp_dir.exists():
        others = sorted(p.name for p in exp_dir.iterdir() if (p / "model_config.json").exists()
                        and read_json(p / "model_config.json").get("model_code") != cfg["model_code"])
        if others:
            raise SystemExit(f"{exp_dir} already holds results from another model ({', '.join(others)}). "
                             "Use a separate --experiment-id for each model.")

    blockers = {r["image_id"]: ([] if allow_unknown_metadata else prompt_metadata_blockers(r)) for r in rows}
    report["skipped_images"] = [(i, "; ".join(b)) for i, b in blockers.items() if b]

    states = {r["image_id"]: {t: trial_state(run_dir / r["image_id"] / f"trial_{t:02d}")
                              for t in range(1, runs + 1)} for r in rows}
    existing = [i for i, s in states.items() if any(st != "pending" for st, _ in s.values())]
    if existing and not resume:
        raise SystemExit(f"Trials already exist in {run_dir} for {existing}. "
                         "Use --resume to continue without repeating completed trials.")

    log(f"Experiment:   {experiment_id}")
    log(f"Model config: {cfg_id}  ({adapter.describe(cfg)})")
    log(f"Results dir:  {run_dir}")
    log(f"Ledger:       {ledger.path}")
    log("Order:        " + ("round-robin across images (A1, B1, A2, B2, ...)" if round_robin
                             else "sequential (all trials of one image, then the next)"))

    plan, total_worst, max_worst, max_input = [], 0.0, 0.0, 0
    log(f"\n{'image_id':<28}{'done':>6}{'to run':>8}{'worst-case/trial':>18}  status")
    for row in rows:
        image_id = row["image_id"]
        prompt = bg.build_prompt(row["site"], row["week"], row["condition"])
        observed = observed_input_tokens(run_dir, image_id)
        input_est = observed or adapter.estimate_input_tokens(prompt, expected_image_tokens(row, adapter))
        worst = worst_case_usd(adapter, prompt, observed, expected_image_tokens(row, adapter))
        todo = [t for t, (st, _) in states[image_id].items() if st != "done"]
        ready = not blockers[image_id]
        plan.append((row, todo, ready))
        total_worst += worst * len(todo)
        max_worst, max_input = max(max_worst, worst), max(max_input, input_est)
        status = "ready" if ready else "BLOCKED: " + "; ".join(blockers[image_id])
        log(f"{image_id:<28}{runs - len(todo):>6}{len(todo):>8}{'$' + format(worst, '.4f'):>18}  {status}")

    n_todo = sum(len(t) for _, t, ready in plan if ready)
    n_blocked = sum(len(t) for _, t, ready in plan if not ready)
    report["planned_trials"] = n_todo
    report["order"] = [(r["image_id"], t) for r, t in execution_order(plan, round_robin)]
    report["blocked_trials"] = n_blocked
    log(f"\nTrials to run now: {n_todo}. Blocked until metadata is confirmed: {n_blocked}.")
    if n_blocked:
        log("  Blocked images need their site/week/condition known and removed from needs_review in the "
            "manifest (or --allow-unknown-metadata, which sends them to the prompt as they stand).")
    log(f"Worst case if every outstanding trial used all {adapter.max_output_tokens()} output tokens: "
        f"${total_worst:.2f}")

    unreconciled = [a for a in all_attempts(run_dir) if attempt_key(a) not in ledger.by_attempt()["finals"]]
    ledger_now = ledger.summary()
    pending_disk = sum(float(a.get("estimated_cost_usd") or a.get("pre_request_worst_case_usd") or 0)
                       for a in unreconciled)
    for line in ledger_report(ledger_now, max_usd):
        log(line)
    if pending_disk:
        log(f"  plus up to ${pending_disk:.4f} from on-disk attempts not yet in the ledger "
            "(added on the next --execute)")
    log(f"  This model config's own records: ${spent_usd(run_dir):.4f}")

    if max_usd is not None and n_todo + n_blocked:
        remaining = max_usd - ledger_now["total_usd"] - pending_disk
        total = n_todo + n_blocked
        log(f"\nProjection for {total} outstanding trials with ${remaining:.2f} remaining. A trial starts only "
            f"if spent + ${max_worst:.4f} (worst case) <= ${max_usd:.2f}. Actual output tokens per trial are "
            "unknown until the first trial; these scenarios are illustrative:")
        for out_tokens in (5_000, 10_000, 20_000, adapter.max_output_tokens()):
            per_trial = adapter.cost_usd({"input_tokens": max_input, "output_tokens": out_tokens})
            n = min(affordable_trials(remaining, max_worst, per_trial), total)
            log(f"  {out_tokens:>6} output tokens/trial (${per_trial:.4f} each): about {n} of {total} trials")

    active_pause, ack_ns = pause_state(run_dir)
    if active_pause:
        log(f"\nPAUSED since {active_pause['created_utc']}: {active_pause['reason']}")
        log("  Review the listed records, then continue with --resume --acknowledge-pause.")

    if rows:
        check_path_lengths(run_dir, [r["image_id"] for r in rows])

    if dry_run:
        log("\nDry run: no API requests made, nothing written.")
        return report
    if max_usd is None:
        raise SystemExit("--max-usd is required with --execute")
    if active_pause:
        if not acknowledge_paused:
            report["stopped"] = f"paused: {active_pause['reason']} (continue with --acknowledge-pause)"
            log(f"\nSTOPPED: {report['stopped']}")
            return report
        acknowledge_pause(run_dir)
        _, ack_ns = pause_state(run_dir)
        log("\nPause acknowledged; the consecutive-outcome count starts again from zero.")
    if n_todo == 0:
        log("\nNothing to run: no ready trials outstanding.")
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

    ledger.reconcile(all_attempts(run_dir))
    spent = ledger.summary()["total_usd"]
    streak = nonsuccess_streak(run_dir, ack_ns)
    consecutive_failed = 0
    images = {}
    try:
        for row, trial in execution_order(plan, round_robin):
            image_id = row["image_id"]
            if image_id not in images:
                image = images[image_id] = prepare_image(row, adapter)
                check_consistency(image, image_attempts(run_dir, image_id), image_id)
                image["image_tokens"] = adapter.image_tokens(image["meta"]["upload_width"],
                                                             image["meta"]["upload_height"])
                log(f"\n{image_id}: upload {image['meta']['upload_width']}x{image['meta']['upload_height']} "
                    f"(~{image['image_tokens']} image tokens; "
                    f"from {image['meta']['original_width']}x{image['meta']['original_height']})")
            image = images[image_id]
            for _ in range(max_attempts):
                worst = worst_case_usd(adapter, image["prompt"], observed_input_tokens(run_dir, image_id),
                                       image["image_tokens"])
                if spent + worst > max_usd:
                    raise BudgetStop(f"next request worst-case ${worst:.4f} would exceed the remaining "
                                     f"budget ${max_usd - spent:.4f} (limit ${max_usd:.2f})")
                rec = run_attempt(adapter, cfg, cfg_id, experiment_id, row, trial, image,
                                  run_dir, worst, manifest_sha, ledger)
                spent = ledger.summary()["total_usd"]
                report["trials_run"] += 1
                usage = rec["usage"] or {}
                label = f"{image_id} trial {trial:02d}" if round_robin else f"trial {trial:02d}"
                log(f"  {label} attempt {rec['attempt']:02d}: {rec['status']:<10} "
                    f"trees={rec['tree_count'] if rec['tree_count'] is not None else '-':<4} "
                    f"in={usage.get('input_tokens', 0)} out={usage.get('output_tokens', 0)} "
                    f"cost=${rec['estimated_cost_usd']:.4f} total=${spent:.4f}"
                    + (f"  [{rec['error']}]" if rec.get("error") else ""))
                if rec["status"] in TERMINAL:
                    break
            if rec["status"] in TERMINAL:
                consecutive_failed = 0
                streak = streak + [rec] if rec["status"] in NONSUCCESS_TERMINAL else []
                if pause_after and len(streak) >= pause_after:
                    write_pause(run_dir, streak, pause_after, exp_dir)
                    raise PauseStop(f"{len(streak)} consecutive trials ended invalid/refusal/max_tokens "
                                    f"({', '.join(r['status'] for r in streak)})")
            else:
                consecutive_failed += 1
                if consecutive_failed >= MAX_CONSECUTIVE_FAILED_TRIALS:
                    raise FatalAdapterError(f"{consecutive_failed} consecutive trials failed; "
                                            "stopping. Re-run with --resume later.")
    except BudgetStop as e:
        report["stopped"] = f"budget: {e}"
    except PauseStop as e:
        report["stopped"] = f"paused: {e}"
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
    for row, _, _ in plan:
        left = [t for t in range(1, runs + 1)
                if trial_state(run_dir / row["image_id"] / f"trial_{t:02d}")[0] != "done"]
        if left:
            remaining[row["image_id"]] = left
    report["incomplete"] = remaining

    log("")
    for line in ledger_report(ledger.summary(), max_usd):
        log(line)
    log(f"  This model config's own records: ${spent_usd(run_dir):.4f}")
    if report["stopped"]:
        log(f"STOPPED: {report['stopped']}")
        if report["stopped"].startswith("paused"):
            log(f"  All trials are kept as recorded. Details: {run_dir / PAUSE_FILE}")
            log("  Inspect the listed records. To continue deliberately: --resume --acknowledge-pause")
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
    ap.add_argument("--collection", help="Only manifest rows with this collection (e.g. low_altitude)")
    ap.add_argument("--runs", type=int, default=10, help="Trials per image (default 10)")
    ap.add_argument("--max-usd", type=float,
                    help="Spending limit across ALL experiments in this project's ledger for this API account")
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true", help="Show the plan and cost bounds; no API calls")
    mode.add_argument("--execute", action="store_true", help="Make the API requests (paid unless --mock)")
    ap.add_argument("--resume", action="store_true", help="Continue; completed trials are skipped")
    ap.add_argument("--output-dir", type=Path,
                    help=f"Default {PRODUCTION_OUTPUT_DIR.name}/ (or {MOCK_OUTPUT_DIR.name}/ with --mock)")
    ap.add_argument("--experiment-id", default="tree_count_consistency_v1")
    ap.add_argument("--model", default="M08", choices=("M08", "M09"),
                    help="M08 = Claude Opus 5.5 (Anthropic); M09 = GPT-6.1 Sol (OpenAI)")
    ap.add_argument("--max-attempts", type=int, default=2,
                    help="Attempts per trial per session after API errors (default 2)")
    ap.add_argument("--pause-after", type=int, default=DEFAULT_PAUSE_AFTER,
                    help="Pause after this many consecutive invalid/refusal/max_tokens trials (0 = never)")
    ap.add_argument("--acknowledge-pause", action="store_true",
                    help="Continue past a recorded pause after reviewing the flagged trials")
    ap.add_argument("--allow-unknown-metadata", action="store_true",
                    help="Run images whose site/week/condition are UNKNOWN or unconfirmed "
                         "(values enter the prompt as they stand)")
    ap.add_argument("--round-robin", action="store_true",
                    help="Cycle through the selected images by trial (A1, B1, A2, B2, ...) instead of "
                         "completing each image first. Same trials, outputs and budget checks")
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
    if args.pause_after < 0:
        ap.error("--pause-after must be 0 or more")

    output_dir = args.output_dir or (MOCK_OUTPUT_DIR if args.mock else PRODUCTION_OUTPUT_DIR)
    if args.mock:
        prod = PRODUCTION_OUTPUT_DIR.resolve()
        out = output_dir.resolve()
        if out == prod or prod in out.parents:
            ap.error(f"--mock results may not be written inside {PRODUCTION_OUTPUT_DIR}")

    image_ids = [s.strip() for s in args.image_ids.split(",") if s.strip()] if args.image_ids else None
    rows = load_manifest(args.manifest, args.image_root, image_ids, args.collection)

    kwargs = {"interrupt_after": args.mock_interrupt_after} if args.mock else {}
    adapter = get_adapter(args.model, mock=args.mock, **kwargs)
    experiment_id = f"MOCK_{args.experiment_id}" if args.mock else args.experiment_id
    # Real spend goes to one project-wide ledger per API account, whatever the output dir.
    ledger_path = (output_dir / "mock_spend_ledger.jsonl" if args.mock
                   else LEDGER_DIR / f"{adapter.billing_account}_ledger.jsonl")

    report = run_benchmark(rows, adapter, output_dir, experiment_id, runs=args.runs,
                           max_usd=args.max_usd, resume=args.resume, dry_run=args.dry_run,
                           max_attempts=args.max_attempts,
                           allow_unknown_metadata=args.allow_unknown_metadata,
                           manifest_path=args.manifest, ledger_path=ledger_path,
                           pause_after=args.pause_after, acknowledge_paused=args.acknowledge_pause,
                           round_robin=args.round_robin)
    if report.get("stopped") == "interrupted by user":
        sys.exit(130)


if __name__ == "__main__":
    main()
