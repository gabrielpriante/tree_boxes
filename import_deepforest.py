"""
Import a lowalt_pilot_v1 DeepForest run (urban-tree-vision-baselines) into benchmark_results/.

The PRIMARY run becomes trial 1 of a new experiment folder laid out like the LLM experiments, so
evaluate_consistency.py can load it. The verification run is NOT imported as a trial: it is a
reproducibility check, and its result (determinism.json) is copied alongside the primary record.

Boxes stay in original-image pixels as floats with their confidence scores. They are never converted to
the 0-1000 integer grid the LLM prompts use.

    python import_deepforest.py --image-dir ../urban-tree-vision-baselines/outputs/lowalt_pilot_v1/m01_native/\
lowalt_pilot_ptr1_27m_m01_v1/M01_9c192941f5/LA_PT-R1_27m

Refuses to write into an existing experiment folder. No API requests.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "benchmark_results"
MANIFEST = ROOT / "manifests" / "manifest.csv"
SCHEMA = "deepforest_import_v1"
MODEL_LABELS = {"M01": "DeepForest M01", "M02v2": "DeepForest M02v2"}
PRIMARY, VERIFICATION = "run_01_primary", "run_02_verification"

COUNTS_LONG_FIELDS = ["experiment_id", "image_id", "site", "week", "condition", "altitude_m", "model", "trial",
                      "tree_count", "input_tokens", "output_tokens", "estimated_cost_usd", "status", "model_code",
                      "model_config_id", "collection", "attempt", "attempts_made", "timestamp_utc", "record_path"]
BOXES_FIELDS = ["experiment_id", "model_config_id", "model", "image_id", "trial", "tree_id",
                "x_min_px", "y_min_px", "x_max_px", "y_max_px", "score"]
SUMMARY_FIELDS = ["image_id", "model", "successful_trials", "mean_tree_count", "standard_deviation", "minimum_count",
                  "maximum_count", "coefficient_of_variation", "experiment_id", "model_code", "model_config_id", "site",
                  "week", "condition", "altitude_m", "collection", "trials_recorded", "invalid_trials",
                  "refusal_trials", "max_tokens_trials", "unresolved_trials", "total_cost_usd"]
ATTEMPT_FIELDS = ["experiment_id", "model_config_id", "image_id", "trial", "attempt", "status", "stop_reason",
                  "tree_count", "raw_box_count", "input_tokens", "output_tokens", "estimated_cost_usd", "cost_basis",
                  "possibly_billed", "api_retries", "error", "started_utc", "finished_utc", "record_path"]
INVALID_FIELDS = ["experiment_id", "model_config_id", "image_id", "trial", "attempt", "status", "errors", "record_path"]


class DeepForestImportError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_csv(path: Path, fields: list, rows: list) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def manifest_row(manifest: Path, image_id: str) -> dict:
    with open(manifest, newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["image_id"] == image_id]
    if len(rows) != 1:
        raise DeepForestImportError(f"{image_id} not found exactly once in {manifest}")
    return rows[0]


def llm_image_hashes(results_dir: Path, image_id: str) -> dict:
    """SHA-256 of the source image as recorded by every existing (LLM) experiment on this image."""
    found = {}
    for rec in sorted(results_dir.glob(f"*/*/{image_id}/trial_*/attempt_*/record.json")):
        r = read_json(rec)
        if r.get("record_schema_version") == SCHEMA:
            continue
        found.setdefault(r["experiment_id"], set()).add(r["image"]["file_sha256"])
    return {k: sorted(v) for k, v in found.items()}


def fmt(v: float) -> str:
    return repr(float(v))


def import_run(image_dir: Path, results_dir: Path = RESULTS_DIR, manifest: Path = MANIFEST,
               image_root: Path = ROOT, baselines_root: Path | None = None) -> Path:
    image_dir = Path(image_dir).resolve()
    primary = image_dir / PRIMARY
    run_rec = read_json(primary / "run_record.json")
    if run_rec.get("status") != "success":
        raise DeepForestImportError(f"{primary}: run status is {run_rec.get('status')!r}")
    if run_rec.get("run") != "primary":
        raise DeepForestImportError(f"{primary}: not a primary run")
    det = read_json(primary / "detections.json")
    if det.get("coordinate_system") != "original_pixels":
        raise DeepForestImportError(f"{primary}/detections.json: expected original-pixel coordinates")
    determinism_path = image_dir / "determinism.json"
    determinism = read_json(determinism_path) if determinism_path.exists() else None
    if determinism is None:
        raise DeepForestImportError(f"{image_dir}: no determinism.json; run the verification and comparison first")
    if determinism["predictions_hash"][0] != run_rec["predictions_hash"]:
        raise DeepForestImportError(f"{determinism_path}: does not describe this primary run")
    cfg_path = image_dir.parent / "model_config.json"   # written by the runner in the <config_id>/ folder
    model_config = read_json(cfg_path)

    experiment_id, image_id = run_rec["experiment_id"], run_rec["image_id"]
    cfg_id, model_code = run_rec["model_config_id"], run_rec["model_code"]
    if det["image_id"] != image_id or det["model_config_id"] != cfg_id:
        raise DeepForestImportError("detections.json does not match run_record.json")

    # Source image: the manifest, the file on disk here and the hash recorded at inference time must agree,
    # and must equal the hash every LLM experiment on this image recorded.
    row = manifest_row(manifest, image_id)
    image_path = image_root / row["image_path"]
    local_sha = sha256_file(image_path)
    if not (local_sha == row["sha256"] == run_rec["source_sha256"]):
        raise DeepForestImportError(f"{image_id}: source hash mismatch (manifest {row['sha256'][:12]}, local {local_sha[:12]}, "
                           f"DeepForest run {run_rec['source_sha256'][:12]})")
    llm_hashes = llm_image_hashes(results_dir, image_id)
    bad = {k: v for k, v in llm_hashes.items() if v != [local_sha]}
    if bad:
        raise DeepForestImportError(f"{image_id}: LLM experiments recorded different source hashes: {bad}")
    W, H = run_rec["original_size"]
    if [int(row["width"]), int(row["height"])] != [W, H]:
        raise DeepForestImportError(f"{image_id}: size differs from the manifest")

    exp_dir = results_dir / experiment_id
    if exp_dir.exists():
        raise DeepForestImportError(f"{exp_dir} already exists; refusing to overwrite")
    trial_dir = exp_dir / cfg_id / image_id / "trial_01" / "attempt_01"
    trial_dir.mkdir(parents=True)

    copies = {"detections": "detections.json", "raw_predictions": "predictions_raw.csv",
              "baselines_run_record": "run_record.json", "run_log": "run.log", "annotated": "annotated_original.jpg"}
    if (primary / "annotated_processed.jpg").exists():
        copies["annotated_processed"] = "annotated_processed.jpg"
    files = {}
    for key, name in copies.items():
        shutil.copy2(primary / name, trial_dir / name)
        files[key] = name
    shutil.copy2(determinism_path, trial_dir / "determinism.json")
    files["determinism"] = "determinism.json"
    pre = run_rec.get("preprocessing")
    if pre:
        src = Path(pre["output_path"])
        local = primary / "preprocessed" / src.name
        shutil.copy2(local, trial_dir / src.name)
        if sha256_file(trial_dir / src.name) != pre["output_sha256"]:
            raise DeepForestImportError("preprocessed image hash differs from its record")
        files["preprocessed_image"] = src.name
    shutil.copy2(cfg_path, exp_dir / cfg_id / "model_config.json")

    trees = det["trees"]
    model = MODEL_LABELS[model_code]
    rel_record = (trial_dir / "record.json").relative_to(exp_dir).as_posix()
    base_rel = None
    if baselines_root:
        try:
            base_rel = image_dir.relative_to(Path(baselines_root).resolve()).as_posix()
        except ValueError:
            base_rel = None
    record = {
        "record_schema_version": SCHEMA,
        "experiment_id": experiment_id, "model_config_id": cfg_id, "model_code": model_code, "model": model,
        "adapter": "deepforest-local", "image_id": image_id, "image_path": row["image_path"],
        "site": row["site"], "week": row["week"], "condition": row["condition"], "altitude_m": row["altitude_m"],
        "coverage_scope": row["coverage_scope"], "collection": row["collection"],
        "manifest_sha256": sha256_file(manifest), "trial": 1, "attempt": 1,
        "trial_note": "single deterministic DeepForest output (primary run); the verification run is a "
                      "reproducibility check recorded in determinism.json, not a second trial",
        "coordinate_system": "original_pixels",
        "image": {"file_sha256": local_sha, "original_width": W, "original_height": H,
                  "upload_width": run_rec["processed_size"][0], "upload_height": run_rec["processed_size"][1],
                  "model_input_size": run_rec["model_input_size"],
                  "note": "upload_* = image size given to DeepForest after preprocessing (no upload took place)"},
        "preprocessing": pre, "padding": run_rec.get("padding"),
        "inference_settings": model_config["inference"], "checkpoint": run_rec["checkpoint"],
        "loaded_weights": run_rec.get("loaded_weights"), "environment": run_rec.get("environment"),
        "status": "success", "stop_reason": "completed", "tree_count": len(trees), "raw_box_count": run_rec["raw_detections"],
        "scores_preserved": True, "estimated_cost_usd": 0.0, "api_requests_made": 0,
        "determinism": {k: determinism[k] for k in ("bitwise_identical", "verdict", "counts", "predictions_hash")},
        "llm_experiments_same_image_hash": sorted(llm_hashes),
        "source": {"repository": "urban-tree-vision-baselines", "image_dir": base_rel or str(image_dir),
                   "run_record_sha256": sha256_file(primary / "run_record.json"),
                   "detections_sha256": sha256_file(primary / "detections.json")},
        "files": files,
        "started_utc": run_rec["started_utc"], "finished_utc": run_rec["finished_utc"],
        "inference_seconds": run_rec.get("inference_seconds"),
        "imported_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (trial_dir / "record.json").write_text(json.dumps(record, indent=2), encoding="utf-8")

    meta = {"site": row["site"], "week": row["week"], "condition": row["condition"], "altitude_m": row["altitude_m"],
            "collection": row["collection"]}
    write_csv(exp_dir / "counts_long.csv", COUNTS_LONG_FIELDS, [{
        "experiment_id": experiment_id, "image_id": image_id, **meta, "model": model, "trial": 1,
        "tree_count": len(trees), "input_tokens": "", "output_tokens": "", "estimated_cost_usd": 0.0,
        "status": "success", "model_code": model_code, "model_config_id": cfg_id, "attempt": 1, "attempts_made": 1,
        "timestamp_utc": run_rec["finished_utc"], "record_path": rel_record}])
    write_csv(exp_dir / "boxes_long.csv", BOXES_FIELDS, [{
        "experiment_id": experiment_id, "model_config_id": cfg_id, "model": model, "image_id": image_id, "trial": 1,
        "tree_id": t["tree_id"], "x_min_px": fmt(t["bbox_px"][0]), "y_min_px": fmt(t["bbox_px"][1]),
        "x_max_px": fmt(t["bbox_px"][2]), "y_max_px": fmt(t["bbox_px"][3]), "score": fmt(t["score"])} for t in trees])
    n = len(trees)
    write_csv(exp_dir / "counts_summary.csv", SUMMARY_FIELDS, [{
        "image_id": image_id, "model": model, "successful_trials": 1, "mean_tree_count": n, "standard_deviation": "",
        "minimum_count": n, "maximum_count": n, "coefficient_of_variation": "", "experiment_id": experiment_id,
        "model_code": model_code, "model_config_id": cfg_id, **meta, "trials_recorded": 1, "invalid_trials": 0,
        "refusal_trials": 0, "max_tokens_trials": 0, "unresolved_trials": 0, "total_cost_usd": 0.0}])
    write_csv(exp_dir / "attempts_log.csv", ATTEMPT_FIELDS, [{
        "experiment_id": experiment_id, "model_config_id": cfg_id, "image_id": image_id, "trial": 1, "attempt": 1,
        "status": "success", "stop_reason": "completed", "tree_count": n, "raw_box_count": run_rec["raw_detections"],
        "input_tokens": "", "output_tokens": "", "estimated_cost_usd": 0.0, "cost_basis": "local inference",
        "possibly_billed": False, "api_retries": 0, "error": "", "started_utc": run_rec["started_utc"],
        "finished_utc": run_rec["finished_utc"], "record_path": rel_record}])
    write_csv(exp_dir / "invalid_trials.csv", INVALID_FIELDS, [])
    (exp_dir / "experiment.json").write_text(json.dumps({
        "experiment_id": experiment_id, "created_utc": record["imported_utc"],
        "trial_convention": "one deterministic DeepForest output imported as trial 1 (primary run); "
                            "not comparable to independent LLM trials",
        "record_schema_version": SCHEMA, "source": record["source"]}, indent=2), encoding="utf-8")
    return exp_dir


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image-dir", type=Path, nargs="+", required=True,
                    help="Baselines image directories holding run_01_primary, run_02_verification and determinism.json")
    ap.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    ap.add_argument("--manifest", type=Path, default=MANIFEST)
    ap.add_argument("--baselines-root", type=Path, default=ROOT.parent / "urban-tree-vision-baselines")
    args = ap.parse_args(argv)
    for d in args.image_dir:
        try:
            out = import_run(d, args.results_dir, args.manifest, ROOT, args.baselines_root)
        except DeepForestImportError as e:
            sys.exit(f"Import refused: {e}")
        print(f"Imported {d} -> {out}")


if __name__ == "__main__":
    main()
