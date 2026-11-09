"""
Determinism check for lowalt_pilot_v1 DeepForest runs.

Compares run_01_primary with run_02_verification in one image directory and writes determinism.json next
to them. Timestamps, durations and paths are ignored. The verification run is a reproducibility check,
not an independent observation.

    .venv/bin/python scripts/compare_deepforest_runs_v1.py outputs/lowalt_pilot_v1/m01_native/<exp>/<cfg>/<image_id>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

PRIMARY, VERIFICATION = "run_01_primary", "run_02_verification"


def load_run(run_dir: Path) -> dict:
    record = json.loads((run_dir / "run_record.json").read_text(encoding="utf-8"))
    dets = json.loads((run_dir / "detections.json").read_text(encoding="utf-8"))
    return {"record": record, "trees": dets["trees"]}


def compare(image_dir: Path, write: bool = True) -> dict:
    image_dir = Path(image_dir)
    out = image_dir / "determinism.json"
    if write and out.exists():
        raise FileExistsError(f"{out} already exists")
    a, b = load_run(image_dir / PRIMARY), load_run(image_dir / VERIFICATION)
    ra, rb = a["record"], b["record"]
    for r in (ra, rb):
        if r.get("status") != "success":
            raise ValueError(f"run {r.get('run')} did not succeed")
    same_inputs = {k: ra.get(k) == rb.get(k) for k in ("source_sha256", "model_config_id", "config", "checkpoint")}
    pre_a, pre_b = ra.get("preprocessing") or {}, rb.get("preprocessing") or {}
    same_inputs["preprocessed_sha256"] = pre_a.get("output_sha256") == pre_b.get("output_sha256")
    ta, tb = a["trees"], b["trees"]
    result = {"image_id": ra["image_id"], "experiment_id": ra["experiment_id"], "model_config_id": ra["model_config_id"],
              "inputs_identical": same_inputs,
              "counts": [len(ta), len(tb)], "raw_counts": [ra["raw_detections"], rb["raw_detections"]],
              "predictions_hash": [ra["predictions_hash"], rb["predictions_hash"]],
              "raw_predictions_hash": [ra["raw_predictions_hash"], rb["raw_predictions_hash"]],
              "inference_seconds": [ra.get("inference_seconds"), rb.get("inference_seconds")],
              "note": "verification run is a reproducibility check, not an independent observation"}
    identical = (ra["predictions_hash"] == rb["predictions_hash"]
                 and ra["raw_predictions_hash"] == rb["raw_predictions_hash"])
    result["bitwise_identical"] = identical
    if not identical:
        # Quantify: match records one-to-one by canonical order when counts agree, else by nearest box.
        result["difference"] = quantify(ta, tb)
    result["verdict"] = "identical" if identical else "differs (both runs preserved)"
    if write:
        out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def quantify(ta: list, tb: list) -> dict:
    A = np.array([t["bbox_processed_px"] + [t["score"]] for t in ta]) if ta else np.zeros((0, 5))
    B = np.array([t["bbox_processed_px"] + [t["score"]] for t in tb]) if tb else np.zeros((0, 5))
    diff = {"count_difference": len(tb) - len(ta)}
    if len(A) and len(B):
        from scipy.optimize import linear_sum_assignment
        from scipy.spatial.distance import cdist
        cost = cdist(A[:, :4], B[:, :4], metric="chebyshev")
        i, j = linear_sum_assignment(cost)
        d = cost[i, j]
        diff.update(matched_pairs=int(len(i)), max_coordinate_abs_diff_px=float(d.max()),
                    pairs_exactly_equal=int((d == 0).sum()), pairs_within_0_01px=int((d <= 0.01).sum()),
                    max_score_abs_diff=float(np.abs(A[i, 4] - B[j, 4]).max()),
                    unmatched=[int(len(A) - len(i)), int(len(B) - len(j))])
    return diff


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image_dir", type=Path)
    args = ap.parse_args(argv)
    print(json.dumps(compare(args.image_dir), indent=2))


if __name__ == "__main__":
    main()
