"""
DeepForest repeated-execution experiment (lowalt_repro_v1): 10 runs of each pilot configuration.

Runs the verified pilot runner (run_deepforest_lowalt_v1.py) unchanged, 10 times for each of the four
image/configuration combinations (40 runs). Each run is a separate Python process (fresh model load, fresh CUDA
context) that writes its own record under outputs/lowalt_repro_v1/. The configurations are exactly the pilot's: the
driver refuses to start if any configuration id differs from the pilot's.

    .venv/bin/python scripts/run_deepforest_repeatability_v1.py --dry-run     # plan and status; runs nothing
    .venv/bin/python scripts/run_deepforest_repeatability_v1.py --prepare     # write experiment_design.json only
    .venv/bin/python scripts/run_deepforest_repeatability_v1.py --execute     # run all outstanding runs (inference)

Order: interleaved by repetition (repeat 1 of all four combinations, then repeat 2, ...), so any drift over the
session is spread evenly across configurations. Completed runs are skipped; the first failed run stops the driver
and must be inspected (a failed run folder is never overwritten).

These are repeated executions of deterministic programs, not independent statistical observations.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_deepforest_lowalt_v1 as rd  # noqa: E402

EXPERIMENT_FAMILY = "lowalt_repro_v1"
OUTPUT_ROOT = Path("outputs/lowalt_repro_v1")
RUNS_PER_COMBINATION = 10
DEFAULT_MANIFEST = Path("/mnt/c/Users/gabpe/OneDrive/Documents/src/tree_boxes/manifests/manifest.csv")
TARGET_GSD_CM = 10.0

# (model, image, experiment id, native GSD, target GSD, pilot experiment id, pilot config id)
COMBINATIONS = [
    ("M01", "LA_PT-R1_27m", "lowalt_repro_ptr1_27m_m01_v1", None, None,
     "lowalt_pilot_ptr1_27m_m01_v1", "M01_9c192941f5"),
    ("M01", "LA_PT-02_64m", "lowalt_repro_pt02_64m_m01_v1", None, None,
     "lowalt_pilot_pt02_64m_m01_v1", "M01_9c192941f5"),
    ("M02v2", "LA_PT-R1_27m", "lowalt_repro_ptr1_27m_m02v2_v1", 0.472, TARGET_GSD_CM,
     "lowalt_pilot_ptr1_27m_m02_v2", "M02v2_b25eccbcc7"),
    ("M02v2", "LA_PT-02_64m", "lowalt_repro_pt02_64m_m02v2_v1", 1.132, TARGET_GSD_CM,
     "lowalt_pilot_pt02_64m_m02_v2", "M02v2_1e05a4037f"),
]


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def check_configurations() -> list:
    """The configuration ids must equal the pilot's: same checkpoint, settings and preprocessing."""
    out = []
    for model, image_id, exp, native, target, pilot_exp, pilot_cid in COMBINATIONS:
        cfg = rd.build_config(model, native, target)
        cid = rd.config_id(cfg)
        if cid != pilot_cid:
            raise SystemExit(f"{exp}: configuration id {cid} differs from the pilot's {pilot_cid}; refusing to run")
        out.append({"model": model, "image_id": image_id, "experiment_id": exp, "config_id": cid,
                    "native_gsd_cm": native, "target_gsd_cm": target, "pilot_experiment_id": pilot_exp})
    return out


def plan(root: Path = OUTPUT_ROOT, runs: int = RUNS_PER_COMBINATION) -> list:
    """Interleaved plan of every run with its folder and current status."""
    items = []
    for rep in range(1, runs + 1):
        for model, image_id, exp, native, target, _, _ in COMBINATIONS:
            label = f"repeat_{rep:02d}"
            run_dir = rd.run_dir_for(root, rd.build_config(model, native, target), exp, image_id, label)
            items.append({"repeat": rep, "model": model, "image_id": image_id, "experiment_id": exp,
                          "native_gsd_cm": native, "target_gsd_cm": target, "run": label, "run_dir": run_dir,
                          "status": run_status(run_dir)})
    return items


def run_status(run_dir: Path) -> str:
    rec = run_dir / "run_record.json"
    if not run_dir.exists():
        return "pending"
    if not rec.exists():
        return "incomplete"           # folder without a record: process died; inspect before re-running
    return json.loads(rec.read_text(encoding="utf-8")).get("status", "unknown")


def command(item: dict, manifest: Path, root: Path) -> list:
    cmd = [sys.executable, str(HERE / "run_deepforest_lowalt_v1.py"), "--model", item["model"],
           "--manifest", str(manifest), "--image-id", item["image_id"], "--experiment-id", item["experiment_id"],
           "--run", item["run"], "--output-root", str(root)]
    if item["native_gsd_cm"] is not None:
        cmd += ["--native-gsd-cm", repr(item["native_gsd_cm"]), "--target-gsd-cm", repr(item["target_gsd_cm"])]
    return cmd


def pilot_reference(root_pilot: Path = rd.OUTPUT_ROOT) -> dict:
    """predictions_hash of the earlier pilot primary/verification runs (read-only), for cross-reference."""
    ref = {}
    for model, image_id, exp, native, target, pilot_exp, pilot_cid in COMBINATIONS:
        entry = {}
        for label in ("primary", "verification"):
            d = rd.run_dir_for(root_pilot, rd.build_config(model, native, target), pilot_exp, image_id, label)
            p = d / "run_record.json"
            if p.exists():
                r = json.loads(p.read_text(encoding="utf-8"))
                entry[label] = {"run_dir": d.as_posix(), "detections": r.get("detections"),
                                "predictions_hash": r.get("predictions_hash"),
                                "raw_predictions_hash": r.get("raw_predictions_hash")}
        ref[exp] = entry
    return ref


def design(manifest: Path, root: Path) -> dict:
    return {
        "experiment_family": EXPERIMENT_FAMILY,
        "purpose": "test whether each deterministic DeepForest configuration reproduces identical detections over "
                   "10 repeated executions on the two approved pilot images",
        "runner": "scripts/run_deepforest_lowalt_v1.py (unchanged inference path and configuration)",
        "combinations": check_configurations(),
        "runs_per_combination": RUNS_PER_COMBINATION,
        "total_runs": RUNS_PER_COMBINATION * len(COMBINATIONS),
        "order": "interleaved by repetition: repeat_01 of all four combinations, then repeat_02, ...",
        "independence": "each run is a separate Python process: fresh model load from the pinned checkpoint (hash and "
                        "tensor-equality verified per run), fresh CUDA context; M02v2 re-resamples the image each run",
        "offline": "HF_HUB_OFFLINE=1; pinned revision " + rd.MODEL_REVISION,
        "manifest": str(manifest),
        "output_root": str(root),
        "prior_pilot_runs": pilot_reference(),
        "interpretation": "repeated executions of deterministic algorithms; not independent statistical observations "
                          "and not comparable as a sample size to the stochastic LLM trials",
    }


def prepare(manifest: Path, root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    path = root / "experiment_design.json"
    d = design(manifest, root)
    if path.exists():
        old = json.loads(path.read_text(encoding="utf-8"))
        old.pop("prepared_utc", None)
        if old != d:
            raise SystemExit(f"{path} exists with a different design; refusing to overwrite")
        return path
    path.write_text(json.dumps({**d, "prepared_utc": now_utc()}, indent=2), encoding="utf-8")
    return path


def subprocess_runner(cmd: list, cwd: Path) -> int:
    env = {**os.environ, "HF_HUB_OFFLINE": "1"}
    return subprocess.run(cmd, cwd=cwd, env=env).returncode


def execute(manifest: Path, root: Path, cwd: Path, runner=subprocess_runner, max_runs: int | None = None,
            log=print) -> dict:
    prepare(manifest, root)
    items = plan(root)
    blocked = [i for i in items if i["status"] not in ("pending", "success")]
    if blocked:
        raise SystemExit("Inspect these runs before continuing (never overwritten): "
                         + ", ".join(f"{i['run_dir']} [{i['status']}]" for i in blocked))
    todo = [i for i in items if i["status"] == "pending"][:max_runs]
    log(f"{len(items) - len([i for i in items if i['status'] == 'pending'])} of {len(items)} runs already complete; "
        f"running {len(todo)} now.")
    driver_log = root / "driver_log.jsonl"
    done = 0
    for item in todo:
        cmd = command(item, manifest, root)
        started, t0 = now_utc(), time.perf_counter()
        code = runner(cmd, cwd)
        status = run_status(item["run_dir"])
        entry = {"repeat": item["repeat"], "experiment_id": item["experiment_id"], "image_id": item["image_id"],
                 "run": item["run"], "run_dir": item["run_dir"].as_posix(), "returncode": code, "status": status,
                 "started_utc": started, "finished_utc": now_utc(),
                 "wall_seconds": round(time.perf_counter() - t0, 2)}
        with open(driver_log, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
        log(f"  {item['experiment_id']} {item['run']}: {status} (exit {code}, {entry['wall_seconds']} s)")
        if code != 0 or status != "success":
            return {"completed_now": done, "stopped": f"{item['run_dir']} ended with status {status} (exit {code})"}
        done += 1
    return {"completed_now": done, "stopped": None}


def quick_summary(root: Path = OUTPUT_ROOT) -> list:
    """Per combination: completed runs, detection counts and distinct canonical prediction hashes."""
    rows, ref = [], pilot_reference()
    for model, image_id, exp, native, target, pilot_exp, _ in COMBINATIONS:
        recs = []
        for rep in range(1, RUNS_PER_COMBINATION + 1):
            p = rd.run_dir_for(root, rd.build_config(model, native, target), exp, image_id,
                               f"repeat_{rep:02d}") / "run_record.json"
            if p.exists():
                recs.append(json.loads(p.read_text(encoding="utf-8")))
        ok = [r for r in recs if r.get("status") == "success"]
        pilot_hashes = {v.get("predictions_hash") for v in ref.get(exp, {}).values()}
        hashes = {r["predictions_hash"] for r in ok}
        rows.append({"experiment_id": exp, "completed": len(ok), "counts": [r["detections"] for r in ok],
                     "distinct_predictions_hashes": len(hashes),
                     "equal_to_pilot_runs": bool(ok) and hashes == pilot_hashes})
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true", help="Show the plan and status; run nothing, write nothing")
    mode.add_argument("--prepare", action="store_true", help="Write outputs/lowalt_repro_v1/experiment_design.json only")
    mode.add_argument("--execute", action="store_true", help="Run every outstanding run (DeepForest inference)")
    mode.add_argument("--summary", action="store_true", help="Print a quick hash/count summary of completed runs")
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    ap.add_argument("--max-runs", type=int, help="Execute at most this many outstanding runs (default: all)")
    args = ap.parse_args(argv)
    root = args.output_root
    if root.resolve() == rd.OUTPUT_ROOT.resolve():
        ap.error(f"the repeatability experiment may not write into the pilot root {rd.OUTPUT_ROOT}")
    cfgs = check_configurations()
    if args.dry_run:
        items = plan(root)
        print(f"{EXPERIMENT_FAMILY}: {len(items)} runs ({RUNS_PER_COMBINATION} per combination), interleaved.")
        for c in cfgs:
            print(f"  {c['experiment_id']}: {c['model']} on {c['image_id']}, config {c['config_id']} (= pilot)")
        counts = {}
        for i in items:
            counts[i["status"]] = counts.get(i["status"], 0) + 1
        print("Status: " + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())))
        nxt = next((i for i in items if i["status"] == "pending"), None)
        if nxt:
            print("Next command: " + " ".join(command(nxt, args.manifest, root)))
        print("Dry run: nothing run, nothing written.")
    elif args.prepare:
        print(f"Wrote {prepare(args.manifest, root)}")
    elif args.execute:
        rep = execute(args.manifest, root, Path.cwd(), max_runs=args.max_runs)
        print(json.dumps(rep, indent=2, default=str))
        for row in quick_summary(root):
            print(json.dumps(row))
        if rep["stopped"]:
            sys.exit(1)
    else:
        for row in quick_summary(root):
            print(json.dumps(row))


if __name__ == "__main__":
    main()
