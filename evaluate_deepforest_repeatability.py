"""
Reproducibility of repeated DeepForest executions (lowalt_repro_v1). Offline; no inference, no API requests.

Reads the 10 repeated runs of each DeepForest configuration/image (written by urban-tree-vision-baselines
scripts/run_deepforest_repeatability_v1.py and mirrored to Windows), and reports for every combination:

* raw-file identity: SHA-256 of each run's predictions_raw.csv (DeepForest's own output order) and of
  detections.json (canonical order), kept separate;
* exact identity of the canonically ordered detections: every coordinate and score compared as floats;
* detection counts per run (mean, SD; SD is reported, not used as a stochastic estimate);
* pairwise agreement for all 45 run pairs with the existing exact-IoU one-to-one matching (0.50, 0.75, 1.00);
* for any pair that is not identical: count difference, unmatched boxes, and the largest coordinate and score
  differences among matched boxes;
* cross-reference with the earlier pilot primary/verification runs (read-only);
* side by side with the frozen M08/M09 within-model statistics (read-only, from an existing evaluation folder).

These are repeated executions of deterministic programs on one machine and software stack. They are not
independent statistical observations and do not add to the sample size of any comparison.

    python evaluate_deepforest_repeatability.py \
        --runs-root ../urban-tree-vision-baselines/outputs/lowalt_repro_v1 \
        --pilot-root ../urban-tree-vision-baselines/outputs/lowalt_pilot_v1 \
        --output-dir evaluation_results_deepforest_repro_v1
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import evaluate_consistency as ev
import evaluation_core as ec

ROOT = Path(__file__).resolve().parent
BASELINES = ROOT.parent / "urban-tree-vision-baselines"
DEFAULT_RUNS_ROOT = BASELINES / "outputs" / "lowalt_repro_v1"
DEFAULT_PILOT_ROOT = BASELINES / "outputs" / "lowalt_pilot_v1"
DEFAULT_LLM_EVAL = ROOT / "evaluation_results"          # frozen M08/M09 evaluation (read-only)
DEFAULT_OUTPUT = ROOT / "evaluation_results_deepforest_repro_v1"
MODEL_DIRS = {"m01_native": "M01", "m02v2_gsd": "M02v2"}
RUN_GLOB = "run_*_repeat"
INTERPRETATION = ("Repeated executions of a deterministic program on one machine and software stack: not "
                  "independent observations, and not comparable as a sample size to the 10 independent stochastic "
                  "LLM trials.")


def sha256_file(path: Path) -> str:
    return ev.sha256_file(path)


def canonical(trees: list) -> list:
    """Detections as exact float tuples (box in original pixels + score), sorted independently of file order."""
    return sorted((tuple(float(v) for v in t["bbox_px"]) + (float(t["score"]),)) for t in trees)


# ------------------------------------------------------------------ loading

def discover(runs_root: Path) -> list:
    """(model_code, experiment_id, config_id, image_id, [run dirs]) for every combination under runs_root."""
    found = []
    for model_dir, code in MODEL_DIRS.items():
        for image_dir in sorted((runs_root / model_dir).glob("*/*/*")):
            runs = sorted(image_dir.glob(RUN_GLOB))
            if runs:
                exp, cfg = image_dir.parent.parent.name, image_dir.parent.name
                found.append((code, exp, cfg, image_dir.name, runs))
    return found


def load_run(run_dir: Path, manifest_rows: dict, image_root: Path) -> dict:
    rec = json.loads((run_dir / "run_record.json").read_text(encoding="utf-8"))
    out = {"run_dir": run_dir, "run": rec.get("run"), "status": rec.get("status"), "record": rec}
    if rec.get("status") != "success":
        return out
    det_path, raw_path = run_dir / "detections.json", run_dir / "predictions_raw.csv"
    det = json.loads(det_path.read_text(encoding="utf-8"))
    if det.get("coordinate_system") != "original_pixels" or det.get("model_config_id") != rec["model_config_id"]:
        raise ev.InputError(f"{det_path}: not a DeepForest original-pixel record of {rec['model_config_id']}")
    if len(det["trees"]) != rec["detections"]:
        raise ev.InputError(f"{det_path}: {len(det['trees'])} boxes but the run record says {rec['detections']}")
    row = manifest_rows.get(rec["image_id"])
    if row is None or row["sha256"] != rec["source_sha256"]:
        raise ev.InputError(f"{run_dir}: source hash does not match the manifest")
    pre = rec.get("preprocessing") or {}
    annotated = run_dir / "annotated_original.jpg"
    out.update(
        trees=det["trees"], canonical=canonical(det["trees"]), image_size=tuple(det["image_size"]),
        image_sha256=rec["source_sha256"], model_config_id=rec["model_config_id"],
        raw_sha256=sha256_file(raw_path), detections_sha256=sha256_file(det_path),
        annotated_sha256=sha256_file(annotated) if annotated.exists() else None,
        preprocessed_sha256=pre.get("output_sha256"), predictions_hash=rec.get("predictions_hash"),
        raw_predictions_hash=rec.get("raw_predictions_hash"), detections=rec["detections"],
        raw_detections=rec.get("raw_detections"), inference_seconds=rec.get("inference_seconds"),
        environment=rec.get("environment") or {}, started_utc=rec.get("started_utc"),
        weights_verified=(rec.get("loaded_weights") or {}).get("all_equal"),
        checkpoint_sha256=(rec.get("checkpoint") or {}).get("model_safetensors_sha256"))
    return out


def detection_set(run: dict, index: int, image_id: str) -> ec.DetectionSet:
    dets = tuple(ec.Detection(f"{image_id}:r{index:02d}:{t['tree_id']}", t["tree_id"], ec.box_from_pixels(t["bbox_px"]))
                 for t in run["trees"])
    w, h = run["image_size"]
    return ec.DetectionSet(f"repeat/{run['model_config_id']}", image_id, run["image_sha256"], w, h, index, dets)


# ------------------------------------------------------------------ comparison

def pair_difference(a: dict, b: dict, matches: list) -> dict:
    """Largest coordinate and score differences among boxes matched at IoU >= 0.50 (0 when identical)."""
    ta, tb = a["trees"], b["trees"]
    coord = score = 0.0
    for i, j, _ in matches:
        coord = max(coord, max(abs(float(x) - float(y)) for x, y in zip(ta[i]["bbox_px"], tb[j]["bbox_px"])))
        score = max(score, abs(float(ta[i]["score"]) - float(tb[j]["score"])))
    return {"max_coordinate_abs_diff_px": coord, "max_score_abs_diff": score}


def analyse_combination(code, exp, cfg, image_id, runs, thresholds=ec.THRESHOLDS) -> dict:
    ok = [r for r in runs if r["status"] == "success"]
    excluded = [(r["run_dir"].name, r["status"]) for r in runs if r["status"] != "success"]
    if not ok:
        return {"model": code, "experiment_id": exp, "model_config_id": cfg, "image_id": image_id,
                "n_runs": 0, "excluded": excluded}
    if len({r["model_config_id"] for r in ok}) != 1 or {r["model_config_id"] for r in ok} != {cfg}:
        raise ev.InputError(f"{exp}/{image_id}: runs disagree on the configuration id")
    sets = [detection_set(r, n, image_id) for n, r in enumerate(ok, start=1)]
    counts = [r["detections"] for r in ok]
    pair_rows = []
    by_pair = {}
    for t in thresholds:
        for res in ec.pairwise_trials(sets, t):
            by_pair.setdefault((res["trial_a"], res["trial_b"]), {})[t] = res
    for (ia, ib), per_t in sorted(by_pair.items()):
        a, b = ok[ia - 1], ok[ib - 1]
        identical = a["canonical"] == b["canonical"]
        idx = {s.trial: {d.uid: k for k, d in enumerate(s.detections)} for s in (sets[ia - 1], sets[ib - 1])}
        half = per_t[thresholds[0]]
        matches = [(idx[ia][ua], idx[ib][ub], v) for ua, ub, v in half["pairs"]]
        diff = {"max_coordinate_abs_diff_px": 0.0, "max_score_abs_diff": 0.0} if identical else pair_difference(a, b, matches)
        row = {"model": code, "experiment_id": exp, "image_id": image_id, "run_a": a["run_dir"].name,
               "run_b": b["run_dir"].name, "n_a": len(a["canonical"]), "n_b": len(b["canonical"]),
               "raw_file_identical": a["raw_sha256"] == b["raw_sha256"],
               "detections_file_identical": a["detections_sha256"] == b["detections_sha256"],
               "canonical_detections_identical": identical,
               "order_only_difference": identical and a["raw_sha256"] != b["raw_sha256"],
               **diff}
        for t in thresholds:
            r = per_t[t]
            tag = ec.threshold_tag(t)
            row[f"agreement_{tag}"] = r["agreement"]
            row[f"matched_{tag}"] = r["matched"]
            row[f"unmatched_a_{tag}"] = r["unmatched_a"]
            row[f"unmatched_b_{tag}"] = r["unmatched_b"]
            row[f"mean_matched_iou_{tag}"] = r["mean_iou"]
        pair_rows.append(row)
    summaries = {t: ec.summarize_pairs([per_t[t] for per_t in by_pair.values()]) for t in thresholds}
    stats = ev.count_stats(counts)
    first = ok[0]
    discrepancies = []
    if len(set(counts)) > 1:
        discrepancies.append(f"detection counts differ: {sorted(set(counts))}")
    for key, label in (("raw_sha256", "predictions_raw.csv"), ("detections_sha256", "detections.json"),
                       ("preprocessed_sha256", "preprocessed image"), ("annotated_sha256", "annotated image")):
        values = {r[key] for r in ok}
        if len(values) > 1:
            discrepancies.append(f"{len(values)} distinct {label} hashes")
    not_identical = [p for p in pair_rows if not p["canonical_detections_identical"]]
    if not_identical:
        discrepancies.append(f"{len(not_identical)} of {len(pair_rows)} run pairs differ in boxes or scores "
                             f"(max coordinate diff {max(p['max_coordinate_abs_diff_px'] for p in not_identical):.6g} px, "
                             f"max score diff {max(p['max_score_abs_diff'] for p in not_identical):.6g})")
    envs = {json.dumps({k: r["environment"].get(k) for k in ("deepforest", "torch", "cudnn", "gpu", "torch_cuda",
                                                             "cudnn_deterministic", "cudnn_benchmark")}, sort_keys=True)
            for r in ok}
    return {
        "model": code, "experiment_id": exp, "model_config_id": cfg, "image_id": image_id, "n_runs": len(ok),
        "excluded": excluded, "runs": ok, "pairs": pair_rows, "summaries": summaries, "counts": counts, "stats": stats,
        "all_raw_files_identical": len({r["raw_sha256"] for r in ok}) == 1,
        "all_detection_files_identical": len({r["detections_sha256"] for r in ok}) == 1,
        "all_canonical_identical": all(r["canonical"] == first["canonical"] for r in ok),
        "all_preprocessed_identical": len({r["preprocessed_sha256"] for r in ok}) == 1,
        "distinct_predictions_hashes": len({r["predictions_hash"] for r in ok}),
        "weights_verified_every_run": all(r["weights_verified"] is True for r in ok),
        "environments": [json.loads(e) for e in sorted(envs)],
        "discrepancies": discrepancies,
    }


def pilot_crossref(pilot_root: Path, combo: dict) -> dict:
    """Compare with the pilot primary/verification runs of the same configuration and image (read-only)."""
    out = {}
    if not combo.get("runs"):
        return out
    model_dir = {v: k for k, v in MODEL_DIRS.items()}[combo["model"]]
    for d in sorted((pilot_root / model_dir).glob(f"*/{combo['model_config_id']}/{combo['image_id']}/run_0[12]_*")):
        p = d / "run_record.json"
        if not p.exists():
            continue
        r = json.loads(p.read_text(encoding="utf-8"))
        same = (d / "detections.json").exists() and canonical(
            json.loads((d / "detections.json").read_text(encoding="utf-8"))["trees"]) == combo["runs"][0]["canonical"]
        out[f"{d.parent.parent.parent.name}/{d.name}"] = {"predictions_hash": r.get("predictions_hash"),
                                                          "detections": r.get("detections"),
                                                          "canonical_identical_to_repeat_runs": same}
    return out


def llm_within_model(llm_eval: Path) -> list:
    """Frozen M08/M09 within-model repeatability on the same images (read-only)."""
    iou, counts = llm_eval / "iou_summary.csv", llm_eval / "count_consistency.csv"
    if not iou.exists() or not counts.exists():
        return []
    cnt = {(r["experiment_id"], r["image_id"]): r for r in ev.read_csv(counts)}
    rows = []
    for r in ev.read_csv(iou):
        if not r["model_config_id"].startswith(("M08", "M09")):
            continue
        c = cnt.get((r["experiment_id"], r["image_id"]), {})
        rows.append({"model": r["model_config_id"][:3], "model_name": r.get("model", ""),
                     "experiment_id": r["experiment_id"], "image_id": r["image_id"], "iou_threshold": r["iou_threshold"],
                     "kind": "independent stochastic trials", "n": int(r["n_trials"]),
                     "pairs": int(r["n_pairs"]), "agreement_mean": float(r["agreement_mean"]),
                     "agreement_sd": float(r["agreement_sd"]) if r["agreement_sd"] else None,
                     "count_mean": float(c["mean_count"]) if c.get("mean_count") else None,
                     "count_sd": float(c["sd_count"]) if c.get("sd_count") else None,
                     "count_cv": float(c["cv_count"]) if c.get("cv_count") else None})
    return rows


# ------------------------------------------------------------------ run

def run(runs_root: Path = DEFAULT_RUNS_ROOT, pilot_root: Path = DEFAULT_PILOT_ROOT, llm_eval: Path = DEFAULT_LLM_EVAL,
        output_dir: Path = DEFAULT_OUTPUT, manifest: Path = ev.MANIFEST, image_root: Path = ROOT,
        figures: bool = True, log=print) -> dict:
    found = discover(Path(runs_root))
    if not found:
        raise ev.InputError(f"no repeated DeepForest runs found under {runs_root} (run the experiment first)")
    manifest_rows = {r["image_id"]: r for r in ev.read_csv(manifest)}
    combos = []
    for code, exp, cfg, image_id, run_dirs in found:
        runs = [load_run(d, manifest_rows, image_root) for d in run_dirs]
        combo = analyse_combination(code, exp, cfg, image_id, runs)
        combo["site"] = manifest_rows.get(image_id, {}).get("site", "")
        combo["pilot"] = pilot_crossref(Path(pilot_root), combo)
        combos.append(combo)
        log(f"{exp} / {image_id}: {combo['n_runs']} runs; canonical identical: {combo.get('all_canonical_identical')}")
    llm_rows = llm_within_model(Path(llm_eval))

    ev.prepare_output_dir(output_dir)
    run_rows, pair_rows, summary_rows = [], [], []
    for c in combos:
        for r in c.get("runs", []):
            env = r["environment"]
            run_rows.append({"model": c["model"], "experiment_id": c["experiment_id"], "image_id": c["image_id"],
                             "site": c["site"], "run": r["run_dir"].name, "status": r["status"],
                             "detections": r["detections"], "raw_detections": r["raw_detections"],
                             "predictions_raw_csv_sha256": r["raw_sha256"], "detections_json_sha256": r["detections_sha256"],
                             "annotated_sha256": r["annotated_sha256"], "preprocessed_sha256": r["preprocessed_sha256"],
                             "predictions_hash": r["predictions_hash"], "raw_predictions_hash": r["raw_predictions_hash"],
                             "weights_verified": r["weights_verified"], "inference_seconds": r["inference_seconds"],
                             "started_utc": r["started_utc"], "gpu": env.get("gpu"), "torch": env.get("torch"),
                             "cudnn": env.get("cudnn")})
        pair_rows += c.get("pairs", [])
        s = c.get("stats") or {}
        row = {"model": c["model"], "experiment_id": c["experiment_id"], "model_config_id": c["model_config_id"],
               "image_id": c["image_id"], "site": c["site"], "runs_successful": c["n_runs"],
               "runs_excluded": ";".join(f"{n}:{st}" for n, st in c["excluded"]),
               "counts_by_run": ";".join(map(str, c.get("counts", []))), "count_mean": s.get("mean"),
               "count_sd": s.get("sd"), "count_min": s.get("min"), "count_max": s.get("max"),
               "all_raw_files_identical": c.get("all_raw_files_identical"),
               "all_detection_files_identical": c.get("all_detection_files_identical"),
               "all_canonical_identical": c.get("all_canonical_identical"),
               "all_preprocessed_identical": c.get("all_preprocessed_identical"),
               "distinct_predictions_hashes": c.get("distinct_predictions_hashes"),
               "weights_verified_every_run": c.get("weights_verified_every_run"),
               "identical_to_pilot_runs": (all(v["canonical_identical_to_repeat_runs"] for v in c["pilot"].values())
                                           if c.get("pilot") else None),
               "discrepancies": "; ".join(c.get("discrepancies", [])) or "none"}
        for t, sm in (c.get("summaries") or {}).items():
            tag = ec.threshold_tag(t)
            row[f"pairs_{tag}"] = sm["n_pairs"]
            row[f"agreement_mean_{tag}"] = sm["agreement_mean"]
            row[f"agreement_min_{tag}"] = sm["agreement_min"]
            row[f"pairs_full_agreement_{tag}"] = sm["pairs_with_full_agreement"]
        summary_rows.append(row)
    ev.write_csv(output_dir / "repro_runs.csv", list(run_rows[0]), run_rows)
    if pair_rows:
        ev.write_csv(output_dir / "repro_pairwise.csv", list(pair_rows[0]), pair_rows)
    ev.write_csv(output_dir / "repro_summary.csv", list(dict.fromkeys(k for r in summary_rows for k in r)), summary_rows)
    if llm_rows:
        ev.write_csv(output_dir / "llm_within_model_reference.csv", list(llm_rows[0]), llm_rows)

    figure_paths = []
    if figures:
        import evaluation_plots_repro as ep
        figure_paths = ep.make_figures(combos, llm_rows, output_dir)
    config = {
        "analysis": "DeepForest repeated-execution reproducibility (lowalt_repro_v1)",
        "algorithm_version": ec.ALGORITHM_VERSION,
        "iou_thresholds": [ec.threshold_label(t) for t in ec.THRESHOLDS],
        "identity_checks": ["SHA-256 of predictions_raw.csv (raw order)", "SHA-256 of detections.json (canonical order)",
                            "exact float equality of canonically sorted boxes and scores", "detection counts",
                            "exact-IoU one-to-one pairwise agreement for every run pair"],
        "interpretation": INTERPRETATION,
        "inputs": {"runs_root": str(runs_root), "pilot_root": str(pilot_root), "llm_evaluation": str(llm_eval),
                   "files": [{"path": ev.rel(r["run_dir"] / n, Path(runs_root).parent),
                              "sha256": sha256_file(r["run_dir"] / n)}
                             for c in combos for r in c.get("runs", []) for n in ("run_record.json", "detections.json",
                                                                                   "predictions_raw.csv")]},
        "software": ev.software_versions(), "code": ev.git_state(ROOT),
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "command": " ".join(sys.argv) if sys.argv else None, "inference_run": False, "api_requests_made": 0,
    }
    (output_dir / ev.MARKER).write_text(json.dumps(config, indent=2), encoding="utf-8")
    (output_dir / "repro_report.md").write_text(report(combos, llm_rows, figure_paths, output_dir), encoding="utf-8")
    log(f"Wrote {output_dir}")
    return {"combos": combos, "llm_rows": llm_rows, "figures": figure_paths, "summary_rows": summary_rows}


def report(combos, llm_rows, figure_paths, out: Path) -> str:
    f = ev.fnum
    L = ["# DeepForest repeated-execution reproducibility (lowalt_repro_v1)\n",
         "Offline analysis of saved runs; no inference and no API requests in this step. " + INTERPRETATION + "\n",
         "## Identity of outputs across runs\n",
         "| Configuration | Image | Runs | Detections per run | Raw files identical | detections.json identical | "
         "Boxes and scores identical (canonical) | Identical to pilot runs | Discrepancies |",
         "|---|---|---|---|---|---|---|---|---|"]
    for c in combos:
        counts = c.get("counts", [])
        shown = f"{counts[0]} in every run" if counts and len(set(counts)) == 1 else ", ".join(map(str, counts)) or "–"
        pil = (all(v["canonical_identical_to_repeat_runs"] for v in c["pilot"].values()) if c.get("pilot") else None)
        L.append(f"| {c['model']} ({c['model_config_id']}) | {c['image_id']} | {c['n_runs']} | {shown} | "
                 f"{c.get('all_raw_files_identical')} | {c.get('all_detection_files_identical')} | "
                 f"{c.get('all_canonical_identical')} | {pil if pil is not None else 'n/a'} | "
                 f"{'; '.join(c.get('discrepancies', [])) or 'none'} |")
    L += ["", "Raw files are DeepForest's own output order; detections.json is sorted canonically by the runner. A "
          "pair whose raw files differ but whose canonical detections match differs only in order.\n",
          "## Pairwise agreement between runs (exact-IoU one-to-one matching)\n",
          "| Configuration | Image | Pairs | Agreement mean (min) at 0.50 | at 0.75 | at 1.00 | Pairs at 100% (1.00) |",
          "|---|---|---|---|---|---|---|"]
    for c in combos:
        s = c.get("summaries")
        if not s:
            continue
        cells = [f"{f(s[t]['agreement_mean'])} ({f(s[t]['agreement_min'])})" for t in ec.THRESHOLDS]
        L.append(f"| {c['model']} | {c['image_id']} | {s[ec.THRESHOLDS[0]]['n_pairs']} | " + " | ".join(cells)
                 + f" | {s[ec.THRESHOLDS[-1]]['pairs_with_full_agreement']} |")
    if llm_rows:
        L += ["\n## Beside the LLMs' within-model repeatability (frozen M08/M09 values, read-only)\n",
              "| Model | Image | Kind | n | Agreement mean ± SD at 0.50 | at 0.75 | at 1.00 | Count mean (CV) |",
              "|---|---|---|---|---|---|---|---|"]
        for c in combos:
            s = c.get("summaries")
            if not s:
                continue
            ag = [f"{f(s[t]['agreement_mean'])} ± {f(s[t]['agreement_sd'])}" for t in ec.THRESHOLDS]
            st = c["stats"]
            L.append(f"| DeepForest {c['model']} | {c['image_id']} | repeated executions | {c['n_runs']} | "
                     + " | ".join(ag) + f" | {f(st['mean'], 1)} ({f(st['cv'] * 100 if st['cv'] is not None else None, 1)}%) |")
        keyed = {}
        for r in llm_rows:
            keyed.setdefault((r["model"], r["image_id"]), {})[r["iou_threshold"]] = r
        for (m, image_id), per in sorted(keyed.items()):
            ag = [f"{f(per[lab]['agreement_mean'])} ± {f(per[lab]['agreement_sd'])}" if lab in per else "–"
                  for lab in ("0.50", "0.75", "1.00")]
            any_r = next(iter(per.values()))
            cv = any_r["count_cv"]
            L.append(f"| {m} | {image_id} | independent stochastic trials | {any_r['n']} | " + " | ".join(ag)
                     + f" | {f(any_r['count_mean'], 1)} ({f(cv * 100 if cv is not None else None, 1)}%) |")
        L.append("\nThe DeepForest rows test reproducibility of one deterministic computation; the LLM rows measure "
                 "run-to-run variation of a stochastic model. Agreement here is consistency, not accuracy.\n")
    L.append("## Environment\n")
    for c in combos:
        for e in c.get("environments", []):
            L.append(f"- {c['experiment_id']}: " + ", ".join(f"{k}={v}" for k, v in e.items()))
    if figure_paths:
        L.append("\n## Figures\n")
        L += [f"- `{ev.rel(p, out)}`" for p in figure_paths]
    return "\n".join(L) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS_ROOT)
    ap.add_argument("--pilot-root", type=Path, default=DEFAULT_PILOT_ROOT)
    ap.add_argument("--llm-evaluation", type=Path, default=DEFAULT_LLM_EVAL)
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument("--manifest", type=Path, default=ev.MANIFEST)
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args(argv)
    try:
        run(args.runs_root, args.pilot_root, args.llm_evaluation, args.output_dir, args.manifest,
            figures=not args.no_figures)
    except ev.InputError as e:
        sys.exit(f"Input problem: {e}")


if __name__ == "__main__":
    main()
