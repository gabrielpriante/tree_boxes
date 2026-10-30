"""
Accuracy of repeated trials against a COMPLETE human reference for one image.

Every trial is matched to the reference independently with the same exact-IoU, one-to-one,
maximum-cardinality matching used for trial-to-trial consistency (evaluation_core.match_boxes).
Nothing here runs unless the reference for the image is declared COMPLETE.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field

import evaluation_core as ec

METRICS = ("precision", "recall", "f1", "true_positives", "false_positives", "false_negatives", "mean_iou_tp")
FP_TYPES = {
    "no_reference_overlap": "does not overlap any reference tree",
    "localization_below_threshold": "overlaps a reference tree, but IoU is below the threshold",
    "reference_matched_by_other_detection": "overlaps a reference tree that another box in the same trial matched",
}


@dataclass
class AccuracyResult:
    image_id: str
    experiment_id: str
    reference: ec.DetectionSet
    per_trial: dict = field(default_factory=dict)     # threshold -> trial -> accuracy dict
    trial_rows: list = field(default_factory=list)
    summary_rows: list = field(default_factory=list)
    reference_rows: list = field(default_factory=list)
    outcome_rows: list = field(default_factory=list)
    containment: dict = field(default_factory=dict)   # threshold label -> share of each matched Claude box inside its reference box


def _stats(values):
    vals = [float(v) for v in values if v is not None]
    if not vals:
        return {"n": 0, "mean": None, "sd": None, "min": None, "max": None}
    return {"n": len(vals), "mean": statistics.fmean(vals), "sd": statistics.stdev(vals) if len(vals) > 1 else None,
            "min": min(vals), "max": max(vals)}


def evaluate_accuracy(u, reference: ec.DetectionSet, status: str, thresholds, group_of: dict | None = None,
                      unit_of: dict | None = None) -> AccuracyResult:
    """u: ImageTrials-like (experiment_id, image_id, sets). group_of: detection uid -> IoU 0.50 group id.
    unit_of: detection uid -> (review_unit_id, human_label or '')."""
    group_of, unit_of = group_of or {}, unit_of or {}
    res = AccuracyResult(u.image_id, u.experiment_id, reference)
    ref_ids = {d.uid: d.label for d in reference.detections}
    sets = sorted(u.sets, key=lambda s: s.trial)
    ids = {"experiment_id": u.experiment_id, "image_id": u.image_id}
    for t in thresholds:
        lab = ec.threshold_label(t)
        res.per_trial[t] = {}
        matched_by_ref = {d.label: {} for d in reference.detections}   # ref label -> trial -> (det uid, iou)
        best_miss = {d.label: {} for d in reference.detections}        # ref label -> trial -> (det uid, iou)
        for s in sets:
            a = ec.accuracy_vs_reference(s, reference, status, t)
            res.per_trial[t][s.trial] = a
            res.trial_rows.append({**ids, "iou_threshold": lab, "trial": s.trial, "n_predictions": len(s.detections),
                                   "n_reference": len(reference.detections),
                                   **{k: a[k] for k in METRICS}})
            pred_to_ref = {p: (ref_ids[r], v) for p, r, v in a["pairs"]}
            for p, (r, v) in pred_to_ref.items():
                matched_by_ref[r][s.trial] = (p, v)
            matched_refs = {r for r, _ in pred_to_ref.values()}
            for d in s.detections:
                ious = [(ec.iou(d.box, rd.box), rd.label) for rd in reference.detections]
                best_iou, best_ref = max(ious, key=lambda x: (x[0], x[1])) if ious else (0, "")
                if d.uid in pred_to_ref:
                    outcome, fp_type = "TP", ""
                elif best_iou == 0:
                    outcome, fp_type = "FP", "no_reference_overlap"
                elif best_iou < t:
                    outcome, fp_type = "FP", "localization_below_threshold"
                else:
                    outcome, fp_type = "FP", "reference_matched_by_other_detection"
                unit, label = unit_of.get(d.uid, ("", ""))
                res.outcome_rows.append({
                    **ids, "iou_threshold": lab, "trial": s.trial, "detection_uid": d.uid, "tree_id": d.label,
                    "group_iou050": group_of.get(d.uid, ""), "review_unit_id": unit, "human_label": label,
                    "outcome": outcome, "fp_type": fp_type,
                    "matched_reference": pred_to_ref.get(d.uid, ("", None))[0],
                    "matched_iou": pred_to_ref.get(d.uid, ("", None))[1],
                    "best_reference": best_ref if best_iou > 0 else "", "best_iou": best_iou})
            for rd in reference.detections:
                if rd.label in matched_refs:
                    continue
                cands = [(ec.iou(d.box, rd.box), d.uid) for d in s.detections]
                v, uid = max(cands, default=(0, ""))
                best_miss[rd.label][s.trial] = (uid if v > 0 else "", v)
        n = len(sets)
        det_box = {d.uid: d.box for s in sets for d in s.detections}
        for rd in reference.detections:
            hits = matched_by_ref[rd.label]
            misses = best_miss[rd.label]
            b = rd.box
            ratios, offsets = [], []
            for p, _ in hits.values():
                pb = det_box[p]
                ratios.append(float(ec.area(pb) / ec.area(b)))
                inter = (min(pb[2], b[2]) - max(pb[0], b[0])) * (min(pb[3], b[3]) - max(pb[1], b[1]))
                res.containment.setdefault(lab, []).append(float(inter / ec.area(pb)))
                offsets.append(float(((pb[0] + pb[2] - b[0] - b[2]) ** 2 + (pb[1] + pb[3] - b[1] - b[3]) ** 2) ** 0.5) / 2)
            res.reference_rows.append({
                **ids, "iou_threshold": lab, "reference_id": rd.label,
                "width_px": float(b[2] - b[0]), "height_px": float(b[3] - b[1]),
                "trials_matched": len(hits), "n_trials": n, "detection_rate": len(hits) / n if n else None,
                "trials_detected": sorted(hits), "trials_missed": sorted(misses),
                "mean_iou_when_matched": _stats(v for _, v in hits.values())["mean"],
                "mean_area_ratio_pred_over_ref": _stats(ratios)["mean"],
                "mean_center_offset_px": _stats(offsets)["mean"],
                "matched_detection_uids": [hits[k][0] for k in sorted(hits)],
                "missed_trials_best_overlap": [f"t{k:02d}:{misses[k][0].rsplit(':', 1)[-1] if misses[k][0] else '-'}"
                                               f"@{float(misses[k][1]):.3f}" for k in sorted(misses)],
            })
        per = list(res.per_trial[t].values())
        tp, fp, fn = (sum(a[k] for a in per) for k in ("true_positives", "false_positives", "false_negatives"))
        pooled_p = tp / (tp + fp) if tp + fp else None
        pooled_r = tp / (tp + fn) if tp + fn else None
        pooled_f1 = (2 * pooled_p * pooled_r / (pooled_p + pooled_r)) if pooled_p and pooled_r else (
            0.0 if pooled_p is not None and pooled_r is not None else None)
        pooled = {"precision": pooled_p, "recall": pooled_r, "f1": pooled_f1,
                  "true_positives": tp, "false_positives": fp, "false_negatives": fn}
        for m in METRICS:
            s = _stats(a[m] for a in per)
            res.summary_rows.append({**ids, "iou_threshold": lab, "metric": m, "n_trials": n, "n_defined": s["n"],
                                     "mean": s["mean"], "sd": s["sd"], "min": s["min"], "max": s["max"],
                                     "pooled_over_trials": pooled.get(m)})
    return res


def summary_value(res: AccuracyResult, threshold, metric: str) -> dict:
    lab = ec.threshold_label(threshold)
    return next(r for r in res.summary_rows if r["iou_threshold"] == lab and r["metric"] == metric)


def _f(v, nd=3):
    return "–" if v is None else f"{float(v):.{nd}f}"


def report_section(res: AccuracyResult, thresholds, status_note: str, labels_by_unit: dict) -> str:
    """Markdown for one image. labels_by_unit: review_unit_id -> (label, note)."""
    L = []
    w = L.append
    n_ref = len(res.reference.detections)
    n_trials = len(next(iter(res.per_trial.values())))
    w(f"## {res.image_id} ({res.experiment_id})\n")
    w(f"Reference: {n_ref} human boxes (H01–H{n_ref:02d}), status COMPLETE. {status_note}\n")
    w(f"Each of the {n_trials} trials is matched to the reference on its own: one-to-one, maximum number of pairs "
      "with IoU ≥ threshold, ties broken by total IoU. IoU = 1.00 requires identical boxes.\n")
    w("### Summary across trials\n")
    w("| IoU ≥ | Precision mean ± SD [range] | Recall mean ± SD [range] | F1 mean ± SD [range] | "
      "TP / FP / FN per trial (mean) | Pooled P / R / F1 | Mean IoU of TPs |")
    w("|---|---|---|---|---|---|---|")
    for t in thresholds:
        cell = {}
        for m in ("precision", "recall", "f1"):
            s = summary_value(res, t, m)
            cell[m] = f"{_f(s['mean'])} ± {_f(s['sd'])} [{_f(s['min'])}–{_f(s['max'])}]"
        tp, fp, fn = (summary_value(res, t, m)["mean"] for m in ("true_positives", "false_positives", "false_negatives"))
        pp, pr, pf = (summary_value(res, t, m)["pooled_over_trials"] for m in ("precision", "recall", "f1"))
        mi = summary_value(res, t, "mean_iou_tp")
        w(f"| {ec.threshold_label(t)} | {cell['precision']} | {cell['recall']} | {cell['f1']} | "
          f"{tp:.1f} / {fp:.1f} / {fn:.1f} | {_f(pp)} / {_f(pr)} / {_f(pf)} | {_f(mi['mean'])} (n={mi['n_defined']}) |")
    w("\nSD is the sample SD (n−1) across trials. Pooled values sum TP, FP and FN over all trials.\n")

    w("### Per-trial results\n")
    w("| IoU ≥ | Trial | Boxes | TP | FP | FN | Precision | Recall | F1 | Mean IoU of TPs |")
    w("|---|---|---|---|---|---|---|---|---|---|")
    for r in res.trial_rows:
        w(f"| {r['iou_threshold']} | {r['trial']:02d} | {r['n_predictions']} | {r['true_positives']} | "
          f"{r['false_positives']} | {r['false_negatives']} | {_f(r['precision'])} | {_f(r['recall'])} | "
          f"{_f(r['f1'])} | {_f(r['mean_iou_tp'])} |")

    w("\n### How often each reference tree was found\n")
    labs = [ec.threshold_label(t) for t in thresholds]
    w(f"| Reference | Size (px) | " + " | ".join(f"Found at IoU ≥ {lab}" for lab in labs)
      + f" | When found at {labs[0]}: mean IoU · area ratio (Claude/human) · centre offset (px) "
        f"| Missed at {labs[0]}: nearest box in that trial |")
    w("|---|---|" + "---|" * len(labs) + "---|---|")
    by = {(r["reference_id"], r["iou_threshold"]): r for r in res.reference_rows}
    for rd in res.reference.detections:
        r50 = by[(rd.label, labs[0])]
        near = [m for m in r50["missed_trials_best_overlap"] if ":-@" not in m]
        none = len(r50["missed_trials_best_overlap"]) - len(near)
        missed = ", ".join(near + ([f"no overlapping box in {none} trial(s)"] if none else [])) or "–"
        found = (f"{_f(r50['mean_iou_when_matched'])} · {_f(r50['mean_area_ratio_pred_over_ref'], 2)} · "
                 f"{_f(r50['mean_center_offset_px'], 0)}" if r50["trials_matched"] else "–")
        w(f"| {rd.label} | {r50['width_px']:.0f} × {r50['height_px']:.0f} | "
          + " | ".join(f"{by[(rd.label, lab)]['trials_matched']}/{n_trials}" for lab in labs)
          + f" | {found} | {missed} |")

    shares = res.containment.get(labs[0], [])
    if shares:
        inside = sum(1 for s in shares if s >= 0.999)
        ratio = _stats(r["mean_area_ratio_pred_over_ref"] for r in res.reference_rows
                       if r["iou_threshold"] == labs[0] and r["trials_matched"])
        w(f"\n**Box extent.** Of the {len(shares)} matched pairs at IoU ≥ {labs[0]}, {inside} Claude boxes lie entirely "
          f"inside the human box (≥ 99.9% of their area). The mean share inside is {statistics.fmean(shares):.3f}. "
          f"Averaged over the found trees, a Claude box covers {_f(ratio['mean'], 2)} of the human box's area. "
          "When one box contains the other, IoU equals this area ratio. Claude's boxes are therefore systematically "
          "tighter than the human boxes, which by itself keeps most matches below IoU 0.75. This is a difference in "
          "box-extent convention, measured against this one annotator; it does not say which extent is correct.")
    w("\n### False positives\n")
    for t in thresholds[:2]:
        lab = ec.threshold_label(t)
        fps = [r for r in res.outcome_rows if r["iou_threshold"] == lab and r["outcome"] == "FP"]
        w(f"**IoU ≥ {lab}: {len(fps)} false-positive boxes in total.**\n")
        if not fps:
            continue
        groups = {}
        for r in fps:
            groups.setdefault((r["group_iou050"], r["fp_type"], r["best_reference"]), []).append(r)
        w("| Group (IoU 0.50) | Human label | FP type | Trials | Overlapping reference | Best IoU with it |")
        w("|---|---|---|---|---|---|")
        for (g, typ, ref), rows in sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[0])):
            labels = sorted({r["human_label"] for r in rows if r["human_label"]})
            ious = [float(r["best_iou"]) for r in rows]
            trials = ", ".join("%02d" % r["trial"] for r in rows)
            rng = f"{min(ious):.3f}" if min(ious) == max(ious) else f"{min(ious):.3f}–{max(ious):.3f}"
            w(f"| {g or '–'} | {', '.join(labels) or 'unlabelled'} | {typ} | {trials} | {ref or '–'} | {rng} |")
        w("")
    w("FP types: " + "; ".join(f"`{k}`: {v}" for k, v in FP_TYPES.items()) + ".\n")

    w("### Human-labelled review units against the reference\n")
    w("Matched through the recorded detection ids, not the group names.\n")
    w("| Review unit | Human label | Note | Detections | " + " | ".join(f"IoU ≥ {lab}: TP / FP (matched reference)" for lab in labs[:2]) + " |")
    w("|---|---|---|---|" + "---|" * 2)
    units = {}
    for r in res.outcome_rows:
        if r["review_unit_id"] and r["review_unit_id"] in labels_by_unit:
            units.setdefault(r["review_unit_id"], {}).setdefault(r["iou_threshold"], []).append(r)
    for uid in sorted(units, key=lambda x: x.rsplit(":", 1)[-1]):
        label, note = labels_by_unit[uid]
        per = units[uid]
        cells = []
        for lab in labs[:2]:
            rows = per.get(lab, [])
            tp = [r for r in rows if r["outcome"] == "TP"]
            refs = sorted({r["matched_reference"] for r in tp})
            fp_types = sorted({r["fp_type"] for r in rows if r["outcome"] == "FP"})
            cells.append(f"{len(tp)} / {len(rows) - len(tp)}" + (f" ({', '.join(refs)})" if refs else "")
                         + (f"; FP: {', '.join(fp_types)}" if fp_types else ""))
        n_det = len(per.get(labs[0], []))
        short = uid.rsplit(":", 1)[-1]
        w(f"| {short} | {label} | {note or ''} | {n_det} | " + " | ".join(cells) + " |")
    w("")
    return "\n".join(L)


LIMITATIONS = """## Limitations

* **One image.** These figures describe `LA_PT-R1_27m` at 26.8 m: a small site with 11 reference trees, mostly
  separate crowns. They do not generalise to denser canopy (PT-02), other altitudes or other models.
* **One annotator, one pass.** The reference is a single expert annotation with no inter-annotator agreement.
  Box extents for irregular crowns are a judgement, which matters most at IoU ≥ 0.75.
* **Coordinate grids differ.** Claude returns a 0–1000 grid (one step ≈ 8.2 × 6.1 original pixels). The human
  boxes use whole pixels. An IoU of exactly 1.00 is therefore practically unreachable. That threshold measures
  exact coordinate reproduction, not detection quality.
* **Matching is per trial.** A trial's box that partly covers a tree but falls below the threshold counts as both
  a false positive and a false negative for that tree. The FP-type column separates these localization errors
  from boxes on non-tree objects.
* **Shared errors.** All trials used the same prompt, settings and downscaled 2212 × 1659 input. Trial-to-trial
  spread therefore shows sampling variability only. It is not an independent estimate of accuracy.
"""
