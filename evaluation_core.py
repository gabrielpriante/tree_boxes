"""
Model-agnostic metrics for repeated bounding-box detections on one image.

Nothing here reads files or knows which model produced a box, so the same functions can score
Claude, DeepForest, ChatGPT or Gemini outputs once each is loaded into DetectionSet objects.

Coordinates
    Every box is (x_min, y_min, x_max, y_max) in pixels of the ORIGINAL source image, held as exact
    fractions. Claude's 0-1000 integers are converted with x * width / 1000 and y * height / 1000,
    the same mapping used to draw the annotated images. IoU is computed exactly, so IoU = 1 holds
    only for identical geometry and threshold tests have no floating-point tolerance.

Two kinds of metric are kept apart
    * Repeatability (this module's consistency functions): do independent trials of one model on one
      image agree with each other? Says nothing about whether any box is a tree.
    * Accuracy (accuracy_vs_reference): agreement with a human reference annotation. Refuses to run
      unless the reference for that image is declared COMPLETE.
"""

from __future__ import annotations

import math
import statistics
from collections import Counter, deque
from dataclasses import dataclass
from fractions import Fraction
from itertools import combinations

import numpy as np
from scipy.optimize import linear_sum_assignment

ALGORITHM_VERSION = "1.0.0"
THRESHOLDS = (Fraction(1, 2), Fraction(3, 4), Fraction(1))
REFERENCE_COMPLETE = "COMPLETE"


class CrossImageComparisonError(ValueError):
    """Raised when boxes from different images would be compared as if they shared coordinates."""


class ReferenceNotAvailableError(RuntimeError):
    """Raised when accuracy metrics are requested without a complete human reference."""


def threshold_label(t) -> str:
    return f"{float(t):.2f}"


def threshold_tag(t) -> str:
    return f"iou{round(float(t) * 100):03d}"


# ------------------------------------------------------------------ data

@dataclass(frozen=True)
class Detection:
    uid: str      # stable identifier, unique across the whole evaluation
    label: str    # the producer's own id within its set, e.g. Claude's tree_id "T001"
    box: tuple    # (x_min, y_min, x_max, y_max) in original-image pixels, Fractions


@dataclass(frozen=True)
class DetectionSet:
    """All boxes from one trial of one producer on one image (or one human reference)."""
    source: str           # e.g. "lowalt_pilot_pt02_64m_v1/M08_high_a28fe01ac5" or "human_reference"
    image_id: str
    image_sha256: str
    width: int
    height: int
    trial: int | None
    detections: tuple

    @property
    def boxes(self) -> list:
        return [d.box for d in self.detections]


def box_from_normalized(bbox, width: int, height: int, scale: int = 1000) -> tuple:
    """0..scale integer box -> exact original-image pixel box."""
    if any(isinstance(v, bool) or not isinstance(v, int) for v in bbox):
        raise ValueError(f"normalized coordinates must be integers: {bbox}")
    x0, y0, x1, y1 = bbox
    box = (Fraction(x0 * width, scale), Fraction(y0 * height, scale),
           Fraction(x1 * width, scale), Fraction(y1 * height, scale))
    validate_box(box)
    return box


def box_from_pixels(values) -> tuple:
    box = tuple(Fraction(str(v)) for v in values)
    validate_box(box)
    return box


def validate_box(box) -> None:
    if len(box) != 4 or not (box[0] < box[2] and box[1] < box[3]):
        raise ValueError(f"box must have x_min < x_max and y_min < y_max: {box}")


# ------------------------------------------------------------------ IoU and matching

def area(b) -> Fraction:
    return (b[2] - b[0]) * (b[3] - b[1])


def iou(a, b) -> Fraction:
    iw = min(a[2], b[2]) - max(a[0], b[0])
    ih = min(a[3], b[3]) - max(a[1], b[1])
    if iw <= 0 or ih <= 0:
        return Fraction(0)
    inter = iw * ih
    return inter / (area(a) + area(b) - inter)


def match_boxes(boxes_a, boxes_b, threshold) -> list:
    """One-to-one assignment between two box lists.

    Returns sorted (i, j, iou) with iou >= threshold (and > 0). Maximizes the NUMBER of matches
    first and breaks ties by the largest total IoU: each eligible pair gets weight K + IoU with
    K = min(n, m) + 1, which exceeds any possible IoU sum, and an optimal assignment
    (scipy linear_sum_assignment) is found on that matrix. Ineligible pairs weigh 0 and are discarded.
    """
    threshold = Fraction(threshold)
    if not boxes_a or not boxes_b:
        return []
    candidates = _candidate_pairs(boxes_a, boxes_b, threshold)
    ious = {}
    for i, j in candidates:   # exact rational IoU for every pair the float prefilter could not rule out
        v = iou(boxes_a[i], boxes_b[j])
        if v > 0 and v >= threshold:
            ious[(i, j)] = v
    if not ious:
        return []
    k = min(len(boxes_a), len(boxes_b)) + 1
    weights = np.zeros((len(boxes_a), len(boxes_b)))
    for (i, j), v in ious.items():
        weights[i, j] = k + float(v)
    rows, cols = linear_sum_assignment(weights, maximize=True)
    return sorted((int(i), int(j), ious[(i, j)]) for i, j in zip(rows, cols) if (i, j) in ious)


PREFILTER_SLACK = 1e-6


def _candidate_pairs(boxes_a, boxes_b, threshold) -> list:
    """Index pairs whose float IoU is within PREFILTER_SLACK of the threshold or above it (and > 0).

    Only a speed-up: float IoU of these coordinates is accurate to far better than 1e-6, so no pair with
    exact IoU >= threshold is discarded, and eligibility is then decided with exact Fractions."""
    A = np.array([[float(v) for v in b] for b in boxes_a])
    B = np.array([[float(v) for v in b] for b in boxes_b])
    area_b = (B[:, 2] - B[:, 0]) * (B[:, 3] - B[:, 1])
    floor = float(threshold) - PREFILTER_SLACK
    out = []
    for start in range(0, len(A), 512):
        a = A[start:start + 512]
        iw = np.minimum(a[:, None, 2], B[None, :, 2]) - np.maximum(a[:, None, 0], B[None, :, 0])
        ih = np.minimum(a[:, None, 3], B[None, :, 3]) - np.maximum(a[:, None, 1], B[None, :, 1])
        inter = np.clip(iw, 0, None) * np.clip(ih, 0, None)
        area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
        union = area_a[:, None] + area_b[None, :] - inter
        ok = (inter > 0) & (inter >= floor * union)
        out += [(start + int(i), int(j)) for i, j in zip(*np.nonzero(ok))]
    return out


def require_same_image(a: DetectionSet, b: DetectionSet, alignment=None) -> None:
    """Boxes are only comparable inside one image's pixel grid. Different images (other altitudes,
    headings or flights) need a validated spatial alignment, which is not implemented."""
    if a.image_sha256 != b.image_sha256 or a.image_id != b.image_id:
        if alignment is None:
            raise CrossImageComparisonError(
                f"{a.image_id} and {b.image_id} are different images; their box coordinates do not "
                "describe the same ground locations. A validated spatial alignment is required.")
        raise NotImplementedError("Cross-image alignment is not implemented (see EVALUATION.md).")
    if (a.width, a.height) != (b.width, b.height):
        raise CrossImageComparisonError(f"{a.image_id}: inconsistent image dimensions between sets")


def _mean(values):
    return statistics.fmean(values) if values else None


def _median(values):
    return statistics.median(values) if values else None


def _sd(values):
    return statistics.stdev(values) if len(values) >= 2 else None


def compare_sets(a: DetectionSet, b: DetectionSet, threshold, alignment=None) -> dict:
    require_same_image(a, b, alignment)
    pairs = match_boxes(a.boxes, b.boxes, threshold)
    n_a, n_b, m = len(a.detections), len(b.detections), len(pairs)
    ious = [float(v) for _, _, v in pairs]
    return {
        "n_a": n_a, "n_b": n_b, "matched": m, "unmatched_a": n_a - m, "unmatched_b": n_b - m,
        # 2m / (n_a + n_b); undefined when both sets are empty
        "agreement": Fraction(2 * m, n_a + n_b) if n_a + n_b else None,
        "mean_iou": _mean(ious), "median_iou": _median(ious),
        "pairs": [(a.detections[i].uid, b.detections[j].uid, v) for i, j, v in pairs],
    }


def pairwise_trials(sets: list, threshold) -> list:
    """Every unordered trial pair (n(n-1)/2 of them), in trial order."""
    ordered = sorted(sets, key=lambda s: s.trial)
    out = []
    for a, b in combinations(ordered, 2):
        r = compare_sets(a, b, threshold)
        r.update(trial_a=a.trial, trial_b=b.trial)
        out.append(r)
    return out


def summarize_pairs(results: list) -> dict:
    agreements = [float(r["agreement"]) for r in results if r["agreement"] is not None]
    ious = [float(v) for r in results for _, _, v in r["pairs"]]
    return {
        "n_pairs": len(results),
        "n_pairs_undefined": sum(1 for r in results if r["agreement"] is None),
        "agreement_mean": _mean(agreements), "agreement_sd": _sd(agreements),
        "agreement_median": _median(agreements),
        "agreement_min": min(agreements) if agreements else None,
        "agreement_max": max(agreements) if agreements else None,
        "pairs_with_full_agreement": sum(1 for a in agreements if a == 1),
        "matched_boxes_total": len(ious),
        "matched_iou_mean": _mean(ious), "matched_iou_median": _median(ious),
        "matched_iou_min": min(ious) if ious else None,
    }


# ------------------------------------------------------------------ persistence

def persistence_bins(n_trials: int) -> list:
    """(label, lo, hi) bins by number of trials. For 10 trials: 10, 8-9, 5-7, 1-4."""
    hi80 = math.ceil(0.8 * n_trials)
    hi50 = math.ceil(0.5 * n_trials)
    bins = [(f"all {n_trials}", n_trials, n_trials),
            (f"{hi80}-{n_trials - 1}", hi80, n_trials - 1),
            (f"{hi50}-{hi80 - 1}", hi50, hi80 - 1),
            (f"1-{hi50 - 1}", 1, hi50 - 1)]
    return [b for b in bins if b[1] <= b[2]]


def persistence_bin(n: int, n_trials: int) -> str:
    for label, lo, hi in persistence_bins(n_trials):
        if lo <= n <= hi:
            return label
    raise ValueError(f"{n} is outside 1..{n_trials}")


@dataclass
class Group:
    group_id: str
    status: str              # CONSISTENT or AMBIGUOUS
    members: list            # [(trial, Detection)], sorted by trial
    missing_links: list      # member uid pairs that were NOT matched to each other
    duplicate_trials: list   # trials contributing more than one member

    @property
    def trials(self) -> list:
        return sorted({t for t, _ in self.members})

    @property
    def n_trials(self) -> int:
        return len(self.trials)


def persistence_groups(sets: list, pair_results: list, id_prefix: str = "") -> list:
    """Group detections that recur across trials, without transitive chaining.

    Edges are the one-to-one matches from every trial pair at one threshold. A connected component
    becomes a CONSISTENT group only if it has at most one detection per trial AND every two members
    were matched directly (the component is a clique). Otherwise (a chain such as A~B, B~C but A!~C,
    or two boxes from one trial) it is AMBIGUOUS: reported with its members and the reason, never
    split or merged automatically. Unmatched detections are consistent groups of one.
    """
    trial_of, det_of = {}, {}
    for s in sets:
        for d in s.detections:
            trial_of[d.uid], det_of[d.uid] = s.trial, d
    adj = {uid: set() for uid in trial_of}
    for r in pair_results:
        for ua, ub, _ in r["pairs"]:
            adj[ua].add(ub)
            adj[ub].add(ua)

    seen, comps = set(), []
    for uid in sorted(adj):
        if uid in seen:
            continue
        comp, queue = [], deque([uid])
        seen.add(uid)
        while queue:
            u = queue.popleft()
            comp.append(u)
            for v in sorted(adj[u]):
                if v not in seen:
                    seen.add(v)
                    queue.append(v)
        comps.append(sorted(comp, key=lambda u: (trial_of[u], u)))

    groups = []
    for comp in comps:
        counts = Counter(trial_of[u] for u in comp)
        dup = sorted(t for t, c in counts.items() if c > 1)
        missing = [(a, b) for a, b in combinations(comp, 2) if b not in adj[a]]
        status = "CONSISTENT" if not dup and not missing else "AMBIGUOUS"
        groups.append(Group("", status, [(trial_of[u], det_of[u]) for u in comp], missing, dup))

    def key(g):
        cx, cy = _median_center(g)
        return (g.status != "CONSISTENT", -g.n_trials, round(cy, 3), round(cx, 3), g.members[0][1].uid)

    groups.sort(key=key)
    n_cons = n_amb = 0
    for g in groups:
        if g.status == "CONSISTENT":
            n_cons += 1
            g.group_id = f"{id_prefix}G{n_cons:02d}"
        else:
            n_amb += 1
            g.group_id = f"{id_prefix}A{n_amb:02d}"
    return groups


def _median_center(g: Group) -> tuple:
    xs = [float(d.box[0] + d.box[2]) / 2 for _, d in g.members]
    ys = [float(d.box[1] + d.box[3]) / 2 for _, d in g.members]
    return statistics.median(xs), statistics.median(ys)


def group_geometry(g: Group, pair_iou: dict) -> dict:
    """Box variation inside a group, in original-image pixels. pair_iou maps frozenset({uid, uid}) -> IoU."""
    boxes = [[float(v) for v in d.box] for _, d in g.members]
    cx = [(b[0] + b[2]) / 2 for b in boxes]
    cy = [(b[1] + b[3]) / 2 for b in boxes]
    w = [b[2] - b[0] for b in boxes]
    h = [b[3] - b[1] for b in boxes]
    member_ious = [float(pair_iou[frozenset((a.uid, b.uid))])
                   for (_, a), (_, b) in combinations(g.members, 2)
                   if frozenset((a.uid, b.uid)) in pair_iou]
    median_box = [statistics.median(col) for col in zip(*boxes)]
    return {
        "median_box_px": median_box,
        "center_x_sd_px": _sd(cx), "center_y_sd_px": _sd(cy),
        "center_max_offset_px": max(math.dist((x, y), (statistics.median(cx), statistics.median(cy)))
                                    for x, y in zip(cx, cy)),
        "width_mean_px": _mean(w), "height_mean_px": _mean(h),
        "width_cv": (_sd(w) / _mean(w)) if len(w) >= 2 else None,
        "height_cv": (_sd(h) / _mean(h)) if len(h) >= 2 else None,
        "member_iou_mean": _mean(member_ious), "member_iou_min": min(member_ious) if member_ious else None,
        "identical_geometry": len({tuple(b) for b in boxes}) == 1,
    }


def detection_match_frequency(sets: list, pair_results: list) -> dict:
    """uid -> number of OTHER trials containing a one-to-one match for that detection."""
    freq = {d.uid: 0 for s in sets for d in s.detections}
    for r in pair_results:
        for ua, ub, _ in r["pairs"]:
            freq[ua] += 1
            freq[ub] += 1
    return freq


# ------------------------------------------------------------------ accuracy (needs a human reference)

def accuracy_vs_reference(pred: DetectionSet, reference: DetectionSet, reference_status: str,
                          threshold) -> dict:
    """Precision / recall / F1 of one prediction set against a human reference at one IoU threshold.
    Refuses to run unless the reference for this image is declared COMPLETE: a partial reference
    would turn unannotated real trees into false positives."""
    if reference_status != REFERENCE_COMPLETE:
        raise ReferenceNotAvailableError(
            f"{reference.image_id}: reference status is {reference_status!r}; accuracy metrics need a "
            f"{REFERENCE_COMPLETE} human reference annotation")
    r = compare_sets(pred, reference, threshold)
    tp, n_pred, n_ref = r["matched"], r["n_a"], r["n_b"]
    precision = tp / n_pred if n_pred else None
    recall = tp / n_ref if n_ref else None
    f1 = (2 * precision * recall / (precision + recall)) if precision and recall else (0.0 if n_pred and n_ref else None)
    return {"true_positives": tp, "false_positives": n_pred - tp, "false_negatives": n_ref - tp,
            "precision": precision, "recall": recall, "f1": f1, "mean_iou_tp": r["mean_iou"],
            "pairs": r["pairs"]}
