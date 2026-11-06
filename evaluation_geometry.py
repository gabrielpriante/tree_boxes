"""
Box geometry, coverage and fragmentation descriptors for evaluate_consistency.py.

Descriptive only: none of these quantities is accuracy. They make outputs with very different box counts
(a few LLM boxes vs thousands of DeepForest boxes) comparable on the same image: how large the boxes are,
how much of the image they cover, how much they overlap each other, and how many boxes of one producer
fall inside one box of another. Float arithmetic; coverage is measured on a raster at 1/COVERAGE_FACTOR of
the original resolution.
"""

from __future__ import annotations

import numpy as np

COVERAGE_FACTOR = 8          # 8192x6144 -> 1024x768 coverage raster
NEAR_DUPLICATE_IOU = 0.3     # within one output, two boxes with IoU >= this are counted as near-duplicates


def as_array(detections) -> np.ndarray:
    return np.array([[float(v) for v in d.box] for d in detections]) if detections else np.zeros((0, 4))


def coverage_mask(boxes: np.ndarray, width: int, height: int, factor: int = COVERAGE_FACTOR) -> np.ndarray:
    """Boolean raster: True where at least one box covers the cell centre."""
    mw, mh = -(-width // factor), -(-height // factor)
    mask = np.zeros((mh, mw), dtype=bool)
    cx = (np.arange(mw) + 0.5) * factor
    cy = (np.arange(mh) + 0.5) * factor
    for x0, y0, x1, y1 in boxes:
        i0, i1 = np.searchsorted(cy, y0), np.searchsorted(cy, y1)
        j0, j1 = np.searchsorted(cx, x0), np.searchsorted(cx, x1)
        mask[i0:i1, j0:j1] = True
    return mask


def coverage_count(boxes: np.ndarray, width: int, height: int, factor: int = COVERAGE_FACTOR) -> np.ndarray:
    """Number of boxes covering each raster cell."""
    mw, mh = -(-width // factor), -(-height // factor)
    cnt = np.zeros((mh, mw), dtype=np.int32)
    cx = (np.arange(mw) + 0.5) * factor
    cy = (np.arange(mh) + 0.5) * factor
    for x0, y0, x1, y1 in boxes:
        cnt[np.searchsorted(cy, y0):np.searchsorted(cy, y1), np.searchsorted(cx, x0):np.searchsorted(cx, x1)] += 1
    return cnt


def near_duplicates(boxes: np.ndarray, threshold: float = NEAR_DUPLICATE_IOU) -> tuple:
    """(pairs with IoU >= threshold, number of boxes in at least one such pair) within one output."""
    n = len(boxes)
    if n < 2:
        return 0, 0
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    involved = np.zeros(n, dtype=bool)
    pairs = 0
    for s in range(0, n, 512):
        a = boxes[s:s + 512]
        iw = np.minimum(a[:, None, 2], boxes[None, :, 2]) - np.maximum(a[:, None, 0], boxes[None, :, 0])
        ih = np.minimum(a[:, None, 3], boxes[None, :, 3]) - np.maximum(a[:, None, 1], boxes[None, :, 1])
        inter = np.clip(iw, 0, None) * np.clip(ih, 0, None)
        iou = inter / (area[s:s + 512, None] + area[None, :] - inter)
        idx = np.arange(s, s + len(a))
        upper = np.arange(n)[None, :] > idx[:, None]          # count each pair once
        hit = (iou >= threshold) & upper
        pairs += int(hit.sum())
        rows, cols = np.nonzero(hit)
        involved[idx[rows]] = True
        involved[cols] = True
    return pairs, int(involved.sum())


def _pct(v, q):
    return float(np.percentile(v, q)) if len(v) else None


def set_geometry(detections, width: int, height: int, gsd_m: float | None) -> dict:
    """Descriptors of one output (one trial, or one DeepForest run) on one image."""
    b = as_array(detections)
    n = len(b)
    w, h = (b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]) if n else (np.zeros(0), np.zeros(0))
    side = np.sqrt(w * h)
    mask = coverage_mask(b, width, height)
    covered = float(mask.mean())
    box_area_sum = float((w * h).sum()) / (width * height)
    pairs, involved = near_duplicates(b)
    image_m2 = width * height * gsd_m ** 2 if gsd_m else None
    out = {"n_boxes": n,
           "box_width_px_median": _pct(w, 50), "box_height_px_median": _pct(h, 50),
           "box_side_px_p10": _pct(side, 10), "box_side_px_median": _pct(side, 50), "box_side_px_p90": _pct(side, 90),
           "box_side_m_p10": _pct(side, 10) * gsd_m if n and gsd_m else None,
           "box_side_m_median": _pct(side, 50) * gsd_m if n and gsd_m else None,
           "box_side_m_p90": _pct(side, 90) * gsd_m if n and gsd_m else None,
           "coverage_fraction": covered,
           "summed_box_area_fraction": box_area_sum,
           "overlap_ratio_summed_over_union": box_area_sum / covered if covered else None,
           "near_duplicate_pairs_iou030": pairs,
           "boxes_in_near_duplicate_pairs_fraction": involved / n if n else None,
           "boxes_per_100_m2": n / image_m2 * 100 if image_m2 else None}
    return out


def fragmentation(fine, coarse) -> dict:
    """How many boxes of `fine` (e.g. DeepForest) have their centre inside each box of `coarse` (e.g. one LLM
    trial), and what share of `fine` centres lies inside any `coarse` box."""
    f, c = as_array(fine), as_array(coarse)
    if not len(c) or not len(f):
        return {"coarse_boxes": len(c), "fine_boxes": len(f), "fine_centres_per_coarse_box_mean": None,
                "coarse_boxes_with_0_fine": None, "coarse_boxes_with_2plus_fine": None,
                "fine_centres_inside_any_coarse_fraction": None}
    cx, cy = (f[:, 0] + f[:, 2]) / 2, (f[:, 1] + f[:, 3]) / 2
    inside = ((cx[None, :] >= c[:, None, 0]) & (cx[None, :] < c[:, None, 2])
              & (cy[None, :] >= c[:, None, 1]) & (cy[None, :] < c[:, None, 3]))
    per = inside.sum(axis=1)
    return {"coarse_boxes": len(c), "fine_boxes": len(f), "fine_centres_per_coarse_box_mean": float(per.mean()),
            "coarse_boxes_with_0_fine": int((per == 0).sum()), "coarse_boxes_with_2plus_fine": int((per >= 2).sum()),
            "fine_centres_inside_any_coarse_fraction": float(inside.any(axis=0).mean())}


def mask_iou(a, b, width: int, height: int) -> float | None:
    ma, mb = coverage_mask(as_array(a), width, height), coverage_mask(as_array(b), width, height)
    union = (ma | mb).sum()
    return float((ma & mb).sum() / union) if union else None
