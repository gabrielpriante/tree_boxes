"""Figures for evaluate_deepforest_repeatability.py (DeepForest repeated executions, lowalt_repro_v1)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import evaluation_core as ec  # noqa: E402
from evaluation_plots import MODEL_COLORS, finish  # noqa: E402

NOTE = ("Repeated executions of a deterministic program (one machine and software stack); not independent "
        "observations and not a larger sample. Agreement = 2 x matched / (nA + nB), exact-IoU one-to-one matching.")
DF_COLORS = {"M01": MODEL_COLORS[0], "M02v2": MODEL_COLORS[1]}
LLM_COLORS = {"M08": MODEL_COLORS[2], "M09": MODEL_COLORS[3]}


def _grid(combos):
    """Rows: configuration (M01, M02v2); columns: image, in a stable order."""
    models = [m for m in ("M01", "M02v2") if any(c["model"] == m for c in combos)]
    images = sorted({c["image_id"] for c in combos}, key=lambda i: (0 if "PT-R1" in i else 1, i))
    cell = {(c["model"], c["image_id"]): c for c in combos}
    return models, images, cell


def make_figures(combos, llm_rows, out: Path) -> list:
    fig_dir = out / "figures"
    paths = [counts_by_run(combos, fig_dir / "P_counts_by_run.png")]
    for t in (ec.THRESHOLDS[0], ec.THRESHOLDS[-1]):
        paths.append(pairwise_heatmaps(combos, t, fig_dir / f"Q_pairwise_agreement_{ec.threshold_tag(t)}.png"))
    paths.append(repeatability_vs_llm(combos, llm_rows, fig_dir / "R_repeatability_vs_llm.png"))
    return paths


def counts_by_run(combos, path):
    models, images, cell = _grid(combos)
    fig, axes = plt.subplots(len(models), len(images), figsize=(6.2 * len(images), 3.6 * len(models) + 1.2),
                             squeeze=False)
    fig.subplots_adjust(top=0.86, bottom=0.14, hspace=0.6, wspace=0.25)
    for r, m in enumerate(models):
        for col, image_id in enumerate(images):
            ax = axes[r][col]
            c = cell.get((m, image_id))
            if not c or not c.get("counts"):
                ax.set_axis_off()
                continue
            counts = c["counts"]
            x = np.arange(1, len(counts) + 1)
            ax.plot(x, counts, "o-", color=DF_COLORS[m], lw=1)
            lo, hi = min(counts), max(counts)
            pad = max(1, 0.02 * hi)
            ax.set_ylim(lo - pad, hi + pad)
            ax.set_xticks(x)
            ax.set_xlabel("run")
            ax.set_ylabel("detections")
            same = (f"{counts[0]} detections in every run" if len(set(counts)) == 1
                    else f"{lo}-{hi} detections, SD {np.std(counts, ddof=1):.2f}")
            exact = "boxes and scores identical" if c.get("all_canonical_identical") else "boxes or scores differ"
            ax.set_title(f"{m} · {image_id}\n{same}; {exact}", fontsize=9)
    fig.suptitle("P. DeepForest detections in each repeated run", fontsize=12)
    return finish(fig, path, NOTE)


def pairwise_heatmaps(combos, t, path):
    models, images, cell = _grid(combos)
    lab = ec.threshold_label(t)
    fig, axes = plt.subplots(len(models), len(images), figsize=(5.4 * len(images), 4.8 * len(models) + 1.0),
                             squeeze=False)
    fig.subplots_adjust(top=0.88, bottom=0.1, hspace=0.45, wspace=0.3)
    im = None
    for r, m in enumerate(models):
        for col, image_id in enumerate(images):
            ax = axes[r][col]
            c = cell.get((m, image_id))
            if not c or not c.get("pairs"):
                ax.set_axis_off()
                continue
            names = [x["run_dir"].name.split("_")[1] for x in c["runs"]]
            n = len(names)
            mat = np.eye(n)
            pos = {x["run_dir"].name: i for i, x in enumerate(c["runs"])}
            key = f"agreement_{ec.threshold_tag(t)}"
            for p in c["pairs"]:
                v = p[key] if p[key] is not None else np.nan
                mat[pos[p["run_a"]], pos[p["run_b"]]] = mat[pos[p["run_b"]], pos[p["run_a"]]] = float(v)
            im = ax.imshow(mat, vmin=0, vmax=1, cmap="viridis")
            ax.set_xticks(range(n), names, fontsize=7)
            ax.set_yticks(range(n), names, fontsize=7)
            off = mat[~np.eye(n, dtype=bool)]
            differing = sum(1 for p in c["pairs"] if not p["canonical_detections_identical"])
            ax.set_title(f"{m} · {image_id}\nmin {np.nanmin(off):.3f}, mean {np.nanmean(off):.3f}; "
                         f"{differing} of {len(c['pairs'])} pairs not identical", fontsize=9)
    if im is not None:
        fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.03, pad=0.02, label="agreement")
    fig.suptitle(f"Q. Pairwise agreement between repeated DeepForest runs at IoU {'≥ ' if t < 1 else '= '}{lab}",
                 fontsize=12)
    return finish(fig, path, NOTE + (" IoU = 1.00 requires identical boxes." if t == 1 else ""))


def repeatability_vs_llm(combos, llm_rows, path):
    images = sorted({c["image_id"] for c in combos}, key=lambda i: (0 if "PT-R1" in i else 1, i))
    thresholds = list(ec.THRESHOLDS)
    fig, axes = plt.subplots(1, len(images), figsize=(7.5 * len(images), 5.6), squeeze=False)
    fig.subplots_adjust(top=0.8, bottom=0.24, wspace=0.2)
    for ax, image_id in zip(axes[0], images):
        series = []
        for c in combos:
            if c["image_id"] == image_id and c.get("summaries"):
                s = c["summaries"]
                series.append((f"DeepForest {c['model']}\n{c['n_runs']} repeated executions", DF_COLORS[c["model"]], "",
                               [s[t]["agreement_mean"] or 0 for t in thresholds], [s[t]["agreement_sd"] or 0 for t in thresholds]))
        for model in ("M08", "M09"):
            per = {r["iou_threshold"]: r for r in llm_rows if r["image_id"] == image_id and r["model"] == model}
            if per:
                n = next(iter(per.values()))["n"]
                series.append((f"{model}\n{n} independent stochastic trials", LLM_COLORS[model], "//",
                               [per.get(ec.threshold_label(t), {}).get("agreement_mean") or 0 for t in thresholds],
                               [per.get(ec.threshold_label(t), {}).get("agreement_sd") or 0 for t in thresholds]))
        x = np.arange(len(thresholds))
        w = 0.8 / max(1, len(series))
        for i, (label, color, hatch, means, sds) in enumerate(series):
            xs = x + (i - (len(series) - 1) / 2) * w
            ax.bar(xs, means, w, yerr=sds, capsize=3, color=color, hatch=hatch, edgecolor="#333", label=label)
            for xi, v in zip(xs, means):
                ax.text(xi, v + 0.03, f"{v:.2f}", ha="center", fontsize=7)
        ax.set_xticks(x, [f"IoU ≥ {ec.threshold_label(t)}" if t < 1 else "IoU = 1.00" for t in thresholds])
        ax.set_ylim(0, 1.15)
        ax.set_ylabel("mean pairwise agreement within the configuration")
        ax.set_title(image_id, fontsize=11)
        ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2, frameon=False)
    fig.suptitle("R. Within-configuration repeatability: deterministic DeepForest runs vs stochastic LLM trials\n"
                 "(solid: repeated executions of one deterministic computation; hatched: frozen M08/M09 values)",
                 fontsize=11)
    return finish(fig, path, NOTE + " Consistency only, not accuracy. LLM values are read from the frozen M08/M09 "
                                   "evaluation and are not recomputed.")
