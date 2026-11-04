"""
Figures, review crops and contact sheets for evaluate_consistency.py. Everything is drawn on the
original full-resolution source image; existing annotated images are never touched.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

import evaluation_core as ec  # noqa: E402

BIN_COLORS = ["#1a9850", "#91cf60", "#fdae61", "#d73027"]  # all, >=80%, >=50%, <50% of trials
AMBIGUOUS_COLOR = "#e7298a"
TRIAL_COLORS = [tuple(int(255 * c) for c in plt.get_cmap("tab10")(i)[:3]) for i in range(10)]
AGREEMENT_DEF = "Pairwise agreement = 2 × matched boxes / (boxes in trial A + boxes in trial B); one-to-one matching."
REPEATABILITY_NOTE = "Repeatability between trials only: not precision, recall or accuracy."
PLOT_WIDTH = 2048
MAX_TILES = 36


def font(size: int):
    for name in ("DejaVuSans-Bold.ttf", "arialbd.ttf", "arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default(size=size)


def hex_rgb(h: str) -> tuple:
    return tuple(int(h[i:i + 2], 16) for i in (1, 3, 5))


def trial_color(trial: int) -> tuple:
    return TRIAL_COLORS[(trial - 1) % len(TRIAL_COLORS)]


def header(u) -> str:
    return f"{u.site} · {u.image_id} · {u.model} ({u.model_config_id}) · {u.n_trials} trials · {u.experiment_id}"


def finish(fig, path: Path, footnote: str) -> Path:
    fig.text(0.01, 0.005, footnote, fontsize=8, color="#444", ha="left", va="bottom", wrap=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130, metadata={"Software": None})
    plt.close(fig)
    return path


def bin_index(n: int, n_trials: int) -> int:
    labels = [b[0] for b in ec.persistence_bins(n_trials)]
    return labels.index(ec.persistence_bin(n, n_trials))


def crop_region(boxes, width, height, margin_frac=0.35, min_margin=100):
    x0 = min(float(b[0]) for b in boxes)
    y0 = min(float(b[1]) for b in boxes)
    x1 = max(float(b[2]) for b in boxes)
    y1 = max(float(b[3]) for b in boxes)
    m = max(min_margin, margin_frac * max(x1 - x0, y1 - y0))
    return (max(0, int(x0 - m)), max(0, int(y0 - m)), min(width, int(x1 + m + 1)), min(height, int(y1 + m + 1)))


# ------------------------------------------------------------------ per-image figures

def make_figures(u, results: dict, stats: dict, out: Path, full_res_maps: bool = True) -> list:
    fig_dir = out / "figures" / u.experiment_id / u.image_id
    with Image.open(u.image_path) as im:
        full = im.convert("RGB")
    scale = PLOT_WIDTH / full.width
    thumb = np.asarray(full.resize((PLOT_WIDTH, round(full.height * scale)), Image.LANCZOS))
    paths = [
        threshold_comparison(u, results, fig_dir / "A_threshold_comparison.png"),
        persistence_overview(u, results, thumb, fig_dir / "B_persistence_overview.png"),
    ]
    if full_res_maps:
        for t, tr in results.items():
            paths.append(persistence_map_full(u, tr, full, fig_dir / f"B_persistence_map_{ec.threshold_tag(t)}.jpg"))
    paths += [
        box_variation(u, results[ec.THRESHOLDS[0]], full, fig_dir / "C_box_variation_iou050.png"),
        agreement_matrix(u, results, fig_dir / "D_agreement_matrix.png"),
        count_consistency(u, stats, fig_dir / "E_count_consistency.png"),
    ]
    return paths


def threshold_comparison(u, results, path):
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13, 5.8))
    fig.subplots_adjust(top=0.84, bottom=0.17, wspace=0.28)
    labels, data = [], []
    for t, tr in results.items():
        labels.append(f"IoU ≥ {ec.threshold_label(t)}" if t < 1 else "IoU = 1.00\n(identical boxes)")
        data.append([float(r["agreement"]) for r in tr.pairs if r["agreement"] is not None])
    rng = np.random.default_rng(0)
    ax.boxplot(data, positions=range(len(data)), widths=0.5, showfliers=False)
    for i, vals in enumerate(data):
        ax.scatter(i + rng.uniform(-0.15, 0.15, len(vals)), vals, s=14, alpha=0.6, color="#2c7fb8", zorder=3)
        if vals:
            ax.text(i, 1.06, f"mean {np.mean(vals):.3f}\nSD {np.std(vals, ddof=1) if len(vals) > 1 else 0:.3f}",
                    ha="center", fontsize=8)
    ax.set_xticks(range(len(labels)), labels)
    ax.set_ylim(-0.03, 1.16)
    ax.set_ylabel("pairwise agreement")
    ax.set_title(f"Agreement of all {len(data[0])} trial pairs", fontsize=10)

    mean_count = np.mean(u.counts)
    matched = [np.mean([r["matched"] for r in tr.pairs]) for tr in results.values()]
    mi = [tr.summary["matched_iou_mean"] for tr in results.values()]
    bars = ax2.bar(range(len(matched)), matched, color="#7fcdbb", edgecolor="#2c7fb8")
    ax2.axhline(mean_count, color="#555", ls="--", lw=1)
    ax2.text(len(matched) - 0.5, mean_count, f" mean boxes per trial {mean_count:.1f}", va="bottom", ha="right", fontsize=8)
    for b, m, v in zip(bars, matched, mi):
        ax2.text(b.get_x() + b.get_width() / 2, m, f"{m:.1f}\nmatched IoU\n{'–' if v is None else f'{v:.3f}'}",
                 ha="center", va="bottom", fontsize=8)
    ax2.set_xticks(range(len(labels)), labels)
    ax2.set_ylabel("mean matched boxes per trial pair")
    ax2.set_ylim(0, max(max(matched), mean_count) * 1.35 + 1)
    ax2.set_title("Matched boxes per pair and their mean IoU", fontsize=10)
    fig.suptitle(f"A. Spatial agreement by IoU threshold\n{header(u)}", fontsize=11)
    return finish(fig, path, f"{AGREEMENT_DEF} {REPEATABILITY_NOTE}")


def _legend_handles(n_trials):
    bins = ec.persistence_bins(n_trials)
    return ([Patch(edgecolor=BIN_COLORS[i], facecolor="none", lw=2, label=f"found in {lab} trials")
             for i, (lab, _, _) in enumerate(bins)]
            + [Patch(edgecolor=AMBIGUOUS_COLOR, facecolor="none", lw=1.5, ls="--", label="ambiguous group (each member box)")])


def persistence_overview(u, results, thumb, path):
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 6.2))
    fig.subplots_adjust(top=0.86, bottom=0.12, left=0.01, right=0.99, wspace=0.03)
    s = thumb.shape[1] / u.width
    for ax, (t, tr) in zip(np.atleast_1d(axes), results.items()):
        ax.imshow(thumb)
        ax.set_axis_off()
        cons = [g for g in tr.groups if g.status == "CONSISTENT"]
        amb = [g for g in tr.groups if g.status == "AMBIGUOUS"]
        for g in amb:
            for _, d in g.members:
                b = [float(v) * s for v in d.box]
                ax.add_patch(Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], fill=False, lw=0.8, ls="--",
                                       edgecolor=AMBIGUOUS_COLOR))
        for g in cons:
            b = [v * s for v in tr.geometry[g.group_id]["median_box_px"]]
            ax.add_patch(Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], fill=False, lw=1.6,
                                   edgecolor=BIN_COLORS[bin_index(g.n_trials, u.n_trials)]))
        ax.set_title(f"IoU {'≥' if t < 1 else '='} {ec.threshold_label(t)}: {len(cons)} groups, {len(amb)} ambiguous",
                     fontsize=10)
    fig.legend(handles=_legend_handles(u.n_trials), loc="lower center", ncol=5, fontsize=9, frameon=False,
               bbox_to_anchor=(0.5, 0.035))
    fig.suptitle(f"B. Detection persistence across trials\n{header(u)}", fontsize=11)
    return finish(fig, path, "Consistent groups are drawn as their median box, coloured by the number of trials that "
                             "detected them. " + REPEATABILITY_NOTE)


def persistence_map_full(u, tr, full: Image.Image, path: Path) -> Path:
    lw = max(3, round(min(full.size) / 320))
    f = font(max(14, round(min(full.size) / 70)))
    probe = ImageDraw.Draw(full)
    lh = round(probe.textbbox((0, 0), "Ag", font=f)[3] * 1.5)
    lines = [(f"{u.site} {u.image_id} | {u.model} ({u.model_config_id}) | {u.n_trials} trials | IoU "
              f"{'>=' if tr.threshold < 1 else '='} {ec.threshold_label(tr.threshold)} | "
              "repeatability between trials, not accuracy", (255, 255, 255))]
    lines += [(f"found in {lab} trials (median box)", hex_rgb(BIN_COLORS[i]))
              for i, (lab, _, _) in enumerate(ec.persistence_bins(u.n_trials))]
    lines.append(("ambiguous group: every member box", hex_rgb(AMBIGUOUS_COLOR)))
    band = lh * len(lines) + 2 * lw
    img = Image.new("RGB", (full.width, full.height + band), (0, 0, 0))
    img.paste(full, (0, 0))
    draw = ImageDraw.Draw(img)
    for g in tr.groups:
        if g.status == "AMBIGUOUS":
            for _, d in g.members:
                draw.rectangle([float(v) for v in d.box], outline=hex_rgb(AMBIGUOUS_COLOR), width=max(2, lw // 2))
    for g in tr.groups:
        if g.status != "CONSISTENT":
            continue
        b = tr.geometry[g.group_id]["median_box_px"]
        color = hex_rgb(BIN_COLORS[bin_index(g.n_trials, u.n_trials)])
        draw.rectangle(b, outline=color, width=lw)
        label = f"{g.group_id} {g.n_trials}/{u.n_trials}"
        tb = draw.textbbox((0, 0), label, font=f)
        draw.rectangle([b[0], b[1], b[0] + tb[2] + 2 * lw, b[1] + tb[3] + 2 * lw], fill=color)
        draw.text((b[0] + lw, b[1] + lw), label, fill=(0, 0, 0), font=f)
    for i, (text, color) in enumerate(lines):
        draw.text((2 * lw, full.height + lw + i * lh), text, fill=color, font=f)
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, quality=85)
    return path


def box_variation(u, tr, full: Image.Image, path: Path) -> Path:
    groups = [g for g in tr.groups if g.status == "CONSISTENT" and g.n_trials >= 2]
    groups += [g for g in tr.groups if g.status == "AMBIGUOUS"]
    shown = groups[:MAX_TILES]
    cols = 6
    rows = max(1, -(-len(shown) // cols))
    height = 3.4 * rows + 1.8
    fig, axes = plt.subplots(rows, cols, figsize=(3.1 * cols, height), squeeze=False)
    fig.subplots_adjust(top=1 - 1.1 / height, bottom=1.0 / height, hspace=0.45, wspace=0.08)
    for ax in axes.flat:
        ax.set_axis_off()
    for ax, g in zip(axes.flat, shown):
        boxes = [d.box for _, d in g.members]
        x0, y0, x1, y1 = crop_region(boxes, u.width, u.height)
        crop = full.crop((x0, y0, x1, y1))
        crop.thumbnail((500, 500))
        ax.imshow(np.asarray(crop), extent=(x0, x1, y1, y0))
        for trial, d in g.members:
            b = [float(v) for v in d.box]
            ax.add_patch(Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], fill=False, lw=1.1,
                                   edgecolor=np.array(trial_color(trial)) / 255))
        geo = tr.geometry[g.group_id]
        m = geo["median_box_px"]
        ax.add_patch(Rectangle((m[0], m[1]), m[2] - m[0], m[3] - m[1], fill=False, lw=1, ls=":", edgecolor="white"))
        ax.set_xlim(x0, x1)
        ax.set_ylim(y1, y0)
        sd = (geo["center_x_sd_px"] or 0, geo["center_y_sd_px"] or 0)
        mi = geo["member_iou_min"]
        title = (f"{g.group_id} · {g.n_trials}/{u.n_trials} trials\ncentre SD {sd[0]:.0f}/{sd[1]:.0f} px · "
                 f"min IoU {'–' if mi is None else f'{mi:.2f}'}")
        ax.set_title(title, fontsize=8, color=AMBIGUOUS_COLOR if g.status == "AMBIGUOUS" else "black")
    handles = [Patch(edgecolor=np.array(trial_color(t)) / 255, facecolor="none", label=f"trial {t:02d}")
               for t in sorted({t for g in shown for t, _ in g.members})]
    fig.legend(handles=handles, loc="lower center", ncol=10, fontsize=8, frameon=False,
               bbox_to_anchor=(0.5, 0.3 / height))
    more = f" Showing {len(shown)} of {len(groups)}." if len(groups) > len(shown) else ""
    fig.suptitle(f"C. Box variation within recurring detections (groups at IoU ≥ 0.50)\n{header(u)}", fontsize=11)
    return finish(fig, path, "Each tile: every trial's box for one group (colour = trial), dotted white = median box. "
                             "Pink titles are ambiguous groups (not all members matched each other). "
                             "Centre SD and IoU in original-image pixels." + more)


def agreement_matrix(u, results, path):
    n = len(results)
    trials = sorted(s.trial for s in u.sets)
    idx = {t: i for i, t in enumerate(trials)}
    fig, axes = plt.subplots(1, n, figsize=(6.2 * n, 6.4))
    fig.subplots_adjust(top=0.82, bottom=0.14, wspace=0.25)
    im = None
    for ax, (t, tr) in zip(np.atleast_1d(axes), results.items()):
        m = np.full((len(trials), len(trials)), np.nan)
        for r in tr.pairs:
            v = np.nan if r["agreement"] is None else float(r["agreement"])
            m[idx[r["trial_a"]], idx[r["trial_b"]]] = m[idx[r["trial_b"]], idx[r["trial_a"]]] = v
        im = ax.imshow(m, vmin=0, vmax=1, cmap="viridis")
        for i in range(len(trials)):
            ax.add_patch(Rectangle((i - 0.5, i - 0.5), 1, 1, color="#ddd"))
            for j in range(len(trials)):
                if i != j and not np.isnan(m[i, j]):
                    ax.text(j, i, f"{m[i, j]:.2f}", ha="center", va="center", fontsize=6.5,
                            color="white" if m[i, j] < 0.6 else "black")
        ax.set_xticks(range(len(trials)), [f"t{t:02d}" for t in trials], fontsize=7, rotation=90)
        ax.set_yticks(range(len(trials)), [f"t{t:02d}" for t in trials], fontsize=7)
        ax.set_title(f"IoU {'≥' if t < 1 else '='} {ec.threshold_label(t)}", fontsize=10)
    fig.colorbar(im, ax=axes, shrink=0.75, label="pairwise agreement")
    fig.suptitle(f"D. Trial-by-trial agreement matrix\n{header(u)}", fontsize=11)
    return finish(fig, path, f"{AGREEMENT_DEF} Diagonal (a trial with itself) left blank. {REPEATABILITY_NOTE}")


def count_consistency(u, stats, path):
    trials = sorted(s.trial for s in u.sets)
    counts = u.counts
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.subplots_adjust(top=0.8, bottom=0.17)
    mean, sd = stats["mean"], stats["sd"] or 0
    ax.axhspan(mean - sd, mean + sd, color="#9ecae1", alpha=0.4, label=f"mean ± SD ({mean:.2f} ± {sd:.3f})")
    ax.axhline(mean, color="#3182bd", lw=1.2)
    ax.plot(trials, counts, "o-", color="#08519c", label="boxes returned")
    for t, c in zip(trials, counts):
        ax.text(t, c + 0.15, str(c), ha="center", va="bottom", fontsize=8)
    ax.set_xticks(trials, [f"t{t:02d}" for t in trials])
    lo, hi = min(counts), max(counts)
    ax.set_ylim(max(0, lo - 3), hi + 3)
    ax.yaxis.get_major_locator().set_params(integer=True)
    ax.set_xlabel("trial")
    ax.set_ylabel("number of boxes")
    ax.legend(fontsize=8, loc="upper right")
    cv = stats["cv"]
    ax.set_title(f"mean {mean:.2f} · SD {sd:.3f} (n−1) · CV {'–' if cv is None else f'{cv:.1%}'} · range {lo}–{hi}",
                 fontsize=10)
    fig.suptitle(f"E. Count consistency across trials\n{header(u)}", fontsize=11)
    return finish(fig, path, "Counts only: equal counts do not mean the same objects were boxed. " + REPEATABILITY_NOTE)


# ------------------------------------------------------------------ review crops and contact sheets

def make_review_images(unit_crops: dict, out: Path) -> list:
    by_image = {}
    for uid, (u, g, d, crop) in unit_crops.items():
        by_image.setdefault(u.key, []).append((uid, u, g, d, crop))
    sheets = []
    for key, items in by_image.items():
        u = items[0][1]
        with Image.open(u.image_path) as im:
            full = im.convert("RGB")
        tiles = []
        for uid, u, g, d, crop in items:
            boxes = [m.box for _, m in g.members]
            region = crop_region([d.box] if d is not None else boxes, u.width, u.height, 0.6, 150)
            img = full.crop(region)
            draw = ImageDraw.Draw(img)
            lw = max(2, round(max(img.size) / 200))
            ox, oy = region[0], region[1]
            for trial, m in g.members:
                b = [float(m.box[0]) - ox, float(m.box[1]) - oy, float(m.box[2]) - ox, float(m.box[3]) - oy]
                if d is None:
                    draw.rectangle(b, outline=trial_color(trial), width=lw)
                elif m.uid != d.uid:
                    draw.rectangle(b, outline=(170, 170, 170), width=max(1, lw // 2))
            if d is not None:
                b = [float(d.box[0]) - ox, float(d.box[1]) - oy, float(d.box[2]) - ox, float(d.box[3]) - oy]
                draw.rectangle(b, outline=hex_rgb(AMBIGUOUS_COLOR), width=lw + 1)
            if max(img.size) > 1200:
                img.thumbnail((1200, 1200))
            elif max(img.size) < 600:
                f = 600 / max(img.size)
                img = img.resize((round(img.width * f), round(img.height * f)), Image.LANCZOS)
            p = out / crop
            p.parent.mkdir(parents=True, exist_ok=True)
            img.save(p, quality=90)
            short = uid.split(f"{u.image_id}:", 1)[1]
            tiles.append((img, short, f"{g.n_trials}/{u.n_trials} trials" + ("" if g.status == "CONSISTENT" else " · ambiguous")))
        sheets.append(contact_sheet(
            f"Review units: {header(u)}",
            "Group crops: box colour = trial. Ambiguous-group crops: pink = the unit under review, grey = other members. "
            "Label in human_review/manual_review.csv.",
            tiles, out / "review" / f"contact_sheet_{u.experiment_id}_{u.image_id}.png"))
    return sheets


def contact_sheet(title: str, subtitle: str, tiles, path: Path, tile=300, cols=6) -> Path:
    cap = 46
    head = 70
    rows = max(1, -(-len(tiles) // cols))
    sheet = Image.new("RGB", (cols * tile, head + rows * (tile + cap)), (255, 255, 255))
    draw = ImageDraw.Draw(sheet)
    draw.text((10, 8), title, fill=(0, 0, 0), font=font(18))
    draw.text((10, 36), subtitle, fill=(60, 60, 60), font=font(13))
    f1, f2 = font(14), font(12)
    for i, (img, title, sub) in enumerate(tiles):
        x, y = (i % cols) * tile, head + (i // cols) * (tile + cap)
        t = img.copy()
        t.thumbnail((tile - 8, tile - 8))
        sheet.paste(t, (x + (tile - t.width) // 2, y + 4 + (tile - 8 - t.height) // 2))
        draw.text((x + 6, y + tile), title, fill=(0, 0, 0), font=f1)
        draw.text((x + 6, y + tile + 20), sub, fill=(90, 90, 90), font=f2)
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path, optimize=False)
    return path


# ------------------------------------------------------------------ human reference (no model predictions)

REFERENCE_COLOR = (0, 229, 255)


def reference_images(image_path: Path, image_id: str, rows: list, out: Path, status: str = "not approved") -> list:
    """Full-resolution map and contact sheet of human reference boxes only (no model output)."""
    boxes = [(r["reference_id"], [float(r[c]) for c in ("x_min_px", "y_min_px", "x_max_px", "y_max_px")]) for r in rows]
    with Image.open(image_path) as im:
        full = im.convert("RGB")
    lw = max(3, round(min(full.size) / 320))
    f = font(max(14, round(min(full.size) / 70)))
    probe = ImageDraw.Draw(full)
    lh = round(probe.textbbox((0, 0), "Ag", font=f)[3] * 1.5)
    annotators = sorted({r.get("annotator") or "?" for r in rows})
    lines = [f"HUMAN REFERENCE ONLY | {image_id} | {len(boxes)} boxes | annotator {', '.join(annotators)} | "
             f"completeness: {status}",
             "Original-image pixels as annotated; no model predictions shown."]
    img = Image.new("RGB", (full.width, full.height + lh * len(lines) + 2 * lw), (0, 0, 0))
    img.paste(full, (0, 0))
    draw = ImageDraw.Draw(img)
    for rid, b in boxes:
        draw.rectangle(b, outline=REFERENCE_COLOR, width=lw)
        tb = draw.textbbox((0, 0), rid, font=f)
        draw.rectangle([b[0], b[1], b[0] + tb[2] + 2 * lw, b[1] + tb[3] + 2 * lw], fill=REFERENCE_COLOR)
        draw.text((b[0] + lw, b[1] + lw), rid, fill=(0, 0, 0), font=f)
    for i, text in enumerate(lines):
        draw.text((2 * lw, full.height + lw + i * lh), text, fill=(255, 255, 255), font=f)
    review = out / "review"
    review.mkdir(parents=True, exist_ok=True)
    map_path = review / f"human_reference_{image_id}.jpg"
    img.save(map_path, quality=85)

    tiles = []
    for rid, b in boxes:
        region = crop_region([b], full.width, full.height, 0.6, 150)
        crop = full.crop(region)
        d = ImageDraw.Draw(crop)
        clw = max(2, round(max(crop.size) / 200))
        d.rectangle([b[0] - region[0], b[1] - region[1], b[2] - region[0], b[3] - region[1]],
                    outline=REFERENCE_COLOR, width=clw)
        tiles.append((crop, rid, f"{b[2] - b[0]:.0f} x {b[3] - b[1]:.0f} px"))
    sheet = contact_sheet(f"Human reference boxes: {image_id} ({len(boxes)} boxes; completeness: {status})",
                          "Each crop shows one human-drawn box (cyan) with surrounding context. No model predictions.",
                          tiles, review / f"human_reference_contact_sheet_{image_id}.png")
    return [map_path, sheet]


# ------------------------------------------------------------------ accuracy against a COMPLETE human reference

OUTCOME_COLORS = {"TP": "#1a9850", "localization_below_threshold": "#fdae61",
                  "reference_matched_by_other_detection": "#ffd92f", "no_reference_overlap": "#d73027"}
OUTCOME_LABELS = {"TP": "true positive", "localization_below_threshold": "FP: overlaps a tree, IoU below threshold",
                  "reference_matched_by_other_detection": "FP: tree already matched by another box",
                  "no_reference_overlap": "FP: no reference tree"}
REF_FOUND_COLOR = "#00e5ff"
MISSED_COLOR = "#e7298a"
ACCURACY_NOTE = "Accuracy against the human reference (COMPLETE). Each trial is matched one-to-one on its own."


def _outcome_key(r):
    return "TP" if r["outcome"] == "TP" else r["fp_type"]


def _n_trials(acc):
    return len(next(iter(acc.per_trial.values())))


def _ref_found(acc, t):
    lab = ec.threshold_label(t)
    return {r["reference_id"]: r["trials_matched"] for r in acc.reference_rows if r["iou_threshold"] == lab}


def _thr_text(t):
    return f"IoU ≥ {ec.threshold_label(t)}" if t < 1 else "IoU = 1.00"


def accuracy_figures(u, acc, out: Path) -> list:
    fig_dir = out / "figures" / u.experiment_id / u.image_id
    thresholds = list(acc.per_trial)
    with Image.open(u.image_path) as im:
        full = im.convert("RGB")
    first = thresholds[0]
    return [accuracy_by_threshold(u, acc, thresholds, fig_dir / "F_accuracy_by_threshold.png"),
            trial_outcomes(u, acc, thresholds, fig_dir / "G_trial_tp_fp_fn.png"),
            reference_frequency(u, acc, thresholds, fig_dir / "H_reference_detection_frequency.png"),
            reference_vs_predictions_overview(u, acc, thresholds[:2], full,
                                              fig_dir / "I_reference_vs_predictions_overview.png"),
            reference_vs_predictions_full(u, acc, first, full,
                                          fig_dir / f"I_reference_vs_predictions_{ec.threshold_tag(first)}.jpg"),
            error_cases(u, acc, first, full, fig_dir / f"J_error_cases_{ec.threshold_tag(first)}.png")]


def accuracy_by_threshold(u, acc, thresholds, path):
    fig, ax = plt.subplots(figsize=(11, 5.6))
    fig.subplots_adjust(top=0.82, bottom=0.16)
    metrics = [("precision", "#2c7fb8"), ("recall", "#d95f02"), ("f1", "#1b9e77")]
    rng = np.random.default_rng(0)
    width = 0.25
    for k, (m, color) in enumerate(metrics):
        for i, t in enumerate(thresholds):
            vals = [a[m] for a in acc.per_trial[t].values() if a[m] is not None]
            if not vals:
                continue
            x = i + (k - 1) * width
            mean, sd = np.mean(vals), (np.std(vals, ddof=1) if len(vals) > 1 else 0.0)
            ax.bar(x, mean, width * 0.9, color=color, alpha=0.35, label=m if i == 0 else None)
            ax.errorbar(x, mean, yerr=sd, color=color, capsize=4)
            ax.scatter(x + rng.uniform(-0.06, 0.06, len(vals)), vals, s=12, color=color, zorder=3)
            ax.text(x, min(1.04, mean + sd + 0.03), f"{mean:.2f}", ha="center", fontsize=8)
    ax.set_xticks(range(len(thresholds)), [_thr_text(t) for t in thresholds])
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("value per trial")
    ax.legend(fontsize=9, loc="upper right")
    fig.suptitle(f"F. Precision, recall and F1 against the human reference ({len(acc.reference.detections)} trees)\n"
                 f"{header(u)}", fontsize=11)
    return finish(fig, path, "Bars: mean over trials; whiskers: SD (n−1); dots: individual trials. " + ACCURACY_NOTE)


def trial_outcomes(u, acc, thresholds, path):
    fig, axes = plt.subplots(1, len(thresholds), figsize=(6 * len(thresholds), 5), sharey=True, squeeze=False)
    fig.subplots_adjust(top=0.8, bottom=0.18, wspace=0.08)
    for ax, t in zip(axes[0], thresholds):
        trials = sorted(acc.per_trial[t])
        x = np.arange(len(trials))
        for off, key, color, name in ((-0.27, "true_positives", "#1a9850", "TP"),
                                      (0.0, "false_positives", "#d73027", "FP"),
                                      (0.27, "false_negatives", "#969696", "FN")):
            ax.bar(x + off, [acc.per_trial[t][k][key] for k in trials], 0.27, color=color, label=name)
        ax.set_xticks(x, [f"t{k:02d}" for k in trials], fontsize=8, rotation=90)
        ax.set_title(_thr_text(t), fontsize=10)
        ax.axhline(len(acc.reference.detections), color="#555", ls=":", lw=1)
    axes[0][0].set_ylabel("boxes")
    axes[0][0].legend(fontsize=8)
    fig.suptitle(f"G. Per-trial true positives, false positives and false negatives\n{header(u)}", fontsize=11)
    return finish(fig, path, f"Dotted line: {len(acc.reference.detections)} reference trees. " + ACCURACY_NOTE)


def reference_frequency(u, acc, thresholds, path):
    refs = [d.label for d in acc.reference.detections]
    n = _n_trials(acc)
    fig, ax = plt.subplots(figsize=(11, 5))
    fig.subplots_adjust(top=0.82, bottom=0.16)
    width = 0.8 / len(thresholds)
    colors = ["#1a9850", "#91cf60", "#bdbdbd"]
    for i, t in enumerate(thresholds):
        found = _ref_found(acc, t)
        ax.bar(np.arange(len(refs)) + (i - (len(thresholds) - 1) / 2) * width, [found[r] for r in refs], width,
               color=colors[i % len(colors)], label=_thr_text(t))
    ax.set_xticks(range(len(refs)), refs)
    ax.set_ylim(0, n + 1)
    ax.set_ylabel(f"trials that found the tree (of {n})")
    ax.axhline(n, color="#555", ls=":", lw=1)
    ax.legend(fontsize=8)
    fig.suptitle(f"H. How often each human reference tree was detected\n{header(u)}", fontsize=11)
    return finish(fig, path, "A tree counts as found in a trial when that trial's one-to-one matching pairs it with a "
                             "box at the threshold. " + ACCURACY_NOTE)


def reference_vs_predictions_overview(u, acc, thresholds, full, path):
    scale = PLOT_WIDTH / full.width
    thumb = np.asarray(full.resize((PLOT_WIDTH, round(full.height * scale)), Image.LANCZOS))
    n = _n_trials(acc)
    det_box = {d.uid: d.box for s in u.sets for d in s.detections}
    fig, axes = plt.subplots(1, len(thresholds), figsize=(8 * len(thresholds), 6.8), squeeze=False)
    fig.subplots_adjust(top=0.86, bottom=0.14, left=0.01, right=0.99, wspace=0.03)
    for ax, t in zip(axes[0], thresholds):
        lab = ec.threshold_label(t)
        ax.imshow(thumb)
        ax.set_axis_off()
        for r in acc.outcome_rows:
            if r["iou_threshold"] == lab:
                b = [float(v) * scale for v in det_box[r["detection_uid"]]]
                ax.add_patch(Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], fill=False, lw=0.7,
                                       edgecolor=OUTCOME_COLORS[_outcome_key(r)]))
        found = _ref_found(acc, t)
        for d in acc.reference.detections:
            b = [float(v) * scale for v in d.box]
            color = REF_FOUND_COLOR if found[d.label] == n else MISSED_COLOR
            ax.add_patch(Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], fill=False, lw=1.8, ls="--", edgecolor=color))
            ax.text(b[0], b[3] + 4, f"{d.label} {found[d.label]}/{n}", color=color, fontsize=7, weight="bold", va="top",
                    bbox={"facecolor": "black", "alpha": 0.6, "pad": 1, "lw": 0})
        ax.set_title(_thr_text(t), fontsize=10)
    handles = [Patch(edgecolor=c, facecolor="none", label=OUTCOME_LABELS[k]) for k, c in OUTCOME_COLORS.items()]
    handles += [Patch(edgecolor=REF_FOUND_COLOR, facecolor="none", ls="--", label="reference tree, found in every trial"),
                Patch(edgecolor=MISSED_COLOR, facecolor="none", ls="--", label="reference tree, missed in some trials")]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=8, frameon=False, bbox_to_anchor=(0.5, 0.03))
    fig.suptitle(f"I. Human reference vs Claude boxes from all {n} trials\n{header(u)}", fontsize=11)
    return finish(fig, path, "Thin boxes: every trial's predictions, coloured by outcome. Dashed: human reference, "
                             "labelled with the number of trials that found it. " + ACCURACY_NOTE)


def reference_vs_predictions_full(u, acc, t, full, path):
    lab = ec.threshold_label(t)
    n = _n_trials(acc)
    det_box = {d.uid: d.box for s in u.sets for d in s.detections}
    lw = max(3, round(min(full.size) / 320))
    f = font(max(14, round(min(full.size) / 70)))
    lh = round(ImageDraw.Draw(full).textbbox((0, 0), "Ag", font=f)[3] * 1.5)
    lines = [(f"{u.site} {u.image_id} | {u.model} | {n} trials | IoU >= {lab} | human reference (COMPLETE, "
              f"{len(acc.reference.detections)} trees) vs Claude boxes from all trials", (255, 255, 255))]
    lines += [(OUTCOME_LABELS[k] + " (thin)", hex_rgb(c)) for k, c in OUTCOME_COLORS.items()]
    lines += [("reference tree found in every trial (thick; label = trials that found it)", hex_rgb(REF_FOUND_COLOR)),
              ("reference tree missed in some trials (thick)", hex_rgb(MISSED_COLOR))]
    img = Image.new("RGB", (full.width, full.height + lh * len(lines) + 2 * lw), (0, 0, 0))
    img.paste(full, (0, 0))
    draw = ImageDraw.Draw(img)
    for r in acc.outcome_rows:
        if r["iou_threshold"] == lab:
            draw.rectangle([float(v) for v in det_box[r["detection_uid"]]],
                           outline=hex_rgb(OUTCOME_COLORS[_outcome_key(r)]), width=max(2, lw // 2))
    found = _ref_found(acc, t)
    for d in acc.reference.detections:
        b = [float(v) for v in d.box]
        color = hex_rgb(REF_FOUND_COLOR) if found[d.label] == n else hex_rgb(MISSED_COLOR)
        draw.rectangle([b[0] - lw, b[1] - lw, b[2] + lw, b[3] + lw], outline=color, width=lw)
        text = f"{d.label} {found[d.label]}/{n}"
        tb = draw.textbbox((0, 0), text, font=f)
        draw.rectangle([b[0], b[3] + lw, b[0] + tb[2] + 2 * lw, b[3] + tb[3] + 3 * lw], fill=color)
        draw.text((b[0] + lw, b[3] + 2 * lw), text, fill=(0, 0, 0), font=f)
    for i, (text, color) in enumerate(lines):
        draw.text((2 * lw, full.height + lw + i * lh), text, fill=color, font=f)
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, quality=85)
    return path


def error_cases(u, acc, t, full, path):
    """Crops of every false-positive location (grouped by IoU 0.50 group) and every reference tree missed in a trial."""
    lab = ec.threshold_label(t)
    n = _n_trials(acc)
    det_box = {d.uid: d.box for s in u.sets for d in s.detections}
    ref_box = {d.label: d.box for d in acc.reference.detections}
    rows = [r for r in acc.outcome_rows if r["iou_threshold"] == lab]
    tiles = []
    fp_groups = {}
    for r in rows:
        if r["outcome"] == "FP":
            fp_groups.setdefault(r["group_iou050"] or r["detection_uid"], []).append(r)
    for g, rs in sorted(fp_groups.items()):
        labels = sorted({r["human_label"] for r in rs if r["human_label"]})
        types = sorted({r["fp_type"].replace("_", " ") for r in rs})
        refs = sorted({r["best_reference"] for r in rs if r["best_reference"]})
        title = (f"{g}: {len(rs)} FP (trial {','.join(str(r['trial']) for r in rs)})\n"
                 f"human label: {'/'.join(labels) or 'none'}\n{'; '.join(types)}")
        tiles.append((title, [det_box[r["detection_uid"]] for r in rs], [_outcome_key(r) for r in rs], refs))
    for r in acc.reference_rows:
        if r["iou_threshold"] == lab and r["trials_matched"] < n:
            near = [x for x in rows if x["best_reference"] == r["reference_id"] and x["outcome"] == "FP"]
            tiles.append((f"{r['reference_id']}: found in {r['trials_matched']}/{n} trials\nmissed in trial "
                          f"{','.join(str(k) for k in r['trials_missed'])}",
                          [det_box[x["detection_uid"]] for x in near], [_outcome_key(x) for x in near],
                          [r["reference_id"]]))
    cols = 4
    nrows = max(1, -(-len(tiles) // cols))
    height = 3.9 * nrows + 1.8
    fig, axes = plt.subplots(nrows, cols, figsize=(4 * cols, height), squeeze=False)
    fig.subplots_adjust(top=1 - 1.1 / height, bottom=1.0 / height, hspace=0.6, wspace=0.08)
    for ax in axes.flat:
        ax.set_axis_off()
    for ax, (title, boxes, keys, refs) in zip(axes.flat, tiles):
        x0, y0, x1, y1 = crop_region(boxes + [ref_box[x] for x in refs], u.width, u.height, 0.5, 150)
        crop = full.crop((x0, y0, x1, y1))
        crop.thumbnail((600, 600))
        ax.imshow(np.asarray(crop), extent=(x0, x1, y1, y0))
        for rid in refs:
            b = [float(v) for v in ref_box[rid]]
            ax.add_patch(Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], fill=False, lw=2, ls="--",
                                   edgecolor=REF_FOUND_COLOR))
            ax.text(b[0], b[1], rid, color="black", fontsize=7, va="bottom",
                    bbox={"facecolor": REF_FOUND_COLOR, "pad": 1, "lw": 0})
        for b, k in zip(boxes, keys):
            b = [float(v) for v in b]
            ax.add_patch(Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], fill=False, lw=1.3,
                                   edgecolor=OUTCOME_COLORS[k]))
        ax.set_xlim(x0, x1)
        ax.set_ylim(y1, y0)
        ax.set_title(title, fontsize=8)
    if not tiles:
        axes[0][0].text(0.5, 0.5, "No false positives and no missed reference trees", ha="center",
                        transform=axes[0][0].transAxes)
    handles = [Patch(edgecolor=c, facecolor="none", label=OUTCOME_LABELS[k]) for k, c in OUTCOME_COLORS.items()]
    handles.append(Patch(edgecolor=REF_FOUND_COLOR, facecolor="none", ls="--", label="human reference box"))
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=8, frameon=False, bbox_to_anchor=(0.5, 0.3 / height))
    fig.suptitle(f"J. Error cases at IoU ≥ {lab}: false-positive locations and missed reference trees\n{header(u)}",
                 fontsize=11)
    return finish(fig, path, "FP tiles group a recurring false positive by its IoU 0.50 consistency group. "
                             "Missed-tree tiles show the reference box and nearby false positives. " + ACCURACY_NOTE)


# ------------------------------------------------------------------ comparison between model configurations (same image)

MODEL_COLORS = ["#4c72b0", "#dd8452", "#55a868", "#c44e52"]
COMPARISON_NOTE = ("Same image, same preprocessed bytes and prompt for every model. Spread between trials is run-to-run "
                   "variation on ONE image, not independent-image performance. Reasoning-effort settings are not "
                   "computationally equivalent across providers.")


def _model_label(u):
    return f"{u.model}\n({u.model_config_id})"


def comparison_figures(image_id: str, units: list, results: dict, stats: dict, acc_by_key: dict, cross_rows: list,
                       out: Path) -> list:
    fig_dir = out / "figures" / "comparison" / image_id
    units = sorted(units, key=lambda u: (u.model_config_id, u.experiment_id))
    with Image.open(units[0].image_path) as im:
        full = im.convert("RGB")
    return [model_comparison_panels(image_id, units, results, stats, acc_by_key, cross_rows,
                                    fig_dir / "K_model_comparison.png"),
            models_side_by_side(image_id, units, acc_by_key, full, fig_dir / "L_models_side_by_side_iou050.png")]


def model_comparison_panels(image_id, units, results, stats, acc_by_key, cross_rows, path):
    thresholds = list(next(iter(results.values())))
    have_acc = all(u.key in acc_by_key for u in units)
    fig, axes = plt.subplots(2, 2, figsize=(15, 10.5))
    fig.subplots_adjust(top=0.88, bottom=0.1, hspace=0.42, wspace=0.22)
    (a1, a2), (a3, a4) = axes
    rng = np.random.default_rng(0)

    for i, u in enumerate(units):
        counts, s = u.counts, stats[u.key]
        a1.scatter(i + rng.uniform(-0.12, 0.12, len(counts)), counts, s=28, color=MODEL_COLORS[i % 4], zorder=3)
        a1.errorbar(i + 0.28, s["mean"], yerr=s["sd"] or 0, fmt="o", color="black", capsize=5)
        cv = "–" if s["cv"] is None else f"{s['cv']:.1%}"
        a1.text(i + 0.33, s["mean"], f" mean {s['mean']:.1f}\n SD {s['sd'] or 0:.2f}\n CV {cv}", fontsize=8, va="center")
    a1.set_xticks(range(len(units)), [_model_label(u) for u in units], fontsize=8)
    a1.set_xlim(-0.5, len(units) - 0.1)
    a1.yaxis.get_major_locator().set_params(integer=True)
    a1.set_ylabel("boxes per trial")
    a1.set_title("Count per trial (dots) and mean ± SD", fontsize=10)

    labels = [_thr_text(t) for t in thresholds]
    x = np.arange(len(thresholds))
    width = 0.8 / (len(units) + 1)
    for i, u in enumerate(units):
        means = [results[u.key][t].summary["agreement_mean"] or 0 for t in thresholds]
        sds = [results[u.key][t].summary["agreement_sd"] or 0 for t in thresholds]
        a2.bar(x + (i - len(units) / 2) * width, means, width, yerr=sds, capsize=3, color=MODEL_COLORS[i % 4],
               label=f"within {u.model}")
    between = [next((r["agreement_mean"] for r in cross_rows if r["iou_threshold"] == ec.threshold_label(t)), 0) or 0
               for t in thresholds]
    between_sd = [next((r["agreement_sd"] for r in cross_rows if r["iou_threshold"] == ec.threshold_label(t)), 0) or 0
                  for t in thresholds]
    a2.bar(x + (len(units) / 2) * width, between, width, yerr=between_sd, capsize=3, color="#bbbbbb", hatch="//",
           edgecolor="#555", label="between models")
    a2.set_xticks(x, labels)
    a2.set_ylim(0, 1.12)
    a2.set_ylabel("pairwise agreement 2M/(nA+nB)")
    a2.legend(fontsize=8, loc="upper right")
    a2.set_title("Spatial agreement: within each model and between models", fontsize=10)

    if have_acc:
        metrics = ("precision", "recall", "f1")
        groups = [(t, m) for t in thresholds[:2] for m in metrics]
        xg = np.arange(len(groups))
        w = 0.8 / len(units)
        for i, u in enumerate(units):
            acc = acc_by_key[u.key]
            vals = [[a[m] for a in acc.per_trial[t].values() if a[m] is not None] for t, m in groups]
            means = [np.mean(v) if v else 0 for v in vals]
            sds = [np.std(v, ddof=1) if len(v) > 1 else 0 for v in vals]
            a3.bar(xg + (i - (len(units) - 1) / 2) * w, means, w, yerr=sds, capsize=3, color=MODEL_COLORS[i % 4],
                   label=u.model)
            for xi, m in zip(xg + (i - (len(units) - 1) / 2) * w, means):
                a3.text(xi, m + 0.03, f"{m:.2f}", ha="center", fontsize=7)
        a3.set_xticks(xg, [f"{m}\n{_thr_text(t)}" for t, m in groups], fontsize=8)
        a3.set_ylim(0, 1.12)
        a3.legend(fontsize=8)
        a3.set_title("Accuracy against the human reference (mean ± SD over trials)", fontsize=10)

        refs = [d.label for d in acc_by_key[units[0].key].reference.detections]
        xr = np.arange(len(refs))
        for i, u in enumerate(units):
            found = _ref_found(acc_by_key[u.key], thresholds[0])
            a4.bar(xr + (i - (len(units) - 1) / 2) * w, [found[r] for r in refs], w, color=MODEL_COLORS[i % 4],
                   label=u.model)
        n = max(_n_trials(acc_by_key[u.key]) for u in units)
        a4.set_xticks(xr, refs)
        a4.set_ylim(0, n + 1)
        a4.axhline(n, color="#555", ls=":", lw=1)
        a4.set_ylabel(f"trials that found the tree (of {n})")
        a4.legend(fontsize=8)
        a4.set_title(f"Each human reference tree: trials that found it at {_thr_text(thresholds[0])}", fontsize=10)
    else:
        for ax in (a3, a4):
            ax.set_axis_off()
        a3.text(0.5, 0.5, "No COMPLETE human reference for this image:\naccuracy is not computed "
                          "(consistency and between-model agreement only).", ha="center", va="center",
                transform=a3.transAxes, fontsize=11)
    fig.suptitle(f"K. Model comparison on {image_id} ({units[0].site}) · "
                 + " vs ".join(f"{u.model} ({u.n_trials} trials)" for u in units), fontsize=12)
    return finish(fig, path, COMPARISON_NOTE)


def models_side_by_side(image_id, units, acc_by_key, full, path, t=ec.THRESHOLDS[0]):
    lab = ec.threshold_label(t)
    scale = PLOT_WIDTH / full.width
    thumb = np.asarray(full.resize((PLOT_WIDTH, round(full.height * scale)), Image.LANCZOS))
    fig, axes = plt.subplots(1, len(units), figsize=(8 * len(units), 6.9), squeeze=False)
    fig.subplots_adjust(top=0.86, bottom=0.14, left=0.01, right=0.99, wspace=0.03)
    any_acc = False
    for ax, u in zip(axes[0], units):
        ax.imshow(thumb)
        ax.set_axis_off()
        acc = acc_by_key.get(u.key)
        if acc:
            any_acc = True
            det_box = {d.uid: d.box for s in u.sets for d in s.detections}
            for r in acc.outcome_rows:
                if r["iou_threshold"] == lab:
                    b = [float(v) * scale for v in det_box[r["detection_uid"]]]
                    ax.add_patch(Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], fill=False, lw=0.7,
                                           edgecolor=OUTCOME_COLORS[_outcome_key(r)]))
            found, n = _ref_found(acc, t), _n_trials(acc)
            for d in acc.reference.detections:
                b = [float(v) * scale for v in d.box]
                color = REF_FOUND_COLOR if found[d.label] == n else MISSED_COLOR
                ax.add_patch(Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], fill=False, lw=1.6, ls="--",
                                       edgecolor=color))
                ax.text(b[0], b[3] + 4, f"{d.label} {found[d.label]}/{n}", color=color, fontsize=7, weight="bold",
                        va="top", bbox={"facecolor": "black", "alpha": 0.6, "pad": 1, "lw": 0})
            p = next(r for r in acc.summary_rows if r["iou_threshold"] == lab and r["metric"] == "precision")["mean"]
            rcl = next(r for r in acc.summary_rows if r["iou_threshold"] == lab and r["metric"] == "recall")["mean"]
            ax.set_title(f"{u.model} · {u.n_trials} trials · precision {p:.2f}, recall {rcl:.2f} at IoU ≥ {lab}",
                         fontsize=10)
        else:
            for s in u.sets:
                for d in s.detections:
                    b = [float(v) * scale for v in d.box]
                    ax.add_patch(Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], fill=False, lw=0.7,
                                           edgecolor=np.array(trial_color(s.trial)) / 255))
            ax.set_title(f"{u.model} · {u.n_trials} trials (box colour = trial)", fontsize=10)
    if any_acc:
        handles = [Patch(edgecolor=c, facecolor="none", label=OUTCOME_LABELS[k]) for k, c in OUTCOME_COLORS.items()]
        handles += [Patch(edgecolor=REF_FOUND_COLOR, facecolor="none", ls="--", label="reference tree, found in every trial"),
                    Patch(edgecolor=MISSED_COLOR, facecolor="none", ls="--", label="reference tree, missed in some trials")]
        fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=8, frameon=False, bbox_to_anchor=(0.5, 0.03))
    fig.suptitle(f"L. Every trial's boxes for each model on {image_id}" + (" vs the human reference" if any_acc else ""),
                 fontsize=12)
    return finish(fig, path, COMPARISON_NOTE)
