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
        sheets.append(contact_sheet(u, tiles, out / "review" / f"contact_sheet_{u.experiment_id}_{u.image_id}.png"))
    return sheets


def contact_sheet(u, tiles, path: Path, tile=300, cols=6) -> Path:
    cap = 46
    head = 70
    rows = max(1, -(-len(tiles) // cols))
    sheet = Image.new("RGB", (cols * tile, head + rows * (tile + cap)), (255, 255, 255))
    draw = ImageDraw.Draw(sheet)
    draw.text((10, 8), f"Review units: {header(u)}", fill=(0, 0, 0), font=font(18))
    draw.text((10, 36), "Group crops: box colour = trial. Ambiguous-group crops: pink = the unit under review, "
                        "grey = other members. Label in human_review/manual_review.csv.", fill=(60, 60, 60), font=font(13))
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
