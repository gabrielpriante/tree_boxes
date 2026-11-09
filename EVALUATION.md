# Offline evaluation of repeated detections

`evaluate_consistency.py` scores saved benchmark trials without any API access. It reads the frozen records in
`benchmark_results/` and writes to `evaluation_results/`, which is regenerated on every run. Human judgements
live in `human_review/`, which the script creates once and never overwrites.

```powershell
python evaluate_consistency.py                                   # every experiment in benchmark_results/
python evaluate_consistency.py --experiments lowalt_pilot_ptr1_27m_v1 lowalt_pilot_pt02_64m_v1
python evaluate_consistency.py --no-full-res-maps                # skip the ~14 MB full-resolution map JPEGs
python -m pytest tests -q                                        # offline tests
```

| Module | Role |
|---|---|
| `evaluation_core.py` | Model-agnostic metrics: exact IoU, one-to-one matching, pairwise agreement, persistence groups, accuracy against a reference. No file I/O. |
| `evaluate_consistency.py` | Loads and verifies saved results, writes the CSVs, review materials, report and config. |
| `evaluation_plots.py` | Figures A–O, review crops and contact sheets. |
| `evaluation_geometry.py` | Box size, coverage, within-output overlap, coverage-mask agreement and fragmentation (descriptive). |
| `import_deepforest.py` | Imports DeepForest runs from `urban-tree-vision-baselines` as pixel-native, single-output experiments. |

## Four kinds of evidence, kept separate

1. **Count consistency.** Variation in the number of boxes across trials (mean, SD n−1, range, CV).
2. **Spatial consistency.** Whether trials put boxes on the same image locations: pairwise IoU matching and persistence.
3. **Count vs field inventory.** The model's mean count beside a site-level field count. Image coverage of the
   site is not established, so this is not an image-level reference. The inventory is never used to filter,
   correct or select predictions.
4. **Human-validated correctness.** Only from human labels (`manual_review.csv`) and human reference boxes.
   Precision, recall and F1 are computed only for images whose reference is declared `COMPLETE`.

Items 1–3 are repeatability or numeric proximity. Trials can agree perfectly on a box that is not a tree, and a
count close to the inventory can hide both misses and false positives.

## Inputs and verification

For each experiment the loader reads `counts_long.csv`, follows each successful trial's `record.json` to its
`detections.json`, and checks:

* the boxes match `boxes_long.csv`, and the box count matches both `counts_long.csv` and `record.json`;
* the source image's SHA-256 matches the hash recorded at inference time, and its size matches the recorded size;
* the number of successful trials matches `counts_summary.csv`, and the recomputed count statistics match it.

Any mismatch stops the evaluation. Non-success trials are listed as excluded. Every input file's SHA-256 is recorded in
`evaluation_config.json`, with software versions, the code commit and the thresholds.

## Coordinates and IoU

Claude returns integers on a 0–1000 grid. Each box is converted to original-image pixels with
`x · width / 1000` and `y · height / 1000`, the mapping used to draw the annotated images, and kept as an exact
fraction. IoU is computed in exact rational arithmetic, so:

* thresholds are applied exactly (`IoU ≥ 1/2`, `IoU ≥ 3/4`), with no floating-point tolerance;
* `IoU = 1` holds only for **identical** boxes. On the 0–1000 grid that means the same four integers. One grid
  step is about 8.2 × 6.1 original pixels (2.2 × 1.7 pixels in the image the model saw), so the 1.00 threshold measures exact
  reproduction of the model's output, not perceptual equivalence.

Thresholds: **0.50, 0.75, 1.00**. Boxes touching only along an edge have IoU 0 and never match.

## Pairwise matching (all n(n−1)/2 trial pairs; 45 for 10 trials)

For each pair of trials and each threshold, boxes are matched **one-to-one**. The assignment first maximizes the
**number** of pairs with IoU ≥ threshold, then the **total IoU** among assignments of that size. This is solved
exactly as a weighted assignment problem: eligible pairs weigh `K + IoU` with `K = min(n, m) + 1`, and ineligible
pairs weigh 0 (`scipy.optimize.linear_sum_assignment`). A box can match at most one box in the other trial.
Greedy highest-IoU-first matching can find fewer matches; a test covers this case, and another checks optimality
against brute force on random box sets.

Per pair: matched boxes, unmatched boxes in each trial, **agreement = 2 × matched / (n_A + n_B)**, and mean and
median IoU of the matched boxes. Agreement is undefined when both trials are empty. Per image and threshold: the
mean, SD, median and range of agreement over all pairs, the number of pairs with full agreement, and the pooled
matched-IoU statistics.

## Detection persistence (recurring objects)

Matches from all pairs at one threshold form a graph whose nodes are detections. Each connected component is
classified as follows:

* **CONSISTENT** if it holds at most one detection per trial **and** every two members were matched to each
  other directly (the component is a clique). The group's size is the number of trials that detected it.
* **AMBIGUOUS** otherwise, for example a chain A~B and B~C where A and C do not match, or two boxes from the same
  trial. Ambiguous components are reported with their members and the reason. They are never split or merged
  automatically, and they are not counted in the persistence bins.

Unmatched detections are consistent groups of one. Bins for 10 trials are 10, 8–9, 5–7 and 1–4. In general
they are all n, ≥80%, ≥50% and <50% of trials.

Group ids are deterministic: consistent groups first (most trials first, then top-to-bottom and left-to-right
by median centre), then ambiguous groups. Ids are assigned separately per threshold: `G03` at 0.50 and `G03` at
0.75 need not be the same object. Each detection's group at every threshold, and the number of other trials in
which it was matched, are in `detections.csv`.

**Limitations.**
* A group is a recurring *box location*, not a verified tree. Two adjacent crowns boxed as one in some trials
  and separately in others produce ambiguous or split groups.
* The rule is conservative. One missing link among 45 makes a 10-trial component ambiguous, so at 0.75, slowly
  drifting boxes on one crown are often flagged. At 1.00, exact equality is transitive, so ambiguity cannot
  arise there.
* Pairwise assignments are optimal per pair but not jointly across all trials. A multi-trial assignment (for
  example, a k-partite clique cover) could resolve some ambiguous components. It is not implemented, because
  any such choice would be an assumption the data do not settle.
* The 0–1000 output grid quantizes boxes (see above).

**Box variation (figure C, `detection_persistence.csv`):** the median box, centre SD in x and y, maximum centre
offset, width and height CV, and mean and minimum IoU between members, all in original-image pixels.

## Human review workflow

1. Run the evaluation. The first run creates `human_review/` with:
   * `manual_review.csv`: one row per review unit;
   * `reference_boxes.csv`: empty;
   * `reference_status.csv`: every image `NONE`;
   * `field_inventory.csv`;
   * `README.md`.
2. Open `evaluation_results/review/contact_sheet_<experiment>_<image>.png` for an overview. Each review unit has
   its own crop at `evaluation_results/review/crops/...`, named in the `crop_path` column.
   * **Review units** are the persistence groups at IoU ≥ 0.50, e.g. `lowalt_pilot_ptr1_27m_v1:LA_PT-R1_27m:G03`.
     Their crops show every trial's box (colour = trial).
   * **Ambiguous groups** are split into individual detections. In those crops, pink is the unit under review
     and grey shows the other members.
   * **Version control:** crops and contact sheets are regenerated exactly by every evaluation run, so new ones
     are git-ignored and kept only locally (since 5 November 2026). Crops committed earlier, including those the
     PT-R1 labels were made from, remain tracked.
3. In `human_review/manual_review.csv`, set `human_label` to `TREE`, `NON_TREE` or `UNCERTAIN`. Optionally add
   `human_note`, `reviewer` and `review_date`. Blank means not reviewed; invalid labels are reported, not used.
4. **Missed trees:** add a row to `human_review/reference_boxes.csv` for each tree, with `x_min_px … y_max_px`
   in original-image pixels (8192 × 6144 for the October 24 images) and `label` = `TREE`. The model's predictions
   are never assumed to contain every tree.
5. Re-run the evaluation. The report then shows a label × persistence table:
   * TREE in all trials: consistently detected trees;
   * NON_TREE in many trials: repeated false positives;
   * TREE in fewer than all trials: inconsistently detected trees;
   * reference trees found in no trial at IoU ≥ 0.50: missed trees (a lower bound until the reference is complete).
6. Only after **every** visible tree in an image has a reference box, set that image to `COMPLETE` in
   `reference_status.csv`. The next run writes `accuracy_vs_reference.csv`: per trial and threshold TP, FP, FN,
   precision, recall and F1. With `NONE` or `PARTIAL` these metrics are not computed, because unannotated real
   trees would count as false positives.

The project lead's preliminary observations are quoted in the report as free text:
* PT-R1 repeatedly detects a plant that is not a tree;
* the seventh box in PT-R1 trial 10 is a real tree;
* PT-02 disagrees more between trials.

They are not linked to detection ids until entered in `manual_review.csv`.

If inputs or the algorithm change, review unit ids may change. The report lists any labelled id it no longer
recognises.

**Threshold of the labels.** Human labels refer to the **IoU ≥ 0.50** review units (`group_id_iou050`) and to
the detections listed in each row. Group ids at 0.75 and 1.00 are separate groupings. For the pilot, only 4 of
35 group ids at 0.75, and none at 1.00, have the same members as at 0.50. Labels are therefore never transferred
by id. A transfer to another threshold would need an explicit detection-level mapping. The evaluation
thresholds themselves remain 0.50, 0.75 and 1.00.

**Validating hand edits.** `manual_review.csv` is checked before every evaluation, or on its own with
`python evaluate_consistency.py --validate-review`, which writes nothing. The check covers:

* `persistence_bin` must match `n_trials`. Date-like values such as `9-Aug`, which Excel produces from `8-9`,
  are flagged.
* `trials_detected` must list `n_trials` distinct trials.
* `review_unit_id` values must be unique.
* Generated columns must still equal the evaluation's template.

Any failure stops the run with a clear error, without writing outputs. The file is never rewritten
automatically. A safe Excel workflow is in `human_review/README.md`.

## Accuracy against a COMPLETE human reference

Reference boxes are imported from Make Sense exports with `import_reference.py`. The import requires the Pascal
VOC and CSV exports to agree. Differences of 1 px or less on `x_max`/`y_max` need an explicit `--use voc|csv`.
The importer never rounds, never replaces existing rows and never touches `reference_status.csv`.

Once an image is marked `COMPLETE`, every evaluation run matches each trial to the reference on its own. It uses
the same exact-IoU, one-to-one, maximum-cardinality matching as the consistency analysis, at 0.50, 0.75 and 1.00.
`evaluation_accuracy.py` writes:

* `accuracy_vs_reference.csv`: per trial and threshold, TP, FP, FN, precision, recall, F1 and the mean IoU of TPs.
* `accuracy_summary.csv`: the mean, sample SD, min and max of each metric across trials, plus pooled values.
* `reference_detection_frequency.csv`: per reference tree, the trials that found it and the matched detection ids.
  It also gives the mean IoU, the area ratio (Claude/human) and the centre offset when found, and the nearest box
  in each trial that missed it.
* `prediction_outcomes.csv`: per predicted box, TP or FP with its matched reference. Each FP gets a type:
  * `no_reference_overlap`;
  * `localization_below_threshold`, where the box overlaps a tree but the IoU is too low;
  * `reference_matched_by_other_detection`.
  Each row also carries the box's IoU 0.50 group, review unit and human label, so error analysis follows
  detection ids rather than group names.
* `accuracy_report.md` and figures F–J, under `figures/<experiment>/<image>/`:
  * F: precision, recall and F1 by threshold;
  * G: per-trial TP/FP/FN;
  * H: how often each reference tree was found;
  * I: the reference against all trials' boxes;
  * J: crops of the error cases.

`human_review/reference_boxes.csv` also produces `review/human_reference_<image>.jpg`, which shows the reference
boxes only.

**Conventions for the paper.**
* **Threshold:** IoU ≥ 0.50 is the primary matching threshold, and IoU ≥ 0.75 a secondary sensitivity check. IoU
  = 1.00 is reported but not interpreted (see below).
* **Summary across trials:** for M08 and M09, the primary summary is the **mean of per-trial values** (with SD and
  range across the 10 trials). The **pooled** values in `accuracy_summary.csv` (`pooled_over_trials`, which sums
  TP, FP and FN over all trials) are secondary and differ slightly: for example, M09 precision at 0.50 is 0.850 as a
  mean against 0.842 pooled. DeepForest has one output per image, so its mean and pooled values coincide.
* **Duplicates:** matching is one-to-one, so a second box on an already-matched reference box is a false positive,
  of type `reference_matched_by_other_detection`.
* **Human labels:** the review labels in `manual_review.csv` annotate the outcome tables only. They never change
  TP, FP or FN.

**The PT-R1 reference.** It is 11 human-annotated image-reference boxes (H01–H11) drawn by one annotator on the
benchmark image. They were drawn after the 10 M08 trials and the review of their detections. They are not traced
tree by tree to any field inventory. They are not field-validated tree positions. PT-02 has no reference, so it
gets no precision, recall or F1.

Against human whole-pixel boxes, IoU = 1.00 is effectively unreachable: Claude's 0–1000 grid has a step of about
8 px. At that threshold the evaluation reports exact coordinate agreement, which is not a measure of detection
quality. The report also measures how much of each matched Claude box lies inside its human box. If one box
contains the other, IoU equals their area ratio, which separates box-extent differences from real misplacement.

## Comparing model configurations on the same image

When one image has results from more than one model configuration, for example M08 and M09 on
`LA_PT-R1_27m`, each is first evaluated on its own, exactly as above. The run then adds:

* `model_comparison.csv`: per image and configuration:
  * trials, count mean/SD/CV;
  * within-model agreement at 0.50, 0.75 and 1.00;
  * mean precision, recall and F1, only where the image has a COMPLETE human reference.
* `cross_model_agreement.csv`: per image, configuration pair and threshold, the agreement between every
  trial of one model and every trial of the other (n_A × n_B pairs). It uses the same one-to-one exact-IoU
  matching.
* A section "Comparison between model configurations" in `evaluation_report.md`.

Comparisons stay within one image (same SHA-256). Between-model agreement measures whether two approaches box
the same places. It says nothing about which is correct. For PT-02, which has no reference, only consistency
and between-model agreement are reported. Reasoning-effort settings are not computationally equivalent across
providers, so model rows are configurations, not "equal-effort" conditions. These outputs appear only when such
data exists. Nothing is produced for a single configuration.

## Deterministic single-output producers (DeepForest) and pixel-native boxes

DeepForest M01 / M02 v2 runs come from `urban-tree-vision-baselines` and are added with `import_deepforest.py`.
The importer:
* takes the **primary** run as trial 1. The bit-identical verification run is a reproducibility check, recorded
  in `determinism.json`, and is not imported as a trial;
* refuses unless the source hash agrees across the manifest, the local file, the DeepForest run record and every
  M08/M09 record on that image;
* refuses to overwrite an existing experiment folder.

**Records.** `record.json` carries `coordinate_system: "original_pixels"`. `boxes_long.csv` has `x_min_px … y_max_px`
and `score` columns at full float precision. Boxes are **never** put on the LLMs' 0–1000 grid: their
`x_*_norm` columns in `detections.csv` are blank. The loader checks `detections.json` against `boxes_long.csv`,
scores included, and turns each float into an exact rational through its decimal string. Every IoU stays exact.

**A unit with one output** (`single_output`):
* has no trial pairs, so count SD/CV, within-model agreement, persistence groups, review units and figures A–E are
  reported as *not applicable*, not as zero;
* still gets accuracy against a COMPLETE reference (figures F–J; the J error-case sheet shows at most 24
  false-positive boxes, nearest to reference trees first, and lists all of them in `prediction_outcomes.csv`);
* is compared between models with each trial of the other model separately: one DeepForest output against 10
  LLM trials gives 10 pairs, which are never duplicated to imitate 10 trials.

**Geometry, coverage and overlap** (all units; descriptive only, never accuracy):
* `geometry_summary.csv`, per output:
  * box count, and box side √(w·h) in px and at the nominal GSD (p10 / median / p90);
  * coverage (share of the image inside any box, on a raster at 1/8 resolution);
  * summed box area / covered area;
  * near-duplicate pairs (IoU ≥ 0.3 within one output);
  * boxes per 100 m².
* `cross_model_coverage.csv`, per pair of outputs from different configurations:
  * coverage-mask IoU;
  * how many box centres of the configuration with more boxes fall inside each box of the other (fragmentation);
  * the share of its centres inside any of them.
* Report section 9.

**Figures** for comparisons with more than two configurations or with a single-output unit. A two-LLM comparison
keeps only K and L, as before.
* **M:** between-model agreement matrices at IoU 0.50 / 0.75. The diagonal is within-model repeatability, n/a for
  a single output.
* **N:** box size, coverage and overlap, one dot per output.
* **O:** coverage maps.
* **K** switches to a log count axis when counts differ by more than 20× and plots per-tree detection as a share
  of outputs.

**Matching speed.** `match_boxes` first runs a vectorised float-IoU prefilter (slack 1e-6). It then computes exact
rational IoU for every surviving pair, so results are identical to exhaustive matching, which a test checks. This
keeps 6,320 × 42 box comparisons fast.

The four-model pilot evaluation is in `evaluation_results_lowalt_v1_deepforest/`:

```powershell
python import_deepforest.py --image-dir <baselines>/outputs/lowalt_pilot_v1/<model>/<experiment>/<config>/<image>
python evaluate_consistency.py --output-dir evaluation_results_lowalt_v1_deepforest
```

It contains every M08/M09 row of `evaluation_results/` unchanged (checked row by row on 5 November 2026; only
`altitude_series.csv`'s `model_outputs` column now also lists the DeepForest experiments) plus the DeepForest
units.

## Repeated DeepForest executions (lowalt_repro_v1; completed 9 November 2026)

`evaluate_deepforest_repeatability.py` (figures in `evaluation_plots_repro.py`) analyses 10 repeated executions of each
DeepForest pilot configuration (M01 and M02v2 at a nominal 10 cm/px) on both pilot images.
* **Where the runs came from:** they were produced with the code preserved in `deepforest_pipeline/` (see its
  README and `docs/lowalt_repro_v1_deepforest.md`) and mirrored to Windows.
* **Not imported as trials:** they are **not** imported into `benchmark_results/`. Identical deterministic repeats
  must not appear as independent trials.
* **Where the raw outputs are:** outside Git. Their checksums are in
  `deepforest_pipeline/provenance/checksums/`.

**Result.** All 40 runs succeeded:
* in each of the four combinations, the raw files, canonical detections, counts (4,836, 6,320, 20 and 183) and
  annotated images were identical across the 10 runs;
* pairwise agreement was 1.000 at 0.50, 0.75 and 1.00;
* every run was identical to the pilot's primary and verification runs.

That is computational reproducibility in one environment (`evaluation_results_deepforest_repro_v1/repro_report.md`),
not accuracy.

```powershell
python evaluate_deepforest_repeatability.py --runs-root ..\urban-tree-vision-baselines\outputs\lowalt_repro_v1 --output-dir evaluation_results_deepforest_repro_v1
```

**Checks per configuration and image**
* **Raw output files:** SHA-256 of `predictions_raw.csv` (DeepForest's own order). This is kept separate from the
  SHA-256 of `detections.json` (canonical order) and from exact float equality of the canonically sorted boxes and
  scores. Raw files that differ while the canonical detections match are reported as an order-only difference.
* **Counts:** detections per run, with mean and SD.
* **Pairwise agreement:** all 45 run pairs, using the same exact-IoU one-to-one matching at 0.50, 0.75 and 1.00.
* **Differences:** for a non-identical pair, the count difference, unmatched boxes, and the largest coordinate and
  score differences among boxes matched at 0.50.
* **Integrity:** the source hash is checked against the manifest and the configuration id must be consistent. Each
  run is cross-referenced with the pilot primary and verification runs.
* **LLM context:** M08/M09 within-model agreement and count CV are read from the frozen `evaluation_results/` CSVs
  and placed beside the DeepForest results. They are not recomputed and nothing in that folder changes.

**Outputs:** `repro_runs.csv`, `repro_pairwise.csv`, `repro_summary.csv`, `llm_within_model_reference.csv`,
`repro_report.md`, `evaluation_config.json`, and figures P (counts per run), Q (pairwise heatmaps at 0.50 and
1.00) and R (deterministic repeats against stochastic LLM trials).

**Matching at this scale.** Two 6,320-box outputs would need a 6,320 × 6,320 assignment matrix. Above 4 million box
pairs, `match_boxes` therefore solves each connected component of the eligible-pair graph separately. This gives
the same optimum, since no eligible pair crosses components; a test checks it against the dense path. Every
earlier evaluation stayed below that size (largest 6,320 × 183), so its results come from the unchanged dense path.

## Adding other models

`evaluation_core` only sees `DetectionSet` objects: boxes in original-image pixels, plus image id, SHA-256,
size and trial number. Adding DeepForest, ChatGPT or Gemini means writing one loader that converts that model's
saved outputs into these objects. Pixel boxes need no conversion; normalized boxes use `box_from_normalized`
with that model's scale. Metrics, persistence, figures and review then work unchanged. Comparing two models on
the **same** image uses the same `compare_sets` call. Comparing across images is refused (next section).

## Across-altitude robustness: design (not part of the present paper; kept for later work)

The October 24 series has five PT-02 images (64–86 m) and five PT-R1 images (27–31 m). So far only
`LA_PT-R1_27m` and `LA_PT-02_64m` have model outputs. `evaluation_results/altitude_series.csv` lists each
image's altitude, gimbal angles, GPS, flagged heading, original and API resolution, estimated GSD and whether
outputs exist.

**Why boxes cannot be compared directly.** A 0–1000 box describes a position in one frame. Between frames, the
camera position, footprint, heading and scale all change; two of the frames are rotated about 90°. The same
coordinates in two images are different ground locations. `compare_sets` raises
`CrossImageComparisonError` for different images and `NotImplementedError` if an alignment is passed, so no
cross-image box comparison can happen by accident.

**What a valid comparison needs.**
1. **Registration** of each image to a common frame. Options, in order of preference:
   * an orthomosaic or georeferenced product, using ground control or RTK positions;
   * feature-based registration (SIFT or ORB with RANSAC homography) between nadir frames of the same site,
     validated with independent check points. A plane homography is only approximate for tall trees, because
     relief displacement grows towards the frame edges and with lower altitude;
   * EXIF/XMP metadata alone (GPS, relative altitude, gimbal yaw, focal length). This gives a rough footprint
     estimate. Consumer GNSS error of several metres is comparable to a crown, so metadata alone is not enough
     to match individual trees.

   OpenCV is not installed, and no registration has been attempted.
2. **A shared comparison area.** Compare only trees inside the intersection of the registered footprints, minus
   a margin for edge distortion. A tree outside an image's footprint, or cut by its edge, is not a false negative
   for that image. `coverage_scope` is `UNKNOWN` for every image until checked.
3. **Orientation.** `LA_PT-R1_30m_angled` (yaw +153.7° vs −116.4°) and `LA_PT-02_86m` (yaw −116.3° vs about +152°)
   are rotated about 90°. Registration must model rotation. Analyse these frames as heading variants, or report
   them separately: for `LA_PT-02_86m` the rotation coincides with the largest altitude change.
4. **Confounds to separate from altitude.**
   * Footprint and framing, i.e. which trees are in view.
   * Ground resolution.
   * Image-edge effects.
   * Sun and shadow: the frames were taken between 09:51 and 10:12, so shadows move little but not zero.
   * Altitude reference: `RelativeAltitude` is height above the takeoff point, not above the canopy. Terrain
     and takeoff location affect the true height.
5. **Matching across images.** After registration, transform each image's boxes into the common frame (rotated
   boxes become polygons). Then match each image's persistence groups to a **human reference tree list in the
   common frame**, not image to image. This keeps "detected in image k" well defined per tree.
6. **Two separate quantities.**
   * *Within-image repeatability*: this evaluation, run per image.
   * *Across-image robustness*: for each reference tree inside the shared area, the fraction of altitudes at
     which it is consistently detected.
   Do not pool them.
7. **Absence is not a false negative.** A tree is "missed" in an image only if it lies inside that image's valid
   area and a human confirms it is visible there.

**Downscaling.** The API sees every 8192 × 6144 image at 2212 × 1659, which is 3.70× coarser. Ground sampling
distance is estimated from relative altitude and the 24 mm 35 mm-equivalent focal length, so it is approximate:

| Image | Altitude | GSD original | GSD as sent to the model |
|---|---|---|---|
| LA_PT-R1_27m | 26.8 m | ≈ 4.7 mm/px | ≈ 17.5 mm/px |
| LA_PT-02_64m | 64.3 m | ≈ 11.3 mm/px | ≈ 41.9 mm/px |
| LA_PT-02_86m | 85.8 m | ≈ 15.1 mm/px | ≈ 55.9 mm/px |

The smallest box Claude returned on PT-02 at 64 m was about 2.1 m across, which is 51 pixels in the image the
model received. At 86 m, a 1.5 m crown would cover only about 27 such pixels. Small or young trees, narrow
crowns, and gaps between touching crowns therefore become harder to resolve as altitude rises, independent of
the model. A fair across-altitude analysis should report crown size, in metres and in model-input pixels,
next to detection outcomes. It should also treat the downscaling as part of the experimental condition. Tiling or
cropping would change it and is out of scope for the current fixed configuration.
