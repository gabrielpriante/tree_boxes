# Detection accuracy against the human reference

Algorithm version 1.0.0. Offline; no API requests. Computed only for images whose human reference is marked COMPLETE in `human_review/reference_status.csv`. Repeatability results are in `evaluation_report.md`. Inputs, hashes and software versions are in `evaluation_config.json`.

Files: `accuracy_vs_reference.csv` (per trial), `accuracy_summary.csv` (across trials), `reference_detection_frequency.csv` (per reference tree), `prediction_outcomes.csv` (per predicted box, with its IoU 0.50 group, review unit and human label).

## LA_PT-R1_27m (lowalt_pilot_ptr1_27m_v1)

Reference: 11 human boxes (H01–H11), status COMPLETE. Approved by Gabe on 2026-10-30.

Each of the 10 trials is matched to the reference on its own: one-to-one, maximum number of pairs with IoU ≥ threshold, ties broken by total IoU. IoU = 1.00 requires identical boxes.

### Summary across trials

| IoU ≥ | Precision mean ± SD [range] | Recall mean ± SD [range] | F1 mean ± SD [range] | TP / FP / FN per trial (mean) | Pooled P / R / F1 | Mean IoU of TPs |
|---|---|---|---|---|---|---|
| 0.50 | 0.786 ± 0.159 [0.333–0.857] | 0.436 ± 0.094 [0.182–0.545] | 0.561 ± 0.117 [0.235–0.667] | 4.8 / 1.3 / 6.2 | 0.787 / 0.436 / 0.561 | 0.663 (n=10) |
| 0.75 | 0.148 ± 0.052 [0.000–0.167] | 0.082 ± 0.029 [0.000–0.091] | 0.105 ± 0.037 [0.000–0.118] | 0.9 / 5.2 / 10.1 | 0.148 / 0.082 / 0.105 | 0.800 (n=9) |
| 1.00 | 0.000 ± 0.000 [0.000–0.000] | 0.000 ± 0.000 [0.000–0.000] | 0.000 ± 0.000 [0.000–0.000] | 0.0 / 6.1 / 11.0 | 0.000 / 0.000 / 0.000 | – (n=0) |

SD is the sample SD (n−1) across trials. Pooled values sum TP, FP and FN over all trials.

### Per-trial results

| IoU ≥ | Trial | Boxes | TP | FP | FN | Precision | Recall | F1 | Mean IoU of TPs |
|---|---|---|---|---|---|---|---|---|---|
| 0.50 | 01 | 6 | 5 | 1 | 6 | 0.833 | 0.455 | 0.588 | 0.665 |
| 0.50 | 02 | 6 | 5 | 1 | 6 | 0.833 | 0.455 | 0.588 | 0.663 |
| 0.50 | 03 | 6 | 5 | 1 | 6 | 0.833 | 0.455 | 0.588 | 0.669 |
| 0.50 | 04 | 6 | 5 | 1 | 6 | 0.833 | 0.455 | 0.588 | 0.663 |
| 0.50 | 05 | 6 | 5 | 1 | 6 | 0.833 | 0.455 | 0.588 | 0.683 |
| 0.50 | 06 | 6 | 2 | 4 | 9 | 0.333 | 0.182 | 0.235 | 0.588 |
| 0.50 | 07 | 6 | 5 | 1 | 6 | 0.833 | 0.455 | 0.588 | 0.661 |
| 0.50 | 08 | 6 | 5 | 1 | 6 | 0.833 | 0.455 | 0.588 | 0.663 |
| 0.50 | 09 | 6 | 5 | 1 | 6 | 0.833 | 0.455 | 0.588 | 0.665 |
| 0.50 | 10 | 7 | 6 | 1 | 5 | 0.857 | 0.545 | 0.667 | 0.707 |
| 0.75 | 01 | 6 | 1 | 5 | 10 | 0.167 | 0.091 | 0.118 | 0.800 |
| 0.75 | 02 | 6 | 1 | 5 | 10 | 0.167 | 0.091 | 0.118 | 0.800 |
| 0.75 | 03 | 6 | 1 | 5 | 10 | 0.167 | 0.091 | 0.118 | 0.800 |
| 0.75 | 04 | 6 | 1 | 5 | 10 | 0.167 | 0.091 | 0.118 | 0.800 |
| 0.75 | 05 | 6 | 1 | 5 | 10 | 0.167 | 0.091 | 0.118 | 0.800 |
| 0.75 | 06 | 6 | 0 | 6 | 11 | 0.000 | 0.000 | 0.000 | – |
| 0.75 | 07 | 6 | 1 | 5 | 10 | 0.167 | 0.091 | 0.118 | 0.793 |
| 0.75 | 08 | 6 | 1 | 5 | 10 | 0.167 | 0.091 | 0.118 | 0.800 |
| 0.75 | 09 | 6 | 1 | 5 | 10 | 0.167 | 0.091 | 0.118 | 0.800 |
| 0.75 | 10 | 7 | 1 | 6 | 10 | 0.143 | 0.091 | 0.111 | 0.813 |
| 1.00 | 01 | 6 | 0 | 6 | 11 | 0.000 | 0.000 | 0.000 | – |
| 1.00 | 02 | 6 | 0 | 6 | 11 | 0.000 | 0.000 | 0.000 | – |
| 1.00 | 03 | 6 | 0 | 6 | 11 | 0.000 | 0.000 | 0.000 | – |
| 1.00 | 04 | 6 | 0 | 6 | 11 | 0.000 | 0.000 | 0.000 | – |
| 1.00 | 05 | 6 | 0 | 6 | 11 | 0.000 | 0.000 | 0.000 | – |
| 1.00 | 06 | 6 | 0 | 6 | 11 | 0.000 | 0.000 | 0.000 | – |
| 1.00 | 07 | 6 | 0 | 6 | 11 | 0.000 | 0.000 | 0.000 | – |
| 1.00 | 08 | 6 | 0 | 6 | 11 | 0.000 | 0.000 | 0.000 | – |
| 1.00 | 09 | 6 | 0 | 6 | 11 | 0.000 | 0.000 | 0.000 | – |
| 1.00 | 10 | 7 | 0 | 7 | 11 | 0.000 | 0.000 | 0.000 | – |

### How often each reference tree was found

| Reference | Size (px) | Found at IoU ≥ 0.50 | Found at IoU ≥ 0.75 | Found at IoU ≥ 1.00 | When found at 0.50: mean IoU · area ratio (Claude/human) · centre offset (px) | Missed at 0.50: nearest box in that trial |
|---|---|---|---|---|---|---|
| H01 | 713 × 795 | 9/10 | 0/10 | 0/10 | 0.745 · 0.74 · 26 | t06:T002@0.333 |
| H02 | 704 × 785 | 9/10 | 0/10 | 0/10 | 0.661 · 0.66 · 19 | t06:T003@0.311 |
| H03 | 2014 × 1850 | 10/10 | 9/10 | 0/10 | 0.782 · 0.78 · 46 | – |
| H04 | 489 × 523 | 0/10 | 0/10 | 0/10 | – | no overlapping box in 10 trial(s) |
| H05 | 165 × 151 | 0/10 | 0/10 | 0/10 | – | no overlapping box in 10 trial(s) |
| H06 | 480 × 552 | 0/10 | 0/10 | 0/10 | – | no overlapping box in 10 trial(s) |
| H07 | 361 × 373 | 1/10 | 0/10 | 0/10 | 0.740 · 0.82 · 40 | no overlapping box in 9 trial(s) |
| H08 | 613 × 758 | 9/10 | 0/10 | 0/10 | 0.529 · 0.53 · 27 | t06:T006@0.451 |
| H09 | 308 × 330 | 0/10 | 0/10 | 0/10 | – | no overlapping box in 10 trial(s) |
| H10 | 221 × 222 | 0/10 | 0/10 | 0/10 | – | no overlapping box in 10 trial(s) |
| H11 | 758 × 751 | 10/10 | 0/10 | 0/10 | 0.612 · 0.61 · 47 | – |

**Box extent.** Of the 48 matched pairs at IoU ≥ 0.50, 46 Claude boxes lie entirely inside the human box (≥ 99.9% of their area). The mean share inside is 0.999. Averaged over the found trees, a Claude box covers 0.69 of the human box's area. When one box contains the other, IoU equals this area ratio. Claude's boxes are therefore systematically tighter than the human boxes, which by itself keeps most matches below IoU 0.75. This is a difference in box-extent convention, measured against this one annotator; it does not say which extent is correct.

### False positives

**IoU ≥ 0.50: 13 false-positive boxes in total.**

| Group (IoU 0.50) | Human label | FP type | Trials | Overlapping reference | Best IoU with it |
|---|---|---|---|---|---|
| G02 | NON_TREE | no_reference_overlap | 01, 02, 03, 04, 05, 06, 07, 08, 09, 10 | – | 0.000 |
| G03 | TREE | localization_below_threshold | 06 | H08 | 0.451 |
| G07 | UNCERTAIN | localization_below_threshold | 06 | H01 | 0.333 |
| G08 | UNCERTAIN | localization_below_threshold | 06 | H02 | 0.311 |

**IoU ≥ 0.75: 52 false-positive boxes in total.**

| Group (IoU 0.50) | Human label | FP type | Trials | Overlapping reference | Best IoU with it |
|---|---|---|---|---|---|
| G02 | NON_TREE | no_reference_overlap | 01, 02, 03, 04, 05, 06, 07, 08, 09, 10 | – | 0.000 |
| G03 | TREE | localization_below_threshold | 01, 02, 03, 04, 05, 06, 07, 08, 09, 10 | H08 | 0.451–0.584 |
| G04 | TREE | localization_below_threshold | 01, 02, 03, 04, 05, 06, 07, 08, 09, 10 | H11 | 0.564–0.716 |
| G05 | TREE | localization_below_threshold | 01, 02, 03, 04, 05, 07, 08, 09, 10 | H01 | 0.722–0.749 |
| G06 | TREE | localization_below_threshold | 01, 02, 03, 04, 05, 07, 08, 09, 10 | H02 | 0.641–0.689 |
| G01 | TREE | localization_below_threshold | 06 | H03 | 0.613 |
| G07 | UNCERTAIN | localization_below_threshold | 06 | H01 | 0.333 |
| G08 | UNCERTAIN | localization_below_threshold | 06 | H02 | 0.311 |
| G09 | TREE | localization_below_threshold | 10 | H07 | 0.740 |

FP types: `no_reference_overlap`: does not overlap any reference tree; `localization_below_threshold`: overlaps a reference tree, but IoU is below the threshold; `reference_matched_by_other_detection`: overlaps a reference tree that another box in the same trial matched.

### Human-labelled review units against the reference

Matched through the recorded detection ids, not the group names.

| Review unit | Human label | Note | Detections | IoU ≥ 0.50: TP / FP (matched reference) | IoU ≥ 0.75: TP / FP (matched reference) |
|---|---|---|---|---|---|
| G01 | TREE |  | 10 | 10 / 0 (H03) | 9 / 1 (H03); FP: localization_below_threshold |
| G02 | NON_TREE | Human-confirmed non-tree object; consistently detected in all ten trials. | 10 | 0 / 10; FP: no_reference_overlap | 0 / 10; FP: no_reference_overlap |
| G03 | TREE |  | 10 | 9 / 1 (H08); FP: localization_below_threshold | 0 / 10; FP: localization_below_threshold |
| G04 | TREE |  | 10 | 10 / 0 (H11) | 0 / 10; FP: localization_below_threshold |
| G05 | TREE |  | 9 | 9 / 0 (H01) | 0 / 9; FP: localization_below_threshold |
| G06 | TREE |  | 9 | 9 / 0 (H02) | 0 / 9; FP: localization_below_threshold |
| G07 | UNCERTAIN | Human review indicates this is the same physical tree represented by G05. The trial 6 bounding box is displaced and includes sidewalk while incompletely capturing the crown. This is a localization issue, not an additional individual tree. | 1 | 0 / 1; FP: localization_below_threshold | 0 / 1; FP: localization_below_threshold |
| G08 | UNCERTAIN | Human review indicates this is the same physical tree represented by G06. The trial 6 bounding box is displaced and includes sidewalk while incompletely capturing the crown. This is a localization issue, not an additional individual tree. | 1 | 0 / 1; FP: localization_below_threshold | 0 / 1; FP: localization_below_threshold |
| G09 | TREE | Human-confirmed tree detected only in trial 10. | 1 | 1 / 0 (H07) | 0 / 1; FP: localization_below_threshold |

### Figures

- `figures/lowalt_pilot_ptr1_27m_v1/LA_PT-R1_27m/F_accuracy_by_threshold.png`
- `figures/lowalt_pilot_ptr1_27m_v1/LA_PT-R1_27m/G_trial_tp_fp_fn.png`
- `figures/lowalt_pilot_ptr1_27m_v1/LA_PT-R1_27m/H_reference_detection_frequency.png`
- `figures/lowalt_pilot_ptr1_27m_v1/LA_PT-R1_27m/I_reference_vs_predictions_overview.png`
- `figures/lowalt_pilot_ptr1_27m_v1/LA_PT-R1_27m/I_reference_vs_predictions_iou050.jpg`
- `figures/lowalt_pilot_ptr1_27m_v1/LA_PT-R1_27m/J_error_cases_iou050.png`

## Limitations

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

