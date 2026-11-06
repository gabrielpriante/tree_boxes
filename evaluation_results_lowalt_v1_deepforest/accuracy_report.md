# Detection accuracy against the human reference

Algorithm version 1.0.0. Offline; no API requests. Computed only for images whose human reference is marked COMPLETE in `human_review/reference_status.csv`. Repeatability results are in `evaluation_report.md`. Inputs, hashes and software versions are in `evaluation_config.json`.

Files: `accuracy_vs_reference.csv` (per trial), `accuracy_summary.csv` (across trials), `reference_detection_frequency.csv` (per reference tree), `prediction_outcomes.csv` (per predicted box, with its IoU 0.50 group, review unit and human label).

## LA_PT-R1_27m (lowalt_pilot_ptr1_27m_m01_v1)

Reference: 11 human boxes (H01–H11), status COMPLETE. Approved by Gabe on 2026-10-30.

Each of the 1 trials is matched to the reference on its own: one-to-one, maximum number of pairs with IoU ≥ threshold, ties broken by total IoU. IoU = 1.00 requires identical boxes.

### Summary across trials

| IoU ≥ | Precision mean ± SD [range] | Recall mean ± SD [range] | F1 mean ± SD [range] | TP / FP / FN per trial (mean) | Pooled P / R / F1 | Mean IoU of TPs |
|---|---|---|---|---|---|---|
| 0.50 | 0.000 ± – [0.000–0.000] | 0.000 ± – [0.000–0.000] | 0.000 ± – [0.000–0.000] | 0.0 / 4836.0 / 11.0 | 0.000 / 0.000 / 0.000 | – (n=0) |
| 0.75 | 0.000 ± – [0.000–0.000] | 0.000 ± – [0.000–0.000] | 0.000 ± – [0.000–0.000] | 0.0 / 4836.0 / 11.0 | 0.000 / 0.000 / 0.000 | – (n=0) |
| 1.00 | 0.000 ± – [0.000–0.000] | 0.000 ± – [0.000–0.000] | 0.000 ± – [0.000–0.000] | 0.0 / 4836.0 / 11.0 | 0.000 / 0.000 / 0.000 | – (n=0) |

SD is the sample SD (n−1) across trials. Pooled values sum TP, FP and FN over all trials.

### Per-trial results

| IoU ≥ | Trial | Boxes | TP | FP | FN | Precision | Recall | F1 | Mean IoU of TPs |
|---|---|---|---|---|---|---|---|---|---|
| 0.50 | 01 | 4836 | 0 | 4836 | 11 | 0.000 | 0.000 | 0.000 | – |
| 0.75 | 01 | 4836 | 0 | 4836 | 11 | 0.000 | 0.000 | 0.000 | – |
| 1.00 | 01 | 4836 | 0 | 4836 | 11 | 0.000 | 0.000 | 0.000 | – |

### How often each reference tree was found

| Reference | Size (px) | Found at IoU ≥ 0.50 | Found at IoU ≥ 0.75 | Found at IoU ≥ 1.00 | When found at 0.50: mean IoU · area ratio (model/human) · centre offset (px) | Missed at 0.50: nearest box in that trial |
|---|---|---|---|---|---|---|
| H01 | 713 × 795 | 0/1 | 0/1 | 0/1 | – | t01:D01665@0.047 |
| H02 | 704 × 785 | 0/1 | 0/1 | 0/1 | – | t01:D03163@0.033 |
| H03 | 2014 × 1850 | 0/1 | 0/1 | 0/1 | – | t01:D04228@0.012 |
| H04 | 489 × 523 | 0/1 | 0/1 | 0/1 | – | t01:D03243@0.054 |
| H05 | 165 × 151 | 0/1 | 0/1 | 0/1 | – | t01:D03105@0.433 |
| H06 | 480 × 552 | 0/1 | 0/1 | 0/1 | – | t01:D00692@0.014 |
| H07 | 361 × 373 | 0/1 | 0/1 | 0/1 | – | t01:D01167@0.086 |
| H08 | 613 × 758 | 0/1 | 0/1 | 0/1 | – | t01:D00762@0.019 |
| H09 | 308 × 330 | 0/1 | 0/1 | 0/1 | – | t01:D03193@0.116 |
| H10 | 221 × 222 | 0/1 | 0/1 | 0/1 | – | t01:D04328@0.237 |
| H11 | 758 × 751 | 0/1 | 0/1 | 0/1 | – | t01:D01704@0.021 |

### False positives

**IoU ≥ 0.50: 4836 false-positive boxes in total.**

| Group (IoU 0.50) | Human label | FP type | Trials | Overlapping reference | Best IoU with it |
|---|---|---|---|---|---|
| – | unlabelled | no_reference_overlap | 4084 boxes in trial(s) 01 | – | 0.000 |
| – | unlabelled | localization_below_threshold | 292 boxes in trial(s) 01 | H03 | 0.000–0.012 |
| – | unlabelled | localization_below_threshold | 168 boxes in trial(s) 01 | H11 | 0.000–0.021 |
| – | unlabelled | localization_below_threshold | 69 boxes in trial(s) 01 | H04 | 0.000–0.054 |
| – | unlabelled | localization_below_threshold | 60 boxes in trial(s) 01 | H08 | 0.000–0.019 |
| – | unlabelled | localization_below_threshold | 39 boxes in trial(s) 01 | H01 | 0.000–0.047 |
| – | unlabelled | localization_below_threshold | 39 boxes in trial(s) 01 | H02 | 0.000–0.033 |
| – | unlabelled | localization_below_threshold | 37 boxes in trial(s) 01 | H07 | 0.000–0.086 |
| – | unlabelled | localization_below_threshold | 18 boxes in trial(s) 01 | H06 | 0.000–0.014 |
| – | unlabelled | localization_below_threshold | 13 boxes in trial(s) 01 | H09 | 0.002–0.116 |
| – | unlabelled | localization_below_threshold | 12 boxes in trial(s) 01 | H10 | 0.002–0.237 |
| – | unlabelled | localization_below_threshold | 5 boxes in trial(s) 01 | H05 | 0.007–0.433 |

**IoU ≥ 0.75: 4836 false-positive boxes in total.**

| Group (IoU 0.50) | Human label | FP type | Trials | Overlapping reference | Best IoU with it |
|---|---|---|---|---|---|
| – | unlabelled | no_reference_overlap | 4084 boxes in trial(s) 01 | – | 0.000 |
| – | unlabelled | localization_below_threshold | 292 boxes in trial(s) 01 | H03 | 0.000–0.012 |
| – | unlabelled | localization_below_threshold | 168 boxes in trial(s) 01 | H11 | 0.000–0.021 |
| – | unlabelled | localization_below_threshold | 69 boxes in trial(s) 01 | H04 | 0.000–0.054 |
| – | unlabelled | localization_below_threshold | 60 boxes in trial(s) 01 | H08 | 0.000–0.019 |
| – | unlabelled | localization_below_threshold | 39 boxes in trial(s) 01 | H01 | 0.000–0.047 |
| – | unlabelled | localization_below_threshold | 39 boxes in trial(s) 01 | H02 | 0.000–0.033 |
| – | unlabelled | localization_below_threshold | 37 boxes in trial(s) 01 | H07 | 0.000–0.086 |
| – | unlabelled | localization_below_threshold | 18 boxes in trial(s) 01 | H06 | 0.000–0.014 |
| – | unlabelled | localization_below_threshold | 13 boxes in trial(s) 01 | H09 | 0.002–0.116 |
| – | unlabelled | localization_below_threshold | 12 boxes in trial(s) 01 | H10 | 0.002–0.237 |
| – | unlabelled | localization_below_threshold | 5 boxes in trial(s) 01 | H05 | 0.007–0.433 |

FP types: `no_reference_overlap`: does not overlap any reference tree; `localization_below_threshold`: overlaps a reference tree, but IoU is below the threshold; `reference_matched_by_other_detection`: overlaps a reference tree that another box in the same trial matched.

### Human-labelled review units against the reference

Matched through the recorded detection ids, not the group names.

| Review unit | Human label | Note | Detections | IoU ≥ 0.50: TP / FP (matched reference) | IoU ≥ 0.75: TP / FP (matched reference) |
|---|---|---|---|---|---|

### Figures

- `figures/lowalt_pilot_ptr1_27m_m01_v1/LA_PT-R1_27m/F_accuracy_by_threshold.png`
- `figures/lowalt_pilot_ptr1_27m_m01_v1/LA_PT-R1_27m/G_trial_tp_fp_fn.png`
- `figures/lowalt_pilot_ptr1_27m_m01_v1/LA_PT-R1_27m/H_reference_detection_frequency.png`
- `figures/lowalt_pilot_ptr1_27m_m01_v1/LA_PT-R1_27m/I_reference_vs_predictions_overview.png`
- `figures/lowalt_pilot_ptr1_27m_m01_v1/LA_PT-R1_27m/I_reference_vs_predictions_iou050.jpg`
- `figures/lowalt_pilot_ptr1_27m_m01_v1/LA_PT-R1_27m/J_error_cases_iou050.png`

## LA_PT-R1_27m (lowalt_pilot_ptr1_27m_m02_v2)

Reference: 11 human boxes (H01–H11), status COMPLETE. Approved by Gabe on 2026-10-30.

Each of the 1 trials is matched to the reference on its own: one-to-one, maximum number of pairs with IoU ≥ threshold, ties broken by total IoU. IoU = 1.00 requires identical boxes.

### Summary across trials

| IoU ≥ | Precision mean ± SD [range] | Recall mean ± SD [range] | F1 mean ± SD [range] | TP / FP / FN per trial (mean) | Pooled P / R / F1 | Mean IoU of TPs |
|---|---|---|---|---|---|---|
| 0.50 | 0.250 ± – [0.250–0.250] | 0.455 ± – [0.455–0.455] | 0.323 ± – [0.323–0.323] | 5.0 / 15.0 / 6.0 | 0.250 / 0.455 / 0.323 | 0.669 (n=1) |
| 0.75 | 0.050 ± – [0.050–0.050] | 0.091 ± – [0.091–0.091] | 0.065 ± – [0.065–0.065] | 1.0 / 19.0 / 10.0 | 0.050 / 0.091 / 0.065 | 0.785 (n=1) |
| 1.00 | 0.000 ± – [0.000–0.000] | 0.000 ± – [0.000–0.000] | 0.000 ± – [0.000–0.000] | 0.0 / 20.0 / 11.0 | 0.000 / 0.000 / 0.000 | – (n=0) |

SD is the sample SD (n−1) across trials. Pooled values sum TP, FP and FN over all trials.

### Per-trial results

| IoU ≥ | Trial | Boxes | TP | FP | FN | Precision | Recall | F1 | Mean IoU of TPs |
|---|---|---|---|---|---|---|---|---|---|
| 0.50 | 01 | 20 | 5 | 15 | 6 | 0.250 | 0.455 | 0.323 | 0.669 |
| 0.75 | 01 | 20 | 1 | 19 | 10 | 0.050 | 0.091 | 0.065 | 0.785 |
| 1.00 | 01 | 20 | 0 | 20 | 11 | 0.000 | 0.000 | 0.000 | – |

### How often each reference tree was found

| Reference | Size (px) | Found at IoU ≥ 0.50 | Found at IoU ≥ 0.75 | Found at IoU ≥ 1.00 | When found at 0.50: mean IoU · area ratio (model/human) · centre offset (px) | Missed at 0.50: nearest box in that trial |
|---|---|---|---|---|---|---|
| H01 | 713 × 795 | 1/1 | 1/1 | 0/1 | 0.785 · 0.78 · 18 | – |
| H02 | 704 × 785 | 1/1 | 0/1 | 0/1 | 0.713 · 0.73 · 44 | – |
| H03 | 2014 × 1850 | 1/1 | 0/1 | 0/1 | 0.606 · 0.61 · 70 | – |
| H04 | 489 × 523 | 1/1 | 0/1 | 0/1 | 0.558 · 1.74 · 122 | – |
| H05 | 165 × 151 | 0/1 | 0/1 | 0/1 | – | t01:D00014@0.054 |
| H06 | 480 × 552 | 0/1 | 0/1 | 0/1 | – | no overlapping box in 1 trial(s) |
| H07 | 361 × 373 | 0/1 | 0/1 | 0/1 | – | no overlapping box in 1 trial(s) |
| H08 | 613 × 758 | 0/1 | 0/1 | 0/1 | – | t01:D00001@0.386 |
| H09 | 308 × 330 | 0/1 | 0/1 | 0/1 | – | no overlapping box in 1 trial(s) |
| H10 | 221 × 222 | 0/1 | 0/1 | 0/1 | – | no overlapping box in 1 trial(s) |
| H11 | 758 × 751 | 1/1 | 0/1 | 0/1 | 0.683 · 0.69 · 65 | – |

**Box extent.** Of the 5 matched pairs at IoU ≥ 0.50, 2 DeepForest M02v2 boxes lie entirely inside the human box (≥ 99.9% of their area). The mean share inside is 0.910. Averaged over the found trees, a DeepForest M02v2 box covers 0.91 of the human box's area. When one box contains the other, IoU equals this area ratio. Most of these boxes are smaller than the human boxes, but they are not consistently nested inside them. This is a difference in box-extent convention, measured against this one annotator; it does not say which extent is correct.

### False positives

**IoU ≥ 0.50: 15 false-positive boxes in total.**

| Group (IoU 0.50) | Human label | FP type | Trials | Overlapping reference | Best IoU with it |
|---|---|---|---|---|---|
| – | unlabelled | no_reference_overlap | 8 boxes in trial(s) 01 | – | 0.000 |
| – | unlabelled | localization_below_threshold | 2 boxes in trial(s) 01 | H04 | 0.024–0.029 |
| – | unlabelled | localization_below_threshold | 2 boxes in trial(s) 01 | H08 | 0.020–0.386 |
| – | unlabelled | localization_below_threshold | 1 boxes in trial(s) 01 | H02 | 0.051 |
| – | unlabelled | localization_below_threshold | 1 boxes in trial(s) 01 | H05 | 0.054 |
| – | unlabelled | localization_below_threshold | 1 boxes in trial(s) 01 | H11 | 0.020 |

**IoU ≥ 0.75: 19 false-positive boxes in total.**

| Group (IoU 0.50) | Human label | FP type | Trials | Overlapping reference | Best IoU with it |
|---|---|---|---|---|---|
| – | unlabelled | no_reference_overlap | 8 boxes in trial(s) 01 | – | 0.000 |
| – | unlabelled | localization_below_threshold | 3 boxes in trial(s) 01 | H04 | 0.024–0.558 |
| – | unlabelled | localization_below_threshold | 2 boxes in trial(s) 01 | H02 | 0.051–0.713 |
| – | unlabelled | localization_below_threshold | 2 boxes in trial(s) 01 | H08 | 0.020–0.386 |
| – | unlabelled | localization_below_threshold | 2 boxes in trial(s) 01 | H11 | 0.020–0.683 |
| – | unlabelled | localization_below_threshold | 1 boxes in trial(s) 01 | H03 | 0.606 |
| – | unlabelled | localization_below_threshold | 1 boxes in trial(s) 01 | H05 | 0.054 |

FP types: `no_reference_overlap`: does not overlap any reference tree; `localization_below_threshold`: overlaps a reference tree, but IoU is below the threshold; `reference_matched_by_other_detection`: overlaps a reference tree that another box in the same trial matched.

### Human-labelled review units against the reference

Matched through the recorded detection ids, not the group names.

| Review unit | Human label | Note | Detections | IoU ≥ 0.50: TP / FP (matched reference) | IoU ≥ 0.75: TP / FP (matched reference) |
|---|---|---|---|---|---|

### Figures

- `figures/lowalt_pilot_ptr1_27m_m02_v2/LA_PT-R1_27m/F_accuracy_by_threshold.png`
- `figures/lowalt_pilot_ptr1_27m_m02_v2/LA_PT-R1_27m/G_trial_tp_fp_fn.png`
- `figures/lowalt_pilot_ptr1_27m_m02_v2/LA_PT-R1_27m/H_reference_detection_frequency.png`
- `figures/lowalt_pilot_ptr1_27m_m02_v2/LA_PT-R1_27m/I_reference_vs_predictions_overview.png`
- `figures/lowalt_pilot_ptr1_27m_m02_v2/LA_PT-R1_27m/I_reference_vs_predictions_iou050.jpg`
- `figures/lowalt_pilot_ptr1_27m_m02_v2/LA_PT-R1_27m/J_error_cases_iou050.png`

## LA_PT-R1_27m (lowalt_pilot_ptr1_27m_m09_v1)

Reference: 11 human boxes (H01–H11), status COMPLETE. Approved by Gabe on 2026-10-30.

Each of the 10 trials is matched to the reference on its own: one-to-one, maximum number of pairs with IoU ≥ threshold, ties broken by total IoU. IoU = 1.00 requires identical boxes.

### Summary across trials

| IoU ≥ | Precision mean ± SD [range] | Recall mean ± SD [range] | F1 mean ± SD [range] | TP / FP / FN per trial (mean) | Pooled P / R / F1 | Mean IoU of TPs |
|---|---|---|---|---|---|---|
| 0.50 | 0.850 ± 0.095 [0.667–1.000] | 0.436 ± 0.038 [0.364–0.455] | 0.575 ± 0.043 [0.471–0.625] | 4.8 / 0.9 / 6.2 | 0.842 / 0.436 / 0.575 | 0.802 (n=10) |
| 0.75 | 0.638 ± 0.131 [0.500–0.833] | 0.327 ± 0.064 [0.273–0.455] | 0.431 ± 0.081 [0.353–0.588] | 3.6 / 2.1 / 7.4 | 0.632 / 0.327 / 0.431 | 0.828 (n=10) |
| 1.00 | 0.000 ± 0.000 [0.000–0.000] | 0.000 ± 0.000 [0.000–0.000] | 0.000 ± 0.000 [0.000–0.000] | 0.0 / 5.7 / 11.0 | 0.000 / 0.000 / 0.000 | – (n=0) |

SD is the sample SD (n−1) across trials. Pooled values sum TP, FP and FN over all trials.

### Per-trial results

| IoU ≥ | Trial | Boxes | TP | FP | FN | Precision | Recall | F1 | Mean IoU of TPs |
|---|---|---|---|---|---|---|---|---|---|
| 0.50 | 01 | 6 | 5 | 1 | 6 | 0.833 | 0.455 | 0.588 | 0.821 |
| 0.50 | 02 | 4 | 4 | 0 | 7 | 1.000 | 0.364 | 0.533 | 0.825 |
| 0.50 | 03 | 6 | 5 | 1 | 6 | 0.833 | 0.455 | 0.588 | 0.780 |
| 0.50 | 04 | 6 | 5 | 1 | 6 | 0.833 | 0.455 | 0.588 | 0.800 |
| 0.50 | 05 | 6 | 5 | 1 | 6 | 0.833 | 0.455 | 0.588 | 0.796 |
| 0.50 | 06 | 6 | 5 | 1 | 6 | 0.833 | 0.455 | 0.588 | 0.805 |
| 0.50 | 07 | 6 | 5 | 1 | 6 | 0.833 | 0.455 | 0.588 | 0.793 |
| 0.50 | 08 | 6 | 4 | 2 | 7 | 0.667 | 0.364 | 0.471 | 0.785 |
| 0.50 | 09 | 5 | 5 | 0 | 6 | 1.000 | 0.455 | 0.625 | 0.820 |
| 0.50 | 10 | 6 | 5 | 1 | 6 | 0.833 | 0.455 | 0.588 | 0.792 |
| 0.75 | 01 | 6 | 4 | 2 | 7 | 0.667 | 0.364 | 0.471 | 0.845 |
| 0.75 | 02 | 4 | 3 | 1 | 8 | 0.750 | 0.273 | 0.400 | 0.855 |
| 0.75 | 03 | 6 | 3 | 3 | 8 | 0.500 | 0.273 | 0.353 | 0.817 |
| 0.75 | 04 | 6 | 4 | 2 | 7 | 0.667 | 0.364 | 0.471 | 0.813 |
| 0.75 | 05 | 6 | 3 | 3 | 8 | 0.500 | 0.273 | 0.353 | 0.839 |
| 0.75 | 06 | 6 | 5 | 1 | 6 | 0.833 | 0.455 | 0.588 | 0.805 |
| 0.75 | 07 | 6 | 3 | 3 | 8 | 0.500 | 0.273 | 0.353 | 0.841 |
| 0.75 | 08 | 6 | 3 | 3 | 8 | 0.500 | 0.273 | 0.353 | 0.809 |
| 0.75 | 09 | 5 | 4 | 1 | 7 | 0.800 | 0.364 | 0.500 | 0.843 |
| 0.75 | 10 | 6 | 4 | 2 | 7 | 0.667 | 0.364 | 0.471 | 0.814 |
| 1.00 | 01 | 6 | 0 | 6 | 11 | 0.000 | 0.000 | 0.000 | – |
| 1.00 | 02 | 4 | 0 | 4 | 11 | 0.000 | 0.000 | 0.000 | – |
| 1.00 | 03 | 6 | 0 | 6 | 11 | 0.000 | 0.000 | 0.000 | – |
| 1.00 | 04 | 6 | 0 | 6 | 11 | 0.000 | 0.000 | 0.000 | – |
| 1.00 | 05 | 6 | 0 | 6 | 11 | 0.000 | 0.000 | 0.000 | – |
| 1.00 | 06 | 6 | 0 | 6 | 11 | 0.000 | 0.000 | 0.000 | – |
| 1.00 | 07 | 6 | 0 | 6 | 11 | 0.000 | 0.000 | 0.000 | – |
| 1.00 | 08 | 6 | 0 | 6 | 11 | 0.000 | 0.000 | 0.000 | – |
| 1.00 | 09 | 5 | 0 | 5 | 11 | 0.000 | 0.000 | 0.000 | – |
| 1.00 | 10 | 6 | 0 | 6 | 11 | 0.000 | 0.000 | 0.000 | – |

### How often each reference tree was found

| Reference | Size (px) | Found at IoU ≥ 0.50 | Found at IoU ≥ 0.75 | Found at IoU ≥ 1.00 | When found at 0.50: mean IoU · area ratio (model/human) · centre offset (px) | Missed at 0.50: nearest box in that trial |
|---|---|---|---|---|---|---|
| H01 | 713 × 795 | 10/10 | 10/10 | 0/10 | 0.805 · 0.80 · 9 | – |
| H02 | 704 × 785 | 10/10 | 2/10 | 0/10 | 0.737 · 0.74 · 9 | – |
| H03 | 2014 × 1850 | 10/10 | 10/10 | 0/10 | 0.849 · 0.85 · 33 | – |
| H04 | 489 × 523 | 0/10 | 0/10 | 0/10 | – | no overlapping box in 10 trial(s) |
| H05 | 165 × 151 | 0/10 | 0/10 | 0/10 | – | no overlapping box in 10 trial(s) |
| H06 | 480 × 552 | 0/10 | 0/10 | 0/10 | – | t08:T006@0.104, no overlapping box in 9 trial(s) |
| H07 | 361 × 373 | 0/10 | 0/10 | 0/10 | – | t06:T004@0.195, t07:T004@0.183, t08:T004@0.309, no overlapping box in 7 trial(s) |
| H08 | 613 × 758 | 8/10 | 4/10 | 0/10 | 0.755 · 0.88 · 47 | no overlapping box in 2 trial(s) |
| H09 | 308 × 330 | 0/10 | 0/10 | 0/10 | – | no overlapping box in 10 trial(s) |
| H10 | 221 × 222 | 0/10 | 0/10 | 0/10 | – | no overlapping box in 10 trial(s) |
| H11 | 758 × 751 | 10/10 | 10/10 | 0/10 | 0.852 · 0.89 · 25 | – |

**Box extent.** Of the 48 matched pairs at IoU ≥ 0.50, 34 gpt-6.1-sol boxes lie entirely inside the human box (≥ 99.9% of their area). The mean share inside is 0.984. Averaged over the found trees, a gpt-6.1-sol box covers 0.83 of the human box's area. When one box contains the other, IoU equals this area ratio. Most of these boxes are smaller than the human boxes, but they are not consistently nested inside them. This is a difference in box-extent convention, measured against this one annotator; it does not say which extent is correct.

### False positives

**IoU ≥ 0.50: 9 false-positive boxes in total.**

| Group (IoU 0.50) | Human label | FP type | Trials | Overlapping reference | Best IoU with it |
|---|---|---|---|---|---|
| G06 | unlabelled | no_reference_overlap | 03, 04, 05, 10 | – | 0.000 |
| G07 | unlabelled | localization_below_threshold | 06, 07 | H07 | 0.183–0.195 |
| G08 | unlabelled | localization_below_threshold | 08 | H07 | 0.309 |
| G09 | unlabelled | localization_below_threshold | 01 | H11 | 0.066 |
| G10 | unlabelled | localization_below_threshold | 08 | H06 | 0.104 |

**IoU ≥ 0.75: 21 false-positive boxes in total.**

| Group (IoU 0.50) | Human label | FP type | Trials | Overlapping reference | Best IoU with it |
|---|---|---|---|---|---|
| G03 | unlabelled | localization_below_threshold | 02, 03, 04, 05, 07, 08, 09, 10 | H02 | 0.706–0.748 |
| G05 | unlabelled | localization_below_threshold | 01, 03, 05, 07 | H08 | 0.704–0.747 |
| G06 | unlabelled | no_reference_overlap | 03, 04, 05, 10 | – | 0.000 |
| G07 | unlabelled | localization_below_threshold | 06, 07 | H07 | 0.183–0.195 |
| G08 | unlabelled | localization_below_threshold | 08 | H07 | 0.309 |
| G09 | unlabelled | localization_below_threshold | 01 | H11 | 0.066 |
| G10 | unlabelled | localization_below_threshold | 08 | H06 | 0.104 |

FP types: `no_reference_overlap`: does not overlap any reference tree; `localization_below_threshold`: overlaps a reference tree, but IoU is below the threshold; `reference_matched_by_other_detection`: overlaps a reference tree that another box in the same trial matched.

### Human-labelled review units against the reference

Matched through the recorded detection ids, not the group names.

| Review unit | Human label | Note | Detections | IoU ≥ 0.50: TP / FP (matched reference) | IoU ≥ 0.75: TP / FP (matched reference) |
|---|---|---|---|---|---|

### Figures

- `figures/lowalt_pilot_ptr1_27m_m09_v1/LA_PT-R1_27m/F_accuracy_by_threshold.png`
- `figures/lowalt_pilot_ptr1_27m_m09_v1/LA_PT-R1_27m/G_trial_tp_fp_fn.png`
- `figures/lowalt_pilot_ptr1_27m_m09_v1/LA_PT-R1_27m/H_reference_detection_frequency.png`
- `figures/lowalt_pilot_ptr1_27m_m09_v1/LA_PT-R1_27m/I_reference_vs_predictions_overview.png`
- `figures/lowalt_pilot_ptr1_27m_m09_v1/LA_PT-R1_27m/I_reference_vs_predictions_iou050.jpg`
- `figures/lowalt_pilot_ptr1_27m_m09_v1/LA_PT-R1_27m/J_error_cases_iou050.png`

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

| Reference | Size (px) | Found at IoU ≥ 0.50 | Found at IoU ≥ 0.75 | Found at IoU ≥ 1.00 | When found at 0.50: mean IoU · area ratio (model/human) · centre offset (px) | Missed at 0.50: nearest box in that trial |
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

**Box extent.** Of the 48 matched pairs at IoU ≥ 0.50, 46 claude-opus-5-5 boxes lie entirely inside the human box (≥ 99.9% of their area). The mean share inside is 0.999. Averaged over the found trees, a claude-opus-5-5 box covers 0.69 of the human box's area. When one box contains the other, IoU equals this area ratio. These boxes are therefore systematically tighter than the human boxes, which by itself limits IoU. This is a difference in box-extent convention, measured against this one annotator; it does not say which extent is correct.

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
* **Coordinate grids differ.** The models return a 0–1000 grid (one step ≈ 8.2 × 6.1 original pixels). The human
  boxes use whole pixels. An IoU of exactly 1.00 is therefore practically unreachable. That threshold measures
  exact coordinate reproduction, not detection quality.
* **Matching is per trial.** A trial's box that partly covers a tree but falls below the threshold counts as both
  a false positive and a false negative for that tree. The FP-type column separates these localization errors
  from boxes on non-tree objects.
* **Shared errors.** All trials used the same prompt, settings and downscaled 2212 × 1659 input. Trial-to-trial
  spread therefore shows sampling variability only. It is not an independent estimate of accuracy.

