# Detection consistency evaluation

Algorithm version 1.0.0. Generated offline from saved benchmark records; no API requests. Configuration, input hashes and software versions: `evaluation_config.json`. Methods: `EVALUATION.md`.

This report keeps four kinds of evidence apart:

1. **Count consistency**: how much the number of boxes varies between independent trials.
2. **Spatial consistency**: whether trials place boxes on the same image locations (IoU matching between trials).
3. **Count vs field inventory**: a site-level field count beside the model's mean count. Not an image-level reference.
4. **Human-validated correctness**: only from human labels and complete human reference boxes.

Items 1–3 measure repeatability or numeric proximity. They are not precision, recall or accuracy, and agreement between trials does not show that a box contains a tree.

## Inputs

| Experiment | Image | Site | Altitude (m) | Model config | Successful trials | Excluded | Counts match saved summary |
|---|---|---|---|---|---|---|---|
| lowalt_pilot_pt02_64m_v1 | LA_PT-02_64m | PT-02 | 64.3 | M08_high_a28fe01ac5 (claude-opus-5-5) | 10 | none | yes |
| lowalt_pilot_ptr1_27m_v1 | LA_PT-R1_27m | PT-R1 | 26.8 | M08_high_a28fe01ac5 (claude-opus-5-5) | 10 | none | yes |

Every trial's `detections.json` was checked against `boxes_long.csv`, `counts_long.csv` and `record.json`, and each source image against the SHA-256 recorded at inference time.

## 1. Count consistency

| Image | Counts by trial | Mean | SD (n−1) | Range | CV |
|---|---|---|---|---|---|
| LA_PT-02_64m | 23, 23, 25, 28, 25, 23, 24, 22, 26, 26 | 24.50 | 1.841 | 22–28 | 7.5% |
| LA_PT-R1_27m | 6, 6, 6, 6, 6, 6, 6, 6, 6, 7 | 6.10 | 0.316 | 6–7 | 5.2% |

## 2. Spatial consistency between trials

Each of the n(n−1)/2 trial pairs is matched one-to-one at each threshold. Agreement = 2 × matched / (boxes in trial A + boxes in trial B). IoU = 1.00 requires identical boxes.

| Image | IoU ≥ | Pairs | Agreement mean | SD | Median | Range | Pairs at 100% | Matched IoU mean | Matched IoU median |
|---|---|---|---|---|---|---|---|---|---|
| LA_PT-02_64m | 0.50 | 45 | 0.875 | 0.049 | 0.875 | 0.766–0.980 | 0 | 0.867 | 0.905 |
| LA_PT-02_64m | 0.75 | 45 | 0.721 | 0.064 | 0.711 | 0.600–0.902 | 0 | 0.913 | 0.928 |
| LA_PT-02_64m | 1.00 | 45 | 0.103 | 0.064 | 0.087 | 0.000–0.333 | 0 | 1.000 | 1.000 |
| LA_PT-R1_27m | 0.50 | 45 | 0.919 | 0.134 | 1.000 | 0.615–1.000 | 28 | 0.935 | 0.967 |
| LA_PT-R1_27m | 0.75 | 45 | 0.885 | 0.199 | 1.000 | 0.462–1.000 | 28 | 0.945 | 0.968 |
| LA_PT-R1_27m | 1.00 | 45 | 0.263 | 0.309 | 0.000 | 0.000–1.000 | 1 | 1.000 | 1.000 |

## 3. Detection persistence

Groups are recurring detections: at most one box per trial, and every two members matched directly (no chaining). Components that fail either rule are **ambiguous** and are listed rather than split.

**LA_PT-02_64m** (10 trials, 245 detections)

| IoU ≥ | groups in all 10 trials | groups in 8-9 trials | groups in 5-7 trials | groups in 1-4 trials | ambiguous groups (detections) | share of detections in all-trial groups |
|---|---|---|---|---|---|---|
| 0.50 | 17 | 2 | 3 | 4 | 5 (36) | 69% |
| 0.75 | 9 | 1 | 4 | 15 | 13 (104) | 37% |
| 1.00 | 0 | 0 | 4 | 171 | 0 (0) | 0% |

**LA_PT-R1_27m** (10 trials, 61 detections)

| IoU ≥ | groups in all 10 trials | groups in 8-9 trials | groups in 5-7 trials | groups in 1-4 trials | ambiguous groups (detections) | share of detections in all-trial groups |
|---|---|---|---|---|---|---|
| 0.50 | 4 | 2 | 0 | 3 | 0 (0) | 66% |
| 0.75 | 3 | 3 | 0 | 4 | 0 (0) | 49% |
| 1.00 | 0 | 0 | 5 | 29 | 0 (0) | 0% |

Ambiguous groups (full membership in `detection_persistence.csv`):

| Image | IoU ≥ | Group | Detections | Trials | Why ambiguous |
|---|---|---|---|---|---|
| LA_PT-02_64m | 0.50 | A01 | 10 | 10 | 23 of 45 member pairs not matched |
| LA_PT-02_64m | 0.50 | A02 | 9 | 9 | 1 of 36 member pairs not matched |
| LA_PT-02_64m | 0.50 | A03 | 9 | 9 | 2 of 36 member pairs not matched |
| LA_PT-02_64m | 0.50 | A04 | 5 | 5 | 3 of 10 member pairs not matched |
| LA_PT-02_64m | 0.50 | A05 | 3 | 3 | 1 of 3 member pairs not matched |
| LA_PT-02_64m | 0.75 | A01 | 10 | 10 | 1 of 45 member pairs not matched |
| LA_PT-02_64m | 0.75 | A02 | 10 | 10 | 2 of 45 member pairs not matched |
| LA_PT-02_64m | 0.75 | A03 | 10 | 10 | 15 of 45 member pairs not matched |
| LA_PT-02_64m | 0.75 | A04 | 10 | 10 | 13 of 45 member pairs not matched |
| LA_PT-02_64m | 0.75 | A05 | 10 | 10 | 1 of 45 member pairs not matched |
| LA_PT-02_64m | 0.75 | A06 | 10 | 10 | 16 of 45 member pairs not matched |
| LA_PT-02_64m | 0.75 | A07 | 10 | 10 | 18 of 45 member pairs not matched |
| LA_PT-02_64m | 0.75 | A08 | 7 | 7 | 8 of 21 member pairs not matched |
| LA_PT-02_64m | 0.75 | A09 | 6 | 6 | 6 of 15 member pairs not matched |
| LA_PT-02_64m | 0.75 | A10 | 6 | 6 | 5 of 15 member pairs not matched |
| LA_PT-02_64m | 0.75 | A11 | 6 | 6 | 2 of 15 member pairs not matched |
| LA_PT-02_64m | 0.75 | A12 | 5 | 5 | 4 of 10 member pairs not matched |
| LA_PT-02_64m | 0.75 | A13 | 4 | 4 | 3 of 6 member pairs not matched |

## 4. Count vs field inventory (site level)

| Image | Mean model count | Site field inventory | Difference | Image coverage |
|---|---|---|---|---|
| LA_PT-02_64m | 24.5 | 24 | +0.5 | UNKNOWN |
| LA_PT-R1_27m | 6.1 | 11 | -4.9 | UNKNOWN |

The inventory counts the whole site. Image coverage is not established, so this is not an image-level reference, and a close count does not show that the boxes are the right trees. The inventory is never used to filter or correct predictions.

## 5. Human validation

Review units: 71 (persistence groups at IoU ≥ 0.50; ambiguous groups are split into their individual detections for review). Labelled so far: 9.


| Image | Persistence | Label | Units |
|---|---|---|---|
| LA_PT-R1_27m | 1-4 | TREE | 1 |
| LA_PT-R1_27m | 1-4 | UNCERTAIN | 2 |
| LA_PT-R1_27m | 8-9 | TREE | 2 |
| LA_PT-R1_27m | all 10 | NON_TREE | 1 |
| LA_PT-R1_27m | all 10 | TREE | 3 |

TREE units found in every trial are consistently detected trees; NON_TREE units found in many trials are repeated false positives; TREE units in fewer than all trials are trees the model detects inconsistently.

- lowalt_pilot_ptr1_27m_v1/LA_PT-R1_27m: 5 of 11 human reference trees were not detected in any trial at IoU ≥ 0.50.

**Accuracy against a human reference:** computed for images with a COMPLETE reference; see `accuracy_report.md`.
- LA_PT-02_64m: reference status NONE; precision/recall/F1 not computed

Preliminary observations reported by the project lead (free text; not machine-readable labels and not linked to detection ids until confirmed in `manual_review.csv`):

- PT-R1: one object detected repeatedly appears to be a plant rather than a tree.
- PT-R1: the extra seventh detection in trial 10 appears to be a genuine tree.
- PT-02: individual detections visibly disagree more between trials.

## 6. Figures

**lowalt_pilot_pt02_64m_v1/LA_PT-02_64m**

- `figures/lowalt_pilot_pt02_64m_v1/LA_PT-02_64m/A_threshold_comparison.png`
- `figures/lowalt_pilot_pt02_64m_v1/LA_PT-02_64m/B_persistence_overview.png`
- `figures/lowalt_pilot_pt02_64m_v1/LA_PT-02_64m/B_persistence_map_iou050.jpg`
- `figures/lowalt_pilot_pt02_64m_v1/LA_PT-02_64m/B_persistence_map_iou075.jpg`
- `figures/lowalt_pilot_pt02_64m_v1/LA_PT-02_64m/B_persistence_map_iou100.jpg`
- `figures/lowalt_pilot_pt02_64m_v1/LA_PT-02_64m/C_box_variation_iou050.png`
- `figures/lowalt_pilot_pt02_64m_v1/LA_PT-02_64m/D_agreement_matrix.png`
- `figures/lowalt_pilot_pt02_64m_v1/LA_PT-02_64m/E_count_consistency.png`

**lowalt_pilot_ptr1_27m_v1/LA_PT-R1_27m**

- `figures/lowalt_pilot_ptr1_27m_v1/LA_PT-R1_27m/A_threshold_comparison.png`
- `figures/lowalt_pilot_ptr1_27m_v1/LA_PT-R1_27m/B_persistence_overview.png`
- `figures/lowalt_pilot_ptr1_27m_v1/LA_PT-R1_27m/B_persistence_map_iou050.jpg`
- `figures/lowalt_pilot_ptr1_27m_v1/LA_PT-R1_27m/B_persistence_map_iou075.jpg`
- `figures/lowalt_pilot_ptr1_27m_v1/LA_PT-R1_27m/B_persistence_map_iou100.jpg`
- `figures/lowalt_pilot_ptr1_27m_v1/LA_PT-R1_27m/C_box_variation_iou050.png`
- `figures/lowalt_pilot_ptr1_27m_v1/LA_PT-R1_27m/D_agreement_matrix.png`
- `figures/lowalt_pilot_ptr1_27m_v1/LA_PT-R1_27m/E_count_consistency.png`
- `figures/lowalt_pilot_ptr1_27m_v1/LA_PT-R1_27m/F_accuracy_by_threshold.png`
- `figures/lowalt_pilot_ptr1_27m_v1/LA_PT-R1_27m/G_trial_tp_fp_fn.png`
- `figures/lowalt_pilot_ptr1_27m_v1/LA_PT-R1_27m/H_reference_detection_frequency.png`
- `figures/lowalt_pilot_ptr1_27m_v1/LA_PT-R1_27m/I_reference_vs_predictions_overview.png`
- `figures/lowalt_pilot_ptr1_27m_v1/LA_PT-R1_27m/I_reference_vs_predictions_iou050.jpg`
- `figures/lowalt_pilot_ptr1_27m_v1/LA_PT-R1_27m/J_error_cases_iou050.png`

Review crops: `review/crops/`; contact sheets: `review/contact_sheet_*.png`.

## 7. Across-altitude organisation (no comparisons made)

| Image | Site | Altitude (m) | Yaw (°) | Heading flagged | Est. GSD original / API (mm/px) | Model outputs |
|---|---|---|---|---|---|---|
| LA_PT-02_64m | PT-02 | 64.3 | +152.40 |  | 11.3 / 41.9 | lowalt_pilot_pt02_64m_v1 |
| LA_PT-02_66m | PT-02 | 66.2 | +152.40 |  | 11.7 / 43.2 | – |
| LA_PT-02_67m | PT-02 | 67.2 | +152.50 |  | 11.8 / 43.8 | – |
| LA_PT-02_71m | PT-02 | 71.2 | +155.50 |  | 12.5 / 46.4 | – |
| LA_PT-02_86m | PT-02 | 85.8 | -116.30 | yes | 15.1 / 55.9 | – |
| LA_PT-R1_27m | PT-R1 | 26.8 | -118.70 |  | 4.7 / 17.5 | lowalt_pilot_ptr1_27m_v1 |
| LA_PT-R1_28m | PT-R1 | 27.8 | -116.40 |  | 4.9 / 18.1 | – |
| LA_PT-R1_30m_angled | PT-R1 | 30.1 | +153.70 | yes | 5.3 / 19.6 | – |
| LA_PT-R1_30m | PT-R1 | 30.2 | -116.40 |  | 5.3 / 19.7 | – |
| LA_PT-R1_31m | PT-R1 | 31.2 | -116.40 |  | 5.5 / 20.3 | – |

GSD is estimated from height above the takeoff point and the 35 mm-equivalent focal length, so it is approximate. Boxes from different images are not compared: no validated alignment exists. The evaluation code refuses such comparisons (`CrossImageComparisonError`). See `EVALUATION.md` for the design.

