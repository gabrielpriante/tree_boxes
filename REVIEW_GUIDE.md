# Review guide

*Frontline Labs urban-tree AI evaluation · experimental phase closed 9 November 2026 · for review by Dr. Elizabeth
Ondula and Dr. Kala Fleming*

This page summarises the study and points to the evidence. It takes about five minutes. All numbers below come from
the frozen result tables linked in each section. Nothing here is new analysis.

## A. Research overview

**Question.** When AI models are asked to find individual trees in urban drone imagery, how do their outputs differ
in four respects?
* **count:** how many trees they report;
* **localization:** where the boxes are;
* **agreement:** with themselves and with each other;
* **reproducibility:** whether a re-run gives the same output.

**Why counts alone mislead.** A tree count can look right for the wrong reasons. On the PT-R1 image the two
vision-language models report about 6 trees per run, with tight counts (CV 5–12%). Yet only about 5 of the 11
human-annotated trees are found, and a non-tree plant is boxed in every Claude run. On PT-02, Claude's mean count
(24.5) is close to the site's inventory count (24), yet no image reference exists there to show which boxes are
trees. DeepForest at native resolution returns thousands of perfectly reproducible boxes that match no reference
tree. Count, localization, agreement and reproducibility therefore have to be measured separately.

This is a focused methodological evaluation of four **existing** model configurations on two fixed images. No new
model was developed or trained.

## B. Experimental design

| Code | Configuration | Kind | Outputs per image |
|---|---|---|---|
| M01 | DeepForest 2.1.0 (pinned checkpoint), original 8,192 × 6,144 image | deterministic detector | 1 (reproduced in 12 executions) |
| M02v2 | Same DeepForest model, image resampled to a nominal 10 cm/pixel | deterministic detector | 1 (reproduced in 12 executions) |
| M08 | Claude Opus 5.5, high adaptive thinking, structured JSON boxes | stochastic vision-language model | 10 independent trials |
| M09 | GPT-6.1 Sol, high reasoning effort, same prompt and schema | stochastic vision-language model | 10 independent trials |

* **Images.** Two frozen drone images, checked by SHA-256 in every run:
  * `LA_PT-R1_27m`: Park Triangle Rain Garden, about 27 m altitude;
  * `LA_PT-02_64m`: Hunter Way, about 64 m altitude.
* **Inputs.** M08 and M09 received byte-identical 2,212 × 1,659 JPEGs and the identical prompt.
* **Two kinds of repetition.** The vision-language models give different answers from run to run, so their 10
  trials measure run-to-run variation. DeepForest gives the same answer every time:
  * its 10 repeated executions (plus 2 pilot runs) test **computational reproducibility**;
  * they count as **one** observation per image, never ten.
* **Reference.** PT-R1 has a human reference: 11 boxes drawn on the image by one annotator (H01–H11). PT-02 has
  none.
* **Matching.** Boxes are compared with exact IoU and one-to-one matching:
  * IoU ≥ 0.50 is primary, 0.75 a sensitivity check;
  * M08/M09 accuracy is the mean of per-trial values.
* **Discontinued.** M10 (Gemini) was optional and is not part of the study.

## C. Methodology

* [RESEARCH_PROGRESS.md](RESEARCH_PROGRESS.md): the full research record. Design (section C), experiments (D),
  findings (E), limitations (F).
* [EVALUATION.md](EVALUATION.md): metric definitions, matching, persistence groups, reference accuracy and its
  conventions, DeepForest single-output handling, reproducibility analysis.
* [BENCHMARK.md](BENCHMARK.md): the frozen M08/M09 configurations, image preprocessing and trial protocol.
* [deepforest_pipeline/README.md](deepforest_pipeline/README.md): the DeepForest code, settings, run records and
  archive checksums. Detailed DeepForest methods are in
  [docs/lowalt_pilot_v1_deepforest.md](deepforest_pipeline/docs/lowalt_pilot_v1_deepforest.md) and
  [docs/lowalt_repro_v1_deepforest.md](deepforest_pipeline/docs/lowalt_repro_v1_deepforest.md).
* **Configuration records:**
  * [manifests/manifest.csv](manifests/manifest.csv) (images and hashes);
  * [M08 config](benchmark_results/lowalt_pilot_ptr1_27m_v1/M08_high_a28fe01ac5/model_config.json);
  * [M09 config](benchmark_results/lowalt_pilot_ptr1_27m_m09_v1/M09_high_90be168b98/model_config.json);
  * [M01 config](benchmark_results/lowalt_pilot_ptr1_27m_m01_v1/M01_9c192941f5/model_config.json);
  * M02v2 config for [PT-R1](benchmark_results/lowalt_pilot_ptr1_27m_m02_v2/M02v2_b25eccbcc7/model_config.json)
    and [PT-02](benchmark_results/lowalt_pilot_pt02_64m_m02_v2/M02v2_1e05a4037f/model_config.json).
* **Human reference:**
  * [human_review/reference_boxes.csv](human_review/reference_boxes.csv);
  * [human_review/reference_status.csv](human_review/reference_status.csv);
  * the original Make Sense exports in [human_review/ground_truth/](human_review/ground_truth/).

## D. Main results

The main four-configuration results are in
[evaluation_results_lowalt_v1_deepforest/](evaluation_results_lowalt_v1_deepforest/evaluation_report.md).

**PT-R1 accuracy against the 11 reference boxes, IoU ≥ 0.50** (mean per trial for M08/M09). Source:
[accuracy_summary.csv](evaluation_results_lowalt_v1_deepforest/accuracy_summary.csv),
[accuracy_vs_reference.csv](evaluation_results_lowalt_v1_deepforest/accuracy_vs_reference.csv),
[reference_detection_frequency.csv](evaluation_results_lowalt_v1_deepforest/reference_detection_frequency.csv),
[accuracy_report.md](evaluation_results_lowalt_v1_deepforest/accuracy_report.md).

| Model | TP / FP / FN | Precision | Recall | F1 | F1 at IoU ≥ 0.75 |
|---|---|---|---|---|---|
| M01 | 0 / 4,836 / 11 | 0.000 | 0.000 | 0.000 | 0.000 |
| M02v2 | 5 / 15 / 6 | 0.250 | 0.455 | 0.323 | 0.065 |
| M08 | 4.8 / 1.3 / 6.2 | 0.786 | 0.436 | 0.561 | 0.105 |
| M09 | 4.8 / 0.9 / 6.2 | 0.850 | 0.436 | 0.575 | 0.431 |

* **Pooled values:** pooled over trials, M08 is 0.787 / 0.436 / 0.561 and M09 is 0.842 / 0.436 / 0.575.
* **Trees nobody found:** H05, H06, H09 and H10 were found by no model.
* **H04:** found only by M02v2.

**Detection counts and within-model agreement.** Sources:
[count_consistency.csv](evaluation_results_lowalt_v1_deepforest/count_consistency.csv),
[iou_summary.csv](evaluation_results_lowalt_v1_deepforest/iou_summary.csv),
[model_comparison.csv](evaluation_results_lowalt_v1_deepforest/model_comparison.csv).

| | M01 | M02v2 | M08 | M09 |
|---|---|---|---|---|
| Boxes per output, PT-R1 | 4,836 | 20 | 6.1 (CV 5.2%) | 5.7 (CV 11.8%) |
| Boxes per output, PT-02 | 6,320 | 183 | 24.5 (CV 7.5%) | 41.8 (CV 8.8%) |
| Agreement between its own trials, IoU ≥ 0.50, PT-R1 / PT-02 | n/a (deterministic) | n/a | 0.919 / 0.875 | 0.838 / 0.794 |

**Cross-model agreement** (agreement = 2 × matched boxes / total boxes, mean over output pairs). Sources:
[cross_model_agreement.csv](evaluation_results_lowalt_v1_deepforest/cross_model_agreement.csv) (box level) and
[cross_model_coverage.csv](evaluation_results_lowalt_v1_deepforest/cross_model_coverage.csv) (where the canopy is).
* **Box level, IoU ≥ 0.50:**
  * M08 vs M09: 0.828 (PT-R1), 0.627 (PT-02);
  * M02v2 vs the LLMs: 0.33–0.37 (PT-R1), 0.04–0.07 (PT-02);
  * M01 vs anything: 0.000.
* **Coverage level (coverage-mask IoU):**
  * M08 vs M09: 0.80 (PT-R1), 0.71 (PT-02);
  * M02v2 vs the LLMs: 0.32–0.33 (PT-R1), 0.47–0.50 (PT-02).

  On PT-02, M02v2 and the LLMs mark similar canopy areas but divide it into very different "trees": boxes of about
  2.7 m against about 10 m ([geometry_summary.csv](evaluation_results_lowalt_v1_deepforest/geometry_summary.csv)).

**DeepForest reproducibility.** Source:
[repro_report.md](evaluation_results_deepforest_repro_v1/repro_report.md),
[repro_summary.csv](evaluation_results_deepforest_repro_v1/repro_summary.csv),
[repro_pairwise.csv](evaluation_results_deepforest_repro_v1/repro_pairwise.csv).
* **Runs:** 40 repeated executions, 10 per configuration and image.
* **Identity:** identical prediction files and counts (4,836, 6,320, 20, 183) within every combination, and
  agreement 1.000 at IoU 0.50, 0.75 and 1.00. All were identical to the earlier pilot runs.
* **Scope:** this is reproducibility in one computing environment, not accuracy.

## E. Key figures

| Figure | What it shows |
|---|---|
| [K: model comparison, PT-R1](evaluation_results_lowalt_v1_deepforest/figures/comparison/LA_PT-R1_27m/K_model_comparison.png) | Counts, within-model agreement, accuracy and per-tree detection for all four configurations on one page, with deterministic outputs and stochastic trials labelled. |
| [L: side by side at IoU ≥ 0.50, PT-R1](evaluation_results_lowalt_v1_deepforest/figures/comparison/LA_PT-R1_27m/L_models_side_by_side_iou050.png) | Every model's boxes on the image, coloured by outcome against the human reference: which trees each model finds, misses or mislocates. |
| [M: agreement matrices, PT-02](evaluation_results_lowalt_v1_deepforest/figures/comparison/LA_PT-02_64m/M_agreement_matrices.png) ([PT-R1](evaluation_results_lowalt_v1_deepforest/figures/comparison/LA_PT-R1_27m/M_agreement_matrices.png)) | Agreement within and between all four configurations at IoU 0.50 and 0.75. Box-level agreement collapses on dense canopy. |
| [O: coverage maps, PT-02](evaluation_results_lowalt_v1_deepforest/figures/comparison/LA_PT-02_64m/O_coverage_maps.png) | Where each model places boxes. Models can agree on where the canopy is while disagreeing on the individual trees. |
| [P: counts by run](evaluation_results_deepforest_repro_v1/figures/P_counts_by_run.png) | DeepForest returns exactly the same number of detections in all 10 runs of each configuration. |
| [Q: pairwise agreement between runs, IoU ≥ 0.50](evaluation_results_deepforest_repro_v1/figures/Q_pairwise_agreement_iou050.png) | Every pair of DeepForest runs agrees completely (1.000). |
| [R: repeatability vs the LLMs](evaluation_results_deepforest_repro_v1/figures/R_repeatability_vs_llm.png) | Deterministic DeepForest repeats beside the LLMs' independent trials, labelled as different kinds of repetition. |

The human reference alone is in
[human_reference_LA_PT-R1_27m.jpg](evaluation_results_lowalt_v1_deepforest/review/human_reference_LA_PT-R1_27m.jpg).

## F. Important limitations

* **Two fixed images**, one per site. Site and altitude are confounded. LLM trial spread is run-to-run variation
  on one input, not variation across images.
* **One human image annotator.** The PT-R1 reference consists of 11 image boxes from one person. They are not
  traced tree by tree to a field inventory.
* **Annotation timing.** The reference was drawn after reviewing M08's detections, so it may not be independent of
  M08. M09 and DeepForest ran after it was frozen.
* **No PT-02 reference.** PT-02 results are counts, consistency, geometry and agreement only, with no accuracy.
* **Different effective image resolutions.** The LLMs received a downscaled 2,212 × 1,659 image. DeepForest
  received the full image (M01) or a nominal 10 cm/pixel resample (M02v2). The providers' "high" reasoning
  settings are not equivalent compute.
* **Reproducibility scope.** DeepForest reproducibility was verified on one GPU and software environment only.
* **Generality.** No conclusion is drawn about model performance in other urban environments, sites or seasons.

The full list is in [RESEARCH_PROGRESS.md, section F](RESEARCH_PROGRESS.md#f-methodological-limitations). The
raw DeepForest output folders are archived outside Git; see
[deepforest_pipeline/provenance/SOURCE.md](deepforest_pipeline/provenance/SOURCE.md).

## G. Questions for reviewers

1. Does the evaluation framework distinguish **count, localization, agreement and reproducibility** clearly
   enough?
2. Are the findings and limitations stated accurately and with appropriate caution?
3. Which results are most useful for the scientific narrative?
4. How should the work be framed for a short conference paper?
