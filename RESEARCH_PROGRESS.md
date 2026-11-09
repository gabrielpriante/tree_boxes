# Research Progress: Repeated-Observation Benchmark of Computer Vision for Urban Trees

*Frontline Labs · Park Triangle urban-tree benchmark · progress report as of 5 November 2026*

This report covers the work held in this repository. It draws only on the saved inference records,
evaluation outputs and human-review files cited in the final section. All findings are **preliminary and
image-specific**: they come from 40 model trials (two model families × two drone images × 10 trials), plus one
deterministic output from each of two DeepForest configurations on the same two images, with one completed human
reference.

**Image approval.** On 5 November 2026 the project lead personally reviewed both pilot images, `LA_PT-R1_27m` and
`LA_PT-02_64m`, and **approved them for the cross-model benchmark** as the closest achievable site-focused images.
The image-suitability decision is closed. The exact files and SHA-256 hashes are preserved for every
matched-image experiment. The frames include some surrounding context, which is recorded as a methodological
limitation (section F), not as a blocker.

---

## A. Executive Summary

**Research question.** *How consistently do different computer-vision approaches characterize urban vegetation
across repeated standardized observations of the same sites?*

**What has been built.** We now have a reproducible, budget-controlled pipeline. It sends a fixed drone image to
a vision-language model many times under identical settings, and records every response with full provenance.
The outputs are then analysed offline in four separate ways, which answer four different questions:

1. **Count consistency:** does the number of trees reported stay the same from trial to trial?
2. **Spatial repeatability:** do the trials put boxes on the *same places*?
3. **Semantic correctness:** are the boxed objects actually trees, as judged by a human reviewer?
4. **Human-reference accuracy:** how many of the trees a human identified does the model find, and how well
   does it outline them?

**What has been run.** The first model in the roster, M08 (Claude Opus 5.5 at a frozen configuration), was run
10 times on each of two images from the 24 October altitude series:
* **PT-R1** at 26.8 m;
* **PT-02** at 64.3 m.

All 20 trials succeeded, for a total API cost of **$0.92**. For PT-R1, a human annotator outlined 11 trees, and
that reference was approved as complete. It is the only image with a completed reference.

The second model, M09 (OpenAI GPT-6.1 Sol at high reasoning effort), was then run 10 times on **each of the same
two images**. It received byte-identical image input and the identical prompt, and all 20 trials succeeded for
**$1.97**. This completes the first two-site comparison between model families: **40 trials in total**.

**Most important results.**

* **Stable counts can hide unstable detections.** Both images gave tight counts: PT-R1 averaged 6.1 trees
  (CV 5.2%) and PT-02 averaged 24.5 (CV 7.5%). Yet at IoU ≥ 0.50, none of PT-02's 45 trial pairs boxed exactly
  the same set of objects. In PT-R1, one trial returned the usual six boxes, but three of them were misplaced.
* **Repeatable is not the same as correct.** On PT-R1, eight of the ten trials produced spatially equivalent
  outputs. Even so, the model found on average **4.8 of the 11** reference trees, giving recall 0.44 at
  IoU ≥ 0.50. It also boxed one non-tree plant **in every trial**. Five reference trees were never touched by
  any box in any trial.
* **Box extent matters as much as detection.** In 46 of the 48 matched cases, the model's box lay entirely
  inside the human box, covering on average 69% of its area. Accuracy therefore drops sharply from IoU ≥ 0.50
  (F1 0.56) to IoU ≥ 0.75 (F1 0.11). This reflects a difference in box-drawing convention as much as
  detection skill.
* **PT-02's count of 24.5 is close to the field inventory of 24 trees, but this is not evidence of accuracy.**
  No human reference exists for PT-02 yet, and the image's coverage of the inventoried site is unconfirmed.
* **Two model families, one image: similar detection accuracy, different repeatability, different box extent.**
  On PT-R1 at IoU ≥ 0.50, the two models perform almost identically (F1 0.56 for Claude, 0.57 for GPT-6.1 Sol;
  recall 0.44 for both).
  * **Same omissions:** both find the same five trees and **both never find the same five reference trees**.
  * **Repeatability:** Claude is more repeatable from run to run (spatial agreement 0.92 vs 0.84).
  * **Box extent:** GPT-6.1 Sol's boxes cover more of each human box (83% vs 69% of its area), so it keeps far
    more matches at IoU ≥ 0.75 (F1 0.43 vs 0.11).

  This is one image, one annotator and two configurations. It does not show that either model is better in
  general.
* **On the dense-canopy image (PT-02), the two models characterize the scene very differently.**
  * **Counts:** GPT-6.1 Sol returns 41.8 boxes per trial on average, against Claude's 24.5.
  * **Coverage:** its boxes cover about 62% of the image per trial, against 46%, at a similar median box size.
  * **Agreement between models** falls from 0.83 on PT-R1 to **0.63** on PT-02 at IoU ≥ 0.50, while each model
    still agrees with itself at 0.79–0.88. On this image, which model is used changes the result more than
    re-running the same model.
  * **No accuracy:** PT-02 has no validated human reference, so **neither model's PT-02 detections, nor their
    closeness to the site inventory, are evidence of accuracy**.
* **DeepForest's result depends almost entirely on input scale (5 November 2026).** The pinned pretrained
  DeepForest checkpoint was run once per configuration on the same two images. Each output was bit-identical on a
  verification rerun.
  * **Native resolution (M01):** 4,836 boxes on PT-R1 and 6,320 on PT-02, with median sides of 8 cm and 18 cm.
    These are leaf- and texture-sized fragments, not crowns. M01 matched **none** of the 11 PT-R1 reference trees
    at any threshold, and none of the LLM boxes on either image.
  * **Resampled to a nominal 10 cm/px (M02 v2):**
    * PT-R1: 20 boxes. Recall at IoU ≥ 0.50 was **0.45**, the LLMs' level, but precision only 0.25 (F1 0.32 vs
      0.56–0.57). It found **H04**, one of the five trees neither LLM ever found, at IoU 0.56. H05, H06, H09 and
      H10 stayed undetected.
    * PT-02: 183 boxes of about 2.7 m, against the LLMs' 25–42 boxes of about 10 m. They mostly subdivide areas
      the LLMs box as single crowns: about 5.8 DeepForest boxes per Claude box. Agreement with either LLM is near
      zero at IoU ≥ 0.50 (0.04–0.07).
  * **Scope:** one output per configuration on two images. It shows how strongly DeepForest's behaviour depends on
    resolution here. It is not a general ranking against the LLMs, which also saw a different (2212 × 1659)
    version of each image.

**State of the study today.**
* **Repeatability and cross-model agreement** are complete for both pilot images with M08 and M09.
* **Human-reference accuracy** is complete for PT-R1 only, for both models.
* **Approved images:** Dr. Elizabeth Ondula advised prioritizing imagery focused on the study site over imagery
  with substantial surrounding areas. The project lead has reviewed both pilot images and approved them as the
  closest achievable site-focused images. They are used unchanged, with no boundary mask or crop, for all
  matched-image comparisons.
* **DeepForest M01 and M02 v2** are complete on both approved images: one verified deterministic output per
  configuration, compared with every M08 and M09 trial, and with the PT-R1 reference.
* **Not yet done:**
  * comparisons across altitudes have a written design only;
  * the 20-image weekly collection is catalogued but not benchmarked.

---

## B. Research Background

### The Park Triangle project and its two sites

The benchmark uses drone imagery of two Park Triangle sites, **PT-R1** and **PT-02**. The broader study asks
how trustworthy environmental knowledge emerges when human observation, computer vision, drone remote sensing
and geospatial systems are combined. This benchmark contributes the computer-vision part. It does not take a
model's output as fact, but measures how stable that output is when nothing about the input changes, and then
checks it against human observation.

This repository does not document the wider project's history, site selection or the intervention behind the
weekly sessions. Those parts of the background should come from the project's own records, and nothing is
assumed about them here. What the repository does record is in the manifest
(`manifests/manifest.csv`):

| Collection | Images | Dates (2026) | Altitude (relative to takeoff) | Notes |
|---|---|---|---|---|
| Weekly sessions (`standard`) | 20: weeks 1, 3, 4, 5 and 6, both sites | 5 Sep – 10 Oct | ≈ 95 m | Weeks 3–6 are labelled BEFORE and AFTER (morning and afternoon flights). Week 1 is labelled MORNING_START / AFTERNOON_END, and how those labels map to BEFORE/AFTER is still unconfirmed. |
| Altitude series (`low_altitude`) | 10: five per site | 24 Oct, 09:51–10:12 | PT-R1 26.8–31.2 m; PT-02 64.3–85.8 m | One frame per site has its heading rotated by about 90° (`LA_PT-R1_30m_angled`, `LA_PT-02_86m`). |

All 30 images are 8192 × 6144-pixel nadir photographs from the same DJI camera (24 mm 35 mm-equivalent lens).
`coverage_scope` (whether a frame shows the whole site) is still `UNKNOWN` for every image.

### How the two sites differ in the benchmarked images

The two benchmarked frames differ in several ways at once:

| | PT-R1 (`LA_PT-R1_27m`) | PT-02 (`LA_PT-02_64m`) |
|---|---|---|
| Relative altitude | 26.8 m | 64.3 m |
| Estimated ground footprint | ≈ 39 × 29 m | ≈ 93 × 70 m |
| Estimated ground sampling distance, original / as seen by the model | ≈ 4.7 / 17.5 mm per pixel | ≈ 11.3 / 41.9 mm per pixel |
| Visual character | small planted lot: young, mostly separated trees, shrubs, sidewalk and street | dense, often touching canopy between houses and streets |
| Site field inventory (site level) | 11 trees | 24 trees |

Footprint and GSD are estimates from relative altitude and focal length (`evaluation_results/altitude_series.csv`).
The visual descriptions come from inspecting the images.

These differences shape everything that follows. Site, altitude, image scale and canopy structure all differ
between the two images, so **no difference between the two results can be attributed to any single one of
these factors**.

### Objectives of the computer-vision benchmark

1. Quantify how much a model's tree detections vary when the *same* image is analysed repeatedly under fixed
   settings.
2. Separate counting stability from spatial stability, and both from correctness.
3. Build human-validated references so that correctness can be measured, not assumed.
4. Prepare a fair, reproducible basis for comparing model families. Later, extend the comparison to repeated
   observations of the same site under different flight conditions.

---

## C. Experimental Design

### Model roster

| Code | Approach | Status in this repository |
|---|---|---|
| **M08** | Claude Opus 5.5 (vision-language model, structured JSON output) | **Completed** for two images (20 trials) |
| **M01, M02 v2** | DeepForest 2.1.0, pinned `weecology/deepforest-tree` checkpoint (M01 native resolution; M02 v2 resampled to a nominal 10 cm/px) | **Completed** on the two pilot images (5 November 2026): one verified deterministic output per configuration, run in `urban-tree-vision-baselines` and imported here. The historical 95 m DeepForest results (M02 at 7.89 cm/px) remain separate and are not compared. |
| **M09** | OpenAI GPT-6.1 Sol (vision-language model, Responses API, high reasoning effort, structured JSON output) | **Completed** for two images (20 trials, 4 November 2026) |
| M10 | Gemini 3.1 Pro Preview | **Discontinued** (9 November 2026). An adapter was implemented and tested offline (5 November), but Google AI Studio requires a $30 minimum initial payment, so the work was replaced by a 10-run DeepForest reproducibility experiment. No Google API request was made and no Gemini result exists. The implementation was removed from the active code (it remains in Git history: commits `833c63e`–`cf2da02`). |
| Copilot | Microsoft Copilot | **Deferred**: no programmatic endpoint identified |
| M03–M07 | — | Not documented in this repository |

### Frozen M08 configuration

M08 has a single configuration. It is stored with every result as configuration id `M08_high_a28fe01ac5`, and
budget pressure can never change it (`BENCHMARK.md`; `model_config.json` in each results folder).

| Setting | Value |
|---|---|
| Model | `claude-opus-5-5` |
| Reasoning effort / thinking | high / adaptive |
| Maximum output tokens | 32,000 |
| Output | Structured JSON against a fixed schema (schema SHA-256 `963a771d…`) |
| Prompt | Fixed annotation prompt template (SHA-256 `e53504b3…`). Site, week and condition labels are inserted. |
| Coordinates | Integer boxes on a 0–1000 grid, `[x_min, y_min, x_max, y_max]` |

### M09 configuration and comparability with M08

M09 uses configuration id `M09_high_90be168b98` (details in `BENCHMARK.md`). It sends exactly what M08 sends:
* the **same image bytes**: the same 2212 × 1659 JPEG, verified by SHA-256 against every M08 trial;
* the **same prompt text**;
* the **same JSON schema**, enforced by strict structured output.

It runs `gpt-6.1-sol` through the OpenAI Responses API with:
* `reasoning.effort = high` and 32,000 maximum output tokens;
* `detail: original` and image context compression disabled;
* `store: false`, prompt caching off, and no tools, instructions or request chaining.

No reasoning summaries are requested. Every trial echoed back the served model `gpt-6.1-sol`, high effort,
compression disabled and `store: false`. Some differences cannot be removed:
* "high" reasoning effort is not the same amount of computation for the two providers;
* each provider encodes the image its own way, and OpenAI's internal processing resolution cannot be
  observed, so token counts are not evidence of resolution;
* Claude's responses contain readable thinking text, whereas M09's contain only opaque, encrypted reasoning
  items and reasoning-token counts.

### Image preprocessing

The API resizes large images to a fixed visual-token budget. In an early version of the pipeline
(preprocessing v1), images were sent at 2576 × 1932 pixels and silently resized again by the API, by a method we
could not see. Version 2, used for every benchmark trial, resizes locally and deterministically:

* **Resize:** each 8192 × 6144 image goes to exactly **2212 × 1659 pixels** (4,740 visual tokens) with LANCZOS
  resampling.
* **Encoding:** saved as JPEG quality 95.
* **Server-side resizing:** each request asks the API to *reject* rather than resize an oversized image.

Every trial of an image therefore sends byte-identical input, and the recorded upload hash confirms this. The
model sees each image at about 1/3.7 of its original linear resolution. All analysis and drawing uses the
original full-resolution images.

### Repeated-trial protocol

Each image is analysed in **10 independent trials**. Every trial is a new, single-message request with the same
image bytes, prompt, schema and settings. No conversation history or earlier result is passed between trials.
Only failures that never produced a result (network or server errors) are retried. A trial that returns an
invalid response, refuses, or hits the token limit stays recorded as that outcome.

For every trial the pipeline stores:
* the raw API response;
* the validated boxes;
* a full-resolution annotated image;
* token usage and cost;
* the image, prompt and schema hashes;
* the configuration.

### Reproducibility, provenance and spending controls

* **Manifest:** a manifest records every image's path, SHA-256 hash, site, week, condition and capture metadata.
  Images whose prompt-relevant labels are unconfirmed are blocked from paid runs.
* **Spending ledger:** a project-wide ledger records a worst-case cost reservation before each request and the
  actual cost after it. A request starts only if the worst case fits under the cumulative limit, which was $5
  for this phase. Spending outside this project (for example in the Console) is not visible to it, so the
  ledger is a local safeguard, not an account-level cap.
* **Run controls:**
  * runs can be resumed without repeating completed trials;
  * a round-robin mode can alternate images;
  * a safeguard pauses after three consecutive invalid, refused or truncated trials.
* **Offline evaluation:** the evaluation runs entirely offline from the saved records. It re-checks every trial
  against the saved CSVs and the source-image hash, and writes its own configuration, input hashes and software
  versions (`evaluation_results/evaluation_config.json`).

---

## D. Completed Experiments

### The 20 paid M08 trials

| Experiment | Image | Trials | Status | Output tokens per trial | Cost |
|---|---|---|---|---|---|
| `lowalt_pilot_ptr1_27m_v1` | `LA_PT-R1_27m` (26.8 m) | 10 | all `success`, no retries | 200–564 (mean 450) | $0.3439 |
| `lowalt_pilot_pt02_64m_v1` | `LA_PT-02_64m` (64.3 m) | 10 | all `success`, no retries | 1,496–1,795 (mean 1,601) | $0.5741 |

Both runs took place on 30 October 2026. Input was about 6,345 tokens per trial for both images. The ledger
records **$0.91794** spent in total, all confirmed from actual API usage.

One earlier single-request run (`output/tree_counts.csv`, 33 boxes) predates the benchmark pipeline and used
preprocessing v1. Its metadata row is also inconsistent: it labels the file `PT_02.JPG` as site PT R1. It is
**not** part of the benchmark and is not used here.

### Count consistency

| Image | Counts, trials 1–10 | Mean | SD (n−1) | Range | CV |
|---|---|---|---|---|---|
| PT-R1 | 6, 6, 6, 6, 6, 6, 6, 6, 6, 7 | 6.10 | 0.316 | 6–7 | 5.2% |
| PT-02 | 23, 23, 25, 28, 25, 23, 24, 22, 26, 26 | 24.50 | 1.841 | 22–28 | 7.5% |

*Source: `evaluation_results/count_consistency.csv`, verified against each experiment's `counts_summary.csv`.*

### Spatial repeatability

Counting tells us nothing about *where* the boxes are. For spatial repeatability, every pair of trials (45 pairs
per image) is compared box by box:
* **Matching:** each box can pair with at most one box in the other trial. The method maximizes the number of
  pairs above an IoU threshold, then their total IoU.
* **Agreement:** a pair's agreement is 2 × matched boxes ÷ total boxes in the two trials. A value of 1.0 means
  the two trials boxed the same set of objects.
* **Thresholds:** IoU ≥ 0.50 (same object, roughly outlined), IoU ≥ 0.75 (closely outlined) and IoU = 1.00
  (identical coordinates).

| Image | IoU | Mean agreement ± SD | Range | Pairs in full agreement | Mean IoU of matched boxes |
|---|---|---|---|---|---|
| PT-R1 | ≥ 0.50 | 0.919 ± 0.134 | 0.615–1.000 | **28 / 45** | 0.935 |
| PT-R1 | ≥ 0.75 | 0.885 ± 0.199 | 0.462–1.000 | 28 / 45 | 0.945 |
| PT-R1 | = 1.00 | 0.263 ± 0.309 | 0–1.000 | 1 / 45 | 1.000 |
| PT-02 | ≥ 0.50 | 0.875 ± 0.049 | 0.766–0.980 | **0 / 45** | 0.867 |
| PT-02 | ≥ 0.75 | 0.721 ± 0.064 | 0.600–0.902 | 0 / 45 | 0.913 |
| PT-02 | = 1.00 | 0.103 ± 0.064 | 0–0.333 | 0 / 45 | 1.000 |

*Source: `evaluation_results/iou_summary.csv`; figure A for each image.*

**Persistence** asks which locations recur across trials. A *recurring location* is a group of boxes, at most one
per trial, in which every member matches every other member directly. Chains (A matches B, B matches C, but A
does not match C) are flagged as ambiguous rather than merged.

| Image | IoU | Found in all 10 trials | 8–9 | 5–7 | 1–4 | Ambiguous groups (boxes) |
|---|---|---|---|---|---|---|
| PT-R1 | ≥ 0.50 | 4 | 2 | 0 | 3 | 0 |
| PT-R1 | ≥ 0.75 | 3 | 3 | 0 | 4 | 0 |
| PT-02 | ≥ 0.50 | 17 | 2 | 3 | 4 | 5 (36 of 245 boxes) |
| PT-02 | ≥ 0.75 | 9 | 1 | 4 | 15 | 13 (104 boxes) |

*Source: `evaluation_results/persistence_summary.csv`, `detection_persistence.csv`; figures B–D.*

![PT-02 spatial agreement across thresholds](evaluation_results/figures/lowalt_pilot_pt02_64m_v1/LA_PT-02_64m/A_threshold_comparison.png)

### The PT-R1 human reference

* **Semantic review:** a human reviewer first labelled the recurring PT-R1 locations as **TREE** (6), **NON_TREE**
  (1) or **UNCERTAIN** (2) (`human_review/manual_review.csv`). Both UNCERTAIN units were later identified as
  misplaced trial-6 boxes on trees that other units already cover.
* **Independent reference:** the same person then annotated the image independently in Make Sense, outlining
  **11 trees** (H01–H11). The VOC export is authoritative.
* **Import checks:** the import compared the Pascal VOC and CSV exports. Five boxes differed by exactly one
  pixel on one edge, apparently from rounding. This was recorded, and the VOC coordinates were used unchanged.
* **Approval:** the reference was approved as **complete** on 30 October 2026
  (`human_review/reference_status.csv`).

The annotator found 11 trees in the image, the same number as the PT-R1 site inventory. Whether these are the
same 11 trees has not been formally recorded.

### Human-reference accuracy (PT-R1 only)

Each trial is matched to the 11 reference boxes on its own, with the same one-to-one matching used above.

| IoU | Precision | Recall | F1 | Mean TP / FP / FN per trial | Mean IoU of matched boxes |
|---|---|---|---|---|---|
| ≥ 0.50 | **0.786** ± 0.159 (0.333–0.857) | **0.436** ± 0.094 (0.182–0.545) | **0.561** ± 0.117 (0.235–0.667) | 4.8 / 1.3 / 6.2 | 0.663 |
| ≥ 0.75 | 0.148 ± 0.052 (0–0.167) | 0.082 ± 0.029 (0–0.091) | 0.105 ± 0.037 (0–0.118) | 0.9 / 5.2 / 10.1 | 0.800 |
| = 1.00 | 0 | 0 | 0 | 0 / 6.1 / 11.0 | — |

*Mean ± SD (n−1) across 10 trials, with the range in brackets. Pooling all trials gives the same precision,
recall and F1. Source: `evaluation_results/accuracy_summary.csv`, `accuracy_vs_reference.csv`,
`accuracy_report.md`.*

At IoU ≥ 0.50, by trial:
* trials 1–5 and 7–9 each scored 5 TP / 1 FP / 6 FN (F1 0.588);
* trial 6 scored 2 / 4 / 9 (F1 0.235);
* trial 10 scored 6 / 1 / 5 (F1 0.667).

### M09 (GPT-6.1 Sol) on PT-R1, compared with M08

Experiment `lowalt_pilot_ptr1_27m_m09_v1` ran on 4 November 2026. The first trial was run alone and reviewed.
Trials 2–10 then ran on resume, with trial 1 kept byte-identical. All 10 trials succeeded on their first
attempt. One trial needed a connection retry before reaching the model, with identical settings.

Both models saw the same pixels, the same prompt and the same 11-tree reference, and are evaluated with the same
code.

| Measure (PT-R1, 10 trials each) | M08 Claude Opus 5.5 | M09 GPT-6.1 Sol |
|---|---|---|
| **Count consistency:** counts, trials 1–10 | 6,6,6,6,6,6,6,6,6,7 | 6,4,6,6,6,6,6,6,5,6 |
| Mean (SD) · CV | 6.1 (0.32) · 5.2% | 5.7 (0.67) · 11.8% |
| **Run-to-run spatial agreement**, IoU ≥ 0.50 (pairs in full agreement) | 0.919 ± 0.134 (28/45) | 0.838 ± 0.101 (7/45) |
| IoU ≥ 0.75 | 0.885 ± 0.199 (28/45) | 0.816 ± 0.097 (4/45) |
| IoU = 1.00 | 0.263 ± 0.309 (1/45) | 0 (0/45) |
| **Human-reference accuracy**, IoU ≥ 0.50: precision / recall / F1 | 0.786 / 0.436 / 0.561 | 0.850 / 0.436 / 0.575 |
| IoU ≥ 0.75 | 0.148 / 0.082 / 0.105 | 0.638 / 0.327 / 0.431 |
| IoU = 1.00 | 0 / 0 / 0 | 0 / 0 / 0 |
| **Localization**: mean IoU of matched boxes (IoU ≥ 0.50) | 0.663 | 0.802 |
| Matched boxes lying entirely inside the human box | 46 of 48 | 34 of 48 |
| Mean share of the human box's area covered | 0.69 | 0.83 |
| Non-tree object (human-labelled NON_TREE) boxed | 10 of 10 trials | 4 of 10 trials |
| Cost · mean output tokens per trial | $0.3439 · 450 | $0.5353 · 4,316 (4,133 reasoning) |

Accuracy values are means over trials. The SDs are in `accuracy_summary.csv`; at IoU ≥ 0.50, M09's F1 is
0.575 ± 0.043 and M08's is 0.561 ± 0.117.

**Agreement between the two models.** Across all 100 M08 × M09 trial pairs, mean agreement is:
* 0.828 ± 0.158 at IoU ≥ 0.50;
* 0.552 ± 0.191 at IoU ≥ 0.75;
* 0 at IoU = 1.00.

The two models therefore usually box the same objects, but outline them differently.

*Sources: `evaluation_results/model_comparison.csv`, `cross_model_agreement.csv`, `accuracy_summary.csv`,
`reference_detection_frequency.csv`, `prediction_outcomes.csv`; figures K and L in
`evaluation_results/figures/comparison/LA_PT-R1_27m/`.*

![PT-R1: M08 vs M09](evaluation_results/figures/comparison/LA_PT-R1_27m/K_model_comparison.png)

### M09 (GPT-6.1 Sol) on PT-02, compared with M08 (consistency only)

Experiment `lowalt_pilot_pt02_64m_m09_v1` ran as one batch of 10 trials on 4 November 2026. All 10 succeeded on
the first attempt with no retries. Image bytes and prompt were identical to the M08 PT-02 run. **PT-02 has no
complete human reference, so no precision, recall or F1 was computed**, and agreement or count closeness is not
accuracy.

| Measure (PT-02, 10 trials each) | M08 Claude Opus 5.5 | M09 GPT-6.1 Sol |
|---|---|---|
| Counts, trials 1–10 | 23, 23, 25, 28, 25, 23, 24, 22, 26, 26 | 43, 45, 37, 39, 39, 48, 40, 38, 45, 44 |
| Mean (SD) · CV | 24.5 (1.84) · 7.5% | 41.8 (3.68) · 8.8% |
| **Within-model agreement**, IoU ≥ 0.50 | 0.875 ± 0.049 | 0.794 ± 0.059 |
| IoU ≥ 0.75 | 0.721 ± 0.064 | 0.475 ± 0.064 |
| IoU = 1.00 | 0.103 ± 0.064 | 0 |
| Trial pairs in full agreement (any threshold) | 0 of 45 | 0 of 45 |
| **Persistence** at 0.50: locations in 10 / 8–9 / 5–7 / 1–4 trials | 17 / 2 / 3 / 4 | 20 / 1 / 2 / 16 |
| Ambiguous chained groups at 0.50 (boxes) | 5 (36 of 245) | 16 (171 of 418) |
| Share of boxes in locations found in every trial | 69% | 48% |
| **Geometry\*:** median box size (ground estimate) | ≈ 11.3 × 9.6 m | ≈ 11.4 × 10.4 m |
| Image area covered by a trial's boxes | 46% (42–50%) | 62% (62–63%) |
| Summed box area ÷ area covered (within-trial overlap) | 1.11 | 1.43 |
| Cost · mean output tokens per trial | $0.0574 · 1,601 | $0.1435 · 13,316 (12,309 reasoning) |

**Agreement between the models**, across all 100 M08 × M09 trial pairs:
* 0.627 ± 0.049 at IoU ≥ 0.50 (range 0.50–0.74);
* 0.198 ± 0.065 at IoU ≥ 0.75;
* 0 at IoU = 1.00.

On PT-R1 the same measure was 0.828 at IoU ≥ 0.50.

\*The geometry rows were computed on 4 November 2026 from `evaluation_results/detections.csv`, using the
estimated ground sampling distance in `altitude_series.csv`. They are not yet a standard pipeline output. Coverage
is the union of a trial's boxes on a 1/16-scale raster. Box sizes in metres inherit the approximations of the GSD
estimate.

*Sources: `evaluation_results/model_comparison.csv`, `cross_model_agreement.csv`, `iou_summary.csv`,
`persistence_summary.csv`; figures K and L in `evaluation_results/figures/comparison/LA_PT-02_64m/` and the M09
figures A–E in `evaluation_results/figures/lowalt_pilot_pt02_64m_m09_v1/LA_PT-02_64m/`.*

![PT-02: M08 vs M09](evaluation_results/figures/comparison/LA_PT-02_64m/K_model_comparison.png)

---

### DeepForest M01 and M02 v2 on both pilot images (5 November 2026)

**Setup.**
- **Checkpoint:** pinned DeepForest checkpoint (`weecology/deepforest-tree` at revision `cc21436b…`, weights
  SHA-256 `d37a7af0…`), loaded offline and checked tensor by tensor against the file.
- **Settings:** the historical frozen inference settings, unchanged (400 px patches, overlap 0.05, cross-patch IoU
  0.15, score 0.10, NMS 0.05).
- **Code and outputs:** run in `urban-tree-vision-baselines` (`scripts/run_deepforest_lowalt_v1.py`; registry
  `docs/lowalt_pilot_v1_deepforest.md`) and imported here with `import_deepforest.py`, keeping float pixel boxes
  and confidence scores.

| Configuration | Input to DeepForest | PT-R1 experiment | PT-02 experiment |
|---|---|---|---|
| M01 (`M01_9c192941f5`) | original 8192 × 6144 image | `lowalt_pilot_ptr1_27m_m01_v1` | `lowalt_pilot_pt02_64m_m01_v1` |
| M02 v2 (`M02v2_b25eccbcc7` / `M02v2_1e05a4037f`) | Lanczos resample to a nominal 10 cm/px: 387 × 290 (padded to 400 × 400) / 927 × 696 | `lowalt_pilot_ptr1_27m_m02_v2` | `lowalt_pilot_pt02_64m_m02_v2` |

**Why 10 cm/px.** It aligns the input with the approximate resolution of DeepForest's training imagery and with
its 400 px windows (about 40 m). It is not an optimised or validated value. The historical 7.89 cm/px was not
reused because its derivation is undocumented. A 7.89 cm/px sensitivity run is possible but was not executed.

**How the nominal native GSD was estimated.** From the camera geometry and the height above the takeoff point:
0.472 cm/px at 26.8 m and 1.132 cm/px at 64.3 m. Crown tops sit closer to the camera, so with 5–10 m crowns the
true crown-level GSD on PT-R1 is about 19–37% finer.

**Why PT-R1 was padded.** The 387 × 290 resample is smaller than one window. Unpadded, DeepForest would have
enlarged it ×2.76 internally instead of the standard ×2.0, so it was padded (black, bottom/right) to 400 × 400.
Coordinates are unchanged, and no box fell in the padding.

**Determinism.** Each configuration was run twice. All four verification runs were **bit-identical** to the primary
runs: same boxes and scores at full precision. The verification run is a reproducibility check, not a second
trial. Each DeepForest configuration is therefore one observation per image, compared with each of the 10 LLM
trials separately (10 pairs), never replicated to look like 10 trials.

**Different inputs.** The LLMs received a 2212 × 1659 JPEG; M01 the full-resolution pixels; M02 v2 the resample.
All boxes are compared in original-image pixels.

**PT-R1 against the 11-tree reference** (LLM values: mean over 10 trials):

| Model | Boxes | P / R / F1 at IoU ≥ 0.50 | P / R / F1 at IoU ≥ 0.75 | At 1.00 | Mean IoU of TPs (0.50) |
|---|---|---|---|---|---|
| M08 Claude Opus 5.5 | 6.1 | 0.786 / 0.436 / 0.561 | 0.148 / 0.082 / 0.105 | 0 | 0.663 |
| M09 GPT-6.1 Sol | 5.7 | 0.850 / 0.436 / 0.575 | 0.638 / 0.327 / 0.431 | 0 | 0.802 |
| DeepForest M01 | 4,836 | 0 / 0 / 0 (0 TP, 4,836 FP, 11 FN) | 0 / 0 / 0 | 0 | – |
| DeepForest M02 v2 | 20 | 0.250 / 0.455 / 0.323 (5 TP, 15 FP, 6 FN) | 0.050 / 0.091 / 0.065 | 0 | 0.669 |

Per reference tree:
- **M02 v2 matched H01, H02, H03, H04 and H11** at IoU ≥ 0.50, with IoU 0.785, 0.713, 0.606, 0.558 and 0.683.
- **H04** was missed by all 20 LLM trials. That makes M02 v2 the only configuration so far that finds any of the
  five trees both LLMs never found.
- **H05, H06, H09 and H10** remain undetected by every model. M02 v2 has no box overlapping H06, H09 or H10 at all,
  and only an IoU of 0.054 with H05.
  - H05, H10 and H09 are the three smallest reference crowns: about 0.7–0.8 m, 1.0 m and 1.5 m across at the
    nominal GSD.
  - H06 is about 2.3–2.6 m across, the same size as H04, which M02 v2 did find. Size alone therefore does not
    explain the misses.
- **H08** (missed by M02 v2, nearest box IoU 0.386) is found by most LLM trials.
- **M02 v2's 15 false positives:** 7 overlap a reference tree below the threshold, and 8 overlap no reference tree.
  Visually, most lie on the shrub and herbaceous beds, which the reference excludes. These boxes have no human
  labels yet.
- **M01's boxes have a median side of 17 px (8 cm).** Its nearest box to each reference tree reaches IoU 0.43 at
  most (H05, the smallest crown). At 0.47 cm/px the detector sees leaves and texture, not crowns.

**PT-02 (no reference: descriptive only).** Means over outputs. Box side = √(width × height), in metres at the
nominal GSD. Coverage is the share of the image inside any box.

| Model | Boxes | Median box side | Coverage | Agreement with M08 / M09 at IoU ≥ 0.50 | Coverage-mask IoU with M08 / M09 |
|---|---|---|---|---|---|
| M08 | 24.5 | 10.2 m | 45% | – / 0.627 | – / 0.714 |
| M09 | 41.8 | 10.8 m | 62% | 0.627 / – | 0.714 / – |
| DeepForest M01 | 6,320 | 0.18 m | 14% | 0.000 / 0.000 | 0.207 / 0.195 |
| DeepForest M02 v2 | 183 | 2.65 m | 34% | 0.068 / 0.035 | 0.470 / 0.498 |

M02 v2 marks largely the same canopy as the LLMs but divides it into much smaller boxes:
- 71% of its box centres fall inside a Claude box, and 91% inside a GPT-6.1 Sol box;
- a Claude box contains on average 5.8 M02 v2 centres, and 21 of Claude's 24.5 boxes contain two or more.

M02 v2 also leaves out some canopy the LLMs include: coverage 34% against 45–62%.

None of this says which granularity matches real crowns. Without a PT-02 reference, M02 v2's 183 boxes and the
LLMs' 25–42 are equally unvalidated. The site inventory of 24 trees is context only.

**Within-output overlap.** No DeepForest output contains near-duplicate boxes (IoU ≥ 0.3 within one output), and
its boxes barely overlap (summed area / covered area ≈ 1.02–1.07). By contrast, 2.5% of M09's PT-02 boxes are
near-duplicates, with an overlap ratio of 1.40.

*Sources: `evaluation_results_lowalt_v1_deepforest/` (`accuracy_summary.csv`, `reference_detection_frequency.csv`,
`prediction_outcomes.csv`, `cross_model_agreement.csv`, `cross_model_coverage.csv`, `geometry_summary.csv`,
`evaluation_report.md`, `accuracy_report.md`); per-run provenance in `benchmark_results/*_m01_v1/` and
`*_m02_v2/`.*

## E. Key Scientific Findings

All findings below come from two full-frame pilot images, two model configurations (10 trials each per image) and
one human reference (PT-R1). Treat them as hypotheses to test, not general conclusions. Both frames are the approved
site-focused images and include some surrounding context (section F). Findings 1–6 were first
established with M08; where M09 adds evidence, it is noted.

### 1. Similar detection counts can conceal spatial instability

The two images have similar count variability (CV 5.2% and 7.5%). Their spatial behaviour is different:

* **PT-R1** is close to all-or-nothing. All 28 pairs among trials 1–5 and 7–9 agree perfectly at both
  IoU ≥ 0.50 and ≥ 0.75. Every imperfect pair involves trial 6 or trial 10.
* **PT-02** never reaches full agreement. Agreement at IoU ≥ 0.50 stays between 0.77 and 0.98 for every pair.
  Only 17 locations recur in all ten trials, although the mean count is 24.5. A further 36 boxes form chains
  in which neighbouring boxes overlap but do not agree as a group. This pattern is consistent with the model
  dividing contiguous canopy differently from one trial to the next (`C_box_variation_iou050.png`).

The clearest single example is PT-R1 **trial 6**. It returned six boxes, like eight other trials, but three of
them were displaced: two onto sidewalk next to street trees, and one only partly covering a tree crown. Its
precision against the reference fell from 0.833 to 0.333, while its count was unchanged. **A count can be
perfectly reproduced while the objects behind it change.**

*Why it matters:* count-based consistency metrics, which are common in canopy and tree-inventory work, would
rate both images, and trial 6, as stable. Spatial and object-level checks are necessary to see the instability.

### 2. High repeatability does not guarantee correct identification

PT-R1 is the most repeatable result in the study: eight of ten trials are spatially equivalent at IoU ≥ 0.50.
Against the human reference, however, the typical trial finds 5 of 11 trees (recall 0.455) and includes one
non-tree. **Repeatability measures whether the model gives the same answer, not whether that answer is right.**
A model that consistently misses the same trees and consistently boxes the same non-tree looks highly
repeatable. The four locations found in all ten trials include one false positive.

### 3. A non-tree object can be consistently misclassified

The most persistent false positive is location **G02**. It is a box found in **all ten trials** with no overlap
with any reference tree, and the reviewer labelled it a plant, not a tree. It accounts for 10 of the 13 false
positives at IoU ≥ 0.50. The model's errors are therefore not random noise that averaging could remove. Some are
systematic: repeating the request makes the same mistake more certain, not less likely.

### 4. Some reference trees are systematically missed

The tree-level results show that omission, not false detection, is the main source of error on PT-R1:

| Reference tree | Size in original pixels | ≈ Size in model input pixels | Found at IoU ≥ 0.50 |
|---|---|---|---|
| H03 | 2014 × 1850 | 544 × 500 | 10/10 |
| H11 | 758 × 751 | 205 × 203 | 10/10 |
| H01, H02, H08 | 613–713 × 758–795 | 166–193 × 205–215 | 9/10 each (missed only in trial 6) |
| H07 | 361 × 373 | 98 × 101 | 1/10 (trial 10 only) |
| H04, H06 | 489 × 523; 480 × 552 | ≈ 130 × 141–149 | **0/10** |
| H09 | 308 × 330 | 83 × 89 | **0/10** |
| H10 | 221 × 222 | 60 × 60 | **0/10** |
| H05 | 165 × 151 | 45 × 41 | **0/10** |

*Source: `evaluation_results/reference_detection_frequency.csv`; figures H–J. Model-input sizes are the original
sizes ÷ 3.70.*

No box from any trial overlapped H04, H05, H06, H09 or H10 at all. In this image, every tree at least about
600 px across was found in at least 9 of 10 trials. Every tree under about 490 px was missed in at least 9 of
10 trials.

Size is the most obvious pattern, but it is not the only difference. Several missed trees are low-contrast
plantings on soil or in grass (figure J). With one image and 11 trees, the size association is a hypothesis to
test, not an established threshold.

**M09 misses the same five trees.** GPT-6.1 Sol never matched H04, H05, H06, H09 or H10 at IoU ≥ 0.50 in any of
its 10 trials. Its only boxes near them were one box overlapping H06 at IoU 0.10 (trial 8). Both models were given
the same pixels and the same prompt. The shared omission is therefore consistent with properties of these trees
in this image, such as size at model-input resolution, contrast or ground cover. It is also consistent with the
shared input itself: the same downscaling and the same instructions.

These data cannot distinguish between these explanations, and they do not establish any of them. Two models are
not a representative sample of detectors, and the reference comes from one annotator. A test would need the same
trees at other scales (altitude series, tiling outside the frozen configuration), other detector families, and
an independent second annotation.

H07, the next-smallest found tree, separates the two models slightly. M08 found it once (trial 10, IoU 0.74).
M09 never matched it, but three M09 trials (6–8) placed a loose box near it (IoU 0.18–0.31), which counts as a
false positive plus a miss.

### 5. Box extent and localization strongly affect IoU-based accuracy

Two different localization effects appear:

* **Systematic tightness.** In 46 of the 48 matches at IoU ≥ 0.50, the model's box lay entirely inside the human
  box. It covered on average 69% of the human box's area. When one box contains another, IoU equals their area
  ratio. This is why only the largest tree, H03, still matches at IoU ≥ 0.75, and why F1 falls from 0.56 to
  0.11 between those thresholds. It reflects a difference in box-drawing convention: the model outlines a
  tighter crown than this annotator did. On its own, it does not show which convention is correct.
* **Occasional displacement.** In trial 6, boxes for H01 and H02 were shifted onto the sidewalk (IoU 0.333 and
  0.311), and the box for H08 fell to IoU 0.451. Each of these counts as both a false positive and a false
  negative.

**M09 draws larger boxes than M08, closer to this annotator's extent.**

| | M08 | M09 |
|---|---|---|
| Matched boxes nested inside the human box | 46 of 48 | 34 of 48 |
| Share of the human box's area covered | 0.69 | 0.83 |
| Mean IoU of matched boxes | 0.66 | 0.80 |

This one difference accounts for most of the gap between the models at IoU ≥ 0.75. Recall at IoU ≥ 0.75 is 0.327
for M09 against 0.082 for M08, although both find the same trees at IoU ≥ 0.50. Three trees (H01, H03, H11)
pass IoU ≥ 0.75 in all 10 M09 trials, against only H03 for M08 (in 9 trials).

This is agreement with one annotator's box-drawing convention. It shows that the IoU ≥ 0.75 comparison depends
on how crowns are outlined. It does not show that one model localizes trees "correctly" and the other does not.

The IoU = 1.00 threshold is effectively unreachable against human boxes. Both models' coordinate grids have a step
of about 8 original pixels, while the human boxes are drawn to the pixel. At that threshold the metric measures
coordinate reproduction, not detection quality.

![PT-R1: human reference vs all trials' boxes](evaluation_results/figures/lowalt_pilot_ptr1_27m_v1/LA_PT-R1_27m/I_reference_vs_predictions_overview.png)

### 6. Site characteristics and image acquisition may affect detection behaviour

PT-02 differs from PT-R1 in several ways:

* about 3.5 times more output tokens per trial;
* about four times as many boxes;
* lower but more uniform spatial agreement (0.875 ± 0.049 vs 0.919 ± 0.134);
* chained, ambiguous groups, which never appear on PT-R1.

The two images also differ in site layout and canopy density, in altitude (64.3 vs 26.8 m), and in ground
resolution as seen by the model (about 42 vs 17.5 mm per pixel). The flights also had different headings. **These
factors are completely confounded in the current data**, so none of the differences can be attributed to altitude,
to site, or to canopy structure. In particular, PT-02 has no completed accuracy reference, so **nothing here shows
that the model is more or less accurate on PT-02 than on PT-R1**.

The comparison with the field inventory needs the same care. PT-02's mean of 24.5 is close to its site inventory
of 24, while PT-R1's mean of 6.1 is well below its inventory of 11. On PT-R1, where a reference exists, the six
boxes per trial include one non-tree and miss six real trees. The count is not close to the inventory, and it is
not composed of the right trees either. For PT-02, the closeness to 24 is a numerical coincidence until shown
otherwise, not validated accuracy.

### 7. Two model families on the same image: separating repeatability, accuracy and localization

The M08–M09 comparison on PT-R1 shows why the evaluation keeps these three questions apart. The two models rank
differently on each.

* **Run-to-run repeatability (same image, repeated trials): M08 is more repeatable.**
  * Its spatial agreement is 0.919 at IoU ≥ 0.50, against 0.838 for M09.
  * 28 of its 45 trial pairs agree fully, against 7 for M09.
  * Its count CV is 5.2%, against 11.8%.
  * M09 varies more between trials, though never in exactly the same way: no two M09 trials share identical
    coordinates.
* **Detection against the human reference at IoU ≥ 0.50: indistinguishable.**
  * Recall is identical (0.436). Both models find the same five trees and miss the same five.
  * M09's precision is slightly higher (0.850 against 0.786), mostly because it boxes the non-tree plant in 4
    trials rather than all 10.
  * The F1 difference (0.575 against 0.561) is well within M08's trial-to-trial spread.
* **Localization against the human reference: M09's box extent is closer to this annotator's** (finding 5).
  This is what drives the large difference at IoU ≥ 0.75.
* **The systematic error is shared.** The non-tree plant fools both models: M08 every time, M09 in 4 of 10
  trials, with nearly the same box (IoU 0.84–0.87 with M08's boxes). The same small trees are missed by both.

**Scope.** These are differences in **run-to-run behaviour on one image**:
* identical input, prompt and reference;
* "high" reasoning effort that is not computationally equivalent across providers;
* one annotator.

They are not estimates of how either model performs across images, sites or altitudes. They do not support
calling either model better in general.

### 8. Agreement between models depends strongly on the scene

On PT-02, a denser and more contiguous canopy seen from 64 m, the two model families diverge far more than on
PT-R1:

| | PT-R1 | PT-02 |
|---|---|---|
| Agreement between the models, IoU ≥ 0.50 | 0.828 ± 0.158 | **0.627 ± 0.049** |
| Agreement within each model, IoU ≥ 0.50 (M08 / M09) | 0.919 / 0.838 | 0.875 / 0.794 |

The differences on PT-02 are structural, not noise:

* **Counts:** GPT-6.1 Sol returns about 70% more boxes (41.8 against 24.5 per trial).
* **Coverage, not subdivision:** its boxes are about the same size as Claude's (median ≈ 11 m across). They cover
  more of the image (about 62% against 46% per trial) and overlap each other more (summed box area 1.43× the
  covered area, against 1.11×). So it is not subdividing Claude's crowns into smaller boxes. It boxes additional
  areas of canopy.
* **Stability:** M09 is also the less stable model on this image:
  * fewer of its boxes recur in every trial (48% against 69%);
  * far more fall into chained, ambiguous groups (171 boxes against 36);
  * its agreement with itself at IoU ≥ 0.75 is much lower (0.48 against 0.72).

  Its count CV is similar to Claude's (8.8% against 7.5%), which again shows that **similar count stability can
  hide very different spatial stability** (finding 1).

**What this does not show.** Without a PT-02 reference, it is unknown whether M09's extra boxes are real crowns
that M08 merges or omits, or non-tree vegetation and other false positives. M08's mean of 24.5 being close to the
site inventory of 24, and M09's 41.8 being far from it, is **not** evidence about either model's accuracy, for
two reasons:
* the frame also contains surrounding context, so image counts are not site-level counts;
* a matching count can hide both misses and false positives (finding 6).

Site, canopy structure, altitude (27 m against 64 m) and heading all differ between the two images, so the drop
in between-model agreement cannot be attributed to any one of them.

**For the research question**, the pilot suggests that the approaches characterize an open, sparse planting
similarly but diverge substantially on dense canopy, in count, in which areas are boxed and in run-to-run
stability. That is a hypothesis for further images, the altitude series and additional model families
to test.

### 9. For a pretrained detector, input scale decides what counts as "a tree"

DeepForest is deterministic, so all of its uncertainty here is about **configuration**, not repetition. Run
twice, each configuration gave bit-identical output. Changing only the input resolution changed everything:

* **At native resolution (0.5–1.1 cm/px),** the detector works at roughly 10–20× finer scale than its training
  imagery. It returns thousands of centimetre-scale boxes (median 8–18 cm). These cover only 14–16% of the image
  and match no reference tree and no LLM box.
* **At about 10 cm/px,** it returns crown-sized objects:
  * **PT-R1:** recall 0.45 at IoU ≥ 0.50, comparable to the LLMs, including one tree the LLMs never found (H04).
    It returns three false positives per true positive, against about 0.2–0.3 for the LLMs.
  * **PT-02:** 4–7× more boxes than the LLMs, with sides about 4× smaller, inside largely the same canopy areas.

The PT-02 contrast is the clearest example yet of the study's central issue. Two approaches can agree **where**
vegetation is, with coverage-mask IoU about 0.5, while disagreeing almost completely on **what the individual
units are**: box agreement 0.04–0.07. Counts from the two families on dense canopy are therefore not
interchangeable.

The 10 cm/px setting was chosen for alignment, not tuned, and was not adjusted after the results. Other scales
(including the historical 7.89 cm/px) remain untested. Note also that the M02 v2 target depends on a nominal GSD
that ignores canopy height.

---

## F. Methodological Limitations

* **Surrounding context in the approved frames.** Dr. Elizabeth Ondula advised prioritizing imagery focused on the
  study site. The project lead reviewed both pilot images and approved them as the closest achievable
  site-focused images (5 November 2026). They are used whole: no site boundary, mask or crop is applied.
  * **PT-02:** the frame (about 93 × 70 m estimated footprint) also shows neighbouring streets and buildings.
  * **PT-R1:** the frame includes street and sidewalk margins.
  * **Consequences:** detections and disagreements in these margins count in every metric, for every model
    alike. Image-level results are therefore not site-level quantities, and the site inventories remain context
    only.
* **One completed human reference.** Every accuracy number comes from one image with 11 trees. It does not extend
  to PT-02, other altitudes, other dates or other models. **PT-02 currently supports repeatability analysis only,
  not validated individual-tree detection accuracy.**
* **Small sample.** Forty paid trials: two models × two images × 10 trials. The trial-to-trial spread estimates *run-to-run* variation
  for one fixed input. It is not *image-to-image* or *site-to-site* variation, and it is not an independent
  estimate of accuracy: all ten trials share the same input, prompt and downscaling, and so share their errors.
* **PT-02 crown boundaries.** In PT-02's dense canopy, individual crowns are often not visually separable. That
  is why the persistence analysis flags chained groups as ambiguous instead of forcing them into single trees.
  Any future PT-02 reference will need an explicit rule for touching or overlapping crowns.
* **Different altitudes and image scales.** The two images differ in ground resolution by a factor of about 2.4.
  Because of the API's fixed visual-token budget, the model sees each image at about 1/3.7 of its original linear
  resolution, which may matter most for small crowns (finding 4).
* **Site and altitude are confounded.** Each site was benchmarked at one altitude, and the two altitudes do not
  overlap (PT-R1 at 27–31 m, PT-02 at 64–86 m). This design cannot separate site effects from altitude effects.
* **Annotation conventions and uncertainty.** The reference is a single annotator's single pass, with no
  agreement check against a second annotator. How much of an irregular crown to include is a judgement, and it
  drives the results at IoU ≥ 0.75 (finding 5).
* **Comparison against site-level inventories.** Inventory counts describe a whole site. Image coverage of each
  site is unconfirmed (`coverage_scope = UNKNOWN`), so inventory comparisons are context, never image-level
  ground truth. The pipeline never uses them to filter or correct predictions.
* **Two model configurations.** M08 and M09 were each run at one frozen configuration with one prompt, on the
  same two images. The results say nothing about other configurations, prompts or model families.
* **Geometry measures.** The PT-02 box-size, coverage and overlap figures for M08/M09 in finding 8 came from a
  one-off analysis of `detections.csv`. Since 5 November these measures are pipeline outputs
  (`geometry_summary.csv`, `cross_model_coverage.csv`), computed on a coverage raster at 1/8 resolution, so
  they differ from the one-off values in the second digit (for example 45.4% against "about 46%" coverage).
  Ground-size conversions inherit the approximations of the GSD estimate.
* **DeepForest comparisons.**
  * **Single output per configuration:** each DeepForest configuration is one deterministic output per image.
    It has no run-to-run spread, and its accuracy on PT-R1 is a single value, not a mean.
  * **Different inputs:** it saw different input pixels from the LLMs (full resolution, or a 10 cm/px resample,
    against a 2212 × 1659 JPEG). The comparison is between whole pipelines, not between models on identical
    input.
  * **Nominal GSD:** the 10 cm/px target is nominal (height above takeoff; canopy height ignored). It is an
    alignment choice, not an optimized one.
  * **PT-R1 padding:** the PT-R1 M02 v2 input was padded to one standard window. This is a documented,
    coordinate-preserving step, but at this scale the whole image is a single window.
  * **No tuning or retraining:** no score-threshold tuning, fine-tuning or crown-size filtering was done, so
    results describe the released checkpoint at its historical settings.
* **Comparability across providers is approximate.**
  * Both models receive identical image bytes, prompt and schema.
  * "High" reasoning effort is a provider-specific setting, not equal computation.
  * Each provider encodes the image differently.
  * OpenAI's internal processing resolution cannot be observed. M09's records document what was submitted and
    requested (`detail: original`, compression disabled), not what was processed.
  * M08's raw responses include readable thinking text. M09's include only opaque encrypted reasoning items.

---

## G. Engineering and Reproducibility

### Pipeline components

| Component | Role |
|---|---|
| `box_generation.py` | Fixed prompt, output schema and v2 image preprocessing; single-image command-line tool |
| `model_adapters.py` | M08 (Anthropic) and M09 (OpenAI Responses API) adapters with retries, error classes, pricing and secret redaction; offline mock adapters; placeholders for planned models |
| `benchmark_runner.py` | Manifest-driven repeated trials; project-wide spending ledger and budget guard; resume, round-robin and pause safeguards |
| `make_manifest.py` | Builds the image manifest from file metadata; non-destructive updates and a review list |
| `evaluate_consistency.py`, `evaluation_core.py` | Offline count, spatial and persistence analysis, with exact-arithmetic IoU and optimal one-to-one matching |
| `evaluation_accuracy.py` | Accuracy against a complete human reference, with error types per box |
| `evaluation_plots.py` | Figures A–J, review crops and contact sheets on the full-resolution images |
| `import_reference.py` | Cross-validated import of human reference boxes from Make Sense exports |
| `import_deepforest.py` | Imports a DeepForest primary run from `urban-tree-vision-baselines` as trial 1, keeping float pixel boxes and scores, after cross-checking the source hash with the manifest and every LLM record |
| `evaluation_geometry.py` | Box size, coverage, within-output overlap, coverage-mask agreement and fragmentation |

### Testing and validation

The suite has **220 offline tests**, all passing, and makes no network calls. The 10 newest (9 November 2026)
cover the DeepForest repeated-execution analysis on synthetic runs:
* identical runs;
* raw-order-only, coordinate, score and count discrepancies;
* failed-run exclusion and input checks;
* component-wise matching against the dense path.

The earlier 210:
* 77 cover the benchmark runner;
* 44 cover the evaluation;
* 55 cover M09, using a fake OpenAI client and an offline capture of the exact HTTP request body;
* 16 cover reference import;
* 18 cover the DeepForest import and evaluation. They check:
  * float boxes and scores, and that nothing is converted to the 0–1000 grid;
  * hash cross-checks and overwrite refusal;
  * single-output handling and the 1 × 10 between-model pairs;
  * reference matching;
  * that frozen LLM records are untouched;
  * the matching prefilter against exhaustive matching;
  * geometry.

`urban-tree-vision-baselines` adds 40 offline tests for the DeepForest runner. A fake model stands in for
inference. 29 cover the checkpoint pin, fidelity to the historical settings, GSD scaling, padding, coordinate
restoration, overwrite protection and determinism reporting. 11 cover the repeated-execution driver: pilot-identical
configurations, the 40-run interleaved plan, resume, stop-on-failure and refusal of the pilot output root.

**Matching speed-up (5 November 2026).** One-to-one matching now computes a float IoU prefilter first, then exact
rational IoU for every pair that could reach the threshold. Results are identical to exhaustive matching, which a
test checks on random boxes. The M08/M09 rows of every evaluation CSV were confirmed unchanged.

A check that runs with the suite confirms that the frozen results, ledgers, human-review files and manifest are
unchanged by the tests.

**Reproducibility check (4 November 2026).** Everything was re-derived offline from the saved records:
* Regenerating the full evaluation into a separate folder reproduced every CSV and both reports byte for byte.
  Only the run timestamp, command line and recorded Git commit differed.
* Re-running the preprocessing reproduced the exact submitted image bytes and prompt hash of all 30 M08 and M09
  trials.

They cover:
* exact IoU at 0.50, 0.75 and 1.00;
* one-to-one matching, checked against brute force;
* ambiguity handling;
* budget accounting across experiments;
* resume and pause behaviour;
* refusal to compare boxes across different images without alignment;
* refusal to compute accuracy against an incomplete reference;
* reproducibility of all outputs.

### Integrity of the human-review data

Two data-integrity problems were caught and resolved without losing any human input:

* **Spreadsheet date conversion.** Opening the review sheet in Excel had turned 40 persistence categories (such as
  `8-9`) into dates (such as `9-Aug`). They were repaired from each row's recorded trial count, after a timestamped
  backup. All nine PT-R1 labels and their reviewer metadata were preserved. Every evaluation now validates the
  review file first and stops with an error rather than using or silently rewriting corrupted values.
  `human_review/README.md` documents a safe Excel workflow.
* **Disagreement between export formats.** The two Make Sense exports differed by one pixel on five boxes. The
  importer refuses such differences unless the user explicitly chooses a source. The choice (VOC) and every
  difference are recorded in the reference rows.

### API expenditure

| Item | Value |
|---|---|
| Item | Anthropic (M08) | OpenAI (M09) |
|---|---|---|
| Paid trials | 20, all successful | 20, all successful; one connection retry before generation (PT-R1) |
| Confirmed spend | $0.3439 (PT-R1) + $0.5741 (PT-02) = **$0.9179** | $0.5353 (PT-R1) + $1.4353 (PT-02) = **$1.9705** |
| Remaining under each provider's separate $5 local limit | $4.0821 | $3.0295 |
| Observed cost per trial | $0.029–0.061 (worst case reserved $0.67) | $0.045–0.074 PT-R1, $0.108–0.170 PT-02 (worst case reserved $0.335) |

Each provider has its own ledger. All entries are priced from usage reported by the API.

*Sources: `spend_ledger/anthropic_api_ledger.jsonl`, `spend_ledger/openai_api_ledger.jsonl`,
`benchmark_results/*/counts_summary.csv`.*

### Repository history note

On 4 November 2026, eight commits were made under the project owner's identity between 14:55 and 15:02. Three
of them, titled "test" (`44b97d1`, `4567eab`, `d24b940`), added M09 trials 2–4. `d24b940` also captured
`evaluation_results/` while the evaluation was regenerating: its `evaluation_config.json` still reads
`{"status": "incomplete"}`, and reports, crops and some figures are missing. Later commits (`a4b495f` through
`254bc4b`) added the remaining trials, the complete evaluation outputs and the code changes.

No commit altered model settings, the manifest, human annotations or frozen M08 records. `d24b940` should not be
cited as an evaluation state. Use the complete outputs in the current tree, which the reproducibility check
regenerates exactly.

On 5 November 2026, the M09 PT-02 work was committed on branch `checkpoint/m09-pt02` in scoped, separate
commits:
* trial records and ledger (`8170b8d`);
* evaluation outputs (`698664b`);
* repository hygiene (`4d4bbd3`);
* this documentation update.

Regenerable review crops and contact sheets are now git-ignored and kept only locally. Crops tracked earlier,
including those the PT-R1 labels were made from, remain in Git.

---

## H. Current Research Status

| Status | Item |
|---|---|
| **Completed** | Frozen M08 configuration and v2 preprocessing; reproducible runner with spending ledger |
| **Completed** | 10 trials each on `LA_PT-R1_27m` and `LA_PT-02_64m`; count, spatial and persistence analysis for both |
| **Completed** | PT-R1 semantic review (9 units) and complete 11-tree human reference; accuracy evaluation at 0.50, 0.75 and 1.00 |
| **Completed** | Offline evaluation, visualization, review-validation and reference-import tooling with tests |
| **In progress** | PT-02 human review (62 review units prepared, none labelled yet) |
| **In progress** | Confirmation of `coverage_scope` for all images, and of the week 1 condition labels |
| **Planned** | M08 runs on more images: the remaining altitude frames and the weekly before/after collection |
| **Planned** | Human references beyond PT-R1 |
| **Completed** | M09 (GPT-6.1 Sol): adapter, separate cost ledger and model-comparison tooling; 10 trials on each pilot image; consistency for both, accuracy for PT-R1, and the M08-vs-M09 comparison on both images (40 pilot trials in total) |
| **Completed** | Pilot image suitability: `LA_PT-R1_27m` and `LA_PT-02_64m` reviewed and approved by the project lead as the closest achievable site-focused images (5 November 2026); decision closed |
| **Completed** | DeepForest M01 (native) and M02 v2 (nominal 10 cm/px) on both pilot images: pinned checkpoint, bit-identical verification runs, import, PT-R1 accuracy and four-model comparison (5 November 2026; not yet committed) |
| **Discontinued** | M10 Gemini (see the model roster) |
| **Prepared, not run** | DeepForest repeated-execution experiment `lowalt_repro_v1`: 10 runs each of M01 and M02v2 (nominal 10 cm/px) on both pilot images, 40 runs, replacing M10 (approved by Dr. Elizabeth Ondula to conclude this phase). Runner, driver, analysis and figures are implemented and tested offline; inference awaits authorization. See `EVALUATION.md` and `urban-tree-vision-baselines/docs/lowalt_repro_v1_deepforest.md`. |
| **Planned** | Local Qwen3-VL via Ollama after hardware checks, then a small held-out DeepForest/NEON-annotated shared-reference set (after the current phase concludes) |
| **Planned** | Across-altitude robustness analysis (design written in `EVALUATION.md`; requires image registration) |
| **Deferred** | Copilot (no programmatic endpoint) |
| **Deferred** | Tiling or cropping to raise model input resolution (outside the frozen configuration) |
| **Deferred** | Cross-image box comparison (refused by the code until a validated alignment exists) |

---

## I. Next Steps

These are **recommendations**, not completed work. They are ordered by scientific value relative to cost. Items
marked *offline* need no API spending.

1. **Localization diagnostics on PT-R1 (offline).** Finding 5 suggests that some of the drop in accuracy at
   higher thresholds comes from how tightly each side draws boxes, not from detection itself. Report
   centre-distance and containment measures alongside IoU, and plot accuracy over a range of IoU thresholds. Keep
   0.50, 0.75 and 1.00 as the primary thresholds.
2. **Measure annotation uncertainty on PT-R1 (low cost, human time only).** A second independent annotation, or a
   repeat by the same annotator, would show how much of the IoU ≥ 0.75 gap is human variability. A short written
   crown-extent convention would make future references consistent.
3. **Label the 62 PT-02 review units.** This gives semantic correctness (tree vs non-tree) for PT-02's recurring
   detections without first solving full crown delineation. It is a feasible step towards an accuracy reference.
4. **Within-site altitude sensitivity at PT-R1 (paid; requires authorization).** Run M08 on the nadir frames with
   the same heading (`LA_PT-R1_28m`, `_30m`, `_31m`). Each image needs either its own human reference, which is
   feasible for about 11 trees, or a validated registration to the 27 m frame. At the observed cost of about
   $0.034 per PT-R1 trial, 30 more trials would cost about $1, within the remaining budget. Holding the site
   fixed removes the site–altitude confound for this narrow altitude range. Treat the rotated `_30m_angled` frame
   as a separate heading test.
5. **A defensible PT-02 reference.** Options include:
   * crown annotation with an explicit rule for touching crowns and an "uncertain boundary" flag;
   * a point-based (stem or crown-centre) reference from the field inventory, if georeferenced stem positions
     exist;
   * restricting the reference to a sub-area where crowns are separable.
6. **Model-family comparison.** M08, M09 and DeepForest M01/M02 v2 are complete on both pilot images. The agreed
   sequence continues with:
   * ~~M10 Gemini~~: discontinued on 9 November 2026 (minimum $30 initial payment); not run. It is replaced by a
     10-run DeepForest repeated-execution experiment (prepared; awaiting authorization), which tests whether the
     deterministic configurations reproduce identical boxes. Those runs are repeated executions, not independent
     observations;
   * **local Qwen3-VL** through Ollama, after hardware checks;
   * a small held-out **DeepForest/NEON-annotated** shared-reference set.

   Each model should be evaluated with the same model-agnostic code. A deterministic model such as DeepForest
   needs one verified run per configuration rather than ten.
7. **Probe the shared misses (offline first).** Before any new paid runs:
   * tabulate the five trees both models never find against their crown size in model-input pixels, contrast and
     background;
   * check whether a second annotator also marks them.

   Only then consider designs that change scale, such as the altitude series or a separately documented tiling
   variant.
8. **Weekly collection.** After the week 1 labels and coverage are confirmed, the 20 weekly images at about 95 m
   allow repeated-observation analysis over time, which is the core of the research question.

---

## Where to Find the Evidence

**Reports and documentation**
- [BENCHMARK.md](BENCHMARK.md): frozen configuration, preprocessing, runner, ledger and budget rules
- [EVALUATION.md](EVALUATION.md): metric definitions, matching and grouping algorithms, human-review workflow,
  accuracy method, across-altitude design
- [evaluation_results/evaluation_report.md](evaluation_results/evaluation_report.md): count, spatial and
  persistence results for both images
- [evaluation_results/accuracy_report.md](evaluation_results/accuracy_report.md): PT-R1 accuracy, per-tree and
  per-error tables
- [manifests/METADATA_REVIEW.md](manifests/METADATA_REVIEW.md): open metadata questions
- [human_review/README.md](human_review/README.md): human-review files and safe editing workflow

**Key data files**
- Inputs: [manifests/manifest.csv](manifests/manifest.csv)
- Saved trials, one folder per trial:
  - M08: `benchmark_results/lowalt_pilot_ptr1_27m_v1/` and `benchmark_results/lowalt_pilot_pt02_64m_v1/`
  - M09: `benchmark_results/lowalt_pilot_ptr1_27m_m09_v1/` and `benchmark_results/lowalt_pilot_pt02_64m_m09_v1/`

  Each holds `counts_long.csv`, `counts_summary.csv`, `boxes_long.csv` and per-trial records. M09 records also
  hold the exact request as sent (`api_request`) and the provider's echoed settings (`provider_response`).
- Spending: `spend_ledger/anthropic_api_ledger.jsonl` (M08) and `spend_ledger/openai_api_ledger.jsonl` (M09)
- Model comparison: [model_comparison.csv](evaluation_results/model_comparison.csv),
  [cross_model_agreement.csv](evaluation_results/cross_model_agreement.csv), and section 8 of
  `evaluation_report.md`
- Count and spatial consistency:
  [count_consistency.csv](evaluation_results/count_consistency.csv),
  [iou_summary.csv](evaluation_results/iou_summary.csv),
  [pairwise_iou.csv](evaluation_results/pairwise_iou.csv),
  [persistence_summary.csv](evaluation_results/persistence_summary.csv),
  [detection_persistence.csv](evaluation_results/detection_persistence.csv)
- Accuracy:
  [accuracy_summary.csv](evaluation_results/accuracy_summary.csv),
  [accuracy_vs_reference.csv](evaluation_results/accuracy_vs_reference.csv),
  [reference_detection_frequency.csv](evaluation_results/reference_detection_frequency.csv),
  [prediction_outcomes.csv](evaluation_results/prediction_outcomes.csv)
- Human input:
  [manual_review.csv](human_review/manual_review.csv),
  [reference_boxes.csv](human_review/reference_boxes.csv),
  [reference_status.csv](human_review/reference_status.csv),
  [field_inventory.csv](human_review/field_inventory.csv), original exports in `human_review/ground_truth/`
- Provenance of the evaluation (inputs, hashes, software versions):
  [evaluation_config.json](evaluation_results/evaluation_config.json)
- Image metadata and estimated ground resolution: [altitude_series.csv](evaluation_results/altitude_series.csv)

**Selected figures**
- Spatial agreement by threshold: `evaluation_results/figures/<experiment>/<image>/A_threshold_comparison.png`
- Persistence maps: `B_persistence_overview.png`, and full-resolution `B_persistence_map_iou050.jpg`
- Box variation within recurring detections: `C_box_variation_iou050.png`
- Trial-by-trial agreement matrix: `D_agreement_matrix.png`
- Count consistency: `E_count_consistency.png`
- PT-R1 accuracy (in `evaluation_results/figures/lowalt_pilot_ptr1_27m_v1/LA_PT-R1_27m/`):
  - `F_accuracy_by_threshold.png`
  - `G_trial_tp_fp_fn.png`
  - `H_reference_detection_frequency.png`
  - `I_reference_vs_predictions_overview.png`
  - `J_error_cases_iou050.png`
- M09 per-model figures: A–J for PT-R1 in `evaluation_results/figures/lowalt_pilot_ptr1_27m_m09_v1/LA_PT-R1_27m/`;
  A–E for PT-02 in `evaluation_results/figures/lowalt_pilot_pt02_64m_m09_v1/LA_PT-02_64m/`
- M08 vs M09 on PT-02: `evaluation_results/figures/comparison/LA_PT-02_64m/K_model_comparison.png` and
  `L_models_side_by_side_iou050.png`
- M08 vs M09 on PT-R1: `evaluation_results/figures/comparison/LA_PT-R1_27m/K_model_comparison.png` and
  `L_models_side_by_side_iou050.png`
- Human reference only: `evaluation_results/review/human_reference_LA_PT-R1_27m.jpg`; review contact sheets in
  `evaluation_results/review/`

**DeepForest M01 / M02 v2 and the four-model comparison** (evaluation in
`evaluation_results_lowalt_v1_deepforest/`. It holds the same M08/M09 rows as `evaluation_results/` plus the
DeepForest units, geometry and coverage.)
- Records: `benchmark_results/lowalt_pilot_{ptr1_27m,pt02_64m}_{m01_v1,m02_v2}/`. Each `record.json` holds the
  checkpoint, weights check, environment and `determinism.json`. Annotated DeepForest images are
  `annotated_original.jpg` there.
- Run code, raw outputs and registry: `urban-tree-vision-baselines/outputs/lowalt_pilot_v1/` and
  `docs/lowalt_pilot_v1_deepforest.md`
- PT-R1 accuracy figures F–J: `evaluation_results_lowalt_v1_deepforest/figures/lowalt_pilot_ptr1_27m_{m01_v1,m02_v2}/LA_PT-R1_27m/`
- Four-model comparison, per image, in `evaluation_results_lowalt_v1_deepforest/figures/comparison/<image>/`:
  - `K_model_comparison.png`: counts on a log scale, within-model agreement, accuracy, per-tree detection;
  - `L_models_side_by_side_iou050.png`;
  - `M_agreement_matrices.png`: between-model agreement at 0.50 and 0.75;
  - `N_geometry_coverage.png`: box size, coverage, overlap and near-duplicates;
  - `O_coverage_maps.png`.
