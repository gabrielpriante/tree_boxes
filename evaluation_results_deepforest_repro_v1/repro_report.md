# DeepForest repeated-execution reproducibility (lowalt_repro_v1)

Offline analysis of saved runs; no inference and no API requests in this step. Repeated executions of a deterministic program on one machine and software stack: not independent observations, and not comparable as a sample size to the 10 independent stochastic LLM trials.

## Identity of outputs across runs

| Configuration | Image | Runs | Detections per run | Raw files identical | detections.json identical | Boxes and scores identical (canonical) | Identical to pilot runs | Discrepancies |
|---|---|---|---|---|---|---|---|---|
| M01 (M01_9c192941f5) | LA_PT-02_64m | 10 | 6320 in every run | True | True | True | True | none |
| M01 (M01_9c192941f5) | LA_PT-R1_27m | 10 | 4836 in every run | True | True | True | True | none |
| M02v2 (M02v2_1e05a4037f) | LA_PT-02_64m | 10 | 183 in every run | True | True | True | True | none |
| M02v2 (M02v2_b25eccbcc7) | LA_PT-R1_27m | 10 | 20 in every run | True | True | True | True | none |

Raw files are DeepForest's own output order; detections.json is sorted canonically by the runner. A pair whose raw files differ but whose canonical detections match differs only in order.

## Pairwise agreement between runs (exact-IoU one-to-one matching)

| Configuration | Image | Pairs | Agreement mean (min) at 0.50 | at 0.75 | at 1.00 | Pairs at 100% (1.00) |
|---|---|---|---|---|---|---|
| M01 | LA_PT-02_64m | 45 | 1.000 (1.000) | 1.000 (1.000) | 1.000 (1.000) | 45 |
| M01 | LA_PT-R1_27m | 45 | 1.000 (1.000) | 1.000 (1.000) | 1.000 (1.000) | 45 |
| M02v2 | LA_PT-02_64m | 45 | 1.000 (1.000) | 1.000 (1.000) | 1.000 (1.000) | 45 |
| M02v2 | LA_PT-R1_27m | 45 | 1.000 (1.000) | 1.000 (1.000) | 1.000 (1.000) | 45 |

## Beside the LLMs' within-model repeatability (frozen M08/M09 values, read-only)

| Model | Image | Kind | n | Agreement mean ± SD at 0.50 | at 0.75 | at 1.00 | Count mean (CV) |
|---|---|---|---|---|---|---|---|
| DeepForest M01 | LA_PT-02_64m | repeated executions | 10 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 6320.0 (0.0%) |
| DeepForest M01 | LA_PT-R1_27m | repeated executions | 10 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 4836.0 (0.0%) |
| DeepForest M02v2 | LA_PT-02_64m | repeated executions | 10 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 183.0 (0.0%) |
| DeepForest M02v2 | LA_PT-R1_27m | repeated executions | 10 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 20.0 (0.0%) |
| M08 | LA_PT-02_64m | independent stochastic trials | 10 | 0.875 ± 0.049 | 0.721 ± 0.064 | 0.103 ± 0.064 | 24.5 (7.5%) |
| M08 | LA_PT-R1_27m | independent stochastic trials | 10 | 0.919 ± 0.134 | 0.885 ± 0.199 | 0.263 ± 0.309 | 6.1 (5.2%) |
| M09 | LA_PT-02_64m | independent stochastic trials | 10 | 0.794 ± 0.059 | 0.475 ± 0.064 | 0.000 ± 0.000 | 41.8 (8.8%) |
| M09 | LA_PT-R1_27m | independent stochastic trials | 10 | 0.838 ± 0.101 | 0.816 ± 0.097 | 0.000 ± 0.000 | 5.7 (11.8%) |

The DeepForest rows test reproducibility of one deterministic computation; the LLM rows measure run-to-run variation of a stochastic model. Agreement here is consistency, not accuracy.

## Environment

- lowalt_repro_pt02_64m_m01_v1: cudnn=92400, cudnn_benchmark=False, cudnn_deterministic=False, deepforest=2.1.0, gpu=NVIDIA GeForce RTX 5070 Laptop GPU, torch=2.14.0+cu130, torch_cuda=13.0
- lowalt_repro_ptr1_27m_m01_v1: cudnn=92400, cudnn_benchmark=False, cudnn_deterministic=False, deepforest=2.1.0, gpu=NVIDIA GeForce RTX 5070 Laptop GPU, torch=2.14.0+cu130, torch_cuda=13.0
- lowalt_repro_pt02_64m_m02v2_v1: cudnn=92400, cudnn_benchmark=False, cudnn_deterministic=False, deepforest=2.1.0, gpu=NVIDIA GeForce RTX 5070 Laptop GPU, torch=2.14.0+cu130, torch_cuda=13.0
- lowalt_repro_ptr1_27m_m02v2_v1: cudnn=92400, cudnn_benchmark=False, cudnn_deterministic=False, deepforest=2.1.0, gpu=NVIDIA GeForce RTX 5070 Laptop GPU, torch=2.14.0+cu130, torch_cuda=13.0

## Figures

- `figures/P_counts_by_run.png`
- `figures/Q_pairwise_agreement_iou050.png`
- `figures/Q_pairwise_agreement_iou100.png`
- `figures/R_repeatability_vs_llm.png`
