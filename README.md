# tree_boxes: evaluating AI tree detection in urban drone imagery

Frontline Labs research repository. It compares how four existing AI configurations detect individual trees in two
fixed urban drone images, and keeps detection counts, localization, agreement and reproducibility separate. The
four configurations are DeepForest at native resolution (M01), DeepForest at a nominal 10 cm/pixel (M02v2), Claude
Opus 5.5 (M08) and GPT-6.1 Sol (M09). The experimental phase is complete (9 November 2026).

**Reviewers: start with [REVIEW_GUIDE.md](REVIEW_GUIDE.md).**

| Document | Purpose |
|---|---|
| [REVIEW_GUIDE.md](REVIEW_GUIDE.md) | Five-minute overview: design, main results, key figures, limitations |
| [RESEARCH_PROGRESS.md](RESEARCH_PROGRESS.md) | Full research record: design, experiments, findings, limitations, status |
| [EVALUATION.md](EVALUATION.md) | Evaluation methods: metrics, matching, reference accuracy, reproducibility analysis |
| [BENCHMARK.md](BENCHMARK.md) | Vision-language-model trial protocol, frozen configurations, cost controls |
| [deepforest_pipeline/README.md](deepforest_pipeline/README.md) | DeepForest code, run records and archive checksums |

Offline tests: `python -m pytest tests -q` (no network access or API charges).
