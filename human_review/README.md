# Human review files

These files hold human judgements. `evaluate_consistency.py` creates each one only if it is
missing and never overwrites it. Everything in `evaluation_results/` is regenerated; nothing here is.

| File | Purpose |
|---|---|
| `manual_review.csv` | One row per review unit (a recurring detection group, or a single detection inside an ambiguous group). Fill `human_label` with `TREE`, `NON_TREE` or `UNCERTAIN`; `human_note`, `reviewer`, `review_date` are optional. Look at the image in `crop_path` (and the contact sheet) to decide. Leave a row blank if not reviewed. |
| `reference_boxes.csv` | Human reference boxes, in **pixels of the original image** (8192×6144 for the October 24 images), e.g. for trees the model missed. One row per tree; `label` is `TREE`. |
| `reference_status.csv` | Per image: `NONE`, `PARTIAL` or `COMPLETE`. Precision, recall and F1 are computed only for images marked `COMPLETE` (every visible tree in the image has a reference box). |
| `field_inventory.csv` | Site-level field counts. Reported next to the model's counts, never used as an image-level reference or to filter predictions. |

`manual_review.csv` is a copy of `evaluation_results/manual_review_template.csv` at the time it was
created. If the evaluation algorithm or inputs change, review unit ids may change; the evaluation
report lists any labelled id it no longer recognises. New units are not added automatically:
copy rows over from the template if needed.
