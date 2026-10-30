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

## Which threshold the labels refer to

Review units are the persistence groups at **IoU >= 0.50** (`group_id_iou050`). A label applies to that
unit's detections (`detection_uids`), not to any group id at IoU 0.75 or 1.00: the same id at another
threshold is generally a different set of detections. Labels are never transferred between thresholds;
any such transfer would need an explicit detection-level mapping (the `detection_uids` column and
`evaluation_results/detections.csv` provide the ids for that).

## Editing in Excel without corrupting the file

Opening the CSV by double-clicking lets Excel convert values: `8-9`, `5-7`, `1-4` in `persistence_bin`
become dates (`9-Aug`, `7-May`, `4-Jan`), and the change is written back when the file is saved.
Only the `human_label`, `human_note`, `reviewer` and `review_date` columns should ever change.

1. Do not double-click the CSV. In Excel: **Data > From Text/CSV**, choose the file, click
   **Transform Data**, select all columns, set **Data Type > Text**, then **Close & Load**.
   (Or **Data > Get Data > Legacy Wizards > From Text (Legacy)**: Delimited, Comma, then mark at least
   `persistence_bin`, `trials_detected`, `n_trials` and `review_date` as **Text** in the last step.)
2. Edit only the four human columns. Save as **CSV UTF-8 (Comma delimited)** to `manual_review.csv`.
3. Close the file in Excel (Excel locks open files, so scripts cannot read or write them).
4. Run `python evaluate_consistency.py --validate-review`. It writes nothing; it reports any
   `persistence_bin` that does not match `n_trials` (flagging date-like values), any `trials_detected` that
   disagrees with `n_trials`, duplicate ids, and any generated column that no longer matches the evaluation.
   A full evaluation run performs the same check first and stops without writing outputs if it fails.
   The check never rewrites your file.

A text editor (VS Code) or LibreOffice Calc with the columns imported as Text avoids the problem entirely.
