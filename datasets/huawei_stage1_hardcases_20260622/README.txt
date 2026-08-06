Stage1 hard cases for Huawei label detector retraining.

Folders:
- raw_images/: copied source photos for annotation
- labels_pending/: empty YOLO txt placeholders to be filled manually
- manifest.csv: per-image target counts and issue notes

Class list:
- 0 = huawei_label

Labeling rule:
- Only label Huawei product labels on the cartons.
- Do not label SF shipping notes / courier slips.
- Keep partially occluded Huawei labels if the visible region is still enough to localize the printed label rectangle.
- Ignore tiny fragments that are too incomplete to define a stable label box.
