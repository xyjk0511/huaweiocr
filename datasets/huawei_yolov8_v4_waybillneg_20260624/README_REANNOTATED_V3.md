# Re-annotated complete product-label dataset (YOLOv8 bbox, resplit)

Class
- 0: product_label

Annotation policy
- Positive: complete visible product labels only. The box covers the full product-label sticker / printed product-label region.
- Negative: SF / Douyin-commerce / logistics waybills, non-product shipping labels, and product labels that are too occluded or edge-truncated to serve as complete-label crops.
- The relabeling goal is to avoid crops that start from Desc / S/N / MAC and miss Part No. / Model.
- Images are physically upright; no EXIF orientation dependency remains.

Counts
- Images: 36
- Boxes: 143
- Empty label files: 2
- Validation issues: 0

Current split
- Train: 28 images, 113 boxes, 1 empty label file
- Valid: 4 images, 11 boxes, 1 empty label file
- Test: 4 images, 19 boxes, 0 empty label files

Resplit note
- The split was adjusted to 28/4/4 so validation and test are no longer trivially small.
- One negative sample is kept in train and one in valid to preserve negative exposure during training and a basic false-positive check during validation.

Train example
```bash
yolo detect train data=data.yaml model=yolov8n.pt imgsz=1280
```
