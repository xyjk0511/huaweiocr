import os
from dataclasses import dataclass

import cv2
import numpy as np


REPO_DIR = os.path.dirname(os.path.abspath(__file__))


@dataclass(frozen=True)
class ModelSpec:
    path: str
    names: tuple[str, ...]


DEFAULT_MODEL_SPECS = {
    "huawei-2ha7t/7": ModelSpec(
        path=os.path.join(REPO_DIR, "local_models", "detectors", "label_detector.onnx"),
        names=("huawei_label",),
    ),
    "sn_model/9": ModelSpec(
        path=os.path.join(REPO_DIR, "local_models", "detectors", "field_detector.onnx"),
        names=("model", "partno", "sn"),
    ),
}


def _read_image(path):
    data = np.fromfile(path, dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def _letterbox(bgr, size):
    h, w = bgr.shape[:2]
    scale = min(size / float(h), size / float(w))
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    resized = cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    pad_w = size - new_w
    pad_h = size - new_h
    left = int(round(pad_w / 2.0 - 0.1))
    right = int(round(pad_w / 2.0 + 0.1))
    top = int(round(pad_h / 2.0 - 0.1))
    bottom = int(round(pad_h / 2.0 + 0.1))

    out = cv2.copyMakeBorder(
        resized,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=(114, 114, 114),
    )
    return out, scale, (left, top)


def _nms_by_class(boxes, scores, class_ids, threshold, max_det):
    keep = []
    for class_id in sorted(set(class_ids.tolist())):
        idxs = np.where(class_ids == class_id)[0]
        class_boxes = boxes[idxs].tolist()
        class_scores = scores[idxs].tolist()
        picked = cv2.dnn.NMSBoxes(class_boxes, class_scores, 0.0, threshold)
        if picked is None or len(picked) == 0:
            continue
        picked = picked.flatten().tolist() if hasattr(picked, "flatten") else list(picked)
        keep.extend(idxs[i] for i in picked)

    keep.sort(key=lambda i: float(scores[i]), reverse=True)
    return keep[:max_det]


def decode_yolov8_output(
    output,
    names,
    original_shape,
    scale,
    pad,
    conf_threshold=0.25,
    nms_threshold=0.45,
    max_det=100,
):
    pred = np.squeeze(output)
    if pred.ndim != 2:
        return []
    if pred.shape[0] < pred.shape[1]:
        pred = pred.T
    if pred.shape[1] < 5:
        return []

    boxes_xywh = pred[:, :4]
    class_scores = pred[:, 4:]
    class_ids = np.argmax(class_scores, axis=1)
    scores = class_scores[np.arange(class_scores.shape[0]), class_ids]
    mask = scores >= conf_threshold
    if not np.any(mask):
        return []

    boxes_xywh = boxes_xywh[mask]
    class_ids = class_ids[mask].astype(np.int32)
    scores = scores[mask].astype(np.float32)

    h, w = original_shape[:2]
    pad_x, pad_y = pad
    x = boxes_xywh[:, 0]
    y = boxes_xywh[:, 1]
    bw = boxes_xywh[:, 2]
    bh = boxes_xywh[:, 3]

    x1 = (x - bw / 2.0 - pad_x) / scale
    y1 = (y - bh / 2.0 - pad_y) / scale
    x2 = (x + bw / 2.0 - pad_x) / scale
    y2 = (y + bh / 2.0 - pad_y) / scale

    x1 = np.clip(x1, 0, w)
    y1 = np.clip(y1, 0, h)
    x2 = np.clip(x2, 0, w)
    y2 = np.clip(y2, 0, h)

    valid = (x2 > x1) & (y2 > y1)
    if not np.any(valid):
        return []

    x1 = x1[valid]
    y1 = y1[valid]
    x2 = x2[valid]
    y2 = y2[valid]
    scores = scores[valid]
    class_ids = class_ids[valid]

    nms_boxes = np.stack([x1, y1, x2 - x1, y2 - y1], axis=1)
    keep = _nms_by_class(nms_boxes, scores, class_ids, nms_threshold, max_det)

    predictions = []
    for i in keep:
        class_id = int(class_ids[i])
        name = names[class_id] if class_id < len(names) else str(class_id)
        width = float(x2[i] - x1[i])
        height = float(y2[i] - y1[i])
        predictions.append(
            {
                "x": float(x1[i] + width / 2.0),
                "y": float(y1[i] + height / 2.0),
                "width": width,
                "height": height,
                "confidence": float(scores[i]),
                "class": name,
                "class_name": name,
                "class_id": class_id,
            }
        )
    return predictions


class LocalYoloDetector:
    def __init__(self, spec, device="auto", conf_threshold=0.25, nms_threshold=0.45):
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError(
                "onnxruntime is required for local YOLO inference. "
                "Install onnxruntime, or on NVIDIA machines install onnxruntime-gpu."
            ) from exc

        self.spec = spec
        self.conf_threshold = float(conf_threshold)
        self.nms_threshold = float(nms_threshold)
        self.path = os.path.abspath(spec.path)
        if not os.path.isfile(self.path):
            raise FileNotFoundError(f"Local YOLO model file not found: {self.path}")

        providers = self._select_providers(ort, device)
        self.session = ort.InferenceSession(self.path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.imgsz = self._input_size()

    @staticmethod
    def _select_providers(ort, device):
        available = set(ort.get_available_providers())
        requested = str(device or "auto").lower()
        wants_cuda = requested not in {"", "auto", "cpu", "-1", "none"}
        if wants_cuda and "CUDAExecutionProvider" in available:
            try:
                device_id = int(requested)
            except ValueError:
                device_id = 0
            return [("CUDAExecutionProvider", {"device_id": device_id}), "CPUExecutionProvider"]
        if requested == "auto" and "CUDAExecutionProvider" in available:
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        return ["CPUExecutionProvider"]

    def _input_size(self):
        shape = self.session.get_inputs()[0].shape
        try:
            h = int(shape[2])
            w = int(shape[3])
            if h == w and h > 0:
                return h
        except (TypeError, ValueError, IndexError):
            pass
        return 640

    def predict(self, image_path):
        bgr = _read_image(image_path)
        if bgr is None:
            raise RuntimeError(f"Failed to read image for local YOLO inference: {image_path}")

        letterboxed, scale, pad = _letterbox(bgr, self.imgsz)
        rgb = cv2.cvtColor(letterboxed, cv2.COLOR_BGR2RGB)
        blob = np.transpose(rgb, (2, 0, 1)).astype(np.float32) / 255.0
        blob = np.expand_dims(blob, axis=0)
        outputs = self.session.run(None, {self.input_name: blob})
        return decode_yolov8_output(
            outputs[0],
            self.spec.names,
            bgr.shape,
            scale,
            pad,
            conf_threshold=self.conf_threshold,
            nms_threshold=self.nms_threshold,
        )


class LocalYoloClient:
    def __init__(self, model_specs=None, detector_cls=LocalYoloDetector):
        self.model_specs = dict(model_specs or DEFAULT_MODEL_SPECS)
        self.detector_cls = detector_cls
        self.detectors = {}
        self.device = os.environ.get("LOCAL_YOLO_DEVICE", "auto")
        self.conf_threshold = float(os.environ.get("LOCAL_YOLO_CONF", "0.25"))
        self.nms_threshold = float(os.environ.get("LOCAL_YOLO_NMS", "0.45"))

    def _detector_for(self, model_id):
        if model_id not in self.model_specs:
            known = ", ".join(sorted(self.model_specs))
            raise KeyError(f"Unknown local YOLO model_id: {model_id}. Known: {known}")
        if model_id not in self.detectors:
            self.detectors[model_id] = self.detector_cls(
                self.model_specs[model_id],
                device=self.device,
                conf_threshold=self.conf_threshold,
                nms_threshold=self.nms_threshold,
            )
        return self.detectors[model_id]

    def infer(self, image_path, model_id=None):
        if not model_id:
            raise ValueError("model_id is required for local YOLO inference.")
        return {"predictions": self._detector_for(model_id).predict(image_path)}
