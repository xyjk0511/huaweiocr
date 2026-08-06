import os
from dataclasses import dataclass

import cv2
import numpy as np


REPO_DIR = os.path.dirname(os.path.abspath(__file__))


@dataclass(frozen=True)
class ModelSpec:
    path: str
    names: tuple[str, ...]
    max_per_class: int | None = None


def _env_flag(name, default=False):
    raw = os.environ.get(name)
    if raw is None:
        return default
    return str(raw).strip().lower() not in {"", "0", "false", "no", "off"}


def _normalized_path(path):
    if not path:
        return ""
    return os.path.normcase(os.path.abspath(path))


_legacy_label_model_path = os.path.join(REPO_DIR, "local_models", "detectors", "label_detector.onnx")
_latest_label_model_dir = os.path.join(
    REPO_DIR,
    "local_models",
    "training",
    "label_detector_yolo26s_960_2class_ignore_v1",
    "weights",
)
_latest_label_model_pt_path = os.path.join(_latest_label_model_dir, "best.pt")
_latest_label_model_onnx_path = os.path.join(_latest_label_model_dir, "best.onnx")
_latest_label_model_path = os.environ.get(
    "LOCAL_YOLO_LATEST_LABEL_MODEL",
    _latest_label_model_pt_path if os.path.exists(_latest_label_model_pt_path) else _latest_label_model_onnx_path,
)
_preferred_label_model_path = os.environ.get("LOCAL_YOLO_LABEL_MODEL")
_prefer_latest_label_model = _env_flag(
    "LOCAL_YOLO_LABEL_MODEL_PREFER_LATEST",
    default=_env_flag("LOCAL_YOLO_LABEL_MODEL_PREFER_HARDCASE", default=True),
)
if _preferred_label_model_path:
    _primary_label_model_path = _preferred_label_model_path
elif _prefer_latest_label_model and os.path.exists(_latest_label_model_path):
    _primary_label_model_path = _latest_label_model_path
else:
    _primary_label_model_path = _legacy_label_model_path
_hardcase_label_model_path = os.environ.get(
    "LOCAL_YOLO_HARDCASE_LABEL_MODEL",
    _primary_label_model_path,
)


DEFAULT_MODEL_SPECS = {
    "huawei-2ha7t/7": ModelSpec(
        path=_primary_label_model_path,
        names=("huawei_label", "shipping_ignore"),
    ),
    "sn_model/9": ModelSpec(
        path=os.path.join(REPO_DIR, "local_models", "detectors", "field_detector.onnx"),
        names=("model", "partno", "sn"),
        max_per_class=1,
    ),
}
if os.path.exists(_hardcase_label_model_path) and _normalized_path(_hardcase_label_model_path) != _normalized_path(_primary_label_model_path):
    DEFAULT_MODEL_SPECS["huawei-2ha7t-hardcase/1"] = ModelSpec(
        path=_hardcase_label_model_path,
        names=("huawei_label", "shipping_ignore"),
    )


def get_model_path(model_id):
    spec = DEFAULT_MODEL_SPECS.get(model_id)
    if spec is None:
        return ""
    return _normalized_path(spec.path)


def model_ids_share_same_path(model_id_a, model_id_b):
    path_a = get_model_path(model_id_a)
    path_b = get_model_path(model_id_b)
    return bool(path_a and path_b and path_a == path_b)


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


def _is_postprocessed_xyxy_output(pred, num_classes):
    if pred.ndim != 2 or pred.shape[1] != 6:
        return False
    if pred.shape[0] == 0:
        return False
    scores = pred[:, 4]
    class_ids = pred[:, 5]
    if not np.all(np.isfinite(scores)) or not np.all(np.isfinite(class_ids)):
        return False
    if np.any(scores < 0.0) or np.any(scores > 1.0):
        return False
    if np.any(class_ids < 0.0) or np.any(class_ids > max(0, num_classes - 1)):
        return False
    rounded = np.round(class_ids)
    if not np.all(np.abs(class_ids - rounded) <= 1e-3):
        return False
    x1 = pred[:, 0]
    y1 = pred[:, 1]
    x2 = pred[:, 2]
    y2 = pred[:, 3]
    return bool(np.any((x2 > x1) & (y2 > y1)))


def _decode_postprocessed_xyxy_output(
    pred,
    names,
    original_shape,
    scale,
    pad,
    conf_threshold=0.25,
    nms_threshold=0.45,
    max_det=100,
):
    h, w = original_shape[:2]
    pad_x, pad_y = pad
    x1 = (pred[:, 0] - pad_x) / scale
    y1 = (pred[:, 1] - pad_y) / scale
    x2 = (pred[:, 2] - pad_x) / scale
    y2 = (pred[:, 3] - pad_y) / scale
    scores = pred[:, 4].astype(np.float32)
    class_ids = np.round(pred[:, 5]).astype(np.int32)

    x1 = np.clip(x1, 0, w)
    y1 = np.clip(y1, 0, h)
    x2 = np.clip(x2, 0, w)
    y2 = np.clip(y2, 0, h)

    valid = (scores >= conf_threshold) & (x2 > x1) & (y2 > y1)
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
    if _is_postprocessed_xyxy_output(pred, len(names)):
        return _decode_postprocessed_xyxy_output(
            pred,
            names,
            original_shape,
            scale,
            pad,
            conf_threshold=conf_threshold,
            nms_threshold=nms_threshold,
            max_det=max_det,
        )
    if pred.shape[0] < pred.shape[1]:
        pred = pred.T
    if pred.shape[1] < 5:
        return []
    if _is_postprocessed_xyxy_output(pred, len(names)):
        return _decode_postprocessed_xyxy_output(
            pred,
            names,
            original_shape,
            scale,
            pad,
            conf_threshold=conf_threshold,
            nms_threshold=nms_threshold,
            max_det=max_det,
        )

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
        self.spec = spec
        self.conf_threshold = float(conf_threshold)
        self.nms_threshold = float(nms_threshold)
        self.path = os.path.abspath(spec.path)
        if not os.path.isfile(self.path):
            raise FileNotFoundError(f"Local YOLO model file not found: {self.path}")
        self.device = str(device or "auto")
        self._predict_impl = None
        self.session = None
        self.input_name = None
        self.imgsz = 640
        self.ultralytics_model = None
        self._init_backend()

    def _init_backend(self):
        suffix = os.path.splitext(self.path)[1].lower()
        if suffix == ".pt":
            self._init_ultralytics_backend()
            self._predict_impl = self._predict_with_ultralytics
            return
        self._init_onnx_backend()
        self._predict_impl = self._predict_with_onnx

    def _init_onnx_backend(self):
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError(
                "onnxruntime is required for local YOLO inference. "
                "Install onnxruntime, or on NVIDIA machines install onnxruntime-gpu."
            ) from exc

        providers = self._select_providers(ort, self.device)
        self.session = ort.InferenceSession(self.path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.imgsz = self._input_size()

    def _init_ultralytics_backend(self):
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "ultralytics is required for local .pt YOLO inference. "
                "Install ultralytics to use PyTorch checkpoints directly."
            ) from exc

        self.ultralytics_model = YOLO(self.path)
        try:
            self.imgsz = int(getattr(self.ultralytics_model.model.args, "imgsz", 0) or 0)
        except Exception:
            self.imgsz = 0
        if self.imgsz <= 0:
            self.imgsz = 960

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
        return self._predict_impl(image_path)

    def supports_original_path_inference(self):
        # Both local backends can read the source image path directly.
        # Restricting ONNX to the temp-JPG fallback costs recall on some edge
        # cases in the packaged app because the extra recompression changes the
        # detector input unnecessarily.
        return os.path.splitext(self.path)[1].lower() in {".pt", ".onnx"}

    def _predict_with_onnx(self, image_path):
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

    def _predict_with_ultralytics(self, image_path):
        result = self.ultralytics_model.predict(
            source=image_path,
            imgsz=self.imgsz,
            conf=self.conf_threshold,
            device=self._ultralytics_device(),
            verbose=False,
        )[0]
        predictions = []
        for box in result.boxes:
            class_id = int(box.cls.item())
            if class_id < 0 or class_id >= len(self.spec.names):
                continue
            x1, y1, x2, y2 = map(float, box.xyxy[0].tolist())
            width = x2 - x1
            height = y2 - y1
            if width <= 0 or height <= 0:
                continue
            name = self.spec.names[class_id]
            predictions.append(
                {
                    "x": float(x1 + width / 2.0),
                    "y": float(y1 + height / 2.0),
                    "width": float(width),
                    "height": float(height),
                    "confidence": float(box.conf.item()),
                    "class": name,
                    "class_name": name,
                    "class_id": class_id,
                }
            )
        return predictions

    def _ultralytics_device(self):
        requested = self.device.strip().lower()
        if requested in {"", "auto"}:
            try:
                import torch
            except Exception:
                return "cpu"
            return 0 if torch.cuda.is_available() else "cpu"
        if requested in {"cpu", "-1", "none", "false", "off"}:
            return "cpu"
        try:
            return int(requested)
        except ValueError:
            return requested


class LocalYoloClient:
    def __init__(self, model_specs=None, detector_cls=LocalYoloDetector):
        self.model_specs = dict(model_specs or DEFAULT_MODEL_SPECS)
        self.detector_cls = detector_cls
        self.detectors = {}
        self.device = os.environ.get("LOCAL_YOLO_DEVICE", "auto")
        self.nms_threshold = float(os.environ.get("LOCAL_YOLO_NMS", "0.45"))

    def _conf_threshold_for(self, model_id):
        raw_global = os.environ.get("LOCAL_YOLO_CONF")
        if raw_global is not None:
            return float(raw_global)
        if model_id in {"huawei-2ha7t/7", "huawei-2ha7t-hardcase/1"}:
            spec = self.model_specs.get(model_id)
            default_label_conf = "0.35"
            if spec and str(spec.path or "").lower().endswith(".onnx"):
                default_label_conf = "0.20"
            return float(os.environ.get("LOCAL_YOLO_LABEL_CONF", default_label_conf))
        if model_id == "sn_model/9":
            return float(os.environ.get("LOCAL_YOLO_FIELD_CONF", "0.25"))
        return 0.25

    def _detector_for(self, model_id):
        if model_id not in self.model_specs:
            known = ", ".join(sorted(self.model_specs))
            raise KeyError(f"Unknown local YOLO model_id: {model_id}. Known: {known}")
        if model_id not in self.detectors:
            self.detectors[model_id] = self.detector_cls(
                self.model_specs[model_id],
                device=self.device,
                conf_threshold=self._conf_threshold_for(model_id),
                nms_threshold=self.nms_threshold,
            )
        return self.detectors[model_id]

    def infer(self, image_path, model_id=None):
        if not model_id:
            raise ValueError("model_id is required for local YOLO inference.")
        return {"predictions": self._detector_for(model_id).predict(image_path)}

    def supports_original_path_inference(self, model_id=None):
        if not model_id:
            raise ValueError("model_id is required for local YOLO inference.")
        return self._detector_for(model_id).supports_original_path_inference()

    def infer_original_path(self, image_path, model_id=None):
        if not model_id:
            raise ValueError("model_id is required for local YOLO inference.")
        return {"predictions": self._detector_for(model_id).predict(image_path)}
