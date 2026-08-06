import os
import json
import re
import sys
import types
import warnings
import inspect
import importlib.machinery
from contextlib import nullcontext

from win_subprocess import hide_subprocess_windows

hide_subprocess_windows()


def configure_paddle_runtime_env():
    """
    Keep PaddleOCR on the conservative CPU executor path in frozen builds.

    Some packaged Paddle/PaddleOCR combinations can fail inside oneDNN/PIR
    conversion before OCR returns a usable fallback.  These flags must be set
    before importing paddle.
    """
    os.environ.setdefault("DISABLE_MODEL_SOURCE_CHECK", "True")
    os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
    os.environ.setdefault("PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT", "False")
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ.setdefault("FLAGS_use_mkldnn", "0")
    os.environ.setdefault("FLAGS_use_onednn", "0")
    os.environ.setdefault("FLAGS_enable_pir_api", "0")
    os.environ.setdefault("FLAGS_enable_pir_in_executor", "0")


configure_paddle_runtime_env()


def preload_torch_runtime_for_paddle():
    if os.name != "nt":
        return
    if "paddle" in sys.modules or "paddleocr" in sys.modules:
        return
    try:
        import torch  # noqa: F401
    except Exception:
        pass


preload_torch_runtime_for_paddle()


def ensure_torch_import_compat():
    """
    PaddleX/modelscope can import torch utilities even on pure OCR paths.

    In the portable CPU build we do not rely on a functional PyTorch runtime.
    If the bundled environment only exposes a partial/namespace ``torch``
    package, install a tiny compatibility stub so ``modelscope.utils.torch_utils``
    can import without pulling the full PyTorch stack into the release.
    """
    try:
        import torch as torch_mod  # type: ignore
        import torch.multiprocessing  # type: ignore  # noqa: F401
        from torch import distributed as _dist  # type: ignore  # noqa: F401
        return torch_mod
    except Exception:
        sys.modules.pop("torch", None)
        sys.modules.pop("torch.multiprocessing", None)
        sys.modules.pop("torch.distributed", None)

    def _stub_spec(name: str, *, is_package: bool = False):
        return importlib.machinery.ModuleSpec(name, loader=None, is_package=is_package)

    class _TorchValueStub:
        def __init__(self, name="torch.stub"):
            self._name = name

        def __call__(self, *args, **kwargs):
            return self

        def __getattr__(self, name):
            return _TorchValueStub(f"{self._name}.{name}")

        def __bool__(self):
            return False

        def __iter__(self):
            return iter(())

        def __repr__(self):
            return f"<_TorchValueStub {self._name}>"

    class _TorchTensorStub:
        def __init__(self, *args, device="cpu", **kwargs):
            self.device = device
            self.shape = ()

        def cpu(self):
            return self

        def numpy(self):
            class _ArrayStub:
                @staticmethod
                def tobytes():
                    return b""

            return _ArrayStub()

        def parameters(self):
            return []

    class _TorchDeviceStub:
        def __init__(self, name="cpu"):
            self.type = str(name)

        def __str__(self):
            return self.type

    class _TorchModuleStub:
        def __init__(self, *args, **kwargs):
            pass

        def parameters(self):
            return []

        def to(self, *args, **kwargs):
            return self

        def eval(self):
            return self

        def train(self, mode=True):
            return self

        def compile(self, **kwargs):
            return self

    def _tensor_factory(*args, **kwargs):
        return _TorchTensorStub(*args, **kwargs)

    def _module_getattr(name):
        if name and name[0].isupper():
            return _TorchModuleStub
        return _TorchValueStub(f"torch.nn.{name}")

    def _value_getattr(name):
        return _TorchValueStub(f"torch.{name}")

    torch_stub = types.ModuleType("torch")
    torch_stub.__dict__["__version__"] = "0.0-stub"
    torch_stub.__package__ = "torch"
    torch_stub.__spec__ = _stub_spec("torch", is_package=True)
    torch_stub.__path__ = []

    class _CudaStub:
        @staticmethod
        def set_device(*args, **kwargs):
            return None

        @staticmethod
        def device_count():
            return 0

        @staticmethod
        def is_available():
            return False

    mp_stub = types.ModuleType("torch.multiprocessing")
    mp_stub.__package__ = "torch"
    mp_stub.__spec__ = _stub_spec("torch.multiprocessing")
    mp_state = {"method": None}

    def _get_start_method(allow_none=False):
        method = mp_state["method"]
        return method if (allow_none or method is not None) else "spawn"

    def _set_start_method(method, force=False):
        if mp_state["method"] is None or force:
            mp_state["method"] = method

    mp_stub.get_start_method = _get_start_method
    mp_stub.set_start_method = _set_start_method

    dist_stub = types.ModuleType("torch.distributed")
    dist_stub.__package__ = "torch"
    dist_stub.__spec__ = _stub_spec("torch.distributed")
    dist_stub.is_available = lambda: False
    dist_stub.is_initialized = lambda: False
    dist_stub.get_rank = lambda group=None: 0
    dist_stub.get_world_size = lambda group=None: 1
    dist_stub.barrier = lambda: None
    dist_stub.init_process_group = lambda *args, **kwargs: None

    nn_stub = types.ModuleType("torch.nn")
    nn_stub.__package__ = "torch"
    nn_stub.__spec__ = _stub_spec("torch.nn", is_package=True)
    nn_stub.__path__ = []
    nn_stub.Module = _TorchModuleStub
    nn_stub.Linear = _TorchModuleStub
    nn_stub.__getattr__ = _module_getattr

    nn_functional_stub = types.ModuleType("torch.nn.functional")
    nn_functional_stub.__package__ = "torch.nn"
    nn_functional_stub.__spec__ = _stub_spec("torch.nn.functional")
    nn_functional_stub.__getattr__ = lambda name: _TorchValueStub(f"torch.nn.functional.{name}")

    nn_utils_stub = types.ModuleType("torch.nn.utils")
    nn_utils_stub.__package__ = "torch.nn"
    nn_utils_stub.__spec__ = _stub_spec("torch.nn.utils", is_package=True)
    nn_utils_stub.__path__ = []

    nn_utils_rnn_stub = types.ModuleType("torch.nn.utils.rnn")
    nn_utils_rnn_stub.__package__ = "torch.nn.utils"
    nn_utils_rnn_stub.__spec__ = _stub_spec("torch.nn.utils.rnn")
    nn_utils_rnn_stub.pad_sequence = lambda *args, **kwargs: _TorchTensorStub()

    nn_utils_stub.rnn = nn_utils_rnn_stub
    nn_stub.functional = nn_functional_stub
    nn_stub.utils = nn_utils_stub

    torch_stub.cuda = _CudaStub()
    torch_stub.multiprocessing = mp_stub
    torch_stub.distributed = dist_stub
    torch_stub.nn = nn_stub
    torch_stub.Tensor = _TorchTensorStub
    torch_stub.ByteTensor = _TorchTensorStub
    torch_stub.LongTensor = _TorchTensorStub
    torch_stub.device = _TorchDeviceStub
    torch_stub.uint8 = "uint8"
    torch_stub.int64 = "int64"
    torch_stub.tensor = _tensor_factory
    torch_stub.empty = _tensor_factory
    torch_stub.zeros = _tensor_factory
    torch_stub.ones = _tensor_factory
    torch_stub.full = _tensor_factory
    torch_stub.arange = _tensor_factory
    torch_stub.cat = _tensor_factory
    torch_stub.manual_seed = lambda *args, **kwargs: None
    torch_stub.no_grad = nullcontext
    torch_stub.compile = lambda model, **kwargs: model
    torch_stub.__getattr__ = _value_getattr
    torch_stub.cuda.manual_seed_all = lambda *args, **kwargs: None

    sys.modules["torch"] = torch_stub
    sys.modules["torch.multiprocessing"] = mp_stub
    sys.modules["torch.distributed"] = dist_stub
    sys.modules["torch.nn"] = nn_stub
    sys.modules["torch.nn.functional"] = nn_functional_stub
    sys.modules["torch.nn.utils"] = nn_utils_stub
    sys.modules["torch.nn.utils.rnn"] = nn_utils_rnn_stub
    return torch_stub


ensure_torch_import_compat()

import paddle  # noqa: E402  (torch compat shim must run before paddle imports)
from paddleocr import PaddleOCR  # noqa: E402
from app_paths import ensure_models_installed, get_resource_path  # noqa: E402

# ================== 配置区 ==================
# 你的 sn 小图目录
IMG_DIR = os.environ.get("OCR_IMG_DIR", os.path.join("stage2_fields", "sn"))

# 输出 JSONL 文件（每行一张图片的结果）
OUT_JSONL = os.environ.get("OCR_OUT_JSONL", os.path.join(IMG_DIR, "sn_ocr_results.jsonl"))

# 置信度阈值（低于这个就当成噪声丢掉）
MIN_SCORE = 0.5

# PaddleOCR 语言：
#   - 只识别数字/大写字母为主：建议 lang="en"
#   - 如果可能有中文：可以改成 lang="ch"
OCR_LANG = "en"

# 是否使用 GPU（你现在一般是 CPU，就 False）
USE_GPU = False
OCR_PROFILE = os.environ.get("HUAWEIOCR_OCR_PROFILE", "mobile").strip().lower()
# ===========================================

LOG_SINK = None


def set_log_sink(sink) -> None:
    global LOG_SINK
    LOG_SINK = sink


def _emit_log(message: str) -> None:
    if LOG_SINK:
        LOG_SINK(message)
    else:
        print(message)


def patch_paddlex_dep_checks():
    """
    In packaged builds, dependency metadata can be missing even if modules exist.
    Patch PaddleX to accept available modules based on importability.
    """
    try:
        import importlib.util
        from paddlex.utils import deps
    except Exception:
        return

    if getattr(deps, "_patched_by_app", False):
        return

    orig_is_dep_available = deps.is_dep_available

    def _module_exists(name):
        try:
            return importlib.util.find_spec(name) is not None
        except Exception:
            return False

    alias_map = {
        "opencv-contrib-python": ["cv2"],
        "opencv-python": ["cv2"],
        "python-bidi": ["bidi"],
        "pyclipper": ["pyclipper"],
    }

    def patched(dep, /, check_version=False):
        try:
            if orig_is_dep_available(dep, check_version=check_version):
                return True
        except Exception:
            pass

        names = list(alias_map.get(dep, []))
        if "-" in dep:
            names.append(dep.replace("-", "_"))

        for name in names:
            if _module_exists(name):
                return True
        return False

    deps.is_dep_available = patched
    deps.require_deps = lambda *args, **kwargs: None
    deps.require_extra = lambda *args, **kwargs: None
    deps._patched_by_app = True


def list_image_files(img_dir):
    """列出目录下所有图片文件（按文件名排序）"""
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
    files = []
    for name in os.listdir(img_dir):
        ext = os.path.splitext(name)[1].lower()
        if ext in exts:
            files.append(name)
    files.sort()
    return files


def _first_existing_model_dir(root, names):
    if not root:
        return None
    for name in names:
        path = os.path.join(root, name)
        if os.path.isdir(path):
            return path
    return None


def _recognition_model_candidates():
    if OCR_PROFILE in {"accurate", "server"}:
        return ["PP-OCRv5_server_rec", "en_PP-OCRv5_mobile_rec", "PP-OCRv5_mobile_rec"]
    return ["en_PP-OCRv5_mobile_rec", "PP-OCRv5_mobile_rec", "PP-OCRv5_server_rec"]


def _local_model_root_fallback():
    for path in (
        get_resource_path("models", "official_models"),
        get_resource_path("bundle", "models", "official_models"),
    ):
        if os.path.isdir(path):
            return path
    return None


def _paddleocr_model_kwargs(model_root):
    det_dir = _first_existing_model_dir(model_root, ["PP-OCRv5_server_det"])
    rec_dir = _first_existing_model_dir(model_root, _recognition_model_candidates())
    cls_dir = _first_existing_model_dir(model_root, ["PP-LCNet_x1_0_textline_ori"])
    rec_name = os.path.basename(rec_dir) if rec_dir else None
    desired = {
        "use_doc_orientation_classify": False,
        "use_doc_unwarping": False,
        "use_textline_orientation": True,
        "use_angle_cls": True,
        "use_mkldnn": False,
        "enable_mkldnn": False,
        "text_detection_model_name": "PP-OCRv5_server_det" if det_dir else None,
        "det_model_dir": det_dir,
        "text_detection_model_dir": det_dir,
        "text_recognition_model_name": rec_name,
        "rec_model_dir": rec_dir,
        "text_recognition_model_dir": rec_dir,
        "textline_orientation_model_name": "PP-LCNet_x1_0_textline_ori" if cls_dir else None,
        "cls_model_dir": cls_dir,
        "textline_orientation_model_dir": cls_dir,
    }
    try:
        params = inspect.signature(PaddleOCR).parameters
    except Exception:
        return {}
    return {name: value for name, value in desired.items() if value is not None and name in params}


def init_ocr():
    """初始化 PaddleOCR 引擎（只初始化一次）"""
    # 可选：关掉一些没必要的 warning
    warnings.filterwarnings("ignore")

    model_root = ensure_models_installed() or _local_model_root_fallback()
    patch_paddlex_dep_checks()

    configure_paddle_runtime_env()

    device = "gpu" if USE_GPU else "cpu"
    paddle.set_device(device)
    _emit_log("🔧 正在初始化 PaddleOCR 引擎（lang='{}', device='{}'）...".format(
        OCR_LANG, device
    ))
    ocr = PaddleOCR(
        lang=OCR_LANG,
        **_paddleocr_model_kwargs(model_root),
    )
    _emit_log("✅ OCR 引擎初始化完成")
    return ocr


def ocr_one_image(ocr, img_path):
    """
    识别单张图片，返回：
      - texts: [{'text': str, 'score': float}, ...]
      - concat: 把高置信度文本拼起来的字符串
    """
    result = ocr.ocr(img_path)

    texts = []
    # paddleocr 3.x 返回 list[dict]，旧版本返回 list[list]
    if result and isinstance(result[0], dict):
        img_result = result[0]
        rec_texts = img_result.get("rec_texts", []) or []
        rec_scores = img_result.get("rec_scores", []) or []
        for text, score in zip(rec_texts, rec_scores):
            texts.append({"text": text, "score": float(score)})
    else:
        # 旧版结构：list[ img_result ]，一张图对应一个 img_result
        for img_result in result:
            for line in img_result:
                # line[1] = (text, score)
                text = line[1][0]
                score = float(line[1][1])
                texts.append({"text": text, "score": score})

    # 只保留高置信度的片段再拼接
    high_conf_texts = [t["text"] for t in texts if t["score"] >= MIN_SCORE]
    concat = "".join(high_conf_texts)

    concat = normalize_sn_text(concat)

    return texts, concat


def normalize_sn_text(text):
    """
    纠正常见的 SN 字段识别错误。
    规则：把 SIN/S1N/SN- 统一为 SN:，并将 ER 后一位若是数字 4/8 纠成 A/B。
    """
    if not text:
        return text

    normalized = text.replace("S1N:", "SN:").replace("SIN:", "SN:").replace("SN-", "SN:")

    # AP162E 常见格式：21500871494 + ER[A-C/9] + 4~7位数字
    # 纠错：...71484... -> ...71494...；ER8/ER4 -> ERB/ERA
    if "21500871484ER" in normalized:
        normalized = normalized.replace("21500871484ER", "21500871494ER")

    normalized = normalized.replace("ERAD", "ERA0")
    normalized = normalized.replace("ER8", "ERB").replace("ER4", "ERA")
    # S380-S8P2T 常见 SN 前缀：4E25A017xxxx，如果识别成 4E28A017 则纠正
    if "4E28A017" in normalized:
        normalized = normalized.replace("4E28A017", "4E25A017")
    # 去掉重复的 SN 前缀，比如 "SN215..." -> "215..."
    normalized = re.sub(r"^SN:?", "", normalized)
    return normalized


def main():
    # 兼容 Windows 控制台中文输出
    try:
        if sys.platform.startswith("win"):
            sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    if not os.path.isdir(IMG_DIR):
        print("❌ 目录不存在：", IMG_DIR)
        return

    files = list_image_files(IMG_DIR)
    if not files:
        print("⚠️ 在目录中没有找到任何图片：", IMG_DIR)
        return

    print("📂 发现图片 {} 张，目录：{}".format(len(files), IMG_DIR))

    ocr = init_ocr()

    # 打开输出文件
    fout = open(OUT_JSONL, "w", encoding="utf-8")

    try:
        for idx, fname in enumerate(files, 1):
            img_path = os.path.join(IMG_DIR, fname)
            print("\n[{:>3}/{}] 识别文件：{}".format(idx, len(files), fname))

            texts, concat = ocr_one_image(ocr, img_path)

            # 控制台打印详细结果
            if not texts:
                print("  ⚠️ 没有识别到任何文本")
            else:
                for i, t in enumerate(texts, 1):
                    print("  [{}] '{}' (score={:.3f})".format(
                        i, t["text"], t["score"]
                    ))
                print("  👉 拼接高置信度文本：", repr(concat))

            # 写一行 JSON 到 jsonl 文件
            record = {
                "file": fname,
                "texts": texts,
                "concat": concat
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            fout.flush()

        print("\n✅ 全部处理完成，结果已保存到：")
        print("   ", OUT_JSONL)

    finally:
        fout.close()


if __name__ == "__main__":
    main()
