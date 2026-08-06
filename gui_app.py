import os
import sys
import threading
import time
import datetime
import json
import re
import openpyxl
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk
import platform
import traceback
import ctypes
import subprocess
import queue
from win_subprocess import hide_subprocess_windows

hide_subprocess_windows()

# GUI 默认走 CPU，避免源码版启动时 onnxruntime 反复探测缺失的 CUDA/cuDNN 依赖。
# 命令行批跑如果显式设置了 LOCAL_YOLO_DEVICE，则仍然以外部设置为准。
os.environ.setdefault("LOCAL_YOLO_DEVICE", "cpu")

from app_paths import get_resource_path, get_barcode_cli_path  # noqa: E402  (env vars must be set before pipeline imports)
from gui_pipeline import copy_images_to_unique_run_dir, load_pipeline_modules, start_ocr_prewarm_thread  # noqa: E402
from gui_i18n import get_strings  # noqa: E402
from huaweiocr.io.feedback import build_feedback_package  # noqa: E402
# 粘贴图片支持（可选）
try:
    from PIL import ImageGrab, Image
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False
    ImageGrab = None
    Image = None
# 尝试导入拖拽支持（可选）
try:
    from tkinterdnd2 import DND_FILES, TkinterDnD
    DND_IMPORT_AVAILABLE = True
except ImportError:
    DND_IMPORT_AVAILABLE = False
    DND_FILES = None
    TkinterDnD = None
# 支持的图片后缀
SUPPORTED_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def _detect_windows_processor_architecture() -> str:
    if os.name != "nt":
        return ""

    class SYSTEM_INFO_UNION(ctypes.Union):
        _fields_ = [
            ("dwOemId", ctypes.c_uint32),
            ("w", ctypes.c_uint16 * 2),
        ]

    class SYSTEM_INFO(ctypes.Structure):
        _anonymous_ = ("u",)
        _fields_ = [
            ("u", SYSTEM_INFO_UNION),
            ("dwPageSize", ctypes.c_uint32),
            ("lpMinimumApplicationAddress", ctypes.c_void_p),
            ("lpMaximumApplicationAddress", ctypes.c_void_p),
            ("dwActiveProcessorMask", ctypes.c_size_t),
            ("dwNumberOfProcessors", ctypes.c_uint32),
            ("dwProcessorType", ctypes.c_uint32),
            ("dwAllocationGranularity", ctypes.c_uint32),
            ("wProcessorLevel", ctypes.c_uint16),
            ("wProcessorRevision", ctypes.c_uint16),
        ]

    try:
        system_info = SYSTEM_INFO()
        ctypes.windll.kernel32.GetNativeSystemInfo(ctypes.byref(system_info))
        arch_code = system_info.w[0]
        if arch_code == 9:
            return "AMD64"
        if arch_code == 0:
            return "x86"
        if arch_code == 12:
            return "ARM64"
    except Exception:
        pass

    machine = (platform.machine() or "").strip().upper()
    if machine in {"AMD64", "X86", "ARM64"}:
        return machine
    if machine in {"X64", "X86_64"}:
        return "AMD64"
    if machine in {"I386", "I686"}:
        return "x86"

    return "AMD64" if ctypes.sizeof(ctypes.c_void_p) == 8 else "x86"


def _prepare_tkdnd_runtime() -> str:
    arch = _detect_windows_processor_architecture()
    if os.name == "nt" and arch and not os.environ.get("PROCESSOR_ARCHITECTURE"):
        os.environ["PROCESSOR_ARCHITECTURE"] = arch
    return arch


def _mask_path_text(text: str) -> str:
    text = "" if text is None else str(text)

    safe_roots = [os.getcwd(), os.path.dirname(os.path.abspath(__file__))]
    if getattr(sys, "frozen", False):
        safe_roots.append(os.path.dirname(os.path.abspath(sys.executable)))
    safe_roots = [
        os.path.normcase(os.path.abspath(root))
        for root in safe_roots
        if root
    ]

    def replace_path(match):
        value = match.group(0)
        trailing = ""
        while value and value[-1] in ".:)]}":
            trailing = value[-1] + trailing
            value = value[:-1]
        abs_value = os.path.normcase(os.path.abspath(value))
        for root in safe_roots:
            try:
                if os.path.commonpath([root, abs_value]) == root:
                    rel = os.path.relpath(value, root)
                    return rel + trailing
            except ValueError:
                continue
        return "[path]" + trailing

    text = re.sub(r"(?i)[a-z]:[\\/][^\s|,;\"']+", replace_path, text)
    return re.sub(r"(?<!\w)/(?:[^/\s]+/)+[^\s|,;\"']+", replace_path, text)


def _display_source(value: str, strings=None) -> str:
    src = str(value or "")
    s = strings or get_strings("zh")
    if src == "barcode":
        return s["source_barcode"]
    if src == "ocr_file":
        return s["source_ocr"]
    if src == "ocr_color":
        return s["source_ocr"]
    if src == "ocr_bin":
        return s["source_ocr"]
    if src == "ocr_top":
        return s["source_ocr"]
    if src == "ocr_no_match":
        return s["source_ocr_no_match"]
    if src.startswith("ocr"):
        return s["source_ocr"]
    if src == "barcode_ambiguous":
        return s["source_barcode_ambiguous"]
    if src == "barcode_parse_fail":
        return s["source_barcode_parse_fail"]
    if src == "barcode_quality_reject":
        return s["source_barcode_quality_reject"]
    if src == "barcode_decoder_miss":
        return s["source_barcode_decoder_miss"]
    if src == "barcode_no_match":
        return s["source_barcode_no_match"]
    if src == "missing":
        return s["source_missing"]
    if src == "none":
        return s["source_none"]
    if "+sn_hint" in src:
        return _display_source(src.replace("+sn_hint", ""), s) + s["source_sn_hint_suffix"]
    return src


def _display_model_src(value: str, strings=None) -> str:
    return _display_source(value, strings)


def _display_sn_src(value: str, strings=None) -> str:
    return _display_source(value, strings)


def _self_check():
    if getattr(sys, "frozen", False):
        base_dir = os.path.dirname(sys.executable)
    else:
        base_dir = os.path.dirname(os.path.abspath(__file__))
    log_path = os.path.join(base_dir, "self_check.log")
    lines = []
    lines.append("=" * 72)
    lines.append(f"time={datetime.datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"frozen={getattr(sys, 'frozen', False)}")
    lines.append(f"python={sys.version}")
    lines.append(f"platform={platform.platform()}")
    lines.append(f"platform_machine={platform.machine()}")
    lines.append(f"processor_arch_env={os.environ.get('PROCESSOR_ARCHITECTURE')}")
    lines.append(f"tkdnd_arch_guess={_detect_windows_processor_architecture()}")
    lines.append(f"base_dir={base_dir}")
    try:
        cli_path = get_barcode_cli_path()
    except Exception:
        cli_path = ""
    lines.append(f"barcode_cli={cli_path}")
    lines.append(f"barcode_cli_exists={os.path.isfile(cli_path)}")
    if os.path.isfile(cli_path):
        try:
            startupinfo = None
            creationflags = 0
            if os.name == "nt":
                startupinfo = subprocess.STARTUPINFO()
                startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                startupinfo.wShowWindow = subprocess.SW_HIDE
                creationflags = subprocess.CREATE_NO_WINDOW | getattr(subprocess, "DETACHED_PROCESS", 0)
            proc = subprocess.run(
                [cli_path, "--version"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="ignore",
                timeout=5,
                startupinfo=startupinfo,
                creationflags=creationflags,
            )
            lines.append(f"barcode_cli_rc={proc.returncode}")
            lines.append(f"barcode_cli_stdout={proc.stdout.strip()}")
            lines.append(f"barcode_cli_stderr={proc.stderr.strip()}")
        except Exception as exc:
            lines.append(f"barcode_cli_run_fail err={exc!r}")
    libiconv = get_resource_path("pyzbar", "libiconv.dll")
    libzbar = get_resource_path("pyzbar", "libzbar-64.dll")
    lines.append(f"libiconv_path={libiconv}")
    lines.append(f"libiconv_exists={os.path.isfile(libiconv)}")
    lines.append(f"libzbar_path={libzbar}")
    lines.append(f"libzbar_exists={os.path.isfile(libzbar)}")
    for dll_path in (libiconv, libzbar):
        if os.path.isfile(dll_path):
            try:
                ctypes.CDLL(dll_path)
                lines.append(f"load_ok={dll_path}")
            except Exception as exc:
                lines.append(f"load_fail={dll_path} err={exc!r}")
    try:
        import pyzbar  # noqa: F401
        lines.append("pyzbar_import=ok")
    except Exception as exc:
        lines.append(f"pyzbar_import=fail err={exc!r}")
    try:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write("\n".join(_mask_path_text(line) for line in lines) + "\n")
    except Exception:
        pass
# ============== GUI 主窗体 ==============
class App(tk.Tk):
    def __init__(self, strings=None):
        super().__init__()
        self.strings = strings or get_strings("zh")
        self.dnd_enabled = False
        self.dnd_error = ""
        self.dnd_arch = _prepare_tkdnd_runtime()
        if DND_IMPORT_AVAILABLE and TkinterDnD is not None:
            try:
                TkinterDnD._require(self)
                self.dnd_enabled = True
            except Exception as exc:
                self.dnd_error = str(exc)
        self.title(self.strings["window_title"])
        self.geometry("900x700")
        # 已选择的图片路径列表
        self.image_paths = []
        self._crop_module = None
        self._scan2_module = None
        self._is_running = False
        self._last_run_dir = ""
        self.result_rows = []
        self._main_thread_ident = threading.get_ident()
        self._log_queue = queue.SimpleQueue()
        self._log_poll_interval_ms = 8
        self._log_max_batch_lines = 10
        self._log_idle_heartbeat_sec = 0.8
        self._last_log_monotonic = time.monotonic()
        self._last_heartbeat_monotonic = 0.0
        self._last_stage_hint = ""
        # ============ 顶部：拖拽区域 ============
        top_frame = tk.Frame(self)
        top_frame.pack(fill=tk.X, padx=10, pady=10)
        if self.dnd_enabled:
            self.drop_area = tk.Label(
                top_frame,
                text=self.strings["drop_hint_enabled"],
                relief="ridge",
                borderwidth=2,
                width=60,
                height=4,
                fg="#555555"
            )
        else:
            hint = self.strings["drop_hint_disabled"]
            if not DND_IMPORT_AVAILABLE:
                hint = self.strings["drop_hint_tkdnd_missing"]
            self.drop_area = tk.Label(
                top_frame,
                text=hint,
                relief="ridge",
                borderwidth=2,
                width=60,
                height=4,
                fg="#aa0000"
            )
        self.drop_area.pack(fill=tk.X, expand=True)
        if self.dnd_enabled:
            # 注册拖拽目标
            self.drop_area.drop_target_register(DND_FILES)
            self.drop_area.dnd_bind("<<Drop>>", self.on_drop)
        # 让 drop_area 可聚焦（可选）
        try:
            self.drop_area.configure(takefocus=True)
            self.drop_area.bind("<Button-1>", lambda e: self.drop_area.focus_set())
        except Exception:
            pass
        # 全局绑定 Ctrl+V：无论焦点在哪都能粘贴
        self.bind_all("<Control-v>", self.on_paste)
        self.bind_all("<Control-V>", self.on_paste)
        # ============ 中间：按钮区域 ============
        mid_frame = tk.Frame(self)
        mid_frame.pack(fill=tk.X, padx=10, pady=5)
        tk.Label(mid_frame, text=self.strings["middle_or"]).pack(side=tk.LEFT, padx=5)
        btn_choose = tk.Button(mid_frame, text=self.strings["btn_select_images"], command=self.choose_files)
        btn_choose.pack(side=tk.LEFT, padx=5)
        btn_clear = tk.Button(mid_frame, text=self.strings["btn_clear_list"], command=self.clear_list)
        btn_clear.pack(side=tk.LEFT, padx=5)
        btn_export = tk.Button(mid_frame, text=self.strings["btn_export_table"], command=self.export_table)
        btn_export.pack(side=tk.RIGHT, padx=5)
        self.btn_export_feedback = tk.Button(
            mid_frame,
            text=self.strings["btn_export_feedback"],
            command=self.export_feedback,
            state="disabled",
        )
        self.btn_export_feedback.pack(side=tk.RIGHT, padx=5)
        btn_clear_table = tk.Button(mid_frame, text=self.strings["btn_clear_table"], command=self.clear_table)
        btn_clear_table.pack(side=tk.RIGHT, padx=5)
        # ============ 已选择文件列表 ============
        list_frame = tk.LabelFrame(self, text=self.strings["selected_images_title"])
        list_frame.pack(fill=tk.BOTH, expand=False, padx=10, pady=5)
        self.listbox = tk.Listbox(list_frame, height=6)
        self.listbox.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        # ============ 识别结果表 ============
        table_frame = tk.LabelFrame(self, text=self.strings["results_table_title"])
        table_frame.pack(fill=tk.BOTH, expand=False, padx=10, pady=5)
        columns = ("label_id", "model", "sn", "model_src", "sn_src")
        self.table = ttk.Treeview(table_frame, columns=columns, show="headings", height=6)
        self.table.heading("label_id", text=self.strings["col_label_id"])
        self.table.heading("model", text=self.strings["col_model"])
        self.table.heading("sn", text=self.strings["col_sn"])
        self.table.heading("model_src", text=self.strings["col_model_src"])
        self.table.heading("sn_src", text=self.strings["col_sn_src"])
        self.table.column("label_id", width=140, anchor="w")
        self.table.column("model", width=120, anchor="w")
        self.table.column("sn", width=200, anchor="w")
        self.table.column("model_src", width=80, anchor="w")
        self.table.column("sn_src", width=80, anchor="w")
        table_scroll = tk.Scrollbar(table_frame, orient=tk.VERTICAL, command=self.table.yview)
        self.table.configure(yscrollcommand=table_scroll.set)
        self.table.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        table_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        # ============ 开始按钮 ============
        btn_frame = tk.Frame(self)
        btn_frame.pack(fill=tk.X, padx=10, pady=5)
        self.btn_start = tk.Button(
            btn_frame,
            text=self.strings["btn_start"],
            command=self.start_run,
            height=2
        )
        self.btn_start.pack(side=tk.LEFT, padx=5)
        # ============ 日志输出 ============
        log_frame = tk.LabelFrame(self, text=self.strings["log_title"])
        log_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        self.log = scrolledtext.ScrolledText(log_frame, state="disabled", font=("Consolas", 9))
        self.log.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        # 启动时给个提示
        if not self.dnd_enabled:
            if not DND_IMPORT_AVAILABLE:
                self.write_log(self.strings["log_tkdnd_missing"])
            else:
                reason = _mask_path_text(self.dnd_error) if self.dnd_error else self.strings["unknown_reason"]
                self.write_log(self.strings["log_tkdnd_init_failed"].format(reason=reason))
        else:
            self.write_log(self.strings["log_drop_hint"])
        if CLIP_AVAILABLE:
            self.write_log(self.strings["log_clip_enabled"])
        else:
            self.write_log(self.strings["log_clip_pillow_missing"])
        self._ocr_prewarm_thread = start_ocr_prewarm_thread(log=self.write_log)
        self.after(self._log_poll_interval_ms, self._poll_log_queue)
    # ========== 工具函数 ==========
    def _append_log_lines(self, lines):
        if not lines:
            return
        self.log.configure(state="normal")
        for line in lines:
            self.log.insert(tk.END, line + "\n")
        self.log.see(tk.END)
        self.log.configure(state="disabled")

    def _poll_log_queue(self):
        lines = []
        drained_to_batch_limit = False
        if hasattr(self, "_log_queue"):
            max_batch = max(1, int(getattr(self, "_log_max_batch_lines", 24)))
            while len(lines) < max_batch:
                try:
                    lines.append(self._log_queue.get_nowait())
                except queue.Empty:
                    break
            drained_to_batch_limit = len(lines) >= max_batch
        if lines:
            App._update_stage_hint(self, lines)
            self._append_log_lines(lines)
            self._last_log_monotonic = time.monotonic()
        try:
            now = time.monotonic()
            if (
                getattr(self, "_is_running", False)
                and not lines
                and (now - getattr(self, "_last_log_monotonic", 0.0)) >= getattr(self, "_log_idle_heartbeat_sec", 0.8)
                and (now - getattr(self, "_last_heartbeat_monotonic", 0.0)) >= getattr(self, "_log_idle_heartbeat_sec", 0.8)
            ):
                heartbeat = App._build_idle_heartbeat(self)
                if heartbeat:
                    self._append_log_lines([heartbeat])
                    self._last_heartbeat_monotonic = now
            backlog = drained_to_batch_limit
            if hasattr(self, "_log_queue"):
                qsize = getattr(self._log_queue, "qsize", None)
                if callable(qsize):
                    try:
                        backlog = backlog or qsize() > 0
                    except Exception:
                        pass
            next_delay = 1 if backlog else self._log_poll_interval_ms
            self.after(next_delay, self._poll_log_queue)
        except Exception:
            pass

    def _flush_log_buffer(self):
        self._poll_log_queue()

    def _update_stage_hint(self, lines):
        for line in lines:
            text = str(line or "")
            if text.startswith("[1/4]") or text.startswith("[2/4]") or text.startswith("[3/4]") or text.startswith("[4/4]"):
                self._last_stage_hint = text
            elif text.startswith("[条码开始]") or text.startswith("[条码完成]") or text.startswith("[OCR开始]") or text.startswith("[OCR完成]"):
                self._last_stage_hint = text

    def _build_idle_heartbeat(self):
        return None

    def write_log(self, text: str):
        """线程安全地往日志窗口里写一行"""
        text = _mask_path_text(text)
        if threading.get_ident() == self._main_thread_ident:
            App._update_stage_hint(self, [text])
            self._append_log_lines([text])
            self._last_log_monotonic = time.monotonic()
            return
        if hasattr(self, "_log_queue"):
            self._log_queue.put(text)
            return
        self._append_log_lines([text])
    def add_files(self, paths):
        """把选中的文件加入列表（过滤非图片、去重）"""
        added = 0
        for p in paths:
            if not os.path.isfile(p):
                continue
            ext = os.path.splitext(p)[1].lower()
            if ext not in SUPPORTED_EXTS:
                continue
            if p in self.image_paths:
                continue
            self.image_paths.append(p)
            self.listbox.insert(tk.END, p)
            added += 1
        if added > 0:
            self.write_log(self.strings["log_added_images"].format(added=added, total=len(self.image_paths)))
    # ========== 事件回调 ==========
    def on_drop(self, event):
        """拖拽文件到 label 上的回调"""
        # event.data 是一个类似 '{C:/a.jpg} {C:/b.png}' 的字符串，用 tk 的 splitlist 解析
        raw = event.data
        files = self.tk.splitlist(raw)
        self.add_files(files)
    def on_paste(self, event=None):
        """
        Ctrl+V 粘贴：
        - 如果剪贴板是文件列表：直接加入
        - 如果剪贴板是图片：保存为 PNG 到独立缓存目录，再加入
        """
        # 粘贴图片先放到独立源目录，避免运行时清理输入目录删掉自己。
        input_dir = os.path.abspath("pasted_images")
        os.makedirs(input_dir, exist_ok=True)
        # 方案 A：优先用 Pillow 直接抓剪贴板（Windows 下最稳）
        if CLIP_AVAILABLE:
            obj = ImageGrab.grabclipboard()  # image / list-of-filenames / None
            # 1) 复制的是文件（Explorer Ctrl+C 文件）
            if isinstance(obj, list) and obj:
                self.add_files(obj)
                return "break"
            # 2) 复制的是图片（截图/网页复制图片）
            if obj is not None:
                # 生成唯一文件名
                fname = f"pasted_{int(time.time() * 1000)}.png"
                save_path = os.path.join(input_dir, fname)
                try:
                    obj.save(save_path, "PNG")
                    self.add_files([save_path])
                    self.write_log(self.strings["log_pasted_image"].format(path=save_path))
                except Exception as e:
                    messagebox.showerror(self.strings["paste_failed_title"], self.strings["paste_save_failed"].format(error=e))
                return "break"
        # 方案 B：没有 Pillow 时，尝试从 Tk 取文本（一般只对“路径文本”有用）
        try:
            data = self.clipboard_get()
        except tk.TclError:
            data = None
        if data:
            # 兼容 '{C:/a.jpg} {C:/b.png}' 这种格式
            files = self.tk.splitlist(data)
            self.add_files(files)
        else:
            messagebox.showinfo(self.strings["clipboard_empty_title"], self.strings["clipboard_empty_message"])
        return "break"
    def choose_files(self):
        """点击“从电脑选择图片…”的回调"""
        files = filedialog.askopenfilenames(
            title=self.strings["choose_images_title"],
            filetypes=[
                (self.strings["filetype_images"], "*.jpg;*.jpeg;*.png;*.bmp;*.webp"),
                (self.strings["filetype_all"], "*.*"),
            ]
        )
        if not files:
            return
        self.add_files(files)
    def clear_list(self):
        """清空已选择图片"""
        self.image_paths.clear()
        self.listbox.delete(0, tk.END)
        self.write_log(self.strings["log_cleared_image_list"])
    # ========== 主流程 ==========
    def clear_table(self):
        """清空识别结果表"""
        self.table.delete(*self.table.get_children())
        self.result_rows = []
        self.write_log(self.strings["log_cleared_table"])
    def _format_issue_summary(self) -> str:
        """汇总未识别项，便于一眼查看问题样本。"""
        missing_sn = []
        missing_model = []
        missing_both = []
        def _norm(v):
            s = "" if v is None else str(v)
            s = s.strip()
            if s.lower() in {"none", "missing", "na", "n/a", "null", "-"}:
                return ""
            return s
        for item in self.table.get_children():
            values = self.table.item(item, "values")
            if not values:
                continue
            label_id = _norm(values[0] if len(values) > 0 else "")
            model = _norm(values[1] if len(values) > 1 else "")
            sn = _norm(values[2] if len(values) > 2 else "")
            if not label_id:
                continue
            if not model and not sn:
                missing_both.append(label_id)
            elif not sn:
                missing_sn.append(label_id)
            elif not model:
                missing_model.append(label_id)
        lines = []
        if missing_sn:
            lines.append(self.strings["issue_missing_sn_prefix"] + self.strings["issue_joiner"].join(missing_sn))
        if missing_model:
            lines.append(self.strings["issue_missing_model_prefix"] + self.strings["issue_joiner"].join(missing_model))
        if missing_both:
            lines.append(self.strings["issue_missing_both_prefix"] + self.strings["issue_joiner"].join(missing_both))
        if not lines:
            lines.append(self.strings["issue_none"])
        return "\n".join(lines)
    def export_table(self):
        """导出结果为 XLSX"""
        if not self.table.get_children():
            messagebox.showinfo(self.strings["no_data_title"], self.strings["no_data_message"])
            return
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        default_name = f"{self.strings['export_default_name_prefix']}_{timestamp}.xlsx"
        path = filedialog.asksaveasfilename(
            title=self.strings["export_title"],
            defaultextension=".xlsx",
            initialfile=default_name,
            filetypes=[(self.strings["filetype_excel"], "*.xlsx"), (self.strings["filetype_all"], "*.*")]
        )
        if not path:
            return
        headers = (self.strings["col_label_id"], self.strings["col_model"], self.strings["col_sn"], self.strings["col_model_src"], self.strings["col_sn_src"])
        try:
            wb = openpyxl.Workbook()
            ws = wb.active
            ws.title = self.strings["excel_sheet_title"]
            ws.append(list(headers))
            if self.result_rows:
                for row in self.result_rows:
                    ws.append([
                        row.get("label_id", ""),
                        row.get("model", ""),
                        row.get("sn", ""),
                        _display_model_src(row.get("model_src", ""), self.strings),
                        _display_sn_src(row.get("sn_src", ""), self.strings),
                    ])
            else:
                for item in self.table.get_children():
                    ws.append(list(self.table.item(item, "values")))
            wb.save(path)
            self.write_log(self.strings["log_results_file"].format(path=path))
            self.write_log(self._format_issue_summary())
        except Exception as e:
            messagebox.showerror(self.strings["export_failed_title"], self.strings["export_save_failed"].format(error=e))
    def export_feedback(self):
        """导出本次运行的失败证据 ZIP"""
        if not self._last_run_dir:
            messagebox.showinfo(self.strings["no_data_title"], self.strings["feedback_no_run"])
            return
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        default_name = f"{self.strings['feedback_default_name_prefix']}_{timestamp}.zip"
        path = filedialog.asksaveasfilename(
            title=self.strings["feedback_title"],
            defaultextension=".zip",
            initialfile=default_name,
            filetypes=[(self.strings["filetype_zip"], "*.zip"), (self.strings["filetype_all"], "*.*")]
        )
        if not path:
            return
        try:
            stats = build_feedback_package(self._last_run_dir, path)
            miss_total = sum(stats.get("misses", {}).values())
            messagebox.showinfo(
                self.strings["feedback_done_title"],
                self.strings["feedback_done"].format(
                    files=stats.get("files", 0),
                    misses=miss_total,
                    skipped=stats.get("skipped", 0),
                    path=stats.get("zip_path", path),
                ),
            )
            self.write_log(self.strings["log_feedback_file"].format(path=stats.get("zip_path", path)))
        except Exception as e:
            messagebox.showerror(self.strings["feedback_failed_title"], self.strings["feedback_failed"].format(error=e))
    def start_run(self):

        """点击“开始识别”按钮"""
        if self._is_running:
            messagebox.showwarning(self.strings["warn_running_title"], self.strings["warn_running_message"])
            return
        if not self.image_paths:
            messagebox.showwarning(self.strings["warn_no_images_title"], self.strings["warn_no_images_message"])
            return
        # 禁用按钮，防止重复点击
        self._is_running = True
        self._last_log_monotonic = time.monotonic()
        self._last_heartbeat_monotonic = 0.0
        self._last_stage_hint = ""
        self._last_run_dir = ""
        self.btn_export_feedback.config(state="disabled")
        self.btn_start.config(state="disabled")
        self.write_log(self.strings["log_start_pipeline"])
        t = threading.Thread(target=self.run_pipeline, daemon=True)
        t.start()
    def run_pipeline(self):
        """在后台线程里跑完整 pipeline：拷贝图片 → crop_labels.main → scan2.main"""
        try:
            try:
                crop_module, scan2_module = load_pipeline_modules()
            except Exception as exc:
                msg = _mask_path_text(str(exc))
                self.write_log(self.strings["log_dependency_load_failed"].format(msg=msg))
                self.after(0, lambda msg=msg: messagebox.showerror(self.strings["dependency_load_failed_title"], msg))
                return

            self._crop_module = crop_module
            self._scan2_module = scan2_module
            input_root = os.path.abspath(getattr(crop_module, "DEFAULT_INPUT_DIR", "new_images"))
            run_dir, copied_records = copy_images_to_unique_run_dir(self.image_paths, input_root)
            input_dir = run_dir
            out_dir = run_dir

            self.write_log(self.strings["stage_prepare_input"])
            self.write_log(self.strings["log_copied_images"].format(count=len(copied_records)))

            old_crop_sink = getattr(crop_module, "LOG_SINK", None)
            old_scan_sink = getattr(scan2_module, "LOG_SINK", None)
            crop_module.set_log_sink(self.write_log)
            scan2_module.set_log_sink(self.write_log)
            try:
                self.write_log(self.strings["stage_crop"])
                crop_stats = crop_module.main(input_dir=input_dir, out_dir=out_dir)
                if not isinstance(crop_stats, dict) or crop_stats.get("label_count", 0) <= 0:
                    raise RuntimeError(self.strings["error_no_label_crops"])
                if crop_stats.get("manifest_rows", 0) <= 0:
                    raise RuntimeError(self.strings["error_no_manifest_rows"])
                self.write_log(self.strings["stage_recognize"])
                result_jsonl = os.path.join(crop_module.STAGE2_DIR, "model_sn_ocr.jsonl")
                debug_log = os.path.join(crop_module.STAGE2_DIR, "debug_ocr_barcode.log")
                scan2_module.main(
                    model_dir=crop_module.OUT_MODEL_DIR,
                    sn_dir=crop_module.OUT_SN_DIR,
                    out_jsonl=result_jsonl,
                    debug_log=debug_log,
                )
                self.write_log(self.strings["stage_refreshing"])
                self.after(0, self.load_results_into_table)
            finally:
                crop_module.set_log_sink(old_crop_sink)
                scan2_module.set_log_sink(old_scan_sink)

            self.write_log(self.strings["stage_complete"])
            self.write_log(self.strings["log_results_file"].format(path=os.path.abspath(scan2_module.OUT_JSONL)))
            self.write_log(self.strings["log_output_folders"].format(stage1=crop_module.STAGE1_DIR, model=crop_module.OUT_MODEL_DIR, sn=crop_module.OUT_SN_DIR))
            self._last_run_dir = run_dir
            self.after(0, lambda: self.write_log(self._format_issue_summary()))
        except Exception as e:
            detail = _mask_path_text(traceback.format_exc())
            self.write_log(self.strings["error_prefix"].format(error=e))
            self.write_log(detail)
            self.after(0, lambda msg=_mask_path_text(str(e)): messagebox.showerror(self.strings["recognition_failed_title"], msg))
        finally:
            def _finish():
                self._is_running = False
                self.btn_start.config(state="normal")
                if self._last_run_dir:
                    self.btn_export_feedback.config(state="normal")
            self.after(0, _finish)
    def load_results_into_table(self):
        """读取 scan2 输出 JSONL 并追加到表格"""
        scan2_module = self._scan2_module
        if scan2_module is None:
            self.write_log(self.strings["warn_scan2_not_loaded"])
            return
        out_path = os.path.abspath(scan2_module.OUT_JSONL)
        if not os.path.isfile(out_path):
            self.write_log(self.strings["warn_output_not_found"].format(path=out_path))
            return
        rows = []
        try:
            with open(out_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            self.write_log(self.strings["warn_read_results_failed"].format(error=e))
            return
        self.result_rows = rows
        def _append():
            self.table.delete(*self.table.get_children())
            for r in rows:
                values = (
                    r.get("label_id", ""),
                    r.get("model", ""),
                    r.get("sn", ""),
                    _display_model_src(r.get("model_src", ""), self.strings),
                    _display_sn_src(r.get("sn_src", ""), self.strings),
                )
                self.table.insert("", tk.END, values=values)
        if threading.get_ident() == self._main_thread_ident:
            _append()
        else:
            self.after(0, _append)
def main(lang: str = "zh"):
    # Windows 控制台中文支持（可选）
    try:
        if sys.platform.startswith("win"):
            sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    _self_check()
    app = App(strings=get_strings(lang))
    app.mainloop()


if __name__ == "__main__":
    main()
