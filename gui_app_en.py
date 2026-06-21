import os
import sys
import threading
import shutil
import time
import datetime
import json
import csv
import re
import warnings
import openpyxl
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk
import platform
import traceback
import ctypes
import subprocess
from win_subprocess import hide_subprocess_windows

hide_subprocess_windows()

from app_paths import get_resource_path, get_barcode_cli_path
from gui_pipeline import copy_images_to_unique_run_dir, load_pipeline_modules, start_ocr_prewarm_thread

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
    DND_AVAILABLE = True
except ImportError:
    DND_AVAILABLE = False
    DND_FILES = None
    TkinterDnD = None

# 支持的图片后缀
SUPPORTED_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def _mask_path_text(text: str) -> str:
    text = "" if text is None else str(text)
    text = re.sub(r"(?i)[a-z]:[\\/][^\r\n\t|,;]+", "[path]", text)
    return re.sub(r"(?<!\w)/(?:[^/\s]+/)+[^\r\n\t|,;]+", "[path]", text)


def _display_sn_src(value: str) -> str:
    src = str(value or "")
    if src == "barcode":
        return "barcode SN"
    if src.startswith("ocr"):
        return "OCR fallback"
    if src == "barcode_ambiguous":
        return "barcode ambiguous"
    if src == "barcode_parse_fail":
        return "barcode parse fail"
    if src == "barcode_quality_reject":
        return "barcode quality reject"
    if src == "barcode_decoder_miss":
        return "barcode miss"
    return src


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
BaseTk = TkinterDnD.Tk if DND_AVAILABLE else tk.Tk


class App(BaseTk):
    def __init__(self):
        super().__init__()

        self.title("Label Recognition Demo (Drag or Select Images)")
        self.geometry("900x700")

        # 已选择的图片路径列表
        self.image_paths = []
        self._crop_module = None
        self._scan2_module = None
        self._is_running = False
        self.result_rows = []

        # ============ 顶部：拖拽区域 ============
        top_frame = tk.Frame(self)
        top_frame.pack(fill=tk.X, padx=10, pady=10)

        if DND_AVAILABLE:
            self.drop_area = tk.Label(
                top_frame,
                text="Drag image files here, or press Ctrl+V to paste",
                relief="ridge",
                borderwidth=2,
                width=60,
                height=4,
                fg="#555555"
            )
        else:
            self.drop_area = tk.Label(
                top_frame,
                text="tkinterdnd2 not installed. Drag-and-drop disabled (pip install tkinterdnd2)",
                relief="ridge",
                borderwidth=2,
                width=60,
                height=4,
                fg="#aa0000"
            )

        self.drop_area.pack(fill=tk.X, expand=True)

        if DND_AVAILABLE:
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

        tk.Label(mid_frame, text="or").pack(side=tk.LEFT, padx=5)

        btn_choose = tk.Button(mid_frame, text="Select Images...", command=self.choose_files)
        btn_choose.pack(side=tk.LEFT, padx=5)

        btn_clear = tk.Button(mid_frame, text="Clear List", command=self.clear_list)
        btn_clear.pack(side=tk.LEFT, padx=5)

        btn_export = tk.Button(mid_frame, text="Export Table", command=self.export_table)
        btn_export.pack(side=tk.RIGHT, padx=5)

        btn_clear_table = tk.Button(mid_frame, text="Clear Table", command=self.clear_table)
        btn_clear_table.pack(side=tk.RIGHT, padx=5)

        # ============ 已选择文件列表 ============
        list_frame = tk.LabelFrame(self, text="Selected Images")
        list_frame.pack(fill=tk.BOTH, expand=False, padx=10, pady=5)

        self.listbox = tk.Listbox(list_frame, height=6)
        self.listbox.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # ============ 识别结果表 ============
        table_frame = tk.LabelFrame(self, text="Results Table")
        table_frame.pack(fill=tk.BOTH, expand=False, padx=10, pady=5)

        columns = ("label_id", "model", "sn", "model_src", "sn_src")
        self.table = ttk.Treeview(table_frame, columns=columns, show="headings", height=6)
        self.table.heading("label_id", text="label_id")
        self.table.heading("model", text="model")
        self.table.heading("sn", text="sn")
        self.table.heading("model_src", text="model_src")
        self.table.heading("sn_src", text="sn_src")
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
            text="Run Recognition (Crop + Barcode + OCR)",
            command=self.start_run,
            height=2
        )
        self.btn_start.pack(side=tk.LEFT, padx=5)

        # ============ 日志输出 ============
        log_frame = tk.LabelFrame(self, text="Log")
        log_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        self.log = scrolledtext.ScrolledText(log_frame, state="disabled", font=("Consolas", 9))
        self.log.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # 启动时给个提示
        if not DND_AVAILABLE:
            self.write_log("Note: tkinterdnd2 is not installed. Drag-and-drop is disabled. To enable it, run: pip install tkinterdnd2")
        else:
            self.write_log("Drag images into the gray area above, or click \"Select Images...\".")
        
        if CLIP_AVAILABLE:
            self.write_log("Ctrl+V paste enabled: supports screenshots, copied web images, and copied files.")
        else:
            self.write_log("Note: pillow is not installed. Ctrl+V can only paste file paths; to paste images run: pip install pillow")
        self._ocr_prewarm_thread = start_ocr_prewarm_thread(log=self.write_log)

    # ========== 工具函数 ==========

    def write_log(self, text: str):
        """线程安全地往日志窗口里写一行"""
        text = _mask_path_text(text)
        def _append():
            self.log.configure(state="normal")
            self.log.insert(tk.END, text + "\n")
            self.log.see(tk.END)
            self.log.configure(state="disabled")

        self.after(0, _append)

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
            self.write_log(f"Added {added} image(s). Total: {len(self.image_paths)}.")

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
        # Keep pasted images outside the pipeline input folder so cleanup cannot delete the source.
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
                    self.write_log(f"Pasted image from clipboard -> {save_path}")
                except Exception as e:
                    messagebox.showerror("Paste Failed", f"Failed to save image: {e}")
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
            messagebox.showinfo("Clipboard Empty/Unsupported", "No usable image or file found in the clipboard.\n\nTip: to paste images, install pillow: pip install pillow")

        return "break"

    def choose_files(self):
        """Handle "Select Images..."."""
        files = filedialog.askopenfilenames(
            title="Select Images",
            filetypes=[
                ("Image files", "*.jpg;*.jpeg;*.png;*.bmp;*.webp"),
                ("All files", "*.*"),
            ]
        )
        if not files:
            return
        self.add_files(files)

    def clear_list(self):
        """Clear selected images."""
        self.image_paths.clear()
        self.listbox.delete(0, tk.END)
        self.write_log("Cleared image list.")

    # ========== 主流程 ==========
    def clear_table(self):
        """Clear results table."""
        self.table.delete(*self.table.get_children())
        self.result_rows = []
        self.write_log("Cleared table.")

    def export_table(self):
        """Export table to XLSX."""
        if not self.table.get_children():
            messagebox.showinfo("No Data", "There are no rows to export.")
            return

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        default_name = f"ocr_results_{timestamp}.xlsx"
        path = filedialog.asksaveasfilename(
            title="Save Results",
            defaultextension=".xlsx",
            initialfile=default_name,
            filetypes=[("Excel", "*.xlsx"), ("All files", "*.*")]
        )
        if not path:
            return

        headers = ("label_id", "model", "sn", "model_src", "sn_src")
        try:
            wb = openpyxl.Workbook()
            ws = wb.active
            ws.title = "results"
            ws.append(list(headers))
            if self.result_rows:
                for row in self.result_rows:
                    ws.append([row.get(name, "") for name in headers])
            else:
                for item in self.table.get_children():
                    ws.append(list(self.table.item(item, "values")))
            wb.save(path)
            self.write_log(f"Exported results to {path}")
        except Exception as e:
            messagebox.showerror("Export Failed", f"Failed to export: {e}")


    def start_run(self):
        """Run recognition."""
        if self._is_running:
            messagebox.showwarning("Running", "A recognition task is already running.")
            return
        if not self.image_paths:
            messagebox.showwarning("No Images", "Please add at least one image.")
            return

        # 禁用按钮，防止重复点击
        self._is_running = True
        self.btn_start.config(state="disabled")
        self.write_log("Running full pipeline...")

        t = threading.Thread(target=self.run_pipeline, daemon=True)
        t.start()

    def run_pipeline(self):
        """Run pipeline in background: copy images -> crop.main -> scan2.main."""
        try:
            try:
                crop_module, scan2_module = load_pipeline_modules()
            except Exception as exc:
                msg = _mask_path_text(str(exc))
                self.write_log(f"ERROR: Failed to load OCR dependencies: {msg}")
                self.after(0, lambda msg=msg: messagebox.showerror("Dependency Load Failed", msg))
                return

            self._crop_module = crop_module
            self._scan2_module = scan2_module
            input_root = os.path.abspath(getattr(crop_module, "DEFAULT_INPUT_DIR", "new_images"))
            run_dir, copied_records = copy_images_to_unique_run_dir(self.image_paths, input_root)
            input_dir = run_dir
            out_dir = run_dir

            self.write_log("[1/4] Preparing input directory")
            self.write_log(f"Copied {len(copied_records)} image(s).")

            old_crop_sink = getattr(crop_module, "LOG_SINK", None)
            old_scan_sink = getattr(scan2_module, "LOG_SINK", None)
            crop_module.set_log_sink(self.write_log)
            scan2_module.set_log_sink(self.write_log)
            try:
                self.write_log("[2/4] Running crop.main(): full image -> label -> model/sn crops")
                crop_stats = crop_module.main(input_dir=input_dir, out_dir=out_dir)
                if not isinstance(crop_stats, dict) or crop_stats.get("label_count", 0) <= 0:
                    raise RuntimeError("No label crops were generated; OCR was stopped.")
                if crop_stats.get("manifest_rows", 0) <= 0:
                    raise RuntimeError("No manifest rows were generated; OCR was stopped.")
                self.write_log("[3/4] Running scan2.main(): read MODEL / SN")
                result_jsonl = os.path.join(crop_module.STAGE2_DIR, "model_sn_ocr.jsonl")
                debug_log = os.path.join(crop_module.STAGE2_DIR, "debug_ocr_barcode.log")
                scan2_module.main(
                    model_dir=crop_module.OUT_MODEL_DIR,
                    sn_dir=crop_module.OUT_SN_DIR,
                    out_jsonl=result_jsonl,
                    debug_log=debug_log,
                )
                self.after(0, self.load_results_into_table)
            finally:
                crop_module.set_log_sink(old_crop_sink)
                scan2_module.set_log_sink(old_scan_sink)

            self.write_log("[4/4] Pipeline complete.")
            self.write_log(f"Results file: {os.path.abspath(scan2_module.OUT_JSONL)}")
            self.write_log(f"Output folders: {crop_module.STAGE1_DIR}/, {crop_module.OUT_MODEL_DIR}/, {crop_module.OUT_SN_DIR}/")

        except Exception as e:
            detail = _mask_path_text(traceback.format_exc())
            self.write_log(f"ERROR: {e}")
            self.write_log(detail)
            self.after(0, lambda msg=_mask_path_text(str(e)): messagebox.showerror("Recognition Failed", msg))

        finally:
            def _finish():
                self._is_running = False
                self.btn_start.config(state="normal")
            self.after(0, _finish)


    def load_results_into_table(self):
        """Read scan2 JSONL output and append to table."""
        scan2_module = self._scan2_module
        if scan2_module is None:
            self.write_log("WARN: scan2 is not loaded yet.")
            return
        out_path = os.path.abspath(scan2_module.OUT_JSONL)
        if not os.path.isfile(out_path):
            self.write_log(f"WARN: Output file not found: {out_path}")
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
            self.write_log(f"WARN: Failed to read results: {e}")
            return

        self.result_rows = rows
        def _append():
            self.table.delete(*self.table.get_children())
            for r in rows:
                values = (
                    r.get("label_id", ""),
                    r.get("model", ""),
                    r.get("sn", ""),
                    r.get("model_src", ""),
                    _display_sn_src(r.get("sn_src", "")),
                )
                self.table.insert("", tk.END, values=values)

        self.after(0, _append)


if __name__ == "__main__":
    # Windows console UTF-8 support (optional)
    try:
        if sys.platform.startswith("win"):
            sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    _self_check()
    app = App()
    app.mainloop()
