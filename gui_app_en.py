import os
import sys
import threading
import shutil
import time
import datetime
import json
import csv
import warnings
import openpyxl
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk
import platform
import traceback
import ctypes
import subprocess

# 你的两个脚本模块名，保持和文件名一致
from app_paths import ensure_paddle_libs_on_path
from app_paths import get_resource_path, get_barcode_cli_path

ensure_paddle_libs_on_path()

import crop
import scan2

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
                creationflags = subprocess.CREATE_NO_WINDOW
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
            f.write("\n".join(lines) + "\n")
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

    def _mask_sn(self, sn: str) -> str:
        s = str(sn or "")
        if len(s) <= 4:
            return s
        if len(s) <= 8:
            return s[:4] + ("*" * (len(s) - 4))
        return s[:4] + ("*" * (len(s) - 8)) + s[-4:]

    def _mask_model(self, model: str) -> str:
        s = str(model or "")
        if len(s) <= 2:
            return s
        if len(s) <= 3:
            return s[:2] + ("*" * (len(s) - 2))
        return s[:2] + ("*" * (len(s) - 3)) + s[-1:]

    # ========== 工具函数 ==========

    def write_log(self, text: str):
        """线程安全地往日志窗口里写一行"""
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
        self.write_log(f"[DND] raw={raw!r}")
        files = self.tk.splitlist(raw)
        self.add_files(files)

    def on_paste(self, event=None):
        """
        Ctrl+V 粘贴：
        - 如果剪贴板是文件列表：直接加入
        - 如果剪贴板是图片：保存为 PNG 到 crop.INPUT_DIR，再加入
        """
        # 目标保存目录：和你 pipeline 输入目录一致
        input_dir = os.path.abspath(crop.INPUT_DIR)
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
            for item in self.table.get_children():
                ws.append(list(self.table.item(item, "values")))
            wb.save(path)
            self.write_log(f"Exported results to {path}")
        except Exception as e:
            messagebox.showerror("Export Failed", f"Failed to export: {e}")


    def start_run(self):
        """Run recognition."""
        if not self.image_paths:
            messagebox.showwarning("No Images", "Please add at least one image.")
            return

        # 禁用按钮，防止重复点击
        self.btn_start.config(state="disabled")
        self.write_log("Running full pipeline...")

        t = threading.Thread(target=self.run_pipeline, daemon=True)
        t.start()

    def run_pipeline(self):
        """Run pipeline in background: copy images -> crop.main -> scan2.main."""

        # 把 crop_labels 的 INPUT_DIR 当作输入目录（你脚本里默认是 "test_images"）
        input_dir = os.path.abspath(crop.INPUT_DIR)

        try:
            # 1) 准备 INPUT_DIR：清空旧图片，拷贝新图片进去
            self.write_log(f"[1/4] Preparing input directory: {input_dir}")

            os.makedirs(input_dir, exist_ok=True)

            # 清理旧的输入图片（只删图片类型）
            for name in os.listdir(input_dir):
                full = os.path.join(input_dir, name)
                if os.path.isfile(full) and os.path.splitext(full)[1].lower() in SUPPORTED_EXTS:
                    try:
                        os.remove(full)
                    except Exception as e:
                        self.write_log(f"WARN: Failed to delete old file {full}: {e}")

            # 拷贝新图片
            for p in self.image_paths:
                dst = os.path.join(input_dir, os.path.basename(p))
                shutil.copy2(p, dst)
            self.write_log(f"Copied {len(self.image_paths)} image(s) to {input_dir}")

            # 2) 把 print 重定向到 GUI 日志
            import builtins
            old_print = builtins.print

            def gui_print(*args, **kwargs):
                s = " ".join(str(a) for a in args)
                self.write_log(s)

            builtins.print = gui_print

            try:
                # 3) 运行 crop_labels（Stage1 + Stage2 裁剪）
                self.write_log("[2/4] Running crop.main(): full image -> label -> model/sn crops")
                crop.main()

                # 4) 运行 scan2（条码 + OCR 提取 MODEL / SN）
                self.write_log("[3/4] Running scan2.main(): read MODEL / SN")
                scan2.main()
                self.load_results_into_table()

            finally:
                # 恢复原来的 print
                builtins.print = old_print

            # 结束提示
            self.write_log("[4/4] Pipeline complete.")
            self.write_log(f"Results file: {os.path.abspath(scan2.OUT_JSONL)}")
            self.write_log("Output folders: stage1_labels/, stage2_fields/model/, stage2_fields/sn/")

        except Exception as e:
            self.write_log(f"ERROR: {e}")

        finally:
            # 重新启用按钮
            self.btn_start.config(state="normal")


    def load_results_into_table(self):
        """Read scan2 JSONL output and append to table."""
        out_path = os.path.abspath(scan2.OUT_JSONL)
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

        def _append():
            for r in rows:
                values = (
                    r.get("label_id", ""),
                    self._mask_model(r.get("model", "")),
                    self._mask_sn(r.get("sn", "")),
                    r.get("model_src", ""),
                    r.get("sn_src", ""),
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
