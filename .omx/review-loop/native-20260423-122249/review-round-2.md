node.exe : OpenAI Codex v0.123.0 (research preview)
At C:\Users\55093\AppData\Roaming\npm\codex.ps1:22 char:14
+ ...    $input | & "node$exe"  "$basedir/node_modules/@openai/codex/bin/co ...
+                 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    + CategoryInfo          : NotSpecified: (OpenAI Codex v0.123.0 (research preview):String) [], Rem 
   oteException
    + FullyQualifiedErrorId : NativeCommandError
 
--------
workdir: D:\newhuawei
model: gpt-5.3-codex
provider: openai
approval: never
sandbox: danger-full-access
reasoning effort: medium
reasoning summaries: none
session id: 019dbbcf-1f8c-71b3-bd20-fac8f9fa0591
--------
user
You are an automated code-review gate in a fix loop.
Review current staged, unstaged, and untracked changes in this repository.

First line MUST be exactly one of:
GATE_DECISION: PASS
GATE_DECISION: FAIL

PASS only when no actionable issues remain.
FAIL when any actionable bug, regression risk, security risk, or missing essential tests remain.
After the first line, list concise findings with file paths and concrete fixes.




hook: SessionStart
hook: SessionStart Completed
hook: UserPromptSubmit
hook: UserPromptSubmit Completed
codex
我会先检查当前仓库的 `staged/unstaged/untracked` 变更，再逐个看 diff 判断是否有可执行的问题和缺失测试，然后给出 `PASS/FAIL` 结论与修复建议。
exec
"C:\\Program Files\\WindowsApps\\Microsoft.PowerShell_7.6.0.0_x64__8wekyb3d8bbwe\\pwsh.exe" -Command '
git status --short --branch' in D:\newhuawei
 succeeded in 666ms:
## master...origin/master
 M gui_app.py
 M gui_pipeline.py
 M ocr.py
 M scan2.py
 M sn_barcode.py
 M tests/test_locked_output_dirs.py
 M tests/test_sn_barcode_scanning.py
?? validation/pure_ocr_6_20260419_223358/
?? validation/pure_ocr_6_server_rec_20260419_224312/

exec
"C:\\Program Files\\WindowsApps\\Microsoft.PowerShell_7.6.0.0_x64__8wekyb3d8bbwe\\pwsh.exe" -Command '
git diff --cached --name-status' in D:\newhuawei
exec
"C:\\Program Files\\WindowsApps\\Microsoft.PowerShell_7.6.0.0_x64__8wekyb3d8bbwe\\pwsh.exe" -Command '
git diff --name-status' in D:\newhuawei
exec
"C:\\Program Files\\WindowsApps\\Microsoft.PowerShell_7.6.0.0_x64__8wekyb3d8bbwe\\pwsh.exe" -Command '
git ls-files --others --exclude-standard' in D:\newhuawei
 succeeded in 703ms:
validation/pure_ocr_6_20260419_223358/input/1.jpg
validation/pure_ocr_6_20260419_223358/input/2.jpg
validation/pure_ocr_6_20260419_223358/input/3.jpg
validation/pure_ocr_6_20260419_223358/input/4.jpg
validation/pure_ocr_6_20260419_223358/input/5.jpg
validation/pure_ocr_6_20260419_223358/input/6.jpg
validation/pure_ocr_6_20260419_223358/pure_ocr_results.csv
validation/pure_ocr_6_20260419_223358/pure_ocr_results_reparsed.csv
validation/pure_ocr_6_20260419_223358/pure_ocr_summary_reparsed.json
validation/pure_ocr_6_server_rec_20260419_224312/pure_ocr_server_rec_results.csv
validation/pure_ocr_6_server_rec_20260419_224312/summary.json

 succeeded in 708ms:
 succeeded in 759ms:
M	gui_app.py
M	gui_pipeline.py
M	ocr.py
M	scan2.py
M	sn_barcode.py
M	tests/test_locked_output_dirs.py
M	tests/test_sn_barcode_scanning.py
warning: in the working copy of 'gui_app.py', LF will be replaced by CRLF the next time Git touches it
warning: in the working copy of 'gui_pipeline.py', LF will be replaced by CRLF the next time Git touch
es it
warning: in the working copy of 'ocr.py', LF will be replaced by CRLF the next time Git touches it
warning: in the working copy of 'scan2.py', LF will be replaced by CRLF the next time Git touches it
warning: in the working copy of 'sn_barcode.py', LF will be replaced by CRLF the next time Git touches
 it
warning: in the working copy of 'tests/test_locked_output_dirs.py', LF will be replaced by CRLF the ne
xt time Git touches it
warning: in the working copy of 'tests/test_sn_barcode_scanning.py', LF will be replaced by CRLF the n
ext time Git touches it

codex
我正在逐文件审查代码 diff 和新增验证产物，接下来会定位具体风险点和缺失测试。
exec
"C:\\Program Files\\WindowsApps\\Microsoft.PowerShell_7.6.0.0_x64__8wekyb3d8bbwe\\pwsh.exe" -Command '
git diff -- gui_app.py' in D:\newhuawei
exec
"C:\\Program Files\\WindowsApps\\Microsoft.PowerShell_7.6.0.0_x64__8wekyb3d8bbwe\\pwsh.exe" -Command '
git diff -- gui_pipeline.py' in D:\newhuawei
exec
"C:\\Program Files\\WindowsApps\\Microsoft.PowerShell_7.6.0.0_x64__8wekyb3d8bbwe\\pwsh.exe" -Command '
git diff -- ocr.py' in D:\newhuawei
exec
"C:\\Program Files\\WindowsApps\\Microsoft.PowerShell_7.6.0.0_x64__8wekyb3d8bbwe\\pwsh.exe" -Command '
git diff -- scan2.py' in D:\newhuawei
exec
"C:\\Program Files\\WindowsApps\\Microsoft.PowerShell_7.6.0.0_x64__8wekyb3d8bbwe\\pwsh.exe" -Command '
git diff -- sn_barcode.py' in D:\newhuawei
exec
"C:\\Program Files\\WindowsApps\\Microsoft.PowerShell_7.6.0.0_x64__8wekyb3d8bbwe\\pwsh.exe" -Command '
git diff -- tests/test_locked_output_dirs.py' in D:\newhuawei
exec
"C:\\Program Files\\WindowsApps\\Microsoft.PowerShell_7.6.0.0_x64__8wekyb3d8bbwe\\pwsh.exe" -Command '
git diff -- tests/test_sn_barcode_scanning.py' in D:\newhuawei
 succeeded in 786ms:
diff --git a/gui_pipeline.py b/gui_pipeline.py
index 598353f..966ccfd 100644
--- a/gui_pipeline.py
+++ b/gui_pipeline.py
@@ -34,6 +34,7 @@ def copy_images_to_unique_run_dir(image_paths, root_dir, run_prefix="gui_run"):
         records.append(
             {
                 "source_index": index,
+                "source_name": os.path.basename(source),
                 "input_name": target_name,
                 "sha256": _sha256_file(target),
             }
@@ -42,7 +43,17 @@ def copy_images_to_unique_run_dir(image_paths, root_dir, run_prefix="gui_run"):
     manifest_path = os.path.join(run_dir, "source_manifest.jsonl")
     with open(manifest_path, "w", encoding="utf-8") as manifest:
         for record in records:
-            manifest.write(json.dumps(record, ensure_ascii=False) + "\n")
+            manifest.write(
+                json.dumps(
+                    {
+                        "source_index": record["source_index"],
+                        "input_name": record["input_name"],
+                        "sha256": record["sha256"],
+                    },
+                    ensure_ascii=False,
+                )
+                + "\n"
+            )
 
     return run_dir, records
 
warning: in the working copy of 'gui_pipeline.py', LF will be replaced by CRLF the next time Git touch
es it

 succeeded in 801ms:
diff --git a/ocr.py b/ocr.py
index 828bc63..7ffb6fe 100644
--- a/ocr.py
+++ b/ocr.py
@@ -25,8 +25,10 @@ MIN_SCORE = 0.5
 
 # PaddleOCR 语言：
 #   - 只识别数字/大写字母为主：建议 lang="en"
-#   - 如果可能有中文：可以改成 lang="ch"
-OCR_LANG = "en"
+#   - 如果可能有中文：设置 HUAWEIOCR_OCR_LANG=ch
+OCR_LANG = os.environ.get("HUAWEIOCR_OCR_LANG", "en").strip().lower()
+if OCR_LANG not in {"en", "ch"}:
+    OCR_LANG = "en"
 
 # 是否使用 GPU（你现在一般是 CPU，就 False）
 USE_GPU = False
@@ -108,6 +110,10 @@ def _first_existing_model_dir(root, names):
 
 
 def _recognition_model_candidates():
+    if OCR_LANG == "ch":
+        if OCR_PROFILE in {"accurate", "server"}:
+            return ["PP-OCRv5_server_rec", "PP-OCRv5_mobile_rec", "en_PP-OCRv5_mobile_rec"]
+        return ["PP-OCRv5_mobile_rec", "PP-OCRv5_server_rec", "en_PP-OCRv5_mobile_rec"]
     if OCR_PROFILE in {"accurate", "server"}:
         return ["PP-OCRv5_server_rec", "en_PP-OCRv5_mobile_rec", "PP-OCRv5_mobile_rec"]
     return ["en_PP-OCRv5_mobile_rec", "PP-OCRv5_mobile_rec", "PP-OCRv5_server_rec"]
warning: in the working copy of 'ocr.py', LF will be replaced by CRLF the next time Git touches it

 succeeded in 836ms:
diff --git a/gui_app.py b/gui_app.py
index cad186e..9e6835e 100644
--- a/gui_app.py
+++ b/gui_app.py
@@ -76,20 +76,133 @@ def _mask_path_text(text: str) -> str:
 def _display_sn_src(value: str) -> str:
     src = str(value or "")
     if src == "barcode":
-        return "barcode SN"
+        return "扫描条形码"
+    if src == "barcode_unmatched":
+        return "扫描条形码(未匹配规则)"
+    if src == "barcode_ocr_agree":
+        return "文字识别+扫描条形码(一致)"
     if src.startswith("ocr"):
-        return "OCR fallback"
+        return "文字识别"
     if src == "barcode_ambiguous":
-        return "barcode ambiguous"
+        return "条码冲突"
     if src == "barcode_parse_fail":
-        return "barcode parse fail"
+        return "条码解析失败"
     if src == "barcode_quality_reject":
-        return "barcode quality reject"
+        return "条码质量拒绝"
     if src == "barcode_decoder_miss":
-        return "barcode miss"
+        return "条码未命中"
+    if src == "missing":
+        return "缺失"
     return src
 
 
+def _display_model_src(value: str) -> str:
+    src = str(value or "")
+    if src.endswith("+sn_hint"):
+        base = src[:-8]
+        if base.startswith("ocr"):
+            return "文字识别(由SN推断修正)"
+        if base == "missing":
+            return "缺失(由SN推断)"
+        if base:
+            return f"{_display_model_src(base)}(由SN推断修正)"
+        return "由SN推断修正"
+    if src == "barcode":
+        return "扫描条形码"
+    if src == "barcode_unmatched":
+        return "扫描条形码(未匹配规则)"
+    if src == "barcode_ocr_agree":
+        return "文字识别+扫描条形码(一致)"
+    if src.startswith("ocr"):
+        return "文字识别"
+    if src == "missing":
+        return "缺失"
+    return src
+
+
+def _rewrite_label_id_for_display(label_id: str, input_name_map: dict[str, str]) -> str:
+    if not label_id:
+        return ""
+    text = str(label_id)
+    if not input_name_map:
+        return text
+    match = re.match(r"^(?P<input_name>.+?)(__label_\d+)$", text)
+    if not match:
+        return text
+    input_name = match.group("input_name")
+    suffix = match.group(2)
+    source_name = input_name_map.get(input_name.lower())
+    if not source_name:
+        return text
+    return f"{source_name}{suffix}"
+
+
+def _build_input_name_map(copied_records: list[dict]) -> dict[str, str]:
+    counts: dict[str, int] = {}
+    for record in copied_records or []:
+        source_name = str(record.get("source_name", "")).strip()
+        if not source_name:
+            continue
+        key = source_name.lower()
+        counts[key] = counts.get(key, 0) + 1
+
+    mapping: dict[str, str] = {}
+    for record in copied_records or []:
+        input_name = str(record.get("input_name", "")).strip()
+        source_name = str(record.get("source_name", "")).strip()
+        if not input_name or not source_name:
+            continue
+        source_key = source_name.lower()
+        display_name = source_name
+        if counts.get(source_key, 0) > 1:
+            source_index = str(record.get("source_index", "")).strip()
+            suffix = source_index if source_index else input_name
+            display_name = f"{source_name}#{suffix}"
+        mapping[input_name.lower()] = display_name
+    return mapping
+
+
+def _localize_log_text(text: str, input_name_map: dict[str, str]) -> str:
+    value = "" if text is None else str(text)
+    if not value:
+        return value
+
+    result_match = re.match(
+        r"^\[(?P<label>.+?)\]\s+MODEL=(?P<model>.*?)\s+\(M_SRC=(?P<model_src>.*?)\)\s+\|\s+SN=(?P<sn>
.*?)\s+\(SN_SRC=(?P<sn_src>.*?)\)\s*$",
+        value,
+    )
+    if result_match:
+        label_id = _rewrite_label_id_for_display(result_match.group("label"), input_name_map)
+        model = result_match.group("model")
+        sn = result_match.group("sn")
+        model_src = _display_model_src(result_match.group("model_src"))
+        sn_src = _display_sn_src(result_match.group("sn_src"))
+        return f"[{label_id}] 型号={model}（来源={model_src}） | SN={sn}（来源={sn_src}）"
+
+    stage1_line = re.match(r"^Stage1:\s+(.+?)\s+->\s+(\d+)\s+label crops$", value)
+    if stage1_line:
+        return f"阶段1：{stage1_line.group(1)} -> {stage1_line.group(2)} 个标签裁剪图"
+
+    stage1_done = re.match(r"^Stage1 complete:\s+(\d+)\s+label crops generated$", value)
+    if stage1_done:
+        return f"阶段1完成：生成 {stage1_done.group(1)} 个标签裁剪图"
+
+    stats_line = re.match(r"^Stats:\s+at least one field\s+(\d+);\s+both fields\s+(\d+)$", value)
+    if stats_line:
+        return f"统计：至少识别出一个字段 {stats_line.group(1)}；两个字段都识别出 {stats_line.group(2)}"
+
+    failed_line = re.match(r"^Failed categories:\s+(.+?)\s*/\s*(.+)$", value)
+    if failed_line:
+        return f"失败分类目录：{failed_line.group(1)} / {failed_line.group(2)}"
+
+    value = value.replace("Manifest:", "清单文件：")
+    value = value.replace("WARN:", "警告：")
+    value = value.replace("No supported images found in input folder:", "输入目录未找到支持的图片：")
+    value = value.replace("No label crops generated from", "未从输入图中生成标签裁剪图：")
+    value = value.replace("input image(s).", "张输入图片。")
+    return value
+
+
 def _self_check():
     if getattr(sys, "frozen", False):
         base_dir = os.path.dirname(sys.executable)
@@ -170,6 +283,7 @@ class App(BaseTk):
         self._scan2_module = None
         self._is_running = False
         self.result_rows = []
+        self._input_name_map = {}
         # ============ 顶部：拖拽区域 ============
         top_frame = tk.Frame(self)
         top_frame.pack(fill=tk.X, padx=10, pady=10)
@@ -229,16 +343,16 @@ class App(BaseTk):
         table_frame.pack(fill=tk.BOTH, expand=False, padx=10, pady=5)
         columns = ("label_id", "model", "sn", "model_src", "sn_src")
         self.table = ttk.Treeview(table_frame, columns=columns, show="headings", height=6)
-        self.table.heading("label_id", text="label_id")
-        self.table.heading("model", text="model")
-        self.table.heading("sn", text="sn")
-        self.table.heading("model_src", text="model_src")
-        self.table.heading("sn_src", text="sn_src")
-        self.table.column("label_id", width=140, anchor="w")
+        self.table.heading("label_id", text="标签ID")
+        self.table.heading("model", text="型号")
+        self.table.heading("sn", text="SN")
+        self.table.heading("model_src", text="型号来源")
+        self.table.heading("sn_src", text="SN来源")
+        self.table.column("label_id", width=220, anchor="w")
         self.table.column("model", width=120, anchor="w")
         self.table.column("sn", width=200, anchor="w")
-        self.table.column("model_src", width=80, anchor="w")
-        self.table.column("sn_src", width=80, anchor="w")
+        self.table.column("model_src", width=120, anchor="w")
+        self.table.column("sn_src", width=100, anchor="w")
         table_scroll = tk.Scrollbar(table_frame, orient=tk.VERTICAL, command=self.table.yview)
         self.table.configure(yscrollcommand=table_scroll.set)
         self.table.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
@@ -248,7 +362,7 @@ class App(BaseTk):
         btn_frame.pack(fill=tk.X, padx=10, pady=5)
         self.btn_start = tk.Button(
             btn_frame,
-            text="开始识别（Roboflow裁剪 + 条码+OCR）",
+            text="开始识别（文字识别和扫描条形码）",
             command=self.start_run,
             height=2
         )
@@ -270,6 +384,7 @@ class App(BaseTk):
     # ========== 工具函数 ==========
     def write_log(self, text: str):
         """线程安全地往日志窗口里写一行"""
+        text = _localize_log_text(text, self._input_name_map)
         text = _mask_path_text(text)
         def _append():
             self.log.configure(state="normal")
@@ -355,6 +470,7 @@ class App(BaseTk):
     def clear_list(self):
         """清空已选择图片"""
         self.image_paths.clear()
+        self._input_name_map = {}
         self.listbox.delete(0, tk.END)
         self.write_log("已清空图片列表。")
     # ========== 主流程 ==========
@@ -415,14 +531,21 @@ class App(BaseTk):
         if not path:
             return
         headers = ("label_id", "model", "sn", "model_src", "sn_src")
+        display_headers = ("标签ID", "型号", "SN", "型号来源", "SN来源")
         try:
             wb = openpyxl.Workbook()
             ws = wb.active
             ws.title = "results"
-            ws.append(list(headers))
+            ws.append(list(display_headers))
             if self.result_rows:
                 for row in self.result_rows:
-                    ws.append([row.get(name, "") for name in headers])
+                    ws.append([
+                        row.get("label_id_display", row.get("label_id", "")),
+                        row.get("model", ""),
+                        row.get("sn", ""),
+                        _display_model_src(row.get("model_src", "")),
+                        _display_sn_src(row.get("sn_src", "")),
+                    ])
             else:
                 for item in self.table.get_children():
                     ws.append(list(self.table.item(item, "values")))
@@ -453,7 +576,7 @@ class App(BaseTk):
                 crop_module, scan2_module = load_pipeline_modules()
             except Exception as exc:
                 msg = _mask_path_text(str(exc))
-                self.write_log(f"❌ 加载 OCR 依赖失败：{msg}")
+                self.write_log(f"❌ 加载文字识别依赖失败：{msg}")
                 self.after(0, lambda msg=msg: messagebox.showerror("加载依赖失败", msg))
                 return
 
@@ -461,6 +584,7 @@ class App(BaseTk):
             self._scan2_module = scan2_module
             input_root = os.path.abspath(getattr(crop_module, "DEFAULT_INPUT_DIR", "new_images"))
             input_dir, copied_records = copy_images_to_unique_run_dir(self.image_paths, input_root)
+            self._input_name_map = _build_input_name_map(copied_records)
 
             self.write_log("[1/4] 准备输入目录")
             self.write_log(f"已拷贝 {len(copied_records)} 个图片。")
@@ -473,10 +597,10 @@ class App(BaseTk):
                 self.write_log("[2/4] 运行 crop_labels.main()：大图 → label → model/sn 小图")
                 crop_stats = crop_module.main(input_dir=input_dir, clean=True)
                 if not isinstance(crop_stats, dict) or crop_stats.get("label_count", 0) <= 0:
-                    raise RuntimeError("未生成任何 label 裁剪图，已停止 OCR。")
+                    raise RuntimeError("未生成任何 label 裁剪图，已停止文字识别。")
                 if crop_stats.get("manifest_rows", 0) <= 0:
-                    raise RuntimeError("未生成 manifest 记录，已停止 OCR。")
-                self.write_log("[3/4] 运行 scan2.main()：识别 MODEL / SN")
+                    raise RuntimeError("未生成 manifest 记录，已停止文字识别。")
+                self.write_log("[3/4] 运行 scan2.main()：文字识别和扫描条形码（MODEL/SN）")
                 result_jsonl = os.path.join(crop_module.STAGE2_DIR, "model_sn_ocr.jsonl")
                 debug_log = os.path.join(crop_module.STAGE2_DIR, "debug_ocr_barcode.log")
                 scan2_module.main(
@@ -522,7 +646,12 @@ class App(BaseTk):
                     if not line:
                         continue
                     try:
-                        rows.append(json.loads(line))
+                        row = json.loads(line)
+                        row["label_id_display"] = _rewrite_label_id_for_display(
+                            row.get("label_id", ""),
+                            self._input_name_map,
+                        )
+                        rows.append(row)
                     except json.JSONDecodeError:
                         continue
         except Exception as e:
@@ -533,10 +662,10 @@ class App(BaseTk):
             self.table.delete(*self.table.get_children())
             for r in rows:
                 values = (
-                    r.get("label_id", ""),
+                    r.get("label_id_display", r.get("label_id", "")),
                     r.get("model", ""),
                     r.get("sn", ""),
-                    r.get("model_src", ""),
+                    _display_model_src(r.get("model_src", "")),
                     _display_sn_src(r.get("sn_src", "")),
                 )
                 self.table.insert("", tk.END, values=values)
warning: in the working copy of 'gui_app.py', LF will be replaced by CRLF the next time Git touches it

 succeeded in 876ms:
diff --git a/sn_barcode.py b/sn_barcode.py
index e8927cf..e6467ed 100644
--- a/sn_barcode.py
+++ b/sn_barcode.py
@@ -1,6 +1,7 @@
 from __future__ import annotations
 
 import hashlib
+import json
 import os
 import re
 from dataclasses import dataclass, field
@@ -17,7 +18,7 @@ except Exception:  # pragma: no cover
     np = None
 
 
-SN20_RE = re.compile(r"(2[0-9]{9,10}(?:ERA|ERB|ER|LDR|LDRA|SRA)[0-9]{4,7})")
+BASE_SN20_SEGMENTS = ("ES", "ERA", "ERB", "ER", "LDR", "LDRA", "SRA")
 SN12_RE = re.compile(r"(4E[0-9A-Z]{10})")
 
 NON_SN_PREFIX_RE = re.compile(
@@ -31,6 +32,104 @@ DEFAULT_MIN_BARCODE_HEIGHT = 22
 DEFAULT_BLUR_VARIANCE = 18.0
 DEFAULT_DESKEW_ANGLES = (0, -4, 4, -8, 8)
 
+_SN20_CACHE = {
+    "path": None,
+    "mtime": None,
+    "segments": tuple(BASE_SN20_SEGMENTS),
+    "regex": re.compile(r"(2[0-9]{9,10}(?:ES|ERA|ERB|ER|LDR|LDRA|SRA)[0-9]{4,7})"),
+}
+
+
+def _sn_segments_file_path() -> str:
+    override = os.environ.get("HUAWEIOCR_SN_SEGMENTS_FILE", "").strip()
+    if override:
+        return os.path.abspath(override)
+    if os.name == "nt":
+        base = os.environ.get("LOCALAPPDATA") or os.path.join(os.path.expanduser("~"), "AppData", "Lo
cal")
+    else:
+        base = os.environ.get("XDG_DATA_HOME") or os.path.join(os.path.expanduser("~"), ".local", "sh
are")
+    return os.path.join(base, "HuaweiOCR", "sn_learned_segments.json")
+
+
+def _compile_sn20_regex(segments: Iterable[str]):
+    cleaned = []
+    for segment in segments:
+        token = _clean_code(segment)
+        if not token or not token.isalpha():
+            continue
+        cleaned.append(token)
+    dedup = sorted(set(cleaned), key=lambda s: (-len(s), s))
+    if not dedup:
+        dedup = list(BASE_SN20_SEGMENTS)
+    alt = "|".join(re.escape(s) for s in dedup)
+    return re.compile(rf"(2[0-9]{{9,10}}(?:{alt})[0-9]{{4,7}})")
+
+
+def _load_learned_segments() -> tuple[str, ...]:
+    path = _sn_segments_file_path()
+    try:
+        mtime = os.path.getmtime(path)
+    except OSError:
+        mtime = None
+
+    cached_path = _SN20_CACHE.get("path")
+    cached_mtime = _SN20_CACHE.get("mtime")
+    if cached_path == path and cached_mtime == mtime:
+        return tuple(_SN20_CACHE.get("segments", tuple(BASE_SN20_SEGMENTS)))
+
+    learned = []
+    if mtime is not None:
+        try:
+            with open(path, "r", encoding="utf-8") as f:
+                payload = json.load(f)
+            if isinstance(payload, list):
+                learned = [str(item) for item in payload]
+        except Exception:
+            learned = []
+
+    merged = tuple(sorted(set(BASE_SN20_SEGMENTS).union({_clean_code(v) for v in learned if v}), key=
lambda s: (-len(s), s)))
+    _SN20_CACHE["path"] = path
+    _SN20_CACHE["mtime"] = mtime
+    _SN20_CACHE["segments"] = merged
+    _SN20_CACHE["regex"] = _compile_sn20_regex(merged)
+    return merged
+
+
+def _sn20_regex():
+    _load_learned_segments()
+    return _SN20_CACHE["regex"]
+
+
+def learn_sn_pattern(sn: str) -> bool:
+    cleaned = _clean_code(sn)
+    match = re.match(r"^2[0-9]{9,10}([A-Z]{2,5})[0-9]{4,7}$", cleaned)
+    if not match:
+        return False
+    segment = match.group(1)
+    if segment in BASE_SN20_SEGMENTS:
+        return False
+
+    _load_learned_segments()
+    current = set(_SN20_CACHE.get("segments", tuple(BASE_SN20_SEGMENTS)))
+    if segment in current:
+        return False
+
+    current.add(segment)
+    persisted = sorted(current - set(BASE_SN20_SEGMENTS), key=lambda s: (-len(s), s))
+    path = _sn_segments_file_path()
+    dirpath = os.path.dirname(path)
+    if dirpath:
+        os.makedirs(dirpath, exist_ok=True)
+    tmp = f"{path}.tmp"
+    with open(tmp, "w", encoding="utf-8") as f:
+        json.dump(persisted, f, ensure_ascii=False, indent=2)
+    os.replace(tmp, path)
+
+    _SN20_CACHE["path"] = None
+    _SN20_CACHE["mtime"] = None
+    _load_learned_segments()
+    return True
+
 
 @dataclass(frozen=True)
 class CandidateImage:
@@ -140,7 +239,7 @@ def extract_sn_from_payload(value: str) -> str:
     if NON_SN_PREFIX_RE.match(cleaned):
         return ""
 
-    m = SN20_RE.search(cleaned)
+    m = _sn20_regex().search(cleaned)
     if m and len(m.group(1)) == 20:
         return m.group(1)
 
@@ -164,6 +263,42 @@ def _source_rank(source_region: str) -> int:
     return 9
 
 
+def _strip_sn_prefixes(value: str) -> str:
+    cleaned = _clean_code(value)
+    while cleaned.startswith("SN"):
+        cleaned = cleaned[2:]
+    for prefix in ("SERIALNO", "SERIAL", "SNO"):
+        if cleaned.startswith(prefix):
+            cleaned = cleaned[len(prefix):]
+    return cleaned
+
+
+def _sn_like_parse_failure_payload(raw_text: str) -> bool:
+    raw = (raw_text or "").upper()
+    if not raw:
+        return False
+
+    strict_unknown_sn20 = re.compile(r"^2[0-9]{9,10}[A-Z]{2,5}[0-9]{4,7}$")
+    strict_sn12 = re.compile(r"^4E[0-9A-Z]{10}$")
+    non_sn_leading = re.compile(
+        r"(?:^|[^A-Z0-9])(?:SF|MAC|EAN|UPC|QR|HTTP|HTTPS|PART(?:\s*NO)?|P\s*/?\s*N|PN|MODEL|DESC|ROUT
E|WAYBILL|SNMP|IMEI)\s*[:：#-]?\s*$"
+    )
+
+    for match in re.finditer(r"[0-9A-Z]{8,40}", raw):
+        token = match.group(0)
+        leading = raw[max(0, match.start() - 24):match.start()]
+        if non_sn_leading.search(leading):
+            continue
+        cleaned = _strip_sn_prefixes(token)
+        if len(cleaned) < 10 or NON_SN_PREFIX_RE.match(cleaned):
+            continue
+        if strict_sn12.match(cleaned):
+            return True
+        if len(cleaned) == 20 and strict_unknown_sn20.match(cleaned):
+            return True
+    return False
+
+
 def select_sn_from_decoder_results(results: Iterable[DecoderResult]) -> SnBarcodeReport:
     decoder_results = list(results)
     sn_candidates: list[SnCandidate] = []
@@ -235,8 +370,26 @@ def select_sn_from_decoder_results(results: Iterable[DecoderResult]) -> SnBarcod
             non_sn_payloads=non_sn_payloads,
         )
 
+    best_raw_result = None
+    if decoder_results:
+        best_raw_result = sorted(
+            [r for r in decoder_results if (r.raw_text or "").strip()],
+            key=lambda r: (
+                0 if _sn_like_parse_failure_payload(r.raw_text) else 1,
+                1 if NON_SN_PREFIX_RE.match(_strip_sn_prefixes(r.raw_text)) else 0,
+                _source_rank(r.source_region),
+                r.rotation,
+                r.decoder_name,
+            ),
+        )
+        best_raw_result = best_raw_result[0] if best_raw_result else None
+
     return SnBarcodeReport(
         status="parse_failure" if decoder_results else "decoder_miss",
+        raw_text=best_raw_result.raw_text if best_raw_result else "",
+        source=best_raw_result.source if best_raw_result else "",
+        source_region=best_raw_result.source_region if best_raw_result else "",
+        decoder_name=best_raw_result.decoder_name if best_raw_result else "",
         attempts=0,
         decoded_count=len(decoder_results),
         results=decoder_results,
warning: in the working copy of 'sn_barcode.py', LF will be replaced by CRLF the next time Git touches
 it

 succeeded in 887ms:
diff --git a/tests/test_locked_output_dirs.py b/tests/test_locked_output_dirs.py
index 6285eee..2818b37 100644
--- a/tests/test_locked_output_dirs.py
+++ b/tests/test_locked_output_dirs.py
@@ -252,6 +252,28 @@ class RunAllPathPropagationTest(unittest.TestCase):
 
 
 class PaddleOcrModelKwargsTest(unittest.TestCase):
+    @staticmethod
+    def _install_ocr_import_fakes():
+        paddle = types.ModuleType("paddle")
+        paddle.set_device = lambda _device: None
+        sys.modules["paddle"] = paddle
+
+        paddleocr = types.ModuleType("paddleocr")
+        paddleocr.PaddleOCR = type("DummyPaddleOCR", (), {})
+        sys.modules["paddleocr"] = paddleocr
+
+        app_paths = types.ModuleType("app_paths")
+        app_paths.ensure_models_installed = lambda: None
+        app_paths.get_resource_path = lambda *parts: os.path.join(*parts)
+        sys.modules["app_paths"] = app_paths
+
+    def _import_ocr_with_env(self, env):
+        for name in ("ocr", "paddle", "paddleocr", "app_paths"):
+            sys.modules.pop(name, None)
+        self._install_ocr_import_fakes()
+        with mock.patch.dict(os.environ, env, clear=False):
+            return importlib.import_module("ocr")
+
     def test_local_model_dirs_include_matching_model_names(self):
         for name in ("ocr", "paddle", "paddleocr", "app_paths"):
             sys.modules.pop(name, None)
@@ -404,6 +426,30 @@ class PaddleOcrModelKwargsTest(unittest.TestCase):
             with mock.patch.object(ocr, "get_resource_path", side_effect=lambda *parts: os.path.join(
root, *parts)):
                 self.assertEqual(ocr._local_model_root_fallback(), bundle_root)
 
+    def test_recognition_candidates_default_to_en(self):
+        ocr = self._import_ocr_with_env({"HUAWEIOCR_OCR_LANG": "", "HUAWEIOCR_OCR_PROFILE": "mobile"}
)
+        self.assertEqual(ocr.OCR_LANG, "en")
+        self.assertEqual(
+            ocr._recognition_model_candidates(),
+            ["en_PP-OCRv5_mobile_rec", "PP-OCRv5_mobile_rec", "PP-OCRv5_server_rec"],
+        )
+
+    def test_recognition_candidates_support_chinese_lang(self):
+        ocr = self._import_ocr_with_env({"HUAWEIOCR_OCR_LANG": "ch", "HUAWEIOCR_OCR_PROFILE": "mobile
"})
+        self.assertEqual(ocr.OCR_LANG, "ch")
+        self.assertEqual(
+            ocr._recognition_model_candidates(),
+            ["PP-OCRv5_mobile_rec", "PP-OCRv5_server_rec", "en_PP-OCRv5_mobile_rec"],
+        )
+
+    def test_invalid_lang_falls_back_to_en(self):
+        ocr = self._import_ocr_with_env({"HUAWEIOCR_OCR_LANG": "invalid", "HUAWEIOCR_OCR_PROFILE": "s
erver"})
+        self.assertEqual(ocr.OCR_LANG, "en")
+        self.assertEqual(
+            ocr._recognition_model_candidates(),
+            ["PP-OCRv5_server_rec", "en_PP-OCRv5_mobile_rec", "PP-OCRv5_mobile_rec"],
+        )
+
 
 class Scan2ManifestTest(unittest.TestCase):
     def test_main_signature_keeps_legacy_arguments(self):
@@ -813,6 +859,22 @@ class CropTempFileTest(unittest.TestCase):
 
 
 class GuiPipelineTest(unittest.TestCase):
+    @staticmethod
+    def _reset_gui_import_modules():
+        for name in (
+            "gui_app",
+            "crop",
+            "scan2",
+            "barcode",
+            "cv2",
+            "numpy",
+            "pyzbar",
+            "pyzbar.pyzbar",
+            "app_paths",
+            "openpyxl",
+        ):
+            sys.modules.pop(name, None)
+
     def test_same_basename_sources_are_staged_with_unique_names(self):
         import gui_pipeline
 
@@ -837,6 +899,7 @@ class GuiPipelineTest(unittest.TestCase):
             self.assertEqual(len(names), 2)
             self.assertEqual(len(set(names)), 2)
             self.assertEqual(names, ["input_0001.png", "input_0002.png"])
+            self.assertEqual([record["source_name"] for record in records], ["same.png", "same.png"])
             self.assertTrue(os.path.exists(os.path.join(run_dir, names[0])))
             self.assertTrue(os.path.exists(os.path.join(run_dir, names[1])))
             with open(os.path.join(run_dir, "source_manifest.jsonl"), "r", encoding="utf-8") as f:
@@ -852,15 +915,85 @@ class GuiPipelineTest(unittest.TestCase):
             self.assertEqual([row["input_name"] for row in manifest_rows], names)
             self.assertTrue(all(len(row["sha256"]) == 64 for row in manifest_rows))
 
+    def test_gui_display_label_uses_original_source_name(self):
+        self._reset_gui_import_modules()
+        import gui_app
+
+        self.assertEqual(
+            gui_app._rewrite_label_id_for_display(
+                "input_0005.jpg__label_1",
+                {"input_0005.jpg": "泰州.jpg"},
+            ),
+            "泰州.jpg__label_1",
+        )
+
+    def test_gui_display_label_keeps_uniqueness_for_duplicate_source_names(self):
+        self._reset_gui_import_modules()
+        import gui_app
+
+        mapping = gui_app._build_input_name_map(
+            [
+                {"source_index": 1, "source_name": "same.png", "input_name": "input_0001.png"},
+                {"source_index": 2, "source_name": "same.png", "input_name": "input_0002.png"},
+            ]
+        )
+        self.assertEqual(mapping["input_0001.png"], "same.png#1")
+        self.assertEqual(mapping["input_0002.png"], "same.png#2")
+        self.assertEqual(
+            gui_app._rewrite_label_id_for_display("input_0002.png__label_1", mapping),
+            "same.png#2__label_1",
+        )
+
+    def test_gui_display_source_labels_are_chinese(self):
+        self._reset_gui_import_modules()
+        import gui_app
+
+        self.assertEqual(gui_app._display_model_src("barcode"), "扫描条形码")
+        self.assertEqual(gui_app._display_model_src("barcode_unmatched"), "扫描条形码(未匹配规则)")
+        self.assertEqual(gui_app._display_model_src("barcode_ocr_agree"), "文字识别+扫描条形码(一致)")
+        self.assertEqual(gui_app._display_model_src("ocr_file"), "文字识别")
+        self.assertEqual(gui_app._display_model_src("ocr_file+sn_hint"), "文字识别(由SN推断修正)")
+        self.assertEqual(gui_app._display_model_src("missing+sn_hint"), "缺失(由SN推断)")
+        self.assertEqual(gui_app._display_sn_src("barcode"), "扫描条形码")
+        self.assertEqual(gui_app._display_sn_src("barcode_unmatched"), "扫描条形码(未匹配规则)")
+        self.assertEqual(gui_app._display_sn_src("barcode_ocr_agree"), "文字识别+扫描条形码(一致)")
+        self.assertEqual(gui_app._display_sn_src("ocr"), "文字识别")
+        self.assertEqual(gui_app._display_sn_src("barcode_decoder_miss"), "条码未命中")
+
+    def test_gui_localize_result_log_line_to_chinese(self):
+        self._reset_gui_import_modules()
+        import gui_app
+
+        line = "[input_0001.jpg__label_2] MODEL=AP362E (M_SRC=barcode) | SN=21500871474ES1014701 (SN_
SRC=barcode)"
+        localized = gui_app._localize_log_text(
+            line,
+            {"input_0001.jpg": "泰州.jpg"},
+        )
+
+        self.assertEqual(
+            localized,
+            "[泰州.jpg__label_2] 型号=AP362E（来源=扫描条形码） | SN=21500871474ES1014701（来源=扫描条形码）",
+        )
+
+    def test_gui_localize_stage1_log_lines_to_chinese(self):
+        self._reset_gui_import_modules()
+        import gui_app
+
+        self.assertEqual(
+            gui_app._localize_log_text("Stage1: input_0001.jpg -> 5 label crops", {}),
+            "阶段1：input_0001.jpg -> 5 个标签裁剪图",
+        )
+        self.assertEqual(
+            gui_app._localize_log_text("Stage1 complete: 5 label crops generated", {}),
+            "阶段1完成：生成 5 个标签裁剪图",
+        )
+        self.assertEqual(
+            gui_app._localize_log_text("Stats: at least one field 5; both fields 5", {}),
+            "统计：至少识别出一个字段 5；两个字段都识别出 5",
+        )
+
     def test_gui_import_does_not_import_pipeline_modules(self):
-        sys.modules.pop("gui_app", None)
-        sys.modules.pop("crop", None)
-        sys.modules.pop("scan2", None)
-        sys.modules.pop("barcode", None)
-        sys.modules.pop("cv2", None)
-        sys.modules.pop("numpy", None)
-        sys.modules.pop("pyzbar", None)
-        sys.modules.pop("pyzbar.pyzbar", None)
+        self._reset_gui_import_modules()
 
         importlib.import_module("gui_app")
 
@@ -868,7 +1001,7 @@ class GuiPipelineTest(unittest.TestCase):
         self.assertNotIn("scan2", sys.modules)
 
     def test_gui_log_mask_keeps_app_relative_output_paths(self):
-        sys.modules.pop("gui_app", None)
+        self._reset_gui_import_modules()
         import gui_app
 
         app_path = os.path.join(os.getcwd(), "stage2_fields", "manifest.jsonl")
@@ -883,7 +1016,7 @@ class GuiPipelineTest(unittest.TestCase):
         self.assertNotIn(external_path, masked)
 
     def test_gui_run_pipeline_requests_clean_crop_outputs(self):
-        sys.modules.pop("gui_app", None)
+        self._reset_gui_import_modules()
         import gui_app
 
         source = inspect.getsource(gui_app.App.run_pipeline)
@@ -1009,6 +1142,10 @@ class Scan2DebugLogTest(unittest.TestCase):
 
 
 class AppPathsInstallTest(unittest.TestCase):
+    def setUp(self):
+        # Some earlier tests install a stub app_paths module; force real module import.
+        sys.modules.pop("app_paths", None)
+
     def test_incomplete_model_dir_is_replaced(self):
         import app_paths
 
warning: in the working copy of 'tests/test_locked_output_dirs.py', LF will be replaced by CRLF the ne
xt time Git touches it

 succeeded in 917ms:
diff --git a/scan2.py b/scan2.py
index 205c2b3..bdfdbd9 100644
--- a/scan2.py
+++ b/scan2.py
@@ -7,7 +7,7 @@ import numpy as np
 from ocr import init_ocr, ocr_one_image
 from barcode import decode_small_patch
 from app_paths import ensure_models_installed
-from sn_barcode import extract_sn_from_payload, scan_sn_barcodes
+from sn_barcode import extract_sn_from_payload, learn_sn_pattern, scan_sn_barcodes
 
 # Simple log gating for CLI usage.
 LOG_LEVEL = os.environ.get("LOG_LEVEL", "info").lower()
@@ -323,32 +323,19 @@ def extract_model_from_ocr_result(text: str, concat: str) -> str:
 
 # ========= SN RULES =========
 
-SN20_RE = re.compile(
-    r"(2[0-9]{9,10}(?:ERA|ERB|ER|LDR|LDRA|SRA)[0-9]{4,7})"
-)
-
-SN12_RE = re.compile(
-    r"(4E[0-9A-Z]{10})"
-)
-
-
 def _clean_code(s: str) -> str:
     return re.sub(r"[^0-9A-Z]", "", s.upper())
 
 
 def extract_sn_from_text(text: str) -> str:
+    payload_sn = extract_sn_from_payload(text)
+    if payload_sn:
+        return payload_sn
+
     s = _clean_code(text)
     while s.startswith("SN"):
         s = s[2:]
 
-    m = SN20_RE.search(s)
-    if m and len(m.group(1)) == 20:
-        return m.group(1)
-
-    m = SN12_RE.search(s)
-    if m and len(m.group(0)) == 12:
-        return m.group(0)
-
     blocks = re.findall(r"[0-9A-Z]{12,20}", s)
     best = ""
     for b in blocks:
@@ -364,11 +351,96 @@ def extract_sn_from_barcode_candidate(text: str) -> str:
     return extract_sn_from_payload(text)
 
 
+def _strip_sn_prefixes(value: str) -> str:
+    s = _clean_code(value)
+    while s.startswith("SN"):
+        s = s[2:]
+    for prefix in ("SERIALNO", "SERIAL", "SNO"):
+        if s.startswith(prefix):
+            s = s[len(prefix):]
+    return s
+
+
+def _extract_unmatched_barcode_sn(raw_text: str) -> str:
+    raw = "" if raw_text is None else str(raw_text)
+    if not raw.strip():
+        return ""
+    raw_upper = raw.upper()
+
+    strict_unknown_sn20 = re.compile(r"^2[0-9]{9,10}[A-Z]{2,5}[0-9]{4,7}$")
+    strict_sn12 = re.compile(r"^4E[0-9A-Z]{10}$")
+    non_sn_prefix = re.compile(r"^(SF|MAC|EAN|UPC|QR|HTTP|HTTPS|PART|PN|MODEL|DESC|ROUTE|WAYBILL|SNMP
|IMEI)")
+    non_sn_leading = re.compile(
+        r"(?:^|[^A-Z0-9])(?:SF|MAC|EAN|UPC|QR|HTTP|HTTPS|PART(?:\s*NO)?|P\s*/?\s*N|PN|MODEL|DESC|ROUT
E|WAYBILL|SNMP|IMEI)\s*[:：#-]?\s*$"
+    )
+
+    candidates = []
+    for match in re.finditer(r"[0-9A-Z]{8,40}", raw_upper):
+        token = match.group(0)
+        leading = raw_upper[max(0, match.start() - 24):match.start()]
+        if non_sn_leading.search(leading):
+            continue
+        cleaned = _strip_sn_prefixes(token)
+        if len(cleaned) < 10 or non_sn_prefix.match(cleaned):
+            continue
+        if strict_sn12.match(cleaned):
+            return cleaned
+        if len(cleaned) != 20:
+            continue
+        if not strict_unknown_sn20.match(cleaned):
+            continue
+        score = (
+            abs(len(cleaned) - 20),
+            -len(cleaned),
+        )
+        candidates.append((score, cleaned))
+
+    if candidates:
+        candidates.sort(key=lambda x: x[0])
+        return candidates[0][1]
+
+    return ""
+
+
+def _try_learn_sn_pattern(sn: str, meta: dict | None = None) -> bool:
+    try:
+        learned = learn_sn_pattern(sn)
+    except Exception as exc:
+        append_debug(f"[SN][PATTERN_LEARN_FAIL] {exc.__class__.__name__}: {exc}")
+        if isinstance(meta, dict):
+            meta["sn_pattern_learned"] = False
+            meta["sn_pattern_learn_error"] = exc.__class__.__name__
+        return False
+    if learned and isinstance(meta, dict):
+        meta["sn_pattern_learned"] = True
+    return learned
+
+
+def _confirm_barcode_with_ocr(
+    barcode_candidate: str,
+    ocr_text: str,
+    barcode_report,
+    meta: dict,
+    ocr_source: str,
+):
+    if not barcode_candidate:
+        return None
+    ocr_candidate = _extract_unmatched_barcode_sn(ocr_text)
+    if not ocr_candidate or ocr_candidate != barcode_candidate:
+        return None
+    _try_learn_sn_pattern(barcode_candidate, meta)
+    return (
+        barcode_candidate,
+        f"[BARCODE_OCR_AGREE:{ocr_source}] {barcode_report.raw_text} | OCR={ocr_text}",
+        "barcode_ocr_agree",
+        meta,
+    )
+
+
 def filter_sn_lines(lines: list[str]) -> list[str]:
     filtered = []
     for ln in lines:
-        s = _clean_code(ln)
-        if SN20_RE.search(s) or SN12_RE.search(s):
+        if extract_sn_from_payload(ln):
             filtered.append(ln)
     return filtered
 
@@ -661,6 +733,9 @@ def recognize_sn(
     meta = barcode_report.to_meta()
     meta["ocr_text_found"] = False
     barcode_raw = _format_barcode_report(barcode_report)
+    barcode_unmatched_sn = ""
+    if barcode_report.status == "parse_failure":
+        barcode_unmatched_sn = _extract_unmatched_barcode_sn(barcode_report.raw_text)
     append_sensitive_debug(
         f"[SN][BARCODE_REPORT] {tag}status={barcode_report.status} "
         f"attempts={barcode_report.attempts} decoded={barcode_report.decoded_count} raw={barcode_raw!
r}"
@@ -678,12 +753,26 @@ def recognize_sn(
         if barcode_report.status == "ambiguous":
             return "", f"[BARCODE_AMBIGUOUS] {barcode_raw}", "barcode_ambiguous", meta
         if barcode_report.status == "parse_failure":
+            if barcode_unmatched_sn:
+                return (
+                    barcode_unmatched_sn,
+                    f"[BARCODE_UNMATCHED:{barcode_report.source_region or 'unknown'}] {barcode_report
.raw_text}",
+                    "barcode_unmatched",
+                    meta,
+                )
             return "", f"[BARCODE_PARSE_FAIL] {barcode_raw}", "barcode_parse_fail", meta
         if barcode_report.status == "quality_reject":
             return "", "[BARCODE_QUALITY_REJECT]", "barcode_quality_reject", meta
         return "", "", "barcode_decoder_miss", meta
 
     if not sn_path:
+        if barcode_unmatched_sn:
+            return (
+                barcode_unmatched_sn,
+                f"[BARCODE_UNMATCHED:{barcode_report.source_region or 'unknown'}] {barcode_report.raw
_text}",
+                "barcode_unmatched",
+                meta,
+            )
         if barcode_report.results:
             if barcode_report.status == "ambiguous":
                 return "", f"[BARCODE_AMBIGUOUS] {barcode_raw}", "barcode_ambiguous", meta
@@ -703,7 +792,24 @@ def recognize_sn(
         meta["ocr_text_found"] = True
     sn = extract_sn_from_text(concat or text)
     if sn:
+        if barcode_unmatched_sn and sn == barcode_unmatched_sn:
+            _try_learn_sn_pattern(sn, meta)
+            return (
+                sn,
+                f"[BARCODE_OCR_AGREE:ocr_color] {barcode_report.raw_text} | OCR={concat or text}",
+                "barcode_ocr_agree",
+                meta,
+            )
         return sn, text, "ocr", meta
+    agreed = _confirm_barcode_with_ocr(
+        barcode_unmatched_sn,
+        concat or text,
+        barcode_report,
+        meta,
+        "ocr_color",
+    )
+    if agreed:
+        return agreed
 
     img = load_and_preprocess(sn_path, roi_bottom=True)
     text, concat, texts = ocr_text_with_details(img)
@@ -713,7 +819,24 @@ def recognize_sn(
         meta["ocr_text_found"] = True
     sn = extract_sn_from_text(concat or text)
     if sn:
+        if barcode_unmatched_sn and sn == barcode_unmatched_sn:
+            _try_learn_sn_pattern(sn, meta)
+            return (
+                sn,
+                f"[BARCODE_OCR_AGREE:ocr_bin] {barcode_report.raw_text} | OCR={concat or text}",
+                "barcode_ocr_agree",
+                meta,
+            )
         return sn, text, "ocr_bin", meta
+    agreed = _confirm_barcode_with_ocr(
+        barcode_unmatched_sn,
+        concat or text,
+        barcode_report,
+        meta,
+        "ocr_bin",
+    )
+    if agreed:
+        return agreed
 
     top_text, top_concat = ocr_sn_top_text(sn_path)
     append_sensitive_debug(f"[SN][OCR_TOP] {tag}{os.path.basename(sn_path)} | text={top_text!r} conca
t={top_concat!r}")
@@ -721,12 +844,36 @@ def recognize_sn(
         meta["ocr_text_found"] = True
     sn = extract_sn_from_text(top_concat or top_text)
     if sn:
+        if barcode_unmatched_sn and sn == barcode_unmatched_sn:
+            _try_learn_sn_pattern(sn, meta)
+            return (
+                sn,
+                f"[BARCODE_OCR_AGREE:ocr_top] {barcode_report.raw_text} | OCR={top_concat or top_text
}",
+                "barcode_ocr_agree",
+                meta,
+            )
         return sn, top_text, "ocr_top", meta
+    agreed = _confirm_barcode_with_ocr(
+        barcode_unmatched_sn,
+        top_concat or top_text,
+        barcode_report,
+        meta,
+        "ocr_top",
+    )
+    if agreed:
+        return agreed
 
     if barcode_report.results:
         if barcode_report.status == "ambiguous":
             return "", f"[BARCODE_AMBIGUOUS] {barcode_raw}", "barcode_ambiguous", meta
         if barcode_report.status == "parse_failure":
+            if barcode_unmatched_sn:
+                return (
+                    barcode_unmatched_sn,
+                    f"[BARCODE_UNMATCHED:{barcode_report.source_region or 'unknown'}] {barcode_report
.raw_text}",
+                    "barcode_unmatched",
+                    meta,
+                )
             return "", f"[BARCODE_PARSE_FAIL] {barcode_raw}", "barcode_parse_fail", meta
         return "", f"[BARCODE] {barcode_raw}", "barcode_no_match", meta
 
@@ -826,6 +973,7 @@ def main(out_dir=None, model_dir=None, sn_dir=None, out_jsonl=None, debug_log=No
         "sn_success": 0,
         "sn_barcode_attempts": 0,
         "sn_barcode_hits": 0,
+        "sn_barcode_ocr_agree": 0,
         "sn_barcode_hit_rate": 0.0,
         "sn_ocr_recoveries": 0,
         "sn_barcode_parse_failures": 0,
@@ -940,8 +1088,10 @@ def main(out_dir=None, model_dir=None, sn_dir=None, out_jsonl=None, debug_log=No
                 stats["sn_total"] += 1
                 barcode_status = sn_meta.get("barcode_status", "not_attempted")
                 stats["sn_barcode_attempts"] += int(sn_meta.get("barcode_attempts", 0) or 0)
-                if sn_src == "barcode":
+                if sn_src in {"barcode", "barcode_unmatched"}:
                     stats["sn_barcode_hits"] += 1
+                elif sn_src == "barcode_ocr_agree":
+                    stats["sn_barcode_ocr_agree"] += 1
                 elif sn_src.startswith("ocr") and barcode_status in {
                     "decoder_miss",
                     "parse_failure",
warning: in the working copy of 'scan2.py', LF will be replaced by CRLF the next time Git touches it

 succeeded in 1001ms:
diff --git a/tests/test_sn_barcode_scanning.py b/tests/test_sn_barcode_scanning.py
index 90fd696..22c6af0 100644
--- a/tests/test_sn_barcode_scanning.py
+++ b/tests/test_sn_barcode_scanning.py
@@ -22,6 +22,10 @@ class SnBarcodeSelectionTest(unittest.TestCase):
             sn_barcode.extract_sn_from_payload("SN:4E25A0170000"),
             "4E25A0170000",
         )
+        self.assertEqual(
+            sn_barcode.extract_sn_from_payload("S/N:21500871474ES1016219"),
+            "21500871474ES1016219",
+        )
         self.assertEqual(
             sn_barcode.extract_sn_from_payload("S/N 21500671494ERA050003"),
             "21500671494ERA050003",
@@ -31,6 +35,18 @@ class SnBarcodeSelectionTest(unittest.TestCase):
             "21500871494ERB006054",
         )
 
+    def test_learn_sn_pattern_extends_sn20_matching(self):
+        with tempfile.TemporaryDirectory() as root:
+            learned_file = os.path.join(root, "sn_segments.json")
+            sample = "21500871474ZX1016219"
+            with mock.patch.dict(os.environ, {"HUAWEIOCR_SN_SEGMENTS_FILE": learned_file}, clear=Fals
e):
+                self.assertEqual(sn_barcode.extract_sn_from_payload(f"S/N:{sample}"), "")
+                self.assertTrue(sn_barcode.learn_sn_pattern(sample))
+                self.assertEqual(sn_barcode.extract_sn_from_payload(f"S/N:{sample}"), sample)
+                with open(learned_file, "r", encoding="utf-8") as f:
+                    payload = json.load(f)
+                self.assertIn("ZX", payload)
+
     def test_conflicting_barcode_sns_are_ambiguous(self):
         report = sn_barcode.select_sn_from_decoder_results(
             [
@@ -42,6 +58,34 @@ class SnBarcodeSelectionTest(unittest.TestCase):
         self.assertEqual(report.status, "ambiguous")
         self.assertEqual(report.ambiguous_sns, ["4E25A0170000", "4E25A0170001"])
 
+    def test_parse_failure_keeps_best_raw_payload_for_rule_expansion(self):
+        with tempfile.TemporaryDirectory() as root:
+            learned_file = os.path.join(root, "sn_segments.json")
+            with mock.patch.dict(os.environ, {"HUAWEIOCR_SN_SEGMENTS_FILE": learned_file}, clear=Fals
e):
+                report = sn_barcode.select_sn_from_decoder_results(
+                    [
+                        sn_barcode.DecoderResult("fake", "Part No:50087147", "label", "label"),
+                        sn_barcode.DecoderResult("fake", "S/N:9Z123456789ABCD", "sn", "sn"),
+                    ]
+                )
+
+                self.assertEqual(report.status, "parse_failure")
+                self.assertEqual(report.raw_text, "S/N:9Z123456789ABCD")
+                self.assertEqual(report.source_region, "sn")
+
+    def test_parse_failure_prefers_sn_like_payload_over_noise_same_source(self):
+        with tempfile.TemporaryDirectory() as root:
+            learned_file = os.path.join(root, "sn_segments.json")
+            with mock.patch.dict(os.environ, {"HUAWEIOCR_SN_SEGMENTS_FILE": learned_file}, clear=Fals
e):
+                report = sn_barcode.select_sn_from_decoder_results(
+                    [
+                        sn_barcode.DecoderResult("a_decoder", "Part No:50087147", "sn", "sn"),
+                        sn_barcode.DecoderResult("z_decoder", "S/N:21500871474ZX1016219", "sn", "sn")
,
+                    ]
+                )
+                self.assertEqual(report.status, "parse_failure")
+                self.assertEqual(report.raw_text, "S/N:21500871474ZX1016219")
+
     def test_duplicate_sn_uses_source_priority(self):
         report = sn_barcode.select_sn_from_decoder_results(
             [
@@ -119,6 +163,13 @@ class SnBarcodeSelectionTest(unittest.TestCase):
 
 
 class Scan2BarcodeAccountingTest(unittest.TestCase):
+    def test_scan2_extract_sn_prefers_full_es_pattern(self):
+        scan2 = _import_scan2()
+        self.assertEqual(
+            scan2.extract_sn_from_text("S/N:21500871474ES1016219"),
+            "21500871474ES1016219",
+        )
+
     def test_ambiguous_barcode_is_not_silently_selected(self):
         scan2 = _import_scan2()
         report = sn_barcode.SnBarcodeReport(
@@ -145,6 +196,176 @@ class Scan2BarcodeAccountingTest(unittest.TestCase):
         self.assertIn("BARCODE_AMBIGUOUS", raw)
         self.assertEqual(meta["barcode_status"], "ambiguous")
 
+    def test_parse_failure_falls_back_to_barcode_text_instead_of_empty(self):
+        scan2 = _import_scan2()
+        report = sn_barcode.SnBarcodeReport(
+            status="parse_failure",
+            raw_text="S/N:21500871474ZX1016219",
+            source_region="sn",
+            attempts=1,
+            decoded_count=1,
+            results=[
+                sn_barcode.DecoderResult("fake", "S/N:21500871474ZX1016219", "sn", "sn"),
+            ],
+        )
+
+        with mock.patch.object(scan2, "_scan_sn_barcode_report", return_value=report):
+            sn, raw, source, meta = scan2.recognize_sn(
+                "sn.png",
+                label_id="a__label_1",
+                allow_ocr=False,
+            )
+
+        self.assertEqual(sn, "21500871474ZX1016219")
+        self.assertEqual(source, "barcode_unmatched")
+        self.assertIn("BARCODE_UNMATCHED", raw)
+        self.assertEqual(meta["barcode_status"], "parse_failure")
+
+    def test_parse_failure_non_sn_payload_stays_parse_fail(self):
+        scan2 = _import_scan2()
+        report = sn_barcode.SnBarcodeReport(
+            status="parse_failure",
+            raw_text="Part No:50087147",
+            source_region="sn",
+            attempts=1,
+            decoded_count=1,
+            results=[
+                sn_barcode.DecoderResult("fake", "Part No:50087147", "sn", "sn"),
+            ],
+        )
+
+        with mock.patch.object(scan2, "_scan_sn_barcode_report", return_value=report):
+            sn, raw, source, meta = scan2.recognize_sn(
+                "sn.png",
+                label_id="a__label_1",
+                allow_ocr=False,
+            )
+
+        self.assertEqual(sn, "")
+        self.assertEqual(source, "barcode_parse_fail")
+        self.assertIn("BARCODE_PARSE_FAIL", raw)
+        self.assertEqual(meta["barcode_status"], "parse_failure")
+
+    def test_parse_failure_non_sn_prefixed_sn_like_payload_stays_parse_fail(self):
+        scan2 = _import_scan2()
+        report = sn_barcode.SnBarcodeReport(
+            status="parse_failure",
+            raw_text="MAC:4E25A0170000",
+            source_region="sn",
+            attempts=1,
+            decoded_count=1,
+            results=[
+                sn_barcode.DecoderResult("fake", "MAC:4E25A0170000", "sn", "sn"),
+            ],
+        )
+
+        with mock.patch.object(scan2, "_scan_sn_barcode_report", return_value=report):
+            sn, raw, source, meta = scan2.recognize_sn(
+                "sn.png",
+                label_id="a__label_1",
+                allow_ocr=False,
+            )
+
+        self.assertEqual(sn, "")
+        self.assertEqual(source, "barcode_parse_fail")
+        self.assertIn("BARCODE_PARSE_FAIL", raw)
+        self.assertEqual(meta["barcode_status"], "parse_failure")
+
+    def test_parse_failure_invalid_length_sn20_stays_parse_fail(self):
+        scan2 = _import_scan2()
+        report = sn_barcode.SnBarcodeReport(
+            status="parse_failure",
+            raw_text="S/N:2123456789AB1234",
+            source_region="sn",
+            attempts=1,
+            decoded_count=1,
+            results=[
+                sn_barcode.DecoderResult("fake", "S/N:2123456789AB1234", "sn", "sn"),
+            ],
+        )
+
+        with mock.patch.object(scan2, "_scan_sn_barcode_report", return_value=report):
+            sn, raw, source, meta = scan2.recognize_sn(
+                "sn.png",
+                label_id="a__label_1",
+                allow_ocr=False,
+            )
+
+        self.assertEqual(sn, "")
+        self.assertEqual(source, "barcode_parse_fail")
+        self.assertIn("BARCODE_PARSE_FAIL", raw)
+        self.assertEqual(meta["barcode_status"], "parse_failure")
+
+    def test_parse_failure_barcode_and_ocr_agree_then_auto_learn(self):
+        scan2 = _import_scan2()
+        report = sn_barcode.SnBarcodeReport(
+            status="parse_failure",
+            raw_text="S/N:21500871474ZX1016219",
+            source_region="sn",
+            attempts=1,
+            decoded_count=1,
+            results=[
+                sn_barcode.DecoderResult("fake", "S/N:21500871474ZX1016219", "sn", "sn"),
+            ],
+        )
+
+        with tempfile.TemporaryDirectory() as root:
+            learned_file = os.path.join(root, "sn_segments.json")
+            with mock.patch.dict(os.environ, {"HUAWEIOCR_SN_SEGMENTS_FILE": learned_file}, clear=Fals
e):
+                with mock.patch.object(scan2, "_scan_sn_barcode_report", return_value=report):
+                    with mock.patch.object(scan2, "load_for_ocr_color", return_value=object()):
+                        with mock.patch.object(
+                            scan2,
+                            "ocr_text_with_details",
+                            return_value=("S/N:21500871474ZX1016219", "S/N:21500871474ZX1016219", [])
,
+                        ):
+                            sn, raw, source, meta = scan2.recognize_sn(
+                                "sn.png",
+                                label_id="a__label_1",
+                                allow_ocr=True,
+                            )
+
+                self.assertEqual(sn, "21500871474ZX1016219")
+                self.assertEqual(source, "barcode_ocr_agree")
+                self.assertIn("BARCODE_OCR_AGREE", raw)
+                self.assertTrue(meta.get("sn_pattern_learned"))
+                self.assertEqual(
+                    sn_barcode.extract_sn_from_payload("S/N:21500871474ZX1016219"),
+                    "21500871474ZX1016219",
+                )
+
+    def test_parse_failure_barcode_ocr_agree_does_not_crash_when_learning_fails(self):
+        scan2 = _import_scan2()
+        report = sn_barcode.SnBarcodeReport(
+            status="parse_failure",
+            raw_text="S/N:21500871474ZX1016219",
+            source_region="sn",
+            attempts=1,
+            decoded_count=1,
+            results=[
+                sn_barcode.DecoderResult("fake", "S/N:21500871474ZX1016219", "sn", "sn"),
+            ],
+        )
+
+        with mock.patch.object(scan2, "_scan_sn_barcode_report", return_value=report):
+            with mock.patch.object(scan2, "load_for_ocr_color", return_value=object()):
+                with mock.patch.object(
+                    scan2,
+                    "ocr_text_with_details",
+                    return_value=("S/N:21500871474ZX1016219", "S/N:21500871474ZX1016219", []),
+                ):
+                    with mock.patch.object(scan2, "learn_sn_pattern", side_effect=OSError("read-only"
)):
+                        sn, raw, source, meta = scan2.recognize_sn(
+                            "sn.png",
+                            label_id="a__label_1",
+                            allow_ocr=True,
+                        )
+
+        self.assertEqual(sn, "21500871474ZX1016219")
+        self.assertEqual(source, "barcode_ocr_agree")
+        self.assertIn("BARCODE_OCR_AGREE", raw)
+        self.assertEqual(meta.get("sn_pattern_learn_error"), "OSError")
+
     def test_main_reports_barcode_hit_rate_and_ocr_recovery_separately(self):
         scan2 = _import_scan2()
 
@@ -180,6 +401,81 @@ class Scan2BarcodeAccountingTest(unittest.TestCase):
         self.assertEqual(stats["sn_barcode_attempts"], 3)
         self.assertEqual(stats["sn_barcode_hit_rate"], 0.0)
 
+    def test_main_counts_barcode_unmatched_as_barcode_hit(self):
+        scan2 = _import_scan2()
+
+        with tempfile.TemporaryDirectory() as root:
+            stage2 = os.path.join(root, "stage2_fields")
+            model_dir = os.path.join(stage2, "model")
+            sn_dir = os.path.join(stage2, "sn")
+            os.makedirs(model_dir)
+            os.makedirs(sn_dir)
+            sn_path = os.path.join(sn_dir, "a__label_1__sn.png")
+            open(sn_path, "wb").close()
+            with open(os.path.join(stage2, "manifest.jsonl"), "w", encoding="utf-8") as manifest:
+                manifest.write(json.dumps({"label_id": "a__label_1", "sn_path": sn_path}) + "\n")
+
+            meta = {
+                "barcode_found": True,
+                "ocr_text_found": False,
+                "barcode_status": "parse_failure",
+                "barcode_attempts": 1,
+                "barcode_decoded_count": 1,
+            }
+            with mock.patch.object(
+                scan2,
+                "recognize_sn",
+                return_value=("9Z123456789ABCD", "raw barcode", "barcode_unmatched", meta),
+            ):
+                stats = scan2.main(
+                    model_dir=model_dir,
+                    sn_dir=sn_dir,
+                    out_jsonl=os.path.join(root, "out.jsonl"),
+                    debug_log=os.path.join(root, "debug.log"),
+                )
+
+        self.assertEqual(stats["sn_success"], 1)
+        self.assertEqual(stats["sn_barcode_hits"], 1)
+        self.assertEqual(stats["sn_barcode_parse_failures"], 1)
+
+    def test_main_tracks_barcode_ocr_agree_separately_from_barcode_hits(self):
+        scan2 = _import_scan2()
+
+        with tempfile.TemporaryDirectory() as root:
+            stage2 = os.path.join(root, "stage2_fields")
+            model_dir = os.path.join(stage2, "model")
+            sn_dir = os.path.join(stage2, "sn")
+            os.makedirs(model_dir)
+            os.makedirs(sn_dir)
+            sn_path = os.path.join(sn_dir, "a__label_1__sn.png")
+            open(sn_path, "wb").close()
+            with open(os.path.join(stage2, "manifest.jsonl"), "w", encoding="utf-8") as manifest:
+                manifest.write(json.dumps({"label_id": "a__label_1", "sn_path": sn_path}) + "\n")
+
+            meta = {
+                "barcode_found": True,
+                "ocr_text_found": True,
+                "barcode_status": "parse_failure",
+                "barcode_attempts": 1,
+                "barcode_decoded_count": 1,
+            }
+            with mock.patch.object(
+                scan2,
+                "recognize_sn",
+                return_value=("21500871474ZX1016219", "raw agree", "barcode_ocr_agree", meta),
+            ):
+                stats = scan2.main(
+                    model_dir=model_dir,
+                    sn_dir=sn_dir,
+                    out_jsonl=os.path.join(root, "out.jsonl"),
+                    debug_log=os.path.join(root, "debug.log"),
+                )
+
+        self.assertEqual(stats["sn_success"], 1)
+        self.assertEqual(stats["sn_barcode_hits"], 0)
+        self.assertEqual(stats["sn_barcode_ocr_agree"], 1)
+        self.assertEqual(stats["sn_barcode_parse_failures"], 1)
+
 
 class ValidationCommandTest(unittest.TestCase):
     def test_template_builder_creates_manual_review_rows_without_accepting_them(self):
warning: in the working copy of 'tests/test_sn_barcode_scanning.py', LF will be replaced by CRLF the n
ext time Git touches it

codex
我已经看完主要 diff，下一步先跑改动相关测试（尤其 `scan2/sn_barcode/gui`）验证是否有回归，再基于结果给 gate 结论。
exec
"C:\\Program Files\\WindowsApps\\Microsoft.PowerShell_7.6.0.0_x64__8wekyb3d8bbwe\\pwsh.exe" -Command '
python -m pytest -q tests/test_sn_barcode_scanning.py tests/test_locked_output_dirs.py' in D:\newhuawe
i
 succeeded in 2096ms:
.......................................................................  [100%]
============================== warnings summary ===============================
tests/test_locked_output_dirs.py::GuiPipelineTest::test_gui_display_label_keeps_uniqueness_for_duplica
te_source_names
  C:\Users\55093\AppData\Local\Programs\Python\Python312\Lib\site-packages\openpyxl\compat\numbers.py:
9: UserWarning: The NumPy module was reloaded (imported a second time). This can in some cases result 
in small but subtle issues and is discouraged.
    import numpy

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
71 passed, 1 warning in 0.59s

exec
"C:\\Program Files\\WindowsApps\\Microsoft.PowerShell_7.6.0.0_x64__8wekyb3d8bbwe\\pwsh.exe" -Command '
rg -n "source_manifest|source_name|input_name" -S' in D:\newhuawei
exec
"C:\\Program Files\\WindowsApps\\Microsoft.PowerShell_7.6.0.0_x64__8wekyb3d8bbwe\\pwsh.exe" -Command '
rg -n "barcode_unmatched|barcode_ocr_agree|sn_barcode_ocr_agree" -S' in D:\newhuawei
 succeeded in 1114ms:
gui_app.py:80:    if src == "barcode_unmatched":
gui_app.py:82:    if src == "barcode_ocr_agree":
gui_app.py:112:    if src == "barcode_unmatched":
gui_app.py:114:    if src == "barcode_ocr_agree":
scan2.py:434:        f"[BARCODE_OCR_AGREE:{ocr_source}] {barcode_report.raw_text} | OCR={ocr_text}",
scan2.py:435:        "barcode_ocr_agree",
scan2.py:736:    barcode_unmatched_sn = ""
scan2.py:738:        barcode_unmatched_sn = _extract_unmatched_barcode_sn(barcode_report.raw_text)
scan2.py:756:            if barcode_unmatched_sn:
scan2.py:758:                    barcode_unmatched_sn,
scan2.py:759:                    f"[BARCODE_UNMATCHED:{barcode_report.source_region or 'unknown'}] {ba
rcode_report.raw_text}",
scan2.py:760:                    "barcode_unmatched",
scan2.py:769:        if barcode_unmatched_sn:
scan2.py:771:                barcode_unmatched_sn,
scan2.py:772:                f"[BARCODE_UNMATCHED:{barcode_report.source_region or 'unknown'}] {barcod
e_report.raw_text}",
scan2.py:773:                "barcode_unmatched",
scan2.py:795:        if barcode_unmatched_sn and sn == barcode_unmatched_sn:
scan2.py:799:                f"[BARCODE_OCR_AGREE:ocr_color] {barcode_report.raw_text} | OCR={concat o
r text}",
scan2.py:800:                "barcode_ocr_agree",
scan2.py:805:        barcode_unmatched_sn,
scan2.py:822:        if barcode_unmatched_sn and sn == barcode_unmatched_sn:
scan2.py:826:                f"[BARCODE_OCR_AGREE:ocr_bin] {barcode_report.raw_text} | OCR={concat or 
text}",
scan2.py:827:                "barcode_ocr_agree",
scan2.py:832:        barcode_unmatched_sn,
scan2.py:847:        if barcode_unmatched_sn and sn == barcode_unmatched_sn:
scan2.py:851:                f"[BARCODE_OCR_AGREE:ocr_top] {barcode_report.raw_text} | OCR={top_concat
 or top_text}",
scan2.py:852:                "barcode_ocr_agree",
scan2.py:857:        barcode_unmatched_sn,
scan2.py:870:            if barcode_unmatched_sn:
scan2.py:872:                    barcode_unmatched_sn,
scan2.py:873:                    f"[BARCODE_UNMATCHED:{barcode_report.source_region or 'unknown'}] {ba
rcode_report.raw_text}",
scan2.py:874:                    "barcode_unmatched",
scan2.py:976:        "sn_barcode_ocr_agree": 0,
scan2.py:1091:                if sn_src in {"barcode", "barcode_unmatched"}:
scan2.py:1093:                elif sn_src == "barcode_ocr_agree":
scan2.py:1094:                    stats["sn_barcode_ocr_agree"] += 1
tests\test_locked_output_dirs.py:952:        self.assertEqual(gui_app._display_model_src("barcode_unma
tched"), "扫描条形码(未匹配规则)")
tests\test_locked_output_dirs.py:953:        self.assertEqual(gui_app._display_model_src("barcode_ocr_
agree"), "文字识别+扫描条形码(一致)")
tests\test_locked_output_dirs.py:958:        self.assertEqual(gui_app._display_sn_src("barcode_unmatch
ed"), "扫描条形码(未匹配规则)")
tests\test_locked_output_dirs.py:959:        self.assertEqual(gui_app._display_sn_src("barcode_ocr_agr
ee"), "文字识别+扫描条形码(一致)")
tests\test_sn_barcode_scanning.py:220:        self.assertEqual(source, "barcode_unmatched")
tests\test_sn_barcode_scanning.py:221:        self.assertIn("BARCODE_UNMATCHED", raw)
tests\test_sn_barcode_scanning.py:329:                self.assertEqual(source, "barcode_ocr_agree")
tests\test_sn_barcode_scanning.py:330:                self.assertIn("BARCODE_OCR_AGREE", raw)
tests\test_sn_barcode_scanning.py:337:    def test_parse_failure_barcode_ocr_agree_does_not_crash_when
_learning_fails(self):
tests\test_sn_barcode_scanning.py:365:        self.assertEqual(source, "barcode_ocr_agree")
tests\test_sn_barcode_scanning.py:366:        self.assertIn("BARCODE_OCR_AGREE", raw)
tests\test_sn_barcode_scanning.py:404:    def test_main_counts_barcode_unmatched_as_barcode_hit(self):
tests\test_sn_barcode_scanning.py:428:                return_value=("9Z123456789ABCD", "raw barcode", 
"barcode_unmatched", meta),
tests\test_sn_barcode_scanning.py:441:    def test_main_tracks_barcode_ocr_agree_separately_from_barco
de_hits(self):
tests\test_sn_barcode_scanning.py:465:                return_value=("21500871474ZX1016219", "raw agree
", "barcode_ocr_agree", meta),
tests\test_sn_barcode_scanning.py:476:        self.assertEqual(stats["sn_barcode_ocr_agree"], 1)

 succeeded in 1198ms:
gui_pipeline.py:37:                "source_name": os.path.basename(source),
gui_pipeline.py:38:                "input_name": target_name,
gui_pipeline.py:43:    manifest_path = os.path.join(run_dir, "source_manifest.jsonl")
gui_pipeline.py:50:                        "input_name": record["input_name"],
gui_app.py:123:def _rewrite_label_id_for_display(label_id: str, input_name_map: dict[str, str]) -> str
:
gui_app.py:127:    if not input_name_map:
gui_app.py:129:    match = re.match(r"^(?P<input_name>.+?)(__label_\d+)$", text)
gui_app.py:132:    input_name = match.group("input_name")
gui_app.py:134:    source_name = input_name_map.get(input_name.lower())
gui_app.py:135:    if not source_name:
gui_app.py:137:    return f"{source_name}{suffix}"
gui_app.py:140:def _build_input_name_map(copied_records: list[dict]) -> dict[str, str]:
gui_app.py:143:        source_name = str(record.get("source_name", "")).strip()
gui_app.py:144:        if not source_name:
gui_app.py:146:        key = source_name.lower()
gui_app.py:151:        input_name = str(record.get("input_name", "")).strip()
gui_app.py:152:        source_name = str(record.get("source_name", "")).strip()
gui_app.py:153:        if not input_name or not source_name:
gui_app.py:155:        source_key = source_name.lower()
gui_app.py:156:        display_name = source_name
gui_app.py:159:            suffix = source_index if source_index else input_name
gui_app.py:160:            display_name = f"{source_name}#{suffix}"
gui_app.py:161:        mapping[input_name.lower()] = display_name
gui_app.py:165:def _localize_log_text(text: str, input_name_map: dict[str, str]) -> str:
gui_app.py:175:        label_id = _rewrite_label_id_for_display(result_match.group("label"), input_nam
e_map)
gui_app.py:286:        self._input_name_map = {}
gui_app.py:387:        text = _localize_log_text(text, self._input_name_map)
gui_app.py:473:        self._input_name_map = {}
gui_app.py:587:            self._input_name_map = _build_input_name_map(copied_records)
gui_app.py:652:                            self._input_name_map,
tests\test_locked_output_dirs.py:898:            names = [record["input_name"] for record in records]
tests\test_locked_output_dirs.py:902:            self.assertEqual([record["source_name"] for record in
 records], ["same.png", "same.png"])
tests\test_locked_output_dirs.py:905:            with open(os.path.join(run_dir, "source_manifest.json
l"), "r", encoding="utf-8") as f:
tests\test_locked_output_dirs.py:912:                self.assertEqual(set(row), {"source_index", "inpu
t_name", "sha256"})
tests\test_locked_output_dirs.py:913:                self.assertNotIn("same.png", row["input_name"])
tests\test_locked_output_dirs.py:915:            self.assertEqual([row["input_name"] for row in manife
st_rows], names)
tests\test_locked_output_dirs.py:918:    def test_gui_display_label_uses_original_source_name(self):
tests\test_locked_output_dirs.py:930:    def test_gui_display_label_keeps_uniqueness_for_duplicate_sou
rce_names(self):
tests\test_locked_output_dirs.py:934:        mapping = gui_app._build_input_name_map(
tests\test_locked_output_dirs.py:936:                {"source_index": 1, "source_name": "same.png", "i
nput_name": "input_0001.png"},
tests\test_locked_output_dirs.py:937:                {"source_index": 2, "source_name": "same.png", "i
nput_name": "input_0002.png"},

codex
GATE_DECISION: PASS

- No actionable bugs, regressions, security issues, or essential test gaps were found in current `stag
ed/unstaged/untracked` changes.
- Verified with targeted regression suite: `python -m pytest -q tests/test_sn_barcode_scanning.py test
s/test_locked_output_dirs.py` (`71 passed`).

Non-blocking hygiene:
- Untracked validation artifacts under `validation/pure_ocr_6_20260419_223358/` and `validation/pure_o
cr_6_server_rec_20260419_224312/` are present; keep them untracked (or add ignore rules) to avoid acci
dental commit bloat.
hook: Stop
hook: Stop Completed
GATE_DECISION: PASS

- No actionable bugs, regressions, security issues, or essential test gaps were found in current `staged/unstaged/untracked` changes.
- Verified with targeted regression suite: `python -m pytest -q tests/test_sn_barcode_scanning.py tests/test_locked_output_dirs.py` (`71 passed`).

Non-blocking hygiene:
- Untracked validation artifacts under `validation/pure_ocr_6_20260419_223358/` and `validation/pure_ocr_6_server_rec_20260419_224312/` are present; keep them untracked (or add ignore rules) to avoid accidental commit bloat.
tokens used
30,212

