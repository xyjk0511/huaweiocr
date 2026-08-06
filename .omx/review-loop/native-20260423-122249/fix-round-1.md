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
session id: 019dbbcd-2fba-7651-9c1c-bd096c85725d
--------
user
You are in auto-remediation round 1/6.
Read and fix all actionable FAIL findings in:
D:\newhuawei\.omx\review-loop\native-20260423-122249\review-round-1.md

Rules:
- Keep changes minimal, safe, and reversible.
- Preserve behavior unless a finding requires a change.

- Run relevant validation after edits (tests/lint/typecheck when available).

Your first line must be exactly:
FIX_STATUS: APPLIED

Then report changed files and validation commands/results.


hook: SessionStart
hook: SessionStart Completed
hook: UserPromptSubmit
hook: UserPromptSubmit Completed
codex
正在执行这一轮自动修复：我先读取 `review-round-1.md` 里的 FAIL 项，按最小改动逐条修复，然后跑仓库里可用的验证命令并回报结果。
exec
"C:\\Program Files\\WindowsApps\\Microsoft.PowerShell_7.6.0.0_x64__8wekyb3d8bbwe\\pwsh.exe" -Command "
rg -n \"newhuawei|HuaweiOCR|review-loop|scan2.py|sn_barcode.py\" C:\\Users\\55093\\.codex\\memories\\M
EMORY.md" in D:\newhuawei
 succeeded in 653ms:
263:# Task Group: D:\newhuawei OpenSpec barcode-first SN hit-rate contract
265:applies_to: cwd=D:\newhuawei; reuse_rule=reuse for this checkout's OpenSpec proposal workflow and 
future barcode-hit-rate planning; verify current `openspec/changes/` state before editing because this
 repo did not originally have an `openspec/` tree.
271:- rollout_summaries/2026-04-19T05-30-16-edJQ-openspec_sn_barcode_hit_rate_proposal.md (cwd=\\?\D:\
newhuawei, rollout_path=C:\Users\55093\.codex\sessions\2026\04\18\rollout-2026-04-18T22-30-16-019da437
-d8e5-7dc1-8bc2-7ddb37b043ce.jsonl, updated_at=2026-04-19T08:41:06+00:00, thread_id=019da437-d8e5-7dc1
-8bc2-7ddb37b043ce, OpenSpec change authored and committed)
275:- D:\newhuawei, openspec, openspec-propose, raise-sn-barcode-scan-hit-rate, barcode-first, exact b
arcode hit rate, 90%, OCR fallback, SHALL NOT be counted as a barcode hit, proposal.md, design.md, tas
ks.md
296:# Task Group: D:\newhuawei HuaweiOCR optimization audit
297:scope: evidence-backed optimization review for the current HuaweiOCR recognition pipeline, especia
lly when deciding whether to tune OCR, barcode candidate selection, or result gating next.
298:applies_to: cwd=D:\newhuawei; reuse_rule=reuse for this checkout's current algorithm-review questi
ons; re-check current `validation/` outputs before claiming the same bottleneck still holds.
304:- rollout_summaries/2026-04-19T08-46-25-2Yg2-huaweiocr_optimization_audit_and_prioritized_improvem
ents.md (cwd=\\?\D:\newhuawei, rollout_path=C:\Users\55093\.codex\sessions\2026\04\19\rollout-2026-04-
19T01-46-25-019da4eb-6dcb-7a82-9355-cefa7289c6e2.jsonl, updated_at=2026-04-20T17:50:54+00:00, thread_i
d=019da4eb-6dcb-7a82-9355-cefa7289c6e2, prioritized audit tied to code and validation artifacts)
308:- D:\newhuawei, HuaweiOCR, optimization audit, crop.py, scan2.py, barcode.py, sn_barcode.py, ocr.p
y, validation/pure_ocr_6_20260419_223358, validation/pure_ocr_6_server_rec_20260419_224312, mobile, se
rver_rec, barcode-first, SN mismatch
318:- `scan2.py` is the decision layer with the most leverage because it decides between barcode and O
CR for `model` and `sn`; this is the first place to inspect before changing OCR internals. [Task 1]
319:- `sn_barcode.py` already has a heavy candidate/decoder cascade, so likely gains are in candidate 
ordering, pruning, caching, and early-exit behavior rather than adding more blind decode attempts. [Ta
sk 1]
985:# Task Group: F:\HuaweiOCR GSD brownfield refactor
986:scope: HuaweiOCR repo clone/init plus Phase 1-4 planning/execution, and later SN/model barcode-fir
st scope reset and OpenSpec proposal work for the same Windows-first OCR pipeline.
987:applies_to: cwd=F:\HuaweiOCR; reuse_rule=reuse for this checkout and phase worktrees; verify curre
nt branch/state before acting because some phases were worktree-based and Phase 3 was executed directl
y on master.
989:## Task 1: Clone HuaweiOCR and initialize brownfield GSD refactor roadmap, success
993:- rollout_summaries/2026-04-18T08-36-06-iuyF-huaweiocr_brownfield_refactor_init.md (cwd=\\?\F:\Hua
weiOCR, rollout_path=C:\Users\55093\.codex\sessions\2026\04\18\rollout-2026-04-18T01-36-06-019d9fbb-a0
64-7f30-9601-28ee6bb677ed.jsonl, updated_at=2026-04-18T08:51:24+00:00, thread_id=019d9fbb-a064-7f30-96
01-28ee6bb677ed, repo setup and 5-phase roadmap)
997:- gh repo clone xyjk0511/huaweiocr, gsd-new-project, brownfield, PROJECT.md, REQUIREMENTS.md, ROAD
MAP.md, STATE.md, phase_count: 5, run_all.py, crop.py, scan2.py, gui_app.py, gui_app_en.py
1003:- rollout_summaries/2026-04-18T09-15-47-NDuK-huaweiocr_phase1_baseline_contract_freeze_and_uat.md
 (cwd=\\?\F:\HuaweiOCR, rollout_path=C:\Users\55093\.codex\sessions\2026\04\18\rollout-2026-04-18T02-1
5-47-019d9fdf-f3ab-7db3-a675-d72500b48286.jsonl, updated_at=2026-04-18T09:31:50+00:00, thread_id=019d9
fdf-f3ab-7db3-a675-d72500b48286, execution and UAT)
1004:- rollout_summaries/2026-04-18T08-51-30-JhZC-huaweiocr_phase_1_gsd_planning_worktree.md (cwd=\\?\
F:\HuaweiOCR, rollout_path=C:\Users\55093\.codex\sessions\2026\04\18\rollout-2026-04-18T01-51-30-019d9
fc9-b70e-74f1-b312-9d259902d698.jsonl, updated_at=2026-04-18T09:15:31+00:00, thread_id=019d9fc9-b70e-7
4f1-b312-9d259902d698, planning worktree and baseline facts)
1014:- rollout_summaries/2026-04-18T11-22-10-1N8g-gsd_phase_2_plan_execute_merge_crop_extraction.md (c
wd=\\?\F:\HuaweiOCR, rollout_path=C:\Users\55093\.codex\sessions\2026\04\18\rollout-2026-04-18T04-22-1
0-019da053-a8cc-73d1-8547-438ba051e01c.jsonl, updated_at=2026-04-18T12:07:11+00:00, thread_id=019da053
-a8cc-73d1-8547-438ba051e01c, Phase 2 merged to master)
1024:- rollout_summaries/2026-04-18T12-21-52-xKdv-phase_3_recognition_core_extraction_plan_execute_and
_merge_c.md (cwd=\\?\F:\HuaweiOCR, rollout_path=C:\Users\55093\.codex\sessions\2026\04\18\rollout-2026
-04-18T05-21-52-019da08a-515b-7590-8560-3be8f7160975.jsonl, updated_at=2026-04-18T12:47:31+00:00, thre
ad_id=019da08a-515b-7590-8560-3be8f7160975, Phase 3 already on master)
1028:- gsd-plan-phase 3, gsd-execute-phase 3, gsd:merge-phase 3, recognition_core, scan2.py, exact leg
acy signature, RecognitionService, RecognitionStats, phase-completeness, TTY error, auth/token error, 
git merge-base --is-ancestor
1034:- rollout_summaries/2026-04-18T12-47-34-FFxS-huaweiocr_phase_4_interface_consolidation.md (cwd=\\
?\F:\HuaweiOCR, rollout_path=C:\Users\55093\.codex\sessions\2026\04\18\rollout-2026-04-18T05-47-34-019
da0a1-d850-7ce1-b992-13b80884c220.jsonl, updated_at=2026-04-18T13:02:56+00:00, thread_id=019da0a1-d850
-7ce1-b992-13b80884c220, latest completed phase)
1038:- gsd-plan-phase 4, gsd-execute-phase 4, interface_core, app_paths.py, HuaweiOCR.spec, run_all.py
, gui_app.py, gui_app_en.py, tests/test_interface_paths.py, tests/test_cli_adapter.py, tests/test_gui_
localization.py, lazy import, ModuleNotFoundError: cv2
1044:- rollout_summaries/2026-04-18T21-15-36-KXm0-sn_model_barcode_priority_openspec_proposal.md (cwd=
\\?\F:\HuaweiOCR, rollout_path=C:\Users\55093\.codex\sessions\2026\04\18\rollout-2026-04-18T14-15-36-0
19da272-f695-7d01-b1d2-b60aee7764b0.jsonl, updated_at=2026-04-19T05:00:39+00:00, thread_id=019da272-f6
95-7d01-b1d2-b60aee7764b0, scope narrowed back to SN/model and proposal artifacts created)
1065:- Repo was cloned from `xyjk0511/huaweiocr` into `F:\HuaweiOCR`, branch `master`, with `origin ht
tps://github.com/xyjk0511/huaweiocr.git`. [Task 1]
1072:- `app_paths.py` is the runtime path authority; `HuaweiOCR.spec` should avoid user-specific absol
ute `C:\Users\...` paths and use relative bundle paths plus PyInstaller collect helpers. [Task 5]

codex
我已快速检索历史记忆里 `D:\\newhuawei` 的上下文，下一步直接打开本轮 `review-round-1.md`，只处理其中可执行的 FAIL 并保持最小改动。
exec
"C:\\Program Files\\WindowsApps\\Microsoft.PowerShell_7.6.0.0_x64__8wekyb3d8bbwe\\pwsh.exe" -Command "
Get-Content -Path 'D:\\newhuawei\\.omx\\review-loop\\native-20260423-122249\\review-round-1.md'" in D:
\newhuawei
exec
"C:\\Program Files\\WindowsApps\\Microsoft.PowerShell_7.6.0.0_x64__8wekyb3d8bbwe\\pwsh.exe" -Command '
Get-ChildItem -Force -Name' in D:\newhuawei
 succeeded in 711ms:
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
session id: 019dbbcb-85b8-79f3-8366-62e72bb3e0d5
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
我会先按代码审查门禁流程检查当前仓库的 `staged / unstaged / untracked` 变更，再逐文件给出可执行修复建议；先快速读取 memory 里与 `D:\newhuawei` 相关
的上下文，然后基于真实 diff 出结论。
exec
"C:\\Program Files\\WindowsApps\\Microsoft.PowerShell_7.6.0.0_x64__8wekyb3d8bbwe\\pwsh.exe" -Command "
Select-String -Path C:\\Users\\55093\\.codex\\memories\\MEMORY.md -Pattern \"newhuawei|HuaweiOCR|barco
de|scan2.py\" -CaseSensitive:"'$false | Select-Object -First 20 | ForEach-Object { "{0}:{1}" -f $_.Lin
eNumber, $_.Line }' in D:\newhuawei
 succeeded in 665ms:
263:# Task Group: D:\newhuawei OpenSpec barcode-first SN hit-rate contract
264:scope: spec-first planning for barcode-first SN recognition where exact barcode hit rate is a rele
ase gate and OCR is fallback-only.
265:applies_to: cwd=D:\newhuawei; reuse_rule=reuse for this checkout's OpenSpec proposal workflow and 
future barcode-hit-rate planning; verify current `openspec/changes/` state before editing because this
 repo did not originally have an `openspec/` tree.
267:## Task 1: Propose a measurable barcode-first SN hit-rate gate, success
271:- rollout_summaries/2026-04-19T05-30-16-edJQ-openspec_sn_barcode_hit_rate_proposal.md (cwd=\\?\D:\
newhuawei, rollout_path=C:\Users\55093\.codex\sessions\2026\04\18\rollout-2026-04-18T22-30-16-019da437
-d8e5-7dc1-8bc2-7ddb37b043ce.jsonl, updated_at=2026-04-19T08:41:06+00:00, thread_id=019da437-d8e5-7dc1
-8bc2-7ddb37b043ce, OpenSpec change authored and committed)
275:- D:\newhuawei, openspec, openspec-propose, raise-sn-barcode-scan-hit-rate, barcode-first, exact b
arcode hit rate, 90%, OCR fallback, SHALL NOT be counted as a barcode hit, proposal.md, design.md, tas
ks.md
280:- when the user said "我们主要需要扫码" / "能扫码优先扫码" -> default to barcode-first behavior, with OCR only as
 fallback and not the primary success path. [Task 1]
281:- when the user said "sn需要扫码命中率至少90% $openspec-propose" -> treat barcode hit rate as an explicit r
elease gate, not a soft quality target. [Task 1]
286:- `openspec new change "<kebab-case-name>"` scaffolds under `openspec/changes/<name>/`; in this re
po the successful change name was `raise-sn-barcode-scan-hit-rate`. [Task 1]
288:- The durable contract here is to measure exact barcode-derived SN hits separately from OCR fallba
ck, and to require at least `90%` exact barcode hit rate before the change can be considered complete.
 [Task 1]
289:- The resulting artifact set was `proposal.md`, `design.md`, `specs/sn-barcode-scanning/spec.md`, 
and `tasks.md`, committed as `8bef275 docs: propose SN barcode hit-rate gate`. [Task 1]
294:- Do not let OCR fallback inflate barcode-success metrics; the spec explicitly requires OCR fallba
ck to be excluded from barcode hit-rate accounting. [Task 1]
296:# Task Group: D:\newhuawei HuaweiOCR optimization audit
297:scope: evidence-backed optimization review for the current HuaweiOCR recognition pipeline, especia
lly when deciding whether to tune OCR, barcode candidate selection, or result gating next.
298:applies_to: cwd=D:\newhuawei; reuse_rule=reuse for this checkout's current algorithm-review questi
ons; re-check current `validation/` outputs before claiming the same bottleneck still holds.
304:- rollout_summaries/2026-04-19T08-46-25-2Yg2-huaweiocr_optimization_audit_and_prioritized_improvem
ents.md (cwd=\\?\D:\newhuawei, rollout_path=C:\Users\55093\.codex\sessions\2026\04\19\rollout-2026-04-
19T01-46-25-019da4eb-6dcb-7a82-9355-cefa7289c6e2.jsonl, updated_at=2026-04-20T17:50:54+00:00, thread_i
d=019da4eb-6dcb-7a82-9355-cefa7289c6e2, prioritized audit tied to code and validation artifacts)
308:- D:\newhuawei, HuaweiOCR, optimization audit, crop.py, scan2.py, barcode.py, sn_barcode.py, ocr.p
y, validation/pure_ocr_6_20260419_223358, validation/pure_ocr_6_server_rec_20260419_224312, mobile, se
rver_rec, barcode-first, SN mismatch
317:- The observed 6-label validation did not support "swap OCR model first" as the highest-leverage m
ove. `mobile` OCR was already very strong (`model_exact_vs_barcode_reparsed 24/24`, `sn_exact_vs_barco
de 32/32`), while `server_rec` had one SN mismatch (`31/32`). [Task 1]
318:- `scan2.py` is the decision layer with the most leverage because it decides between barcode and O
CR for `model` and `sn`; this is the first place to inspect before changing OCR internals. [Task 1]
319:- `sn_barcode.py` already has a heavy candidate/decoder cascade, so likely gains are in candidate 
ordering, pruning, caching, and early-exit behavior rather than adding more blind decode attempts. [Ta
sk 1]

codex
已拿到与该仓库相关的历史约束（SN 条码命中率与 OCR fallback 分离）。接下来我会直接审阅当前 git 变更内容并定位可执行修复点。
exec
"C:\\Program Files\\WindowsApps\\Microsoft.PowerShell_7.6.0.0_x64__8wekyb3d8bbwe\\pwsh.exe" -Command '
git status --porcelain=v1' in D:\newhuawei
exec
"C:\\Program Files\\WindowsApps\\Microsoft.PowerShell_7.6.0.0_x64__8wekyb3d8bbwe\\pwsh.exe" -Command '
git diff --stat' in D:\newhuawei
exec
"C:\\Program Files\\WindowsApps\\Microsoft.PowerShell_7.6.0.0_x64__8wekyb3d8bbwe\\pwsh.exe" -Command '
git diff --cached --stat' in D:\newhuawei
 succeeded in 691ms:
 gui_app.py                        | 177 +++++++++++++++++++----
 gui_pipeline.py                   |  13 +-
 ocr.py                            |  10 +-
 scan2.py                          | 189 +++++++++++++++++++++---
 sn_barcode.py                     | 155 +++++++++++++++++++-
 tests/test_locked_output_dirs.py  | 107 ++++++++++++--
 tests/test_sn_barcode_scanning.py | 295 ++++++++++++++++++++++++++++++++++++++
 7 files changed, 886 insertions(+), 60 deletions(-)
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

 succeeded in 709ms:
 succeeded in 719ms:
 M gui_app.py
 M gui_pipeline.py
 M ocr.py
 M scan2.py
 M sn_barcode.py
 M tests/test_locked_output_dirs.py
 M tests/test_sn_barcode_scanning.py
?? validation/pure_ocr_6_20260419_223358/
?? validation/pure_ocr_6_server_rec_20260419_224312/

codex
当前只有未暂存改动、没有 staged。下面逐文件看 diff（含测试），重点找回归风险和测试缺口。
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
 succeeded in 706ms:
diff --git a/scan2.py b/scan2.py
index 205c2b3..2ad3d5d 100644
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
 
@@ -940,7 +1087,7 @@ def main(out_dir=None, model_dir=None, sn_dir=None, out_jsonl=None, debug_log=No
                 stats["sn_total"] += 1
                 barcode_status = sn_meta.get("barcode_status", "not_attempted")
                 stats["sn_barcode_attempts"] += int(sn_meta.get("barcode_attempts", 0) or 0)
-                if sn_src == "barcode":
+                if sn_src in {"barcode", "barcode_unmatched", "barcode_ocr_agree"}:
                     stats["sn_barcode_hits"] += 1
                 elif sn_src.startswith("ocr") and barcode_status in {
                     "decoder_miss",
warning: in the working copy of 'scan2.py', LF will be replaced by CRLF the next time Git touches it

 succeeded in 710ms:
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

 succeeded in 738ms:
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

 succeeded in 776ms:
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

exec
"C:\\Program Files\\WindowsApps\\Microsoft.PowerShell_7.6.0.0_x64__8wekyb3d8bbwe\\pwsh.exe" -Command '
git diff -- sn_barcode.py' in D:\newhuawei
exec
"C:\\Program Files\\WindowsApps\\Microsoft.PowerShell_7.6.0.0_x64__8wekyb3d8bbwe\\pwsh.exe" -Command '
git diff -- tests/test_locked_output_dirs.py' in D:\newhuawei
exec
"C:\\Program Files\\WindowsApps\\Microsoft.PowerShell_7.6.0.0_x64__8wekyb3d8bbwe\\pwsh.exe" -Command '
git diff -- tests/test_sn_barcode_scanning.py' in D:\newhuawei
 succeeded in 665ms:
diff --git a/tests/test_locked_output_dirs.py b/tests/test_locked_output_dirs.py
index 6285eee..4e70450 100644
--- a/tests/test_locked_output_dirs.py
+++ b/tests/test_locked_output_dirs.py
@@ -813,6 +813,22 @@ class CropTempFileTest(unittest.TestCase):
 
 
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
 
@@ -837,6 +853,7 @@ class GuiPipelineTest(unittest.TestCase):
             self.assertEqual(len(names), 2)
             self.assertEqual(len(set(names)), 2)
             self.assertEqual(names, ["input_0001.png", "input_0002.png"])
+            self.assertEqual([record["source_name"] for record in records], ["same.png", "same.png"])
             self.assertTrue(os.path.exists(os.path.join(run_dir, names[0])))
             self.assertTrue(os.path.exists(os.path.join(run_dir, names[1])))
             with open(os.path.join(run_dir, "source_manifest.jsonl"), "r", encoding="utf-8") as f:
@@ -852,15 +869,85 @@ class GuiPipelineTest(unittest.TestCase):
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
 
@@ -868,7 +955,7 @@ class GuiPipelineTest(unittest.TestCase):
         self.assertNotIn("scan2", sys.modules)
 
     def test_gui_log_mask_keeps_app_relative_output_paths(self):
-        sys.modules.pop("gui_app", None)
+        self._reset_gui_import_modules()
         import gui_app
 
         app_path = os.path.join(os.getcwd(), "stage2_fields", "manifest.jsonl")
@@ -883,7 +970,7 @@ class GuiPipelineTest(unittest.TestCase):
         self.assertNotIn(external_path, masked)
 
     def test_gui_run_pipeline_requests_clean_crop_outputs(self):
-        sys.modules.pop("gui_app", None)
+        self._reset_gui_import_modules()
         import gui_app
 
         source = inspect.getsource(gui_app.App.run_pipeline)
warning: in the working copy of 'tests/test_locked_output_dirs.py', LF will be replaced by CRLF the ne
xt time Git touches it

 succeeded in 682ms:
diff --git a/tests/test_sn_barcode_scanning.py b/tests/test_sn_barcode_scanning.py
index 90fd696..8a90ae5 100644
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
 
@@ -180,6 +401,80 @@ class Scan2BarcodeAccountingTest(unittest.TestCase):
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
+    def test_main_counts_barcode_ocr_agree_as_barcode_hit(self):
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
+        self.assertEqual(stats["sn_barcode_hits"], 1)
+        self.assertEqual(stats["sn_barcode_parse_failures"], 1)
+
 
 class ValidationCommandTest(unittest.TestCase):
     def test_template_builder_creates_manual_review_rows_without_accepting_them(self):
warning: in the working copy of 'tests/test_sn_barcode_scanning.py', LF will be replaced by CRLF the n
ext time Git touches it

 succeeded in 716ms:
diff --git a/sn_barcode.py b/sn_barcode.py
index e8927cf..17e7dc3 100644
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
@@ -31,6 +32,102 @@ DEFAULT_MIN_BARCODE_HEIGHT = 22
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
+    os.makedirs(os.path.dirname(path), exist_ok=True)
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
@@ -140,7 +237,7 @@ def extract_sn_from_payload(value: str) -> str:
     if NON_SN_PREFIX_RE.match(cleaned):
         return ""
 
-    m = SN20_RE.search(cleaned)
+    m = _sn20_regex().search(cleaned)
     if m and len(m.group(1)) == 20:
         return m.group(1)
 
@@ -164,6 +261,42 @@ def _source_rank(source_region: str) -> int:
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
@@ -235,8 +368,26 @@ def select_sn_from_decoder_results(results: Iterable[DecoderResult]) -> SnBarcod
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

exec
"C:\\Program Files\\WindowsApps\\Microsoft.PowerShell_7.6.0.0_x64__8wekyb3d8bbwe\\pwsh.exe" -Command "
Get-ChildItem -Recurse -File validation\\pure_ocr_6_20260419_223358, validation\\pure_ocr_6_server_rec
_20260419_224312 | Select-Object -ExpandProperty FullName" in D:\newhuawei
 succeeded in 723ms:
D:\newhuawei\validation\pure_ocr_6_20260419_223358\pure_ocr_results_reparsed.csv
D:\newhuawei\validation\pure_ocr_6_20260419_223358\pure_ocr_results_reparsed.jsonl
D:\newhuawei\validation\pure_ocr_6_20260419_223358\pure_ocr_results.csv
D:\newhuawei\validation\pure_ocr_6_20260419_223358\pure_ocr_results.jsonl
D:\newhuawei\validation\pure_ocr_6_20260419_223358\pure_ocr_summary_reparsed.json
D:\newhuawei\validation\pure_ocr_6_20260419_223358\input\1.jpg
D:\newhuawei\validation\pure_ocr_6_20260419_223358\input\2.jpg
D:\newhuawei\validation\pure_ocr_6_20260419_223358\input\3.jpg
D:\newhuawei\validation\pure_ocr_6_20260419_223358\input\4.jpg
D:\newhuawei\validation\pure_ocr_6_20260419_223358\input\5.jpg
D:\newhuawei\validation\pure_ocr_6_20260419_223358\input\6.jpg
D:\newhuawei\validation\pure_ocr_6_20260419_223358\stage1_labels\1.jpg__label_1.png
D:\newhuawei\validation\pure_ocr_6_20260419_223358\stage1_labels\1.jpg__label_2.png
D:\newhuawei\validation\pure_ocr_6_20260419_223358\stage1_labels\1.jpg__label_3.png
D:\newhuawei\validation\pure_ocr_6_20260419_223358\stage1_labels\1.jpg__label_4.png
D:\newhuawei\validation\pure_ocr_6_20260419_223358\stage1_labels\1.jpg__label_5.png
D:\newhuawei\validation\pure_ocr_6_20260419_223358\stage1_labels\1.jpg__label_6.png
D:\newhuawei\validation\pure_ocr_6_20260419_223358\stage1_labels\1.jpg__label_7.png
D:\newhuawei\validation\pure_ocr_6_20260419_223358\stage1_labels\2.jpg__label_1.png
D:\newhuawei\validation\pure_ocr_6_20260419_223358\stage1_labels\2.jpg__label_2.png
D:\newhuawei\validation\pure_ocr_6_20260419_223358\stage1_labels\2.jpg__label_3.png
D:\newhuawei\validation\pure_ocr_6_20260419_223358\stage1_labels\2.jpg__label_4.png
D:\newhuawei\validation\pure_ocr_6_20260419_223358\stage1_labels\2.jpg__label_5.png
D:\newhuawei\validation\pure_ocr_6_20260419_223358\stage1_labels\2.jpg__label_6.png
D:\newhuawei\validation\pure_ocr_6_20260419_223358\stage1_labels\3.jpg__label_1.png
D:\newhuawei\validation\pure_ocr_6_20260419_223358\stage1_labels\3.jpg__label_2.png
D:\newhuawei\validation\pure_ocr_6_20260419_223358\stage1_labels\3.jpg__label_3.png
D:\newhuawei\validation\pure_ocr_6_20260419_223358\stage1_labels\3.jpg__label_4.png
D:\newhuawei\validation\pure_ocr_6_20260419_223358\stage1_labels\4.jpg__label_1.png
D:\newhuawei\validation\pure_ocr_6_20260419_223358\stage1_labels\4.jpg__label_2.png
D:\newhuawei\validation\pure_ocr_6_20260419_223358\stage1_labels\4.jpg__label_3.png
D:\newhuawei\validation\pure_ocr_6_20260419_223358\stage1_labels\4.jpg__label_4.png
D:\newhuawei\validation\pure_ocr_6_20260419_223358\stage1_labels\4.jpg__label_5.png
D:\newhuawei\validation\pure_ocr_6_20260419_223358\stage1_labels\4.jpg__label_6.png
D:\newhuawei\validation\pure_ocr_6_20260419_223358\stage1_labels\5.jpg__label_1.png
D:\newhuawei\validation\pure_ocr_6_20260419_223358\stage1_labels\5.jpg__label_2.png
D:\newhuawei\validation\pure_ocr_6_20260419_223358\stage1_labels\5.jpg__label_3.png
D:\newhuawei\validation\pure_ocr_6_20260419_223358\stage1_labels\5.jpg__label_4.png
D:\newhuawei\validation\pure_ocr_6_20260419_223358\stage1_labels\5.jpg__label_5.png
D:\newhuawei\validation\pure_ocr_6_20260419_223358\stage1_labels\6.jpg__label_1.png
D:\newhuawei\validation\pure_ocr_6_20260419_223358\stage1_labels\6.jpg__label_2.png
D:\newhuawei\validation\pure_ocr_6_20260419_223358\stage1_labels\6.jpg__label_3.png
D:\newhuawei\validation\pure_ocr_6_20260419_223358\stage1_labels\6.jpg__label_4.png
D:\newhuawei\validation\pure_ocr_6_20260419_223358\stage1_labels\6.jpg__label_5.png
D:\newhuawei\validation\pure_ocr_6_20260419_223358\stage2_fields\manifest.jsonl
D:\newhuawei\validation\pure_ocr_6_20260419_223358\stage2_fields\model\1.jpg__label_1__model.png
D:\newhuawei\validation\pure_ocr_6_20260419_223358\stage2_fields\model\1.jpg__label_2__model.png
D:\newhuawei\validation\pure_ocr_6_20260419_223358\stage2_fields\model\1.jpg__label_3__model.png
D:\newhuawei\validation\pure_ocr_6_20260419_223358\stage2_fields\model\1.jpg__label_4__model.png
D:\newhuawei\validation\pure_ocr_6_20260419_223358\stage2_fields\model\1.jpg__label_5__model.png
D:\newhuawei\validation\pure_ocr_6_20260419_223358\stage2_fields\model\1.jpg__label_6__model.png
D:\newhuawei\validation\pure_ocr_6_20260419_223358\stage2_fields\model\1.jpg__label_7__model.png
D:\newhuawei\validation\pure_ocr_6_20260419_223358\stage2_fields\model\2.jpg__label_1__model.png
D:\newhuawei\validation\pure_ocr_6_20260419_223358\stage2_fields\model\2.jpg__label_2__model.png
D:\newhuawei\validation\pure_ocr_6_20260419_223358\stage2_fields\model\2.jpg__label_3__model.png
D:\newhuawei\validation\pure_ocr_6_20260419_223358\stage2_fields\model\2.jpg__label_4__model.png
D:\newhuawei\validation\pure_ocr_6_20260419_223358\stage2_fields\model\2.jpg__label_5__model.png
D:\newhuawei\validation\pure_ocr_6_20260419_223358\stage2_fields\model\2.jpg__label_6__model.png
D:\newhuawei\validation\pure_ocr_6_20260419_223358\stage2_fields\model\3.jpg__label_1__model.png
D:\newhuawei\validation\pure_ocr_6_20260419_223358\stage2_fields\model\3.jpg__label_2__model.png
D:\newhuawei\validation\pure_ocr_6_20260419_223358\stage2_fields\model\3.jpg__label_3__model.png
D:\newhuawei\validation\pure_ocr_6_20260419_223358\stage2_fields\model\3.jpg__label_4__model.png
D:\newhuawei\validation\pure_ocr_6_20260419_223358\stage2_fields\model\4.jpg__label_1__model.png
D:\newhuawei\validation\pure_ocr_6_20260419_223358\stage2_fields\model\4.jpg__label_2__model.png
D:\newhuawei\validation\pure_ocr_6_20260419_223358\stage2_fields\model\4.jpg__label_3__model.png
D:\newhuawei\validation\pure_ocr_6_20260419_223358\stage2_fields\model\4.jpg__label_4__model.png
D:\newhuawei\validation\pure_ocr_6_20260419_223358\stage2_fields\model\4.jpg__label_5__model.png
D:\newhuawei\validation\pure_ocr_6_20260419_223358\stage2_fields\model\4.jpg__label_6__model.png
D:\newhuawei\validation\pure_ocr_6_20260419_223358\stage2_fields\model\5.jpg__label_1__model.png
D:\newhuawei\validation\pure_ocr_6_20260419_223358\stage2_fields\model\5.jpg__label_2__model.png
D:\newhuawei\validation\pure_ocr_6_20260419_223358\stage2_fields\model\5.jpg__label_3__model.png
D:\newhuawei\validation\pure_ocr_6_20260419_223358\stage2_fields\model\5.jpg__label_4__model.png
D:\newhuawei\validation\pure_ocr_6_20260419_223358\stage2_fields\model\5.jpg__label_5__model.png
D:\newhuawei\validation\pure_ocr_6_20260419_223358\stage2_fields\model\6.jpg__label_1__model.png
D:\newhuawei\validation\pure_ocr_6_20260419_223358\stage2_fields\model\6.jpg__label_2__model.png
D:\newhuawei\validation\pure_ocr_6_20260419_223358\stage2_fields\model\6.jpg__label_3__model.png
D:\newhuawei\validation\pure_ocr_6_20260419_223358\stage2_fields\model\6.jpg__label_4__model.png
D:\newhuawei\validation\pure_ocr_6_20260419_223358\stage2_fields\model\6.jpg__label_5__model.png
D:\newhuawei\validation\pure_ocr_6_20260419_223358\stage2_fields\sn\1.jpg__label_1__sn.png
D:\newhuawei\validation\pure_ocr_6_20260419_223358\stage2_fields\sn\1.jpg__label_2__sn.png
D:\newhuawei\validation\pure_ocr_6_20260419_223358\stage2_fields\sn\1.jpg__label_3__sn.png
D:\newhuawei\validation\pure_ocr_6_20260419_223358\stage2_fields\sn\1.jpg__label_4__sn.png
D:\newhuawei\validation\pure_ocr_6_20260419_223358\stage2_fields\sn\1.jpg__label_5__sn.png
D:\newhuawei\validation\pure_ocr_6_20260419_223358\stage2_fields\sn\1.jpg__label_6__sn.png
D:\newhuawei\validation\pure_ocr_6_20260419_223358\stage2_fields\sn\1.jpg__label_7__sn.png
D:\newhuawei\validation\pure_ocr_6_20260419_223358\stage2_fields\sn\2.jpg__label_1__sn.png
D:\newhuawei\validation\pure_ocr_6_20260419_223358\stage2_fields\sn\2.jpg__label_2__sn.png
D:\newhuawei\validation\pure_ocr_6_20260419_223358\stage2_fields\sn\2.jpg__label_3__sn.png
D:\newhuawei\validation\pure_ocr_6_20260419_223358\stage2_fields\sn\2.jpg__label_4__sn.png
D:\newhuawei\validation\pure_ocr_6_20260419_223358\stage2_fields\sn\2.jpg__label_5__sn.png
D:\newhuawei\validation\pure_ocr_6_20260419_223358\stage2_fields\sn\2.jpg__label_6__sn.png
D:\newhuawei\validation\pure_ocr_6_20260419_223358\stage2_fields\sn\3.jpg__label_1__sn.png
D:\newhuawei\validation\pure_ocr_6_20260419_223358\stage2_fields\sn\3.jpg__label_2__sn.png
D:\newhuawei\validation\pure_ocr_6_20260419_223358\stage2_fields\sn\3.jpg__label_3__sn.png
D:\newhuawei\validation\pure_ocr_6_20260419_223358\stage2_fields\sn\3.jpg__label_4__sn.png
D:\newhuawei\validation\pure_ocr_6_20260419_223358\stage2_fields\sn\4.jpg__label_1__sn.png
D:\newhuawei\validation\pure_ocr_6_20260419_223358\stage2_fields\sn\4.jpg__label_2__sn.png
D:\newhuawei\validation\pure_ocr_6_20260419_223358\stage2_fields\sn\4.jpg__label_3__sn.png
D:\newhuawei\validation\pure_ocr_6_20260419_223358\stage2_fields\sn\4.jpg__label_4__sn.png
D:\newhuawei\validation\pure_ocr_6_20260419_223358\stage2_fields\sn\4.jpg__label_5__sn.png
D:\newhuawei\validation\pure_ocr_6_20260419_223358\stage2_fields\sn\4.jpg__label_6__sn.png
D:\newhuawei\validation\pure_ocr_6_20260419_223358\stage2_fields\sn\5.jpg__label_1__sn.png
D:\newhuawei\validation\pure_ocr_6_20260419_223358\stage2_fields\sn\5.jpg__label_2__sn.png
D:\newhuawei\validation\pure_ocr_6_20260419_223358\stage2_fields\sn\5.jpg__label_3__sn.png
D:\newhuawei\validation\pure_ocr_6_20260419_223358\stage2_fields\sn\5.jpg__label_4__sn.png
D:\newhuawei\validation\pure_ocr_6_20260419_223358\stage2_fields\sn\5.jpg__label_5__sn.png
D:\newhuawei\validation\pure_ocr_6_20260419_223358\stage2_fields\sn\6.jpg__label_1__sn.png
D:\newhuawei\validation\pure_ocr_6_20260419_223358\stage2_fields\sn\6.jpg__label_2__sn.png
D:\newhuawei\validation\pure_ocr_6_20260419_223358\stage2_fields\sn\6.jpg__label_3__sn.png
D:\newhuawei\validation\pure_ocr_6_20260419_223358\stage2_fields\sn\6.jpg__label_4__sn.png
D:\newhuawei\validation\pure_ocr_6_20260419_223358\stage2_fields\sn\6.jpg__label_5__sn.png
D:\newhuawei\validation\pure_ocr_6_server_rec_20260419_224312\pure_ocr_server_rec_results.csv
D:\newhuawei\validation\pure_ocr_6_server_rec_20260419_224312\pure_ocr_server_rec_results.jsonl
D:\newhuawei\validation\pure_ocr_6_server_rec_20260419_224312\summary.json

codex
我已经完成静态 diff 审阅，接下来跑受影响测试文件确认这些改动是否自证，并据此判断是否存在“缺失关键测试”的门禁问题。
exec
"C:\\Program Files\\WindowsApps\\Microsoft.PowerShell_7.6.0.0_x64__8wekyb3d8bbwe\\pwsh.exe" -Command '
python -m pytest tests/test_sn_barcode_scanning.py tests/test_locked_output_dirs.py -q' in D:\newhuawe
i
2026-04-23T19:23:31.630744Z ERROR codex_core::tools::router: error=Exit code: 1
Wall time: 2.2 seconds
Output:
.................................................................FFF     [100%]
================================== FAILURES ===================================
__________ AppPathsInstallTest.test_incomplete_model_dir_is_replaced __________

self = <tests.test_locked_output_dirs.AppPathsInstallTest testMethod=test_incomplete_model_dir_is_repl
aced>

    def test_incomplete_model_dir_is_replaced(self):
        import app_paths
    
        with tempfile.TemporaryDirectory() as root:
            bundled = os.path.join(root, "bundled", "models", "official_models")
            source_model = os.path.join(bundled, "model_a")
            os.makedirs(source_model)
            with open(os.path.join(source_model, "weights.bin"), "wb") as f:
                f.write(b"complete")
    
            data_dir = os.path.join(root, "data")
            target = os.path.join(data_dir, "models", "official_models", "model_a")
            os.makedirs(target)
            with open(os.path.join(target, "partial.bin"), "wb") as f:
                f.write(b"partial")
    
            def fake_resource_path(*parts):
                return os.path.join(root, "bundled", *parts)
    
>           with mock.patch.object(app_paths, "get_resource_path", side_effect=fake_resource_path):
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

tests\test_locked_output_dirs.py:1118: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
C:\Users\55093\AppData\Local\Programs\Python\Python312\Lib\unittest\mock.py:1467: in __enter__
    original, local = self.get_original()
                      ^^^^^^^^^^^^^^^^^^^
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _

self = <unittest.mock._patch object at 0x0000024E380FB740>

    def get_original(self):
        target = self.getter()
        name = self.attribute
    
        original = DEFAULT
        local = False
    
        try:
            original = target.__dict__[name]
        except (AttributeError, KeyError):
            original = getattr(target, name, DEFAULT)
        else:
            local = True
    
        if name in _builtins and isinstance(target, ModuleType):
            self.create = True
    
        if not self.create and original is DEFAULT:
>           raise AttributeError(
                "%s does not have the attribute %r" % (target, name)
            )
E           AttributeError: <module 'app_paths'> does not have the attribute 'get_resource_path'

C:\Users\55093\AppData\Local\Programs\Python\Python312\Lib\unittest\mock.py:1437: AttributeError
__ AppPathsInstallTest.test_stale_lock_reclaim_does_not_remove_changed_lock ___

self = <tests.test_locked_output_dirs.AppPathsInstallTest testMethod=test_stale_lock_reclaim_does_not_
remove_changed_lock>

    def test_stale_lock_reclaim_does_not_remove_changed_lock(self):
        import app_paths
    
        with tempfile.TemporaryDirectory() as root:
            lock_path = os.path.join(root, ".huaweiocr_install.lock")
            with open(lock_path, "w", encoding="utf-8") as f:
                f.write("")
            old = time.time() - 10
            os.utime(lock_path, (old, old))
>           real_read = app_paths._read_lock_snapshot
                        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E           AttributeError: module 'app_paths' has no attribute '_read_lock_snapshot'

tests\test_locked_output_dirs.py:1166: AttributeError
_______ AppPathsInstallTest.test_stale_model_install_lock_is_recovered ________

self = <tests.test_locked_output_dirs.AppPathsInstallTest testMethod=test_stale_model_install_lock_is_
recovered>

    def test_stale_model_install_lock_is_recovered(self):
        import app_paths
    
        with tempfile.TemporaryDirectory() as root:
            bundled = os.path.join(root, "bundled", "models", "official_models")
            source_model = os.path.join(bundled, "model_a")
            os.makedirs(source_model)
            with open(os.path.join(source_model, "weights.bin"), "wb") as f:
                f.write(b"complete")
    
            data_dir = os.path.join(root, "data")
            target_root = os.path.join(data_dir, "models", "official_models")
            os.makedirs(target_root)
            lock_path = os.path.join(target_root, ".huaweiocr_install.lock")
            with open(lock_path, "w", encoding="utf-8") as f:
                f.write("")
            old = time.time() - 10
            os.utime(lock_path, (old, old))
    
            def fake_resource_path(*parts):
                return os.path.join(root, "bundled", *parts)
    
>           with mock.patch.object(app_paths, "get_resource_path", side_effect=fake_resource_path):
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

tests\test_locked_output_dirs.py:1148: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
C:\Users\55093\AppData\Local\Programs\Python\Python312\Lib\unittest\mock.py:1467: in __enter__
    original, local = self.get_original()
                      ^^^^^^^^^^^^^^^^^^^
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _

self = <unittest.mock._patch object at 0x0000024E384A15E0>

    def get_original(self):
        target = self.getter()
        name = self.attribute
    
        original = DEFAULT
        local = False
    
        try:
            original = target.__dict__[name]
        except (AttributeError, KeyError):
            original = getattr(target, name, DEFAULT)
        else:
            local = True
    
        if name in _builtins and isinstance(target, ModuleType):
            self.create = True
    
        if not self.create and original is DEFAULT:
>           raise AttributeError(
                "%s does not have the attribute %r" % (target, name)
            )
E           AttributeError: <module 'app_paths'> does not have the attribute 'get_resource_path'

C:\Users\55093\AppData\Local\Programs\Python\Python312\Lib\unittest\mock.py:1437: AttributeError
============================== warnings summary ===============================
tests/test_locked_output_dirs.py::GuiPipelineTest::test_gui_display_label_keeps_uniqueness_for_duplica
te_source_names
  C:\Users\55093\AppData\Local\Programs\Python\Python312\Lib\site-packages\openpyxl\compat\numbers.py:
9: UserWarning: The NumPy module was reloaded (imported a second time). This can in some cases result 
in small but subtle issues and is discouraged.
    import numpy

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
=========================== short test summary info ===========================
FAILED tests/test_locked_output_dirs.py::AppPathsInstallTest::test_incomplete_model_dir_is_replaced
FAILED tests/test_locked_output_dirs.py::AppPathsInstallTest::test_stale_lock_reclaim_does_not_remove_
changed_lock
FAILED tests/test_locked_output_dirs.py::AppPathsInstallTest::test_stale_model_install_lock_is_recover
ed
3 failed, 65 passed, 1 warning in 0.70s

 exited 1 in 2158ms:
.................................................................FFF     [100%]
================================== FAILURES ===================================
__________ AppPathsInstallTest.test_incomplete_model_dir_is_replaced __________

self = <tests.test_locked_output_dirs.AppPathsInstallTest testMethod=test_incomplete_model_dir_is_repl
aced>

    def test_incomplete_model_dir_is_replaced(self):
        import app_paths
    
        with tempfile.TemporaryDirectory() as root:
            bundled = os.path.join(root, "bundled", "models", "official_models")
            source_model = os.path.join(bundled, "model_a")
            os.makedirs(source_model)
            with open(os.path.join(source_model, "weights.bin"), "wb") as f:
                f.write(b"complete")
    
            data_dir = os.path.join(root, "data")
            target = os.path.join(data_dir, "models", "official_models", "model_a")
            os.makedirs(target)
            with open(os.path.join(target, "partial.bin"), "wb") as f:
                f.write(b"partial")
    
            def fake_resource_path(*parts):
                return os.path.join(root, "bundled", *parts)
    
>           with mock.patch.object(app_paths, "get_resource_path", side_effect=fake_resource_path):
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

tests\test_locked_output_dirs.py:1118: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
C:\Users\55093\AppData\Local\Programs\Python\Python312\Lib\unittest\mock.py:1467: in __enter__
    original, local = self.get_original()
                      ^^^^^^^^^^^^^^^^^^^
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _

self = <unittest.mock._patch object at 0x0000024E380FB740>

    def get_original(self):
        target = self.getter()
        name = self.attribute
    
        original = DEFAULT
        local = False
    
        try:
            original = target.__dict__[name]
        except (AttributeError, KeyError):
            original = getattr(target, name, DEFAULT)
        else:
            local = True
    
        if name in _builtins and isinstance(target, ModuleType):
            self.create = True
    
        if not self.create and original is DEFAULT:
>           raise AttributeError(
                "%s does not have the attribute %r" % (target, name)
            )
E           AttributeError: <module 'app_paths'> does not have the attribute 'get_resource_path'

C:\Users\55093\AppData\Local\Programs\Python\Python312\Lib\unittest\mock.py:1437: AttributeError
__ AppPathsInstallTest.test_stale_lock_reclaim_does_not_remove_changed_lock ___

self = <tests.test_locked_output_dirs.AppPathsInstallTest testMethod=test_stale_lock_reclaim_does_not_
remove_changed_lock>

    def test_stale_lock_reclaim_does_not_remove_changed_lock(self):
        import app_paths
    
        with tempfile.TemporaryDirectory() as root:
            lock_path = os.path.join(root, ".huaweiocr_install.lock")
            with open(lock_path, "w", encoding="utf-8") as f:
                f.write("")
            old = time.time() - 10
            os.utime(lock_path, (old, old))
>           real_read = app_paths._read_lock_snapshot
                        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E           AttributeError: module 'app_paths' has no attribute '_read_lock_snapshot'

tests\test_locked_output_dirs.py:1166: AttributeError
_______ AppPathsInstallTest.test_stale_model_install_lock_is_recovered ________

self = <tests.test_locked_output_dirs.AppPathsInstallTest testMethod=test_stale_model_install_lock_is_
recovered>

    def test_stale_model_install_lock_is_recovered(self):
        import app_paths
    
        with tempfile.TemporaryDirectory() as root:
            bundled = os.path.join(root, "bundled", "models", "official_models")
            source_model = os.path.join(bundled, "model_a")
            os.makedirs(source_model)
            with open(os.path.join(source_model, "weights.bin"), "wb") as f:
                f.write(b"complete")
    
            data_dir = os.path.join(root, "data")
            target_root = os.path.join(data_dir, "models", "official_models")
            os.makedirs(target_root)
            lock_path = os.path.join(target_root, ".huaweiocr_install.lock")
            with open(lock_path, "w", encoding="utf-8") as f:
                f.write("")
            old = time.time() - 10
            os.utime(lock_path, (old, old))
    
            def fake_resource_path(*parts):
                return os.path.join(root, "bundled", *parts)
    
>           with mock.patch.object(app_paths, "get_resource_path", side_effect=fake_resource_path):
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

tests\test_locked_output_dirs.py:1148: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
C:\Users\55093\AppData\Local\Programs\Python\Python312\Lib\unittest\mock.py:1467: in __enter__
    original, local = self.get_original()
                      ^^^^^^^^^^^^^^^^^^^
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _

self = <unittest.mock._patch object at 0x0000024E384A15E0>

    def get_original(self):
        target = self.getter()
        name = self.attribute
    
        original = DEFAULT
        local = False
    
        try:
            original = target.__dict__[name]
        except (AttributeError, KeyError):
            original = getattr(target, name, DEFAULT)
        else:
            local = True
    
        if name in _builtins and isinstance(target, ModuleType):
            self.create = True
    
        if not self.create and original is DEFAULT:
>           raise AttributeError(
                "%s does not have the attribute %r" % (target, name)
            )
E           AttributeError: <module 'app_paths'> does not have the attribute 'get_resource_path'

C:\Users\55093\AppData\Local\Programs\Python\Python312\Lib\unittest\mock.py:1437: AttributeError
============================== warnings summary ===============================
tests/test_locked_output_dirs.py::GuiPipelineTest::test_gui_display_label_keeps_uniqueness_for_duplica
te_source_names
  C:\Users\55093\AppData\Local\Programs\Python\Python312\Lib\site-packages\openpyxl\compat\numbers.py:
9: UserWarning: The NumPy module was reloaded (imported a second time). This can in some cases result 
in small but subtle issues and is discouraged.
    import numpy

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
=========================== short test summary info ===========================
FAILED tests/test_locked_output_dirs.py::AppPathsInstallTest::test_incomplete_model_dir_is_replaced
FAILED tests/test_locked_output_dirs.py::AppPathsInstallTest::test_stale_lock_reclaim_does_not_remove_
changed_lock
FAILED tests/test_locked_output_dirs.py::AppPathsInstallTest::test_stale_model_install_lock_is_recover
ed
3 failed, 65 passed, 1 warning in 0.70s

codex
我已定位到一个明确的实现缺陷和一个提交流程风险，正在补充精确行号后给出最终 gate 结论。
exec
"C:\\Program Files\\WindowsApps\\Microsoft.PowerShell_7.6.0.0_x64__8wekyb3d8bbwe\\pwsh.exe" -Command "
rg -n \"os\\.makedirs\\(os\\.path\\.dirname\\(path\\), exist_ok=True\\)\" sn_barcode.py" in D:\newhuaw
ei
exec
"C:\\Program Files\\WindowsApps\\Microsoft.PowerShell_7.6.0.0_x64__8wekyb3d8bbwe\\pwsh.exe" -Command "
rg -n \"\\?\\? validation/|validation/pure_ocr_6\" -n .gitignore gitignore* 2>"'$null' in D:\newhuawei
exec
"C:\\Program Files\\WindowsApps\\Microsoft.PowerShell_7.6.0.0_x64__8wekyb3d8bbwe\\pwsh.exe" -Command '
rg -n "HUAWEIOCR_OCR_LANG|_recognition_model_candidates|OCR_LANG" tests ocr.py' in D:\newhuawei
2026-04-23T19:23:58.558108Z ERROR codex_core::tools::router: error=Exit code: 1
Wall time: 0.7 seconds
Output:

 exited 1 in 725ms:
 succeeded in 692ms:
ocr.py:28:#   - 如果可能有中文：设置 HUAWEIOCR_OCR_LANG=ch
ocr.py:29:OCR_LANG = os.environ.get("HUAWEIOCR_OCR_LANG", "en").strip().lower()
ocr.py:30:if OCR_LANG not in {"en", "ch"}:
ocr.py:31:    OCR_LANG = "en"
ocr.py:112:def _recognition_model_candidates():
ocr.py:113:    if OCR_LANG == "ch":
ocr.py:134:    rec_dir = _first_existing_model_dir(model_root, _recognition_model_candidates())
ocr.py:174:        OCR_LANG, device
ocr.py:177:        lang=OCR_LANG,

 succeeded in 762ms:
120:    os.makedirs(os.path.dirname(path), exist_ok=True)

exec
"C:\\Program Files\\WindowsApps\\Microsoft.PowerShell_7.6.0.0_x64__8wekyb3d8bbwe\\pwsh.exe" -Command '
rg -n "sn_barcode_hits|barcode_ocr_agree|barcode_unmatched" scan2.py' in D:\newhuawei
 succeeded in 665ms:
435:        "barcode_ocr_agree",
736:    barcode_unmatched_sn = ""
738:        barcode_unmatched_sn = _extract_unmatched_barcode_sn(barcode_report.raw_text)
756:            if barcode_unmatched_sn:
758:                    barcode_unmatched_sn,
760:                    "barcode_unmatched",
769:        if barcode_unmatched_sn:
771:                barcode_unmatched_sn,
773:                "barcode_unmatched",
795:        if barcode_unmatched_sn and sn == barcode_unmatched_sn:
800:                "barcode_ocr_agree",
805:        barcode_unmatched_sn,
822:        if barcode_unmatched_sn and sn == barcode_unmatched_sn:
827:                "barcode_ocr_agree",
832:        barcode_unmatched_sn,
847:        if barcode_unmatched_sn and sn == barcode_unmatched_sn:
852:                "barcode_ocr_agree",
857:        barcode_unmatched_sn,
870:            if barcode_unmatched_sn:
872:                    barcode_unmatched_sn,
874:                    "barcode_unmatched",
975:        "sn_barcode_hits": 0,
1090:                if sn_src in {"barcode", "barcode_unmatched", "barcode_ocr_agree"}:
1091:                    stats["sn_barcode_hits"] += 1
1120:        stats["sn_barcode_hit_rate"] = stats["sn_barcode_hits"] / float(stats["sn_total"])

codex
GATE_DECISION: FAIL

- `scan2.py:1090` Regression risk: `sn_barcode_hits` now counts `barcode_ocr_agree` as a barcode hit, 
but that source explicitly depends on OCR confirmation, which inflates barcode-only hit-rate metrics. 
 
  Fix: count only pure barcode-derived sources (for example `{"barcode", "barcode_unmatched"}`), and k
eep `barcode_ocr_agree` in a separate metric like `sn_barcode_ocr_agree`.

- `sn_barcode.py:120` Bug: `learn_sn_pattern()` calls `os.makedirs(os.path.dirname(path), exist_ok=Tru
e)`; this crashes when `HUAWEIOCR_SN_SEGMENTS_FILE` is set to a filename without a directory component
 (dirname is empty).  
  Fix: guard directory creation, e.g. `dirpath = os.path.dirname(path); if dirpath: os.makedirs(dirpat
h, exist_ok=True)`.

- `ocr.py:29-31,112-116` Missing essential tests: new `HUAWEIOCR_OCR_LANG` behavior (`en/ch` selection
 and invalid-value fallback) has no corresponding unit tests, so model-selection regressions can slip 
in unnoticed.  
  Fix: add tests that monkeypatch env and assert `_recognition_model_candidates()` order for `en`, `ch
`, and invalid values.

- `tests/test_locked_output_dirs.py` Verification blocker: current test run (`pytest tests/test_sn_bar
code_scanning.py tests/test_locked_output_dirs.py -q`) has 3 failing tests in `AppPathsInstallTest` du
e missing `app_paths.get_resource_path` / `app_paths._read_lock_snapshot`.  
  Fix: either restore these APIs (or compatibility shims) in `app_paths.py`, or update/remove those te
sts if the API was intentionally removed.


hook: Stop
hook: Stop Completed
tokens used
54,233
GATE_DECISION: FAIL

- `scan2.py:1090` Regression risk: `sn_barcode_hits` now counts `barcode_ocr_agree` as a barcode hit, 
but that source explicitly depends on OCR confirmation, which inflates barcode-only hit-rate metrics. 
 
  Fix: count only pure barcode-derived sources (for example `{"barcode", "barcode_unmatched"}`), and k
eep `barcode_ocr_agree` in a separate metric like `sn_barcode_ocr_agree`.

- `sn_barcode.py:120` Bug: `learn_sn_pattern()` calls `os.makedirs(os.path.dirname(path), exist_ok=Tru
e)`; this crashes when `HUAWEIOCR_SN_SEGMENTS_FILE` is set to a filename without a directory component
 (dirname is empty).  
  Fix: guard directory creation, e.g. `dirpath = os.path.dirname(path); if dirpath: os.makedirs(dirpat
h, exist_ok=True)`.

- `ocr.py:29-31,112-116` Missing essential tests: new `HUAWEIOCR_OCR_LANG` behavior (`en/ch` selection
 and invalid-value fallback) has no corresponding unit tests, so model-selection regressions can slip 
in unnoticed.  
  Fix: add tests that monkeypatch env and assert `_recognition_model_candidates()` order for `en`, `ch
`, and invalid values.

- `tests/test_locked_output_dirs.py` Verification blocker: current test run (`pytest tests/test_sn_bar
code_scanning.py tests/test_locked_output_dirs.py -q`) has 3 failing tests in `AppPathsInstallTest` du
e missing `app_paths.get_resource_path` / `app_paths._read_lock_snapshot`.  
  Fix: either restore these APIs (or compatibility shims) in `app_paths.py`, or update/remove those te
sts if the API was intentionally removed.




 succeeded in 749ms:
__pycache__
_tmp_barcode_debug
.git
.mypy_cache
.omx
.planning
.pytest_cache
.ruff_cache
1
assets
build
bundle
clearimage_cli
dist
dist_rebuild
legacy
new_images
new_images_uploaded_20260419_0801
openspec
pasted_images
runs_retest_barcode_20260419
runs_uploaded_20260419_0801
stage1_labels
stage2_fields
tests
validation
.env
.env.example
.gitignore
app_paths.py
barcode.py
crop.py
debug.py
gui_app_en.py
gui_app.py
gui_pipeline.py
HuaweiOCR.spec
LICENSE
ocr.py
README_EN.md
README_ZH.md
README.md
requirements.txt
run_all.py
scan2.py
self_check.log
sn_barcode.py
start.bat
validate_sn_barcodes.py
win_subprocess.py

codex
我已提取到 4 个可修复 FAIL：`scan2.py` 计数口径、`sn_barcode.py` 空目录崩溃、`ocr.py` 缺测试、以及 `app_paths.py` 缺失兼容 API。接下来按最小
改动依次修复代码并补测试，然后跑相关 pytest 进行回归验证。
exec
"C:\\Program Files\\WindowsApps\\Microsoft.PowerShell_7.6.0.0_x64__8wekyb3d8bbwe\\pwsh.exe" -Command '
rg -n "sn_barcode_hits|sn_barcode_hit_rate|barcode_ocr_agree|barcode_unmatched" scan2.py' in D:\newhua
wei
exec
"C:\\Program Files\\WindowsApps\\Microsoft.PowerShell_7.6.0.0_x64__8wekyb3d8bbwe\\pwsh.exe" -Command '
Get-Content -Path sn_barcode.py | Select-Object -First 220' in D:\newhuawei
exec
"C:\\Program Files\\WindowsApps\\Microsoft.PowerShell_7.6.0.0_x64__8wekyb3d8bbwe\\pwsh.exe" -Command '
Get-Content -Path app_paths.py | Select-Object -First 320' in D:\newhuawei
exec
"C:\\Program Files\\WindowsApps\\Microsoft.PowerShell_7.6.0.0_x64__8wekyb3d8bbwe\\pwsh.exe" -Command "
Get-Content -Path tests\\test_locked_output_dirs.py | Select-Object -First 1300" in D:\newhuawei
exec
"C:\\Program Files\\WindowsApps\\Microsoft.PowerShell_7.6.0.0_x64__8wekyb3d8bbwe\\pwsh.exe" -Command "
Get-Content -Path tests\\test_sn_barcode_scanning.py | Select-Object -First 420" in D:\newhuawei
 succeeded in 732ms:
from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

try:
    import cv2
except Exception:  # pragma: no cover - exercised only in stripped test envs
    cv2 = None

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None


BASE_SN20_SEGMENTS = ("ES", "ERA", "ERB", "ER", "LDR", "LDRA", "SRA")
SN12_RE = re.compile(r"(4E[0-9A-Z]{10})")

NON_SN_PREFIX_RE = re.compile(
    r"^(SF|MAC|EAN|UPC|QR|HTTP|HTTPS|PART|PN|MODEL|DESC|ROUTE|WAYBILL|SNMP|IMEI)"
)

DEFAULT_MAX_CANDIDATES = 96
DEFAULT_MAX_DECODER_ATTEMPTS = 96
DEFAULT_MIN_BARCODE_WIDTH = 120
DEFAULT_MIN_BARCODE_HEIGHT = 22
DEFAULT_BLUR_VARIANCE = 18.0
DEFAULT_DESKEW_ANGLES = (0, -4, 4, -8, 8)

_SN20_CACHE = {
    "path": None,
    "mtime": None,
    "segments": tuple(BASE_SN20_SEGMENTS),
    "regex": re.compile(r"(2[0-9]{9,10}(?:ES|ERA|ERB|ER|LDR|LDRA|SRA)[0-9]{4,7})"),
}


def _sn_segments_file_path() -> str:
    override = os.environ.get("HUAWEIOCR_SN_SEGMENTS_FILE", "").strip()
    if override:
        return os.path.abspath(override)
    if os.name == "nt":
        base = os.environ.get("LOCALAPPDATA") or os.path.join(os.path.expanduser("~"), "AppData", "Loc
al")
    else:
        base = os.environ.get("XDG_DATA_HOME") or os.path.join(os.path.expanduser("~"), ".local", "sha
re")
    return os.path.join(base, "HuaweiOCR", "sn_learned_segments.json")


def _compile_sn20_regex(segments: Iterable[str]):
    cleaned = []
    for segment in segments:
        token = _clean_code(segment)
        if not token or not token.isalpha():
            continue
        cleaned.append(token)
    dedup = sorted(set(cleaned), key=lambda s: (-len(s), s))
    if not dedup:
        dedup = list(BASE_SN20_SEGMENTS)
    alt = "|".join(re.escape(s) for s in dedup)
    return re.compile(rf"(2[0-9]{{9,10}}(?:{alt})[0-9]{{4,7}})")


def _load_learned_segments() -> tuple[str, ...]:
    path = _sn_segments_file_path()
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        mtime = None

    cached_path = _SN20_CACHE.get("path")
    cached_mtime = _SN20_CACHE.get("mtime")
    if cached_path == path and cached_mtime == mtime:
        return tuple(_SN20_CACHE.get("segments", tuple(BASE_SN20_SEGMENTS)))

    learned = []
    if mtime is not None:
        try:
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            if isinstance(payload, list):
                learned = [str(item) for item in payload]
        except Exception:
            learned = []

    merged = tuple(sorted(set(BASE_SN20_SEGMENTS).union({_clean_code(v) for v in learned if v}), key=l
ambda s: (-len(s), s)))
    _SN20_CACHE["path"] = path
    _SN20_CACHE["mtime"] = mtime
    _SN20_CACHE["segments"] = merged
    _SN20_CACHE["regex"] = _compile_sn20_regex(merged)
    return merged


def _sn20_regex():
    _load_learned_segments()
    return _SN20_CACHE["regex"]


def learn_sn_pattern(sn: str) -> bool:
    cleaned = _clean_code(sn)
    match = re.match(r"^2[0-9]{9,10}([A-Z]{2,5})[0-9]{4,7}$", cleaned)
    if not match:
        return False
    segment = match.group(1)
    if segment in BASE_SN20_SEGMENTS:
        return False

    _load_learned_segments()
    current = set(_SN20_CACHE.get("segments", tuple(BASE_SN20_SEGMENTS)))
    if segment in current:
        return False

    current.add(segment)
    persisted = sorted(current - set(BASE_SN20_SEGMENTS), key=lambda s: (-len(s), s))
    path = _sn_segments_file_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(persisted, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)

    _SN20_CACHE["path"] = None
    _SN20_CACHE["mtime"] = None
    _load_learned_segments()
    return True


@dataclass(frozen=True)
class CandidateImage:
    image: Any
    source: str
    source_region: str
    variant: str
    rotation: int = 0
    deskew_angle: int = 0
    rect: tuple[int, int, int, int] | None = None


@dataclass(frozen=True)
class DecoderResult:
    decoder_name: str
    raw_text: str
    source: str
    source_region: str
    rotation: int = 0
    confidence: float | None = None
    rect: tuple[int, int, int, int] | None = None
    barcode_type: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "decoder": self.decoder_name,
            "raw_text": self.raw_text,
            "source": self.source,
            "source_region": self.source_region,
            "rotation": self.rotation,
            "confidence": self.confidence,
            "rect": list(self.rect) if self.rect else None,
            "type": self.barcode_type,
        }


@dataclass(frozen=True)
class SnCandidate:
    sn: str
    raw_text: str
    source: str
    source_region: str
    decoder_name: str
    rotation: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "sn": self.sn,
            "raw_text": self.raw_text,
            "source": self.source,
            "source_region": self.source_region,
            "decoder": self.decoder_name,
            "rotation": self.rotation,
        }


@dataclass
class SnBarcodeReport:
    status: str
    sn: str = ""
    raw_text: str = ""
    source: str = ""
    source_region: str = ""
    decoder_name: str = ""
    attempts: int = 0
    decoded_count: int = 0
    results: list[DecoderResult] = field(default_factory=list)
    sn_candidates: list[SnCandidate] = field(default_factory=list)
    non_sn_payloads: list[str] = field(default_factory=list)
    ambiguous_sns: list[str] = field(default_factory=list)
    quality_issues: list[dict[str, Any]] = field(default_factory=list)
    decoder_errors: list[str] = field(default_factory=list)

    def to_meta(self) -> dict[str, Any]:
        return {
            "barcode_status": self.status,
            "barcode_found": self.decoded_count > 0,
            "barcode_attempts": self.attempts,
            "barcode_decoded_count": self.decoded_count,
            "barcode_sources": sorted({r.source for r in self.results}),
            "barcode_source_regions": sorted({r.source_region for r in self.results}),
            "barcode_decoder_names": sorted({r.decoder_name for r in self.results}),
            "barcode_non_sn_payloads": self.non_sn_payloads,
            "barcode_ambiguous_sns": self.ambiguous_sns,
            "barcode_quality_issues": self.quality_issues,
            "barcode_decoder_errors": self.decoder_errors,
            "barcode_results": [r.to_dict() for r in self.results],
            "barcode_sn_candidates": [c.to_dict() for c in self.sn_candidates],
        }


 succeeded in 719ms:
import os
import sys
import shutil
import tempfile
import time
import ctypes

MODEL_INSTALL_MARKER = ".huaweiocr_complete"
MODEL_ROOT_MARKER = ".huaweiocr_models_root"
MODEL_INSTALL_LOCK_MALFORMED_GRACE_SECONDS = 1


def _pid_is_running(pid):
    if not pid or pid <= 0:
        return False
    if pid == os.getpid():
        return True
    if os.name == "nt":
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(0x1000, False, int(pid))
        if not handle:
            return False
        try:
            exit_code = ctypes.c_ulong()
            if not kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
                return True
            return exit_code.value == 259
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _read_lock_snapshot(lock_path):
    try:
        stat = os.stat(lock_path)
        with open(lock_path, "rb") as f:
            data = f.read()
        return {
            "mtime_ns": getattr(stat, "st_mtime_ns", int(stat.st_mtime * 1_000_000_000)),
            "size": stat.st_size,
            "data": data,
        }
    except FileNotFoundError:
        return None


def _lock_metadata_from_snapshot(snapshot):
    if not snapshot:
        return None, None
    try:
        lines = snapshot["data"].decode("utf-8").splitlines()
        return int(lines[0].strip()), float(lines[1].strip())
    except Exception:
        return None, None


def _lock_snapshot_is_stale(snapshot):
    if snapshot is None:
        return True
    now = time.time()
    pid, created = _lock_metadata_from_snapshot(snapshot)
    if pid is not None and created is not None:
        return not _pid_is_running(pid)
    age = now - (snapshot["mtime_ns"] / 1_000_000_000)
    return age > MODEL_INSTALL_LOCK_MALFORMED_GRACE_SECONDS


def _reclaim_stale_lock(lock_path):
    first = _read_lock_snapshot(lock_path)
    if not _lock_snapshot_is_stale(first):
        return False
    time.sleep(0.05)
    second = _read_lock_snapshot(lock_path)
    if first != second or not _lock_snapshot_is_stale(second):
        return False
    try:
        os.remove(lock_path)
        return True
    except FileNotFoundError:
        return True
    except PermissionError:
        return False


def _write_lock_metadata(lock_fd):
    payload = f"{os.getpid()}\n{time.time()}\n".encode("ascii")
    os.write(lock_fd, payload)
    os.fsync(lock_fd)


def get_base_dir():
    if getattr(sys, "frozen", False):
        exe_dir = os.path.dirname(sys.executable)
        internal = os.path.join(exe_dir, "_internal")
        if os.path.isdir(internal):
            return internal
        return exe_dir
    return os.path.dirname(os.path.abspath(__file__))


def get_resource_path(*parts):
    return os.path.join(get_base_dir(), *parts)


def get_user_data_dir():
    override = os.environ.get("HUAWEIOCR_DATA_DIR")
    if override:
        return os.path.abspath(override)
    if os.name == "nt":
        base = os.environ.get("LOCALAPPDATA") or os.path.join(os.path.expanduser("~"), "AppData", "Loc
al")
    else:
        base = os.environ.get("XDG_DATA_HOME") or os.path.join(os.path.expanduser("~"), ".local", "sha
re")
    return os.path.join(base, "HuaweiOCR")


def get_model_install_root():
    override = os.environ.get("HUAWEIOCR_MODEL_DIR")
    if override:
        return os.path.abspath(override)
    return os.path.join(get_user_data_dir(), "models", "official_models")


def ensure_models_installed():
    bundled = get_resource_path("models", "official_models")
    if not os.path.isdir(bundled):
        return None

    target_root = get_model_install_root()
    os.makedirs(target_root, exist_ok=True)
    root_marker = os.path.join(target_root, MODEL_ROOT_MARKER)
    with open(root_marker, "a", encoding="utf-8"):
        pass

    os.environ["HUAWEIOCR_INSTALLED_MODEL_DIR"] = target_root
    os.environ["PADDLE_PDX_OFFICIAL_MODEL_DIR"] = target_root
    lock_path = os.path.join(target_root, ".huaweiocr_install.lock")
    lock_fd = None
    deadline = time.time() + 30
    while lock_fd is None:
        try:
            lock_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_RDWR)
            _write_lock_metadata(lock_fd)
        except FileExistsError:
            if _reclaim_stale_lock(lock_path):
                continue
            if time.time() >= deadline:
                raise TimeoutError(f"Timed out waiting for model install lock: {lock_path}")
            time.sleep(0.2)

    try:
        for name in os.listdir(bundled):
            src = os.path.join(bundled, name)
            dst = os.path.join(target_root, name)
            marker = os.path.join(dst, MODEL_INSTALL_MARKER)
            if not os.path.isdir(src):
                continue
            if os.path.isdir(dst) and os.path.isfile(marker):
                continue
            if os.path.exists(dst):
                if not os.path.isfile(root_marker):
                    raise RuntimeError(f"Refusing to modify unmanaged model directory: {dst}")
                shutil.rmtree(dst)
            tmp = tempfile.mkdtemp(prefix=f".{name}.", dir=target_root)
            try:
                shutil.copytree(src, tmp, dirs_exist_ok=True)
                with open(os.path.join(tmp, MODEL_INSTALL_MARKER), "w", encoding="utf-8") as f:
                    f.write("ok\n")
                os.replace(tmp, dst)
            finally:
                if os.path.exists(tmp):
                    shutil.rmtree(tmp, ignore_errors=True)
    finally:
        if lock_fd is not None:
            os.close(lock_fd)
        try:
            os.remove(lock_path)
        except FileNotFoundError:
            pass
    return target_root


def get_barcode_cli_path():
    candidates = [
        get_resource_path("BarcodeReaderCLI", "bin", "BarcodeReaderCLI.exe"),
        get_resource_path("bundle", "BarcodeReaderCLI", "bin", "BarcodeReaderCLI.exe"),
        get_resource_path("dist", "HuaweiOCR", "_internal", "BarcodeReaderCLI", "bin", "BarcodeReaderC
LI.exe"),
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    return candidates[0]


def ensure_paddle_libs_on_path():
    libs_dir = get_resource_path("paddle", "libs")
    if not os.path.isdir(libs_dir):
        return
    path = os.environ.get("PATH", "")
    if libs_dir not in path:
        os.environ["PATH"] = libs_dir + os.pathsep + path

 succeeded in 786ms:
435:        "barcode_ocr_agree",
736:    barcode_unmatched_sn = ""
738:        barcode_unmatched_sn = _extract_unmatched_barcode_sn(barcode_report.raw_text)
756:            if barcode_unmatched_sn:
758:                    barcode_unmatched_sn,
760:                    "barcode_unmatched",
769:        if barcode_unmatched_sn:
771:                barcode_unmatched_sn,
773:                "barcode_unmatched",
795:        if barcode_unmatched_sn and sn == barcode_unmatched_sn:
800:                "barcode_ocr_agree",
805:        barcode_unmatched_sn,
822:        if barcode_unmatched_sn and sn == barcode_unmatched_sn:
827:                "barcode_ocr_agree",
832:        barcode_unmatched_sn,
847:        if barcode_unmatched_sn and sn == barcode_unmatched_sn:
852:                "barcode_ocr_agree",
857:        barcode_unmatched_sn,
870:            if barcode_unmatched_sn:
872:                    barcode_unmatched_sn,
874:                    "barcode_unmatched",
975:        "sn_barcode_hits": 0,
976:        "sn_barcode_hit_rate": 0.0,
1090:                if sn_src in {"barcode", "barcode_unmatched", "barcode_ocr_agree"}:
1091:                    stats["sn_barcode_hits"] += 1
1120:        stats["sn_barcode_hit_rate"] = stats["sn_barcode_hits"] / float(stats["sn_total"])

 succeeded in 799ms:
import importlib
import inspect
import json
import os
import sys
import tempfile
import time
import types
import unittest
import subprocess
from unittest import mock


def _install_crop_import_fakes():
    cv2 = types.ModuleType("cv2")
    cv2.IMREAD_COLOR = 1
    cv2.IMWRITE_JPEG_QUALITY = 1
    cv2.imdecode = lambda *args, **kwargs: None
    cv2.imencode = lambda *args, **kwargs: (False, b"")
    cv2.dnn = types.SimpleNamespace(NMSBoxes=lambda *args, **kwargs: [])
    sys.modules["cv2"] = cv2

    numpy = types.ModuleType("numpy")
    numpy.uint8 = object()
    numpy.fromfile = lambda *args, **kwargs: b""
    sys.modules["numpy"] = numpy

    inference_sdk = types.ModuleType("inference_sdk")

    class DummyInferenceHTTPClient:
        def __init__(self, *args, **kwargs):
            pass

        def infer(self, *args, **kwargs):
            return {"predictions": []}

    inference_sdk.InferenceHTTPClient = DummyInferenceHTTPClient
    sys.modules["inference_sdk"] = inference_sdk


def _install_scan2_import_fakes():
    cv2 = types.ModuleType("cv2")
    cv2.IMREAD_COLOR = 1
    cv2.THRESH_BINARY = 0
    cv2.THRESH_OTSU = 0
    cv2.COLOR_BGR2GRAY = 1
    cv2.COLOR_GRAY2BGR = 2
    cv2.BORDER_CONSTANT = 0
    cv2.ROTATE_90_COUNTERCLOCKWISE = 0
    cv2.INTER_CUBIC = 0
    cv2.imread = lambda *args, **kwargs: None
    sys.modules["cv2"] = cv2

    numpy = types.ModuleType("numpy")
    sys.modules["numpy"] = numpy

    ocr = types.ModuleType("ocr")
    ocr.init_ocr = lambda: object()
    ocr.ocr_one_image = lambda *args, **kwargs: ([], "")
    sys.modules["ocr"] = ocr

    barcode = types.ModuleType("barcode")
    barcode.decode_small_patch = lambda *args, **kwargs: {"results": []}
    sys.modules["barcode"] = barcode

    app_paths = types.ModuleType("app_paths")
    app_paths.ensure_models_installed = lambda: None
    sys.modules["app_paths"] = app_paths


def _import_crop():
    os.environ["API_KEY"] = "test-key"
    sys.modules.pop("crop", None)
    _install_crop_import_fakes()
    return importlib.import_module("crop")


def _import_scan2():
    sys.modules.pop("scan2", None)
    _install_scan2_import_fakes()
    return importlib.import_module("scan2")


def _install_barcode_import_fakes():
    cv2 = types.ModuleType("cv2")
    cv2.COLOR_BGR2GRAY = 1
    cv2.COLOR_GRAY2BGR = 2
    cv2.BORDER_CONSTANT = 0
    cv2.INTER_CUBIC = 0
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C = 0
    cv2.THRESH_BINARY = 0
    cv2.MORPH_RECT = 0
    cv2.MORPH_CLOSE = 0
    cv2.ROTATE_90_CLOCKWISE = 0
    cv2.cvtColor = lambda img, *args, **kwargs: img
    cv2.resize = lambda img, *args, **kwargs: img
    cv2.bitwise_not = lambda img: img
    cv2.copyMakeBorder = lambda img, *args, **kwargs: img
    cv2.createCLAHE = lambda *args, **kwargs: types.SimpleNamespace(apply=lambda img: img)
    cv2.GaussianBlur = lambda img, *args, **kwargs: img
    cv2.addWeighted = lambda img, *args, **kwargs: img
    cv2.adaptiveThreshold = lambda img, *args, **kwargs: img
    cv2.getStructuringElement = lambda *args, **kwargs: object()
    cv2.morphologyEx = lambda img, *args, **kwargs: img
    cv2.imwrite = lambda *args, **kwargs: True
    sys.modules["cv2"] = cv2

    numpy = types.ModuleType("numpy")
    numpy.ndarray = object
    numpy.rot90 = lambda img, k: img
    sys.modules["numpy"] = numpy

    pyzbar_pkg = types.ModuleType("pyzbar")
    pyzbar_mod = types.ModuleType("pyzbar.pyzbar")
    pyzbar_mod.ZBarSymbol = types.SimpleNamespace(CODE128=object())
    pyzbar_mod.decode = lambda *args, **kwargs: []
    pyzbar_pkg.pyzbar = pyzbar_mod
    sys.modules["pyzbar"] = pyzbar_pkg
    sys.modules["pyzbar.pyzbar"] = pyzbar_mod


def _import_barcode():
    sys.modules.pop("barcode", None)
    _install_barcode_import_fakes()
    return importlib.import_module("barcode")


@unittest.skipUnless(os.name == "nt", "Windows file locking behavior only")
class LockedOutputDirsTest(unittest.TestCase):
    def test_locked_manifest_switches_stage2_to_run_directory(self):
        crop = _import_crop()
        with tempfile.TemporaryDirectory() as root:
            input_dir = os.path.join(root, "new_images")
            original_stage2 = os.path.join(root, "stage2_fields")
            os.makedirs(original_stage2)
            locked_manifest = os.path.join(original_stage2, "manifest.jsonl")

            with open(locked_manifest, "w", encoding="utf-8") as locked:
                locked.write("locked\n")
                locked.flush()

                crop.configure_paths(input_dir=input_dir, out_dir=root)
                crop.ensure_dirs()

                self.assertNotEqual(
                    os.path.abspath(original_stage2),
                    os.path.abspath(crop.STAGE2_DIR),
                )
                self.assertTrue(os.path.isdir(crop.OUT_MODEL_DIR))
                self.assertTrue(os.path.isdir(crop.OUT_SN_DIR))
                self.assertTrue(os.path.isdir(crop.FAILED_DIR))
                self.assertEqual(
                    crop.MANIFEST_PATH,
                    os.path.join(crop.STAGE2_DIR, "manifest.jsonl"),
                )
                self.assertTrue(os.path.exists(locked_manifest))


class RunAllPathPropagationTest(unittest.TestCase):
    def test_scan2_reads_actual_crop_output_dirs(self):
        import run_all

        with tempfile.TemporaryDirectory() as root:
            input_dir = os.path.join(root, "input")
            out_dir = os.path.join(root, "out")
            os.makedirs(input_dir)
            with open(os.path.join(input_dir, "sample.png"), "wb") as image:
                image.write(b"not-a-real-image")

            crop = types.ModuleType("crop")
            crop.OUT_MODEL_DIR = os.path.join(out_dir, "stage2_fields_run_x", "model")
            crop.OUT_SN_DIR = os.path.join(out_dir, "stage2_fields_run_x", "sn")
            crop.STAGE2_DIR = os.path.join(out_dir, "stage2_fields_run_x")
            crop.STAGE1_DIR = os.path.join(out_dir, "stage1_labels_run_x")
            crop.set_log_level = lambda level: None
            crop.main = lambda **kwargs: {"label_count": 1, "manifest_rows": 1}

            calls = {}
            scan2 = types.ModuleType("scan2")
            scan2.set_log_level = lambda level: None

            def fake_scan2_main(**kwargs):
                calls.update(kwargs)
                return {"sn_total": 0}

            scan2.main = fake_scan2_main

            argv = [
                "run_all.py",
                "--input",
                input_dir,
                "--out",
                out_dir,
            ]
            with mock.patch.dict(sys.modules, {"crop": crop, "scan2": scan2}):
                with mock.patch.object(sys, "argv", argv):
                    self.assertEqual(run_all.main(), 0)

            self.assertEqual(calls["model_dir"], crop.OUT_MODEL_DIR)
            self.assertEqual(calls["sn_dir"], crop.OUT_SN_DIR)
            self.assertEqual(
                calls["out_jsonl"],
                os.path.join(crop.STAGE2_DIR, "model_sn_ocr.jsonl"),
            )

    def test_zero_label_crop_returns_nonzero_without_scanning(self):
        import run_all

        with tempfile.TemporaryDirectory() as root:
            input_dir = os.path.join(root, "input")
            out_dir = os.path.join(root, "out")
            os.makedirs(input_dir)
            with open(os.path.join(input_dir, "sample.png"), "wb") as image:
                image.write(b"not-a-real-image")

            crop = types.ModuleType("crop")
            crop.set_log_level = lambda level: None
            crop.main = mock.Mock(return_value={"label_count": 0, "manifest_rows": 0})

            scan2 = types.ModuleType("scan2")
            scan2.set_log_level = lambda level: None
            scan2.main = mock.Mock()

            argv = ["run_all.py", "--input", input_dir, "--out", out_dir]
            with mock.patch.dict(sys.modules, {"crop": crop, "scan2": scan2}):
                with mock.patch.object(sys, "argv", argv):
                    self.assertEqual(run_all.main(), 1)

            scan2.main.assert_not_called()

    def test_empty_input_returns_nonzero_without_running_pipeline(self):
        import run_all

        with tempfile.TemporaryDirectory() as root:
            input_dir = os.path.join(root, "empty")
            os.makedirs(input_dir)
            crop = types.ModuleType("crop")
            crop.set_log_level = lambda level: None
            crop.main = mock.Mock()

            scan2 = types.ModuleType("scan2")
            scan2.set_log_level = lambda level: None
            scan2.main = mock.Mock()

            argv = ["run_all.py", "--input", input_dir, "--out", os.path.join(root, "out")]
            with mock.patch.dict(sys.modules, {"crop": crop, "scan2": scan2}):
                with mock.patch.object(sys, "argv", argv):
                    self.assertEqual(run_all.main(), 2)

            crop.main.assert_not_called()
            scan2.main.assert_not_called()


class PaddleOcrModelKwargsTest(unittest.TestCase):
    def test_local_model_dirs_include_matching_model_names(self):
        for name in ("ocr", "paddle", "paddleocr", "app_paths"):
            sys.modules.pop(name, None)

        paddle = types.ModuleType("paddle")
        paddle.set_device = lambda _device: None
        sys.modules["paddle"] = paddle

        paddleocr = types.ModuleType("paddleocr")

        class DummyPaddleOCR:
            def __init__(
                self,
                use_doc_orientation_classify=None,
                use_doc_unwarping=None,
                use_textline_orientation=None,
                text_detection_model_name=None,
                text_detection_model_dir=None,
                text_recognition_model_name=None,
                text_recognition_model_dir=None,
                textline_orientation_model_name=None,
                textline_orientation_model_dir=None,
            ):
                pass

        paddleocr.PaddleOCR = DummyPaddleOCR
        sys.modules["paddleocr"] = paddleocr

        app_paths = types.ModuleType("app_paths")
        app_paths.ensure_models_installed = lambda: None
        app_paths.get_resource_path = lambda *parts: os.path.join(*parts)
        sys.modules["app_paths"] = app_paths

        import ocr

        with tempfile.TemporaryDirectory() as root:
            models = os.path.join(root, "official_models")
            for name in (
                "PP-OCRv5_server_det",
                "PP-OCRv5_server_rec",
                "en_PP-OCRv5_mobile_rec",
                "PP-LCNet_x1_0_textline_ori",
            ):
                os.makedirs(os.path.join(models, name))

            kwargs = ocr._paddleocr_model_kwargs(models)

        self.assertEqual(kwargs["text_detection_model_name"], "PP-OCRv5_server_det")
        self.assertEqual(kwargs["text_recognition_model_name"], "en_PP-OCRv5_mobile_rec")
        self.assertEqual(kwargs["textline_orientation_model_name"], "PP-LCNet_x1_0_textline_ori")
        self.assertIs(kwargs["use_doc_orientation_classify"], False)
        self.assertIs(kwargs["use_doc_unwarping"], False)
        self.assertIs(kwargs["use_textline_orientation"], True)
        self.assertTrue(kwargs["text_detection_model_dir"].endswith("PP-OCRv5_server_det"))
        self.assertTrue(kwargs["text_recognition_model_dir"].endswith("en_PP-OCRv5_mobile_rec"))
        self.assertTrue(kwargs["textline_orientation_model_dir"].endswith("PP-LCNet_x1_0_textline_ori"
))

    def test_fast_profile_prefers_mobile_recognition_model(self):
        for name in ("ocr", "paddle", "paddleocr", "app_paths"):
            sys.modules.pop(name, None)

        paddle = types.ModuleType("paddle")
        paddle.set_device = lambda _device: None
        sys.modules["paddle"] = paddle

        paddleocr = types.ModuleType("paddleocr")

        class DummyPaddleOCR:
            def __init__(self, text_recognition_model_name=None, text_recognition_model_dir=None):
                pass

        paddleocr.PaddleOCR = DummyPaddleOCR
        sys.modules["paddleocr"] = paddleocr

        app_paths = types.ModuleType("app_paths")
        app_paths.ensure_models_installed = lambda: None
        app_paths.get_resource_path = lambda *parts: os.path.join(*parts)
        sys.modules["app_paths"] = app_paths

        with mock.patch.dict(os.environ, {"HUAWEIOCR_OCR_PROFILE": "fast"}, clear=False):
            import ocr

        with tempfile.TemporaryDirectory() as root:
            models = os.path.join(root, "official_models")
            for name in ("PP-OCRv5_server_rec", "en_PP-OCRv5_mobile_rec"):
                os.makedirs(os.path.join(models, name))

            kwargs = ocr._paddleocr_model_kwargs(models)

        self.assertEqual(kwargs["text_recognition_model_name"], "en_PP-OCRv5_mobile_rec")
        self.assertTrue(kwargs["text_recognition_model_dir"].endswith("en_PP-OCRv5_mobile_rec"))

    def test_server_profile_prefers_server_recognition_model(self):
        for name in ("ocr", "paddle", "paddleocr", "app_paths"):
            sys.modules.pop(name, None)

        paddle = types.ModuleType("paddle")
        paddle.set_device = lambda _device: None
        sys.modules["paddle"] = paddle

        paddleocr = types.ModuleType("paddleocr")

        class DummyPaddleOCR:
            def __init__(self, text_recognition_model_name=None, text_recognition_model_dir=None):
                pass

        paddleocr.PaddleOCR = DummyPaddleOCR
        sys.modules["paddleocr"] = paddleocr

        app_paths = types.ModuleType("app_paths")
        app_paths.ensure_models_installed = lambda: None
        app_paths.get_resource_path = lambda *parts: os.path.join(*parts)
        sys.modules["app_paths"] = app_paths

        with mock.patch.dict(os.environ, {"HUAWEIOCR_OCR_PROFILE": "server"}, clear=False):
            import ocr

        with tempfile.TemporaryDirectory() as root:
            models = os.path.join(root, "official_models")
            for name in ("PP-OCRv5_server_rec", "en_PP-OCRv5_mobile_rec"):
                os.makedirs(os.path.join(models, name))

            kwargs = ocr._paddleocr_model_kwargs(models)

        self.assertEqual(kwargs["text_recognition_model_name"], "PP-OCRv5_server_rec")
        self.assertTrue(kwargs["text_recognition_model_dir"].endswith("PP-OCRv5_server_rec"))

    def test_local_source_model_root_falls_back_to_bundle_models(self):
        for name in ("ocr", "paddle", "paddleocr", "app_paths"):
            sys.modules.pop(name, None)

        paddle = types.ModuleType("paddle")
        paddle.set_device = lambda _device: None
        sys.modules["paddle"] = paddle

        paddleocr = types.ModuleType("paddleocr")
        paddleocr.PaddleOCR = type("DummyPaddleOCR", (), {})
        sys.modules["paddleocr"] = paddleocr

        app_paths = types.ModuleType("app_paths")
        app_paths.ensure_models_installed = lambda: None
        app_paths.get_resource_path = lambda *parts: os.path.join(*parts)
        sys.modules["app_paths"] = app_paths

        import ocr

        with tempfile.TemporaryDirectory() as root:
            bundle_root = os.path.join(root, "bundle", "models", "official_models")
            os.makedirs(bundle_root)
            with mock.patch.object(ocr, "get_resource_path", side_effect=lambda *parts: os.path.join(r
oot, *parts)):
                self.assertEqual(ocr._local_model_root_fallback(), bundle_root)


class Scan2ManifestTest(unittest.TestCase):
    def test_main_signature_keeps_legacy_arguments(self):
        scan2 = _import_scan2()

        signature = inspect.signature(scan2.main)
        self.assertEqual(
            list(signature.parameters),
            ["out_dir", "model_dir", "sn_dir", "out_jsonl", "debug_log", "log_level"],
        )
        self.assertEqual(
            str(signature),
            "(out_dir=None, model_dir=None, sn_dir=None, out_jsonl=None, debug_log=None, log_level='in
fo')",
        )

    def test_manifest_keeps_labels_without_model_or_sn_crops(self):
        scan2 = _import_scan2()

        with tempfile.TemporaryDirectory() as root:
            stage2 = os.path.join(root, "stage2_fields")
            model_dir = os.path.join(stage2, "model")
            sn_dir = os.path.join(stage2, "sn")
            os.makedirs(model_dir)
            os.makedirs(sn_dir)

            manifest_path = os.path.join(stage2, "manifest.jsonl")
            with open(manifest_path, "w", encoding="utf-8") as manifest:
                manifest.write(json.dumps({"label_id": "img__with__sep__label_1"}) + "\n")
                manifest.write(json.dumps({"label_id": "missing_both__label_1"}) + "\n")

            with open(os.path.join(model_dir, "img__with__sep__label_1__model.png"), "wb") as image:
                image.write(b"model")

            out_jsonl = os.path.join(root, "out.jsonl")
            with mock.patch.object(scan2, "recognize_model", return_value=("M1", "raw", "test")):
                stats = scan2.main(
                    model_dir=model_dir,
                    sn_dir=sn_dir,
                    out_jsonl=out_jsonl,
                    debug_log=os.path.join(root, "debug.log"),
                )

            with open(out_jsonl, "r", encoding="utf-8") as f:
                rows = [json.loads(line) for line in f if line.strip()]

            self.assertEqual([row["label_id"] for row in rows], ["img__with__sep__label_1", "missing_b
oth__label_1"])
            self.assertEqual(rows[1]["model_src"], "missing")
            self.assertEqual(rows[1]["sn_src"], "missing")
            self.assertEqual(stats["sn_total"], 0)

    def test_manifest_bad_json_fails_fast(self):
        scan2 = _import_scan2()

        with tempfile.TemporaryDirectory() as root:
            stage2 = os.path.join(root, "stage2_fields")
            os.makedirs(os.path.join(stage2, "model"))
            os.makedirs(os.path.join(stage2, "sn"))
            with open(os.path.join(stage2, "manifest.jsonl"), "w", encoding="utf-8") as manifest:
                manifest.write("{bad json}\n")

            with self.assertRaisesRegex(ValueError, "Invalid manifest JSON"):
                scan2.main(
                    model_dir=os.path.join(stage2, "model"),
                    sn_dir=os.path.join(stage2, "sn"),
                    out_jsonl=os.path.join(root, "out.jsonl"),
                    debug_log=os.path.join(root, "debug.log"),
                )

    def test_manifest_missing_label_id_fails_fast(self):
        scan2 = _import_scan2()

        with tempfile.TemporaryDirectory() as root:
            stage2 = os.path.join(root, "stage2_fields")
            os.makedirs(os.path.join(stage2, "model"))
            os.makedirs(os.path.join(stage2, "sn"))
            with open(os.path.join(stage2, "manifest.jsonl"), "w", encoding="utf-8") as manifest:
                manifest.write(json.dumps({"model_path": None}) + "\n")

            with self.assertRaisesRegex(ValueError, "missing label_id"):
                scan2.main(
                    model_dir=os.path.join(stage2, "model"),
                    sn_dir=os.path.join(stage2, "sn"),
                    out_jsonl=os.path.join(root, "out.jsonl"),
                    debug_log=os.path.join(root, "debug.log"),
                )

    def test_raw_fields_keep_full_values_by_default(self):
        scan2 = _import_scan2()

        with tempfile.TemporaryDirectory() as root:
            stage2 = os.path.join(root, "stage2_fields")
            model_dir = os.path.join(stage2, "model")
            sn_dir = os.path.join(stage2, "sn")
            os.makedirs(model_dir)
            os.makedirs(sn_dir)
            model_path = os.path.join(model_dir, "a__label_1__model.png")
            sn_path = os.path.join(sn_dir, "a__label_1__sn.png")
            open(model_path, "wb").close()
            open(sn_path, "wb").close()
            with open(os.path.join(stage2, "manifest.jsonl"), "w", encoding="utf-8") as manifest:
                manifest.write(json.dumps({"label_id": "a__label_1", "model_path": model_path, "sn_pat
h": sn_path}) + "\n")

            out_jsonl = os.path.join(root, "out.jsonl")
            with mock.patch.object(scan2, "recognize_model", return_value=("MODEL1", "RAW_MODEL_SECRET
_123456", "test")):
                with mock.patch.object(scan2, "recognize_sn", return_value=("SN1", "RAW_SN_SECRET_1234
56", "test", {})):
                    with mock.patch.dict(os.environ, {"SCAN2_MASK_RAW": "", "SCAN2_UNSAFE_RAW": ""}, c
lear=False):
                        scan2.main(
                            model_dir=model_dir,
                            sn_dir=sn_dir,
                            out_jsonl=out_jsonl,
                            debug_log=os.path.join(root, "debug.log"),
                        )

            with open(out_jsonl, "r", encoding="utf-8") as f:
                row = json.loads(f.readline())
            self.assertEqual(row["model"], "MODEL1")
            self.assertEqual(row["sn"], "SN1")
            self.assertEqual(row["model_raw"], "RAW_MODEL_SECRET_123456")
            self.assertEqual(row["sn_raw"], "RAW_SN_SECRET_123456")

    def test_raw_fields_can_be_masked_with_env_flag(self):
        scan2 = _import_scan2()

        with tempfile.TemporaryDirectory() as root:
            stage2 = os.path.join(root, "stage2_fields")
            model_dir = os.path.join(stage2, "model")
            sn_dir = os.path.join(stage2, "sn")
            os.makedirs(model_dir)
            os.makedirs(sn_dir)
            model_path = os.path.join(model_dir, "a__label_1__model.png")
            sn_path = os.path.join(sn_dir, "a__label_1__sn.png")
            open(model_path, "wb").close()
            open(sn_path, "wb").close()
            with open(os.path.join(stage2, "manifest.jsonl"), "w", encoding="utf-8") as manifest:
                manifest.write(json.dumps({"label_id": "a__label_1", "model_path": model_path, "sn_pat
h": sn_path}) + "\n")

            out_jsonl = os.path.join(root, "out.jsonl")
            with mock.patch.object(scan2, "recognize_model", return_value=("MODEL1", "RAW_MODEL_SECRET
_123456", "test")):
                with mock.patch.object(scan2, "recognize_sn", return_value=("SN1", "RAW_SN_SECRET_1234
56", "test", {})):
                    with mock.patch.dict(os.environ, {"SCAN2_MASK_RAW": "1", "SCAN2_UNSAFE_RAW": ""}, 
clear=False):
                        scan2.main(
                            model_dir=model_dir,
                            sn_dir=sn_dir,
                            out_jsonl=out_jsonl,
                            debug_log=os.path.join(root, "debug.log"),
                        )

            with open(out_jsonl, "r", encoding="utf-8") as f:
                row = json.loads(f.readline())
            self.assertNotIn("RAW_MODEL_SECRET_123456", row["model_raw"])
            self.assertNotIn("RAW_SN_SECRET_123456", row["sn_raw"])

    def test_info_log_keeps_full_model_and_sn_values(self):
        scan2 = _import_scan2()

        with tempfile.TemporaryDirectory() as root:
            stage2 = os.path.join(root, "stage2_fields")
            model_dir = os.path.join(stage2, "model")
            sn_dir = os.path.join(stage2, "sn")
            os.makedirs(model_dir)
            os.makedirs(sn_dir)
            model_path = os.path.join(model_dir, "a__label_1__model.png")
            sn_path = os.path.join(sn_dir, "a__label_1__sn.png")
            open(model_path, "wb").close()
            open(sn_path, "wb").close()
            with open(os.path.join(stage2, "manifest.jsonl"), "w", encoding="utf-8") as manifest:
                manifest.write(json.dumps({"label_id": "a__label_1", "model_path": model_path, "sn_pat
h": sn_path}) + "\n")

            logs = []
            old_sink = scan2.LOG_SINK
            scan2.set_log_sink(logs.append)
            try:
                with mock.patch.object(scan2, "recognize_model", return_value=("S380-S8P2T", "raw", "o
cr_color")):
                    with mock.patch.object(scan2, "recognize_sn", return_value=("4E25B0105849", "raw",
 "barcode", {})):
                        scan2.main(
                            model_dir=model_dir,
                            sn_dir=sn_dir,
                            out_jsonl=os.path.join(root, "out.jsonl"),
                            debug_log=os.path.join(root, "debug.log"),
                        )
            finally:
                scan2.set_log_sink(old_sink)

            joined = "\n".join(logs)
            self.assertIn("MODEL=S380-S8P2T", joined)
            self.assertIn("SN=4E25B0105849", joined)
            self.assertNotIn("S380**8P2T", joined)
            self.assertNotIn("4E25****5849", joined)

    def test_label_crop_barcode_is_used_when_sn_crop_is_missing(self):
        scan2 = _import_scan2()

        with tempfile.TemporaryDirectory() as root:
            stage2 = os.path.join(root, "stage2_fields")
            model_dir = os.path.join(stage2, "model")
            sn_dir = os.path.join(stage2, "sn")
            os.makedirs(model_dir)
            os.makedirs(sn_dir)
            label_path = os.path.join(root, "label.png")
            open(label_path, "wb").close()
            with open(os.path.join(stage2, "manifest.jsonl"), "w", encoding="utf-8") as manifest:
                manifest.write(json.dumps({"label_id": "a__label_1", "label_crop": label_path}) + "\n"
)

            out_jsonl = os.path.join(root, "out.jsonl")
            with mock.patch.object(scan2, "read_barcodes", return_value=["SN:4E25A0170000"]):
                with mock.patch.object(scan2, "load_for_ocr_color") as load_ocr:
                    stats = scan2.main(
                        model_dir=model_dir,
                        sn_dir=sn_dir,
                        out_jsonl=out_jsonl,
                        debug_log=os.path.join(root, "debug.log"),
                    )

            with open(out_jsonl, "r", encoding="utf-8") as f:
                row = json.loads(f.readline())
            self.assertEqual(row["sn"], "4E25A0170000")
            self.assertEqual(row["sn_src"], "barcode")
            self.assertEqual(stats["sn_total"], 1)
            self.assertEqual(stats["sn_success"], 1)
            load_ocr.assert_not_called()

    def test_model_barcode_is_enabled_by_default(self):
        scan2 = _import_scan2()

        with tempfile.TemporaryDirectory() as root:
            stage2 = os.path.join(root, "stage2_fields")
            model_dir = os.path.join(stage2, "model")
            sn_dir = os.path.join(stage2, "sn")
            os.makedirs(model_dir)
            os.makedirs(sn_dir)
            model_path = os.path.join(model_dir, "a__label_1__model.png")
            open(model_path, "wb").close()
            with open(os.path.join(stage2, "manifest.jsonl"), "w", encoding="utf-8") as manifest:
                manifest.write(json.dumps({"label_id": "a__label_1", "model_path": model_path}) + "\n"
)

            with mock.patch.object(scan2, "recognize_model", return_value=("MODEL1", "raw", "barcode")
) as recognize_model:
                with mock.patch.dict(os.environ, {}, clear=True):
                    stats = scan2.main(
                        model_dir=model_dir,
                        sn_dir=sn_dir,
                        out_jsonl=os.path.join(root, "out.jsonl"),
                        debug_log=os.path.join(root, "debug.log"),
                    )

            self.assertTrue(recognize_model.call_args.kwargs["use_barcode"])
            self.assertEqual(stats["model_total"], 1)
            self.assertEqual(stats["model_success"], 1)
            self.assertEqual(stats["model_barcode_hits"], 1)
            self.assertEqual(stats["model_barcode_hit_rate"], 1.0)

    def test_model_barcode_can_be_disabled_with_env_flag(self):
        scan2 = _import_scan2()

        with tempfile.TemporaryDirectory() as root:
            stage2 = os.path.join(root, "stage2_fields")
            model_dir = os.path.join(stage2, "model")
            sn_dir = os.path.join(stage2, "sn")
            os.makedirs(model_dir)
            os.makedirs(sn_dir)
            model_path = os.path.join(model_dir, "a__label_1__model.png")
            open(model_path, "wb").close()
            with open(os.path.join(stage2, "manifest.jsonl"), "w", encoding="utf-8") as manifest:
                manifest.write(json.dumps({"label_id": "a__label_1", "model_path": model_path}) + "\n"
)

            with mock.patch.object(scan2, "recognize_model", return_value=("MODEL1", "raw", "test")) a
s recognize_model:
                with mock.patch.dict(os.environ, {"SCAN2_MODEL_BARCODE": "0"}, clear=True):
                    scan2.main(
                        model_dir=model_dir,
                        sn_dir=sn_dir,
                        out_jsonl=os.path.join(root, "out.jsonl"),
                        debug_log=os.path.join(root, "debug.log"),
                    )

            self.assertFalse(recognize_model.call_args.kwargs["use_barcode"])

    def test_model_recognition_skips_barcode_cli_when_disabled(self):
        scan2 = _import_scan2()
        fake_img = object()

        with mock.patch.object(scan2, "try_model_from_barcode") as barcode_mock:
            with mock.patch.object(scan2, "load_for_ocr_color", return_value=None):
                with mock.patch.object(scan2, "load_and_preprocess", return_value=fake_img):
                    with mock.patch.object(scan2, "ocr_text_with_details", return_value=("", "", [])):
                        model, _raw, source = scan2.recognize_model("model.png", use_barcode=False)

        self.assertEqual(model, "")
        self.assertEqual(source, "none")
        barcode_mock.assert_not_called()

    def test_model_recognition_prefers_file_path_ocr(self):
        scan2 = _import_scan2()

        with mock.patch.object(
            scan2,
            "ocr_text_with_details",
            return_value=(
                "Model: S380-S8P2T",
                "Model:S380-S8P2T",
                [{"text": "Model: S380-S8P2T", "score": 0.99}],
            ),
        ) as ocr_mock:
            with mock.patch.object(scan2, "load_for_ocr_color") as color_mock:
                with mock.patch.object(scan2, "load_and_preprocess") as bin_mock:
                    model, raw, source = scan2.recognize_model("model.png", use_barcode=False)

        self.assertEqual(model, "S380-S8P2T")
        self.assertEqual(raw, "Model: S380-S8P2T")
        self.assertEqual(source, "ocr_file")
        ocr_mock.assert_called_once_with("model.png")
        color_mock.assert_not_called()
        bin_mock.assert_not_called()

    def test_model_ocr_result_prefers_spaced_text_over_concat(self):
        scan2 = _import_scan2()

        model = scan2.extract_model_from_ocr_result(
            "Model: AP162E 9SC:AD1625(1",
            "Model: AP162E9SC:AD1625(1",
        )

        self.assertEqual(model, "AP162E")

    def test_s380_s8p2t_ocr_noise_is_normalized(self):
        scan2 = _import_scan2()

        self.assertEqual(scan2.extract_model_from_text("MO8S-0802 Wac"), "S380-S8P2T")
        self.assertEqual(scan2.extract_model_from_text("M08S-0802 Wac"), "S380-S8P2T")


class CropTempFileTest(unittest.TestCase):
    def test_original_path_for_label_id_resolves_current_input_dir(self):
        crop = _import_crop()
        with tempfile.TemporaryDirectory() as root:
            input_dir = os.path.join(root, "input")
            os.makedirs(input_dir)
            original = os.path.join(input_dir, "image_01.jpg")
            with open(original, "wb") as f:
                f.write(b"image")

            crop.configure_paths(input_dir=input_dir, out_dir=root)

            self.assertEqual(
                crop.original_path_for_label_id("image_01.jpg__label_3"),
                original,
            )

    def test_stage1_uses_extension_in_label_name_to_avoid_same_stem_collision(self):
        crop = _import_crop()
        fake_img = types.SimpleNamespace(shape=(100, 100, 3), size=1)
        pred = {"x": 50, "y": 50, "width": 20, "height": 20, "class": crop.MODEL1_LABEL_CLASS}

        with tempfile.TemporaryDirectory() as root:
            crop.configure_paths(input_dir=os.path.join(root, "input"), out_dir=root)
            os.makedirs(crop.INPUT_DIR)
            path_png = os.path.join(root, "a", "same.png")
            path_jpg = os.path.join(root, "b", "same.jpg")

            with mock.patch.object(crop, "read_image", return_value=fake_img):
                with mock.patch.object(crop, "infer_with_resize", return_value=[pred]):
                    with mock.patch.object(crop, "crop_from_pred", return_value=fake_img):
                        with mock.patch.object(crop, "save_png_required", side_effect=lambda path, _im
g, _ctx: path):
                            out_png = crop.stage1_crop_labels(path_png)
                            out_jpg = crop.stage1_crop_labels(path_jpg)

            self.assertEqual(os.path.basename(out_png[0]), "same.png__label_1.png")
            self.assertEqual(os.path.basename(out_jpg[0]), "same.jpg__label_1.png")
            self.assertNotEqual(os.path.basename(out_png[0]), os.path.basename(out_jpg[0]))

    def test_save_png_required_raises_when_write_fails(self):
        crop = _import_crop()
        with tempfile.TemporaryDirectory() as root:
            target = os.path.join(root, "out.png")
            with mock.patch.object(crop, "save_png", return_value=False):
                with self.assertRaisesRegex(RuntimeError, "Failed to write test crop"):
                    crop.save_png_required(target, object(), "test crop")

    def test_infer_with_resize_uses_unique_temp_file_and_cleans_it(self):
        crop = _import_crop()

        calls = []

        class FakeClient:
            def infer(self, path, model_id):
                calls.append(path)
                self.seen_exists = os.path.exists(path)
                return {"predictions": []}

        fake_client = FakeClient()
        fake_img = types.SimpleNamespace(shape=(100, 200, 3))

        def write_tmp(_bgr, path, quality=85):
            with open(path, "wb") as f:
                f.write(b"tmp")
            return True

        with tempfile.TemporaryDirectory() as root:
            crop.TMP_DIR = root
            with mock.patch.object(crop, "get_inference_client", return_value=fake_client):
                with mock.patch.object(crop, "_write_tmp_jpg", side_effect=write_tmp):
                    self.assertEqual(crop.infer_with_resize(fake_img, "same__name.png", "model/1"), []
)
                    self.assertEqual(crop.infer_with_resize(fake_img, "same__name.png", "model/1"), []
)

            self.assertEqual(len(calls), 2)
            self.assertNotEqual(calls[0], calls[1])
            self.assertFalse(os.path.exists(calls[0]))
            self.assertFalse(os.path.exists(calls[1]))


class GuiPipelineTest(unittest.TestCase):
    @staticmethod
    def _reset_gui_import_modules():
        for name in (
            "gui_app",
            "crop",
            "scan2",
            "barcode",
            "cv2",
            "numpy",
            "pyzbar",
            "pyzbar.pyzbar",
            "app_paths",
            "openpyxl",
        ):
            sys.modules.pop(name, None)

    def test_same_basename_sources_are_staged_with_unique_names(self):
        import gui_pipeline

        with tempfile.TemporaryDirectory() as root:
            source_a = os.path.join(root, "a")
            source_b = os.path.join(root, "b")
            os.makedirs(source_a)
            os.makedirs(source_b)
            path_a = os.path.join(source_a, "same.png")
            path_b = os.path.join(source_b, "same.png")
            with open(path_a, "wb") as f:
                f.write(b"a")
            with open(path_b, "wb") as f:
                f.write(b"b")

            run_dir, records = gui_pipeline.copy_images_to_unique_run_dir(
                [path_a, path_b],
                os.path.join(root, "new_images"),
            )

            names = [record["input_name"] for record in records]
            self.assertEqual(len(names), 2)
            self.assertEqual(len(set(names)), 2)
            self.assertEqual(names, ["input_0001.png", "input_0002.png"])
            self.assertEqual([record["source_name"] for record in records], ["same.png", "same.png"])
            self.assertTrue(os.path.exists(os.path.join(run_dir, names[0])))
            self.assertTrue(os.path.exists(os.path.join(run_dir, names[1])))
            with open(os.path.join(run_dir, "source_manifest.jsonl"), "r", encoding="utf-8") as f:
                manifest_rows = [json.loads(line) for line in f]
            manifest_text = json.dumps(manifest_rows, ensure_ascii=False)
            self.assertNotIn("source_path", manifest_text)
            self.assertNotIn("input_path", manifest_text)
            self.assertNotIn(os.path.abspath(root), manifest_text)
            for row in manifest_rows:
                self.assertEqual(set(row), {"source_index", "input_name", "sha256"})
                self.assertNotIn("same.png", row["input_name"])
            self.assertEqual([row["source_index"] for row in manifest_rows], [1, 2])
            self.assertEqual([row["input_name"] for row in manifest_rows], names)
            self.assertTrue(all(len(row["sha256"]) == 64 for row in manifest_rows))

    def test_gui_display_label_uses_original_source_name(self):
        self._reset_gui_import_modules()
        import gui_app

        self.assertEqual(
            gui_app._rewrite_label_id_for_display(
                "input_0005.jpg__label_1",
                {"input_0005.jpg": "泰州.jpg"},
            ),
            "泰州.jpg__label_1",
        )

    def test_gui_display_label_keeps_uniqueness_for_duplicate_source_names(self):
        self._reset_gui_import_modules()
        import gui_app

        mapping = gui_app._build_input_name_map(
            [
                {"source_index": 1, "source_name": "same.png", "input_name": "input_0001.png"},
                {"source_index": 2, "source_name": "same.png", "input_name": "input_0002.png"},
            ]
        )
        self.assertEqual(mapping["input_0001.png"], "same.png#1")
        self.assertEqual(mapping["input_0002.png"], "same.png#2")
        self.assertEqual(
            gui_app._rewrite_label_id_for_display("input_0002.png__label_1", mapping),
            "same.png#2__label_1",
        )

    def test_gui_display_source_labels_are_chinese(self):
        self._reset_gui_import_modules()
        import gui_app

        self.assertEqual(gui_app._display_model_src("barcode"), "扫描条形码")
        self.assertEqual(gui_app._display_model_src("barcode_unmatched"), "扫描条形码(未匹配规则)")
        self.assertEqual(gui_app._display_model_src("barcode_ocr_agree"), "文字识别+扫描条形码(一致)")
        self.assertEqual(gui_app._display_model_src("ocr_file"), "文字识别")
        self.assertEqual(gui_app._display_model_src("ocr_file+sn_hint"), "文字识别(由SN推断修正)")
        self.assertEqual(gui_app._display_model_src("missing+sn_hint"), "缺失(由SN推断)")
        self.assertEqual(gui_app._display_sn_src("barcode"), "扫描条形码")
        self.assertEqual(gui_app._display_sn_src("barcode_unmatched"), "扫描条形码(未匹配规则)")
        self.assertEqual(gui_app._display_sn_src("barcode_ocr_agree"), "文字识别+扫描条形码(一致)")
        self.assertEqual(gui_app._display_sn_src("ocr"), "文字识别")
        self.assertEqual(gui_app._display_sn_src("barcode_decoder_miss"), "条码未命中")

    def test_gui_localize_result_log_line_to_chinese(self):
        self._reset_gui_import_modules()
        import gui_app

        line = "[input_0001.jpg__label_2] MODEL=AP362E (M_SRC=barcode) | SN=21500871474ES1014701 (SN_S
RC=barcode)"
        localized = gui_app._localize_log_text(
            line,
            {"input_0001.jpg": "泰州.jpg"},
        )

        self.assertEqual(
            localized,
            "[泰州.jpg__label_2] 型号=AP362E（来源=扫描条形码） | SN=21500871474ES1014701（来源=扫描条形码）",
        )

    def test_gui_localize_stage1_log_lines_to_chinese(self):
        self._reset_gui_import_modules()
        import gui_app

        self.assertEqual(
            gui_app._localize_log_text("Stage1: input_0001.jpg -> 5 label crops", {}),
            "阶段1：input_0001.jpg -> 5 个标签裁剪图",
        )
        self.assertEqual(
            gui_app._localize_log_text("Stage1 complete: 5 label crops generated", {}),
            "阶段1完成：生成 5 个标签裁剪图",
        )
        self.assertEqual(
            gui_app._localize_log_text("Stats: at least one field 5; both fields 5", {}),
            "统计：至少识别出一个字段 5；两个字段都识别出 5",
        )

    def test_gui_import_does_not_import_pipeline_modules(self):
        self._reset_gui_import_modules()

        importlib.import_module("gui_app")

        self.assertNotIn("crop", sys.modules)
        self.assertNotIn("scan2", sys.modules)

    def test_gui_log_mask_keeps_app_relative_output_paths(self):
        self._reset_gui_import_modules()
        import gui_app

        app_path = os.path.join(os.getcwd(), "stage2_fields", "manifest.jsonl")
        external_path = r"F:\wechat\xwechat_files\sample.jpg"

        masked = gui_app._mask_path_text(
            f"Manifest: {app_path}; source: {external_path}"
        )

        self.assertIn(os.path.join("stage2_fields", "manifest.jsonl"), masked)
        self.assertIn("source: [path]", masked)
        self.assertNotIn(external_path, masked)

    def test_gui_run_pipeline_requests_clean_crop_outputs(self):
        self._reset_gui_import_modules()
        import gui_app

        source = inspect.getsource(gui_app.App.run_pipeline)

        self.assertIn("crop_module.main(input_dir=input_dir, clean=True)", source)


class BarcodeCliBudgetTest(unittest.TestCase):
    @unittest.skipUnless(os.name == "nt", "Windows process flags only")
    def test_global_subprocess_patch_hides_check_output_children(self):
        sys.modules.pop("win_subprocess", None)
        import win_subprocess

        original_popen = subprocess.Popen
        try:
            class FakePopen:
                calls = []

                def __init__(self, *args, **kwargs):
                    self.args = args
                    self.kwargs = kwargs
                    self.returncode = 0
                    FakePopen.calls.append((args, kwargs))

                def __enter__(self):
                    return self

                def __exit__(self, *_exc_info):
                    return False

                def communicate(self, input=None, timeout=None):
                    return b"ok", b""

                def poll(self):
                    return self.returncode

                def kill(self):
                    self.returncode = -1

                def wait(self, timeout=None):
                    return self.returncode

            subprocess.Popen = FakePopen
            if hasattr(subprocess, "_huaweiocr_hidden_windows"):
                delattr(subprocess, "_huaweiocr_hidden_windows")
            win_subprocess.hide_subprocess_windows()

            self.assertTrue(issubclass(subprocess.Popen, FakePopen))

            class LibraryPopenSubclass(subprocess.Popen):
                pass

            self.assertTrue(issubclass(LibraryPopenSubclass, FakePopen))
            subprocess.check_output(["cmd.exe", "/c", "echo", "ok"])

            kwargs = FakePopen.calls[-1][1]
            self.assertTrue(kwargs["creationflags"] & subprocess.CREATE_NO_WINDOW)
            self.assertTrue(kwargs["creationflags"] & subprocess.DETACHED_PROCESS)
            self.assertEqual(kwargs["startupinfo"].wShowWindow, subprocess.SW_HIDE)
        finally:
            subprocess.Popen = original_popen
            if hasattr(subprocess, "_huaweiocr_hidden_windows"):
                delattr(subprocess, "_huaweiocr_hidden_windows")

    @unittest.skipUnless(os.name == "nt", "Windows process flags only")
    def test_run_cli_uses_detached_hidden_process_flags(self):
        barcode = _import_barcode()

        with mock.patch.object(barcode.subprocess, "run") as run_mock:
            barcode._run_cli(["BarcodeReaderCLI.exe", "--version"])

        kwargs = run_mock.call_args.kwargs
        self.assertTrue(kwargs["creationflags"] & barcode.subprocess.CREATE_NO_WINDOW)
        self.assertTrue(kwargs["creationflags"] & barcode.subprocess.DETACHED_PROCESS)
        self.assertEqual(kwargs["startupinfo"].wShowWindow, barcode.subprocess.SW_HIDE)

    def test_decode_small_patch_caps_cli_attempts(self):
        barcode = _import_barcode()
        fake_img = types.SimpleNamespace(shape=(10, 10))
        calls = []

        def fake_cli(_img, _tag):
            calls.append(_tag)
            return []

        with mock.patch.object(barcode, "decode_with_transforms", return_value=[]):
            with mock.patch.object(barcode, "crop_bar_band", return_value=fake_img):
                with mock.patch.object(barcode, "enhance_band", return_value=(fake_img, fake_img)):
                    with mock.patch.object(barcode, "_rotate90", side_effect=lambda img, _k: img):
                        with mock.patch.object(barcode, "decode_with_cli", side_effect=fake_cli):
                            barcode.decode_small_patch(fake_img)

        self.assertEqual(len(calls), barcode.CLI_MAX_CALLS_PER_PATCH)


class Scan2DebugLogTest(unittest.TestCase):
    def test_info_mode_does_not_create_debug_log(self):
        scan2 = _import_scan2()
        with tempfile.TemporaryDirectory() as root:
            log_path = os.path.join(root, "debug.log")
            scan2.DEBUG_LOG_PATH = log_path
            scan2.set_log_level("info")

            scan2.append_debug("MODEL path C:/secret/customer_4E25A017ABCDEFG")

            self.assertFalse(os.path.exists(log_path))

    def test_debug_log_masks_sensitive_text(self):
        scan2 = _import_scan2()
        with tempfile.TemporaryDirectory() as root:
            log_path = os.path.join(root, "debug.log")
            scan2.DEBUG_LOG_PATH = log_path
            scan2.set_log_level("debug")

            scan2.append_sensitive_debug(r"SN C:\customer\asset_4E25A017ABCDEFG.png | 4E25A017ABCDEFG"
)

            with open(log_path, "r", encoding="utf-8") as f:
                data = f.read()
            self.assertNotIn("4E25A017ABCDEFG", data)
            self.assertNotIn("customer", data)
            self.assertNotIn("asset_4E25A017ABCDEFG.png", data)
            self.assertIn("4E25", data)


class AppPathsInstallTest(unittest.TestCase):
    def test_incomplete_model_dir_is_replaced(self):
        import app_paths

        with tempfile.TemporaryDirectory() as root:
            bundled = os.path.join(root, "bundled", "models", "official_models")
            source_model = os.path.join(bundled, "model_a")
            os.makedirs(source_model)
            with open(os.path.join(source_model, "weights.bin"), "wb") as f:
                f.write(b"complete")

            data_dir = os.path.join(root, "data")
            target = os.path.join(data_dir, "models", "official_models", "model_a")
            os.makedirs(target)
            with open(os.path.join(target, "partial.bin"), "wb") as f:
                f.write(b"partial")

            def fake_resource_path(*parts):
                return os.path.join(root, "bundled", *parts)

            with mock.patch.object(app_paths, "get_resource_path", side_effect=fake_resource_path):
                with mock.patch.dict(os.environ, {"HUAWEIOCR_DATA_DIR": data_dir, "HUAWEIOCR_MODEL_DIR
": ""}, clear=False):
                    app_paths.ensure_models_installed()

            self.assertTrue(os.path.exists(os.path.join(target, "weights.bin")))
            self.assertTrue(os.path.exists(os.path.join(target, app_paths.MODEL_INSTALL_MARKER)))
            self.assertFalse(os.path.exists(os.path.join(target, "partial.bin")))

    def test_stale_model_install_lock_is_recovered(self):
        import app_paths

        with tempfile.TemporaryDirectory() as root:
            bundled = os.path.join(root, "bundled", "models", "official_models")
            source_model = os.path.join(bundled, "model_a")
            os.makedirs(source_model)
            with open(os.path.join(source_model, "weights.bin"), "wb") as f:
                f.write(b"complete")

            data_dir = os.path.join(root, "data")
            target_root = os.path.join(data_dir, "models", "official_models")
            os.makedirs(target_root)
            lock_path = os.path.join(target_root, ".huaweiocr_install.lock")
            with open(lock_path, "w", encoding="utf-8") as f:
                f.write("")
            old = time.time() - 10
            os.utime(lock_path, (old, old))

            def fake_resource_path(*parts):
                return os.path.join(root, "bundled", *parts)

            with mock.patch.object(app_paths, "get_resource_path", side_effect=fake_resource_path):
                with mock.patch.dict(os.environ, {"HUAWEIOCR_DATA_DIR": data_dir, "HUAWEIOCR_MODEL_DIR
": ""}, clear=False):
                    app_paths.ensure_models_installed()

            target = os.path.join(target_root, "model_a")
            self.assertTrue(os.path.exists(os.path.join(target, "weights.bin")))
            self.assertTrue(os.path.exists(os.path.join(target, app_paths.MODEL_INSTALL_MARKER)))
            self.assertFalse(os.path.exists(lock_path))

    def test_stale_lock_reclaim_does_not_remove_changed_lock(self):
        import app_paths

        with tempfile.TemporaryDirectory() as root:
            lock_path = os.path.join(root, ".huaweiocr_install.lock")
            with open(lock_path, "w", encoding="utf-8") as f:
                f.write("")
            old = time.time() - 10
            os.utime(lock_path, (old, old))
            real_read = app_paths._read_lock_snapshot
            calls = {"count": 0}

            def racing_read(path):
                snapshot = real_read(path)
                calls["count"] += 1
                if calls["count"] == 1:
                    with open(lock_path, "w", encoding="utf-8") as f:
                        f.write(f"{os.getpid()}\n{time.time()}\n")
                return snapshot

            with mock.patch.object(app_paths, "_read_lock_snapshot", side_effect=racing_read):
                self.assertFalse(app_paths._reclaim_stale_lock(lock_path))

            self.assertTrue(os.path.exists(lock_path))
            with open(lock_path, "r", encoding="utf-8") as f:
                data = f.read()
            self.assertIn(str(os.getpid()), data)


if __name__ == "__main__":
    unittest.main()

 succeeded in 878ms:
import json
import os
import tempfile
import types
import unittest
from unittest import mock

import sn_barcode
import validate_sn_barcodes
from tests.test_locked_output_dirs import _import_scan2


class SnBarcodeSelectionTest(unittest.TestCase):
    def test_non_sn_payloads_are_rejected(self):
        self.assertEqual(sn_barcode.extract_sn_from_payload("SF3260776146675"), "")
        self.assertEqual(sn_barcode.extract_sn_from_payload("EAN: 6971234567890"), "")
        self.assertEqual(sn_barcode.extract_sn_from_payload("MAC:E49024187A70"), "")
        self.assertEqual(sn_barcode.extract_sn_from_payload("Part No: 50087149"), "")

    def test_valid_sn_payload_is_extracted(self):
        self.assertEqual(
            sn_barcode.extract_sn_from_payload("SN:4E25A0170000"),
            "4E25A0170000",
        )
        self.assertEqual(
            sn_barcode.extract_sn_from_payload("S/N:21500871474ES1016219"),
            "21500871474ES1016219",
        )
        self.assertEqual(
            sn_barcode.extract_sn_from_payload("S/N 21500671494ERA050003"),
            "21500671494ERA050003",
        )
        self.assertEqual(
            sn_barcode.extract_sn_from_payload("[)>06 1P50087149 18VLEHWT S21500871494ERB006054"),
            "21500871494ERB006054",
        )

    def test_learn_sn_pattern_extends_sn20_matching(self):
        with tempfile.TemporaryDirectory() as root:
            learned_file = os.path.join(root, "sn_segments.json")
            sample = "21500871474ZX1016219"
            with mock.patch.dict(os.environ, {"HUAWEIOCR_SN_SEGMENTS_FILE": learned_file}, clear=False
):
                self.assertEqual(sn_barcode.extract_sn_from_payload(f"S/N:{sample}"), "")
                self.assertTrue(sn_barcode.learn_sn_pattern(sample))
                self.assertEqual(sn_barcode.extract_sn_from_payload(f"S/N:{sample}"), sample)
                with open(learned_file, "r", encoding="utf-8") as f:
                    payload = json.load(f)
                self.assertIn("ZX", payload)

    def test_conflicting_barcode_sns_are_ambiguous(self):
        report = sn_barcode.select_sn_from_decoder_results(
            [
                sn_barcode.DecoderResult("fake", "SN:4E25A0170000", "sn", "sn"),
                sn_barcode.DecoderResult("fake", "SN:4E25A0170001", "sn", "sn.region.1"),
            ]
        )

        self.assertEqual(report.status, "ambiguous")
        self.assertEqual(report.ambiguous_sns, ["4E25A0170000", "4E25A0170001"])

    def test_parse_failure_keeps_best_raw_payload_for_rule_expansion(self):
        with tempfile.TemporaryDirectory() as root:
            learned_file = os.path.join(root, "sn_segments.json")
            with mock.patch.dict(os.environ, {"HUAWEIOCR_SN_SEGMENTS_FILE": learned_file}, clear=False
):
                report = sn_barcode.select_sn_from_decoder_results(
                    [
                        sn_barcode.DecoderResult("fake", "Part No:50087147", "label", "label"),
                        sn_barcode.DecoderResult("fake", "S/N:9Z123456789ABCD", "sn", "sn"),
                    ]
                )

                self.assertEqual(report.status, "parse_failure")
                self.assertEqual(report.raw_text, "S/N:9Z123456789ABCD")
                self.assertEqual(report.source_region, "sn")

    def test_parse_failure_prefers_sn_like_payload_over_noise_same_source(self):
        with tempfile.TemporaryDirectory() as root:
            learned_file = os.path.join(root, "sn_segments.json")
            with mock.patch.dict(os.environ, {"HUAWEIOCR_SN_SEGMENTS_FILE": learned_file}, clear=False
):
                report = sn_barcode.select_sn_from_decoder_results(
                    [
                        sn_barcode.DecoderResult("a_decoder", "Part No:50087147", "sn", "sn"),
                        sn_barcode.DecoderResult("z_decoder", "S/N:21500871474ZX1016219", "sn", "sn"),
                    ]
                )
                self.assertEqual(report.status, "parse_failure")
                self.assertEqual(report.raw_text, "S/N:21500871474ZX1016219")

    def test_duplicate_sn_uses_source_priority(self):
        report = sn_barcode.select_sn_from_decoder_results(
            [
                sn_barcode.DecoderResult("fake", "SN:4E25A0170000", "label", "label"),
                sn_barcode.DecoderResult("fake", "SN:4E25A0170000", "sn", "sn"),
            ]
        )

        self.assertEqual(report.status, "hit")
        self.assertEqual(report.sn, "4E25A0170000")
        self.assertEqual(report.source_region, "sn")

    def test_unique_higher_priority_source_wins_over_original_conflicts(self):
        report = sn_barcode.select_sn_from_decoder_results(
            [
                sn_barcode.DecoderResult("fake", "SN:21500872884ERA005572", "sn", "sn"),
                sn_barcode.DecoderResult("fake", "SN:21500872884ERA005405", "original", "original"),
                sn_barcode.DecoderResult("fake", "SN:21500872884ERA005765", "original", "original"),
            ]
        )

        self.assertEqual(report.status, "hit")
        self.assertEqual(report.sn, "21500872884ERA005572")
        self.assertEqual(report.source_region, "sn")

    def test_label_source_wins_over_original_region_conflicts(self):
        report = sn_barcode.select_sn_from_decoder_results(
            [
                sn_barcode.DecoderResult(
                    "fake",
                    "SN:21500872884ERA005572",
                    "label",
                    "label.rot0.full",
                ),
                sn_barcode.DecoderResult(
                    "fake",
                    "SN:21500872884ERA005405",
                    "original",
                    "original.rot0.region.1",
                ),
            ]
        )

        self.assertEqual(report.status, "hit")
        self.assertEqual(report.sn, "21500872884ERA005572")
        self.assertEqual(report.source_region, "label.rot0.full")

    def test_attempt_budget_is_isolated_per_source(self):
        candidates_by_source = {
            "sn": [sn_barcode.CandidateImage(object(), "sn", "sn", "raw")],
            "label": [sn_barcode.CandidateImage(object(), "label", "label", "raw")],
        }

        def fake_candidates(_image, source, max_candidates=96):
            return candidates_by_source[source]

        def fake_decode(candidate):
            if candidate.source == "label":
                return [
                    sn_barcode.DecoderResult("fake", "SN:4E25A0170000", "label", "label")
                ], []
            return [], []

        with mock.patch.object(sn_barcode, "_read_image", return_value=object()):
            with mock.patch.object(sn_barcode, "generate_candidate_images", side_effect=fake_candidate
s):
                with mock.patch.object(sn_barcode, "_decode_pyzbar", side_effect=fake_decode):
                    report = sn_barcode.scan_sn_barcodes(
                        [("sn", "sn.png"), ("label", "label.png")],
                        max_decoder_attempts=1,
                    )

        self.assertEqual(report.status, "hit")
        self.assertEqual(report.sn, "4E25A0170000")
        self.assertEqual(report.attempts, 2)


class Scan2BarcodeAccountingTest(unittest.TestCase):
    def test_scan2_extract_sn_prefers_full_es_pattern(self):
        scan2 = _import_scan2()
        self.assertEqual(
            scan2.extract_sn_from_text("S/N:21500871474ES1016219"),
            "21500871474ES1016219",
        )

    def test_ambiguous_barcode_is_not_silently_selected(self):
        scan2 = _import_scan2()
        report = sn_barcode.SnBarcodeReport(
            status="ambiguous",
            attempts=2,
            decoded_count=2,
            ambiguous_sns=["4E25A0170000", "4E25A0170001"],
            results=[
                sn_barcode.DecoderResult("fake", "SN:4E25A0170000", "sn", "sn"),
                sn_barcode.DecoderResult("fake", "SN:4E25A0170001", "label", "label"),
            ],
        )

        with mock.patch.object(scan2, "_scan_sn_barcode_report", return_value=report):
            sn, raw, source, meta = scan2.recognize_sn(
                "sn.png",
                label_id="a__label_1",
                label_path="label.png",
                allow_ocr=False,
            )

        self.assertEqual(sn, "")
        self.assertEqual(source, "barcode_ambiguous")
        self.assertIn("BARCODE_AMBIGUOUS", raw)
        self.assertEqual(meta["barcode_status"], "ambiguous")

    def test_parse_failure_falls_back_to_barcode_text_instead_of_empty(self):
        scan2 = _import_scan2()
        report = sn_barcode.SnBarcodeReport(
            status="parse_failure",
            raw_text="S/N:21500871474ZX1016219",
            source_region="sn",
            attempts=1,
            decoded_count=1,
            results=[
                sn_barcode.DecoderResult("fake", "S/N:21500871474ZX1016219", "sn", "sn"),
            ],
        )

        with mock.patch.object(scan2, "_scan_sn_barcode_report", return_value=report):
            sn, raw, source, meta = scan2.recognize_sn(
                "sn.png",
                label_id="a__label_1",
                allow_ocr=False,
            )

        self.assertEqual(sn, "21500871474ZX1016219")
        self.assertEqual(source, "barcode_unmatched")
        self.assertIn("BARCODE_UNMATCHED", raw)
        self.assertEqual(meta["barcode_status"], "parse_failure")

    def test_parse_failure_non_sn_payload_stays_parse_fail(self):
        scan2 = _import_scan2()
        report = sn_barcode.SnBarcodeReport(
            status="parse_failure",
            raw_text="Part No:50087147",
            source_region="sn",
            attempts=1,
            decoded_count=1,
            results=[
                sn_barcode.DecoderResult("fake", "Part No:50087147", "sn", "sn"),
            ],
        )

        with mock.patch.object(scan2, "_scan_sn_barcode_report", return_value=report):
            sn, raw, source, meta = scan2.recognize_sn(
                "sn.png",
                label_id="a__label_1",
                allow_ocr=False,
            )

        self.assertEqual(sn, "")
        self.assertEqual(source, "barcode_parse_fail")
        self.assertIn("BARCODE_PARSE_FAIL", raw)
        self.assertEqual(meta["barcode_status"], "parse_failure")

    def test_parse_failure_non_sn_prefixed_sn_like_payload_stays_parse_fail(self):
        scan2 = _import_scan2()
        report = sn_barcode.SnBarcodeReport(
            status="parse_failure",
            raw_text="MAC:4E25A0170000",
            source_region="sn",
            attempts=1,
            decoded_count=1,
            results=[
                sn_barcode.DecoderResult("fake", "MAC:4E25A0170000", "sn", "sn"),
            ],
        )

        with mock.patch.object(scan2, "_scan_sn_barcode_report", return_value=report):
            sn, raw, source, meta = scan2.recognize_sn(
                "sn.png",
                label_id="a__label_1",
                allow_ocr=False,
            )

        self.assertEqual(sn, "")
        self.assertEqual(source, "barcode_parse_fail")
        self.assertIn("BARCODE_PARSE_FAIL", raw)
        self.assertEqual(meta["barcode_status"], "parse_failure")

    def test_parse_failure_invalid_length_sn20_stays_parse_fail(self):
        scan2 = _import_scan2()
        report = sn_barcode.SnBarcodeReport(
            status="parse_failure",
            raw_text="S/N:2123456789AB1234",
            source_region="sn",
            attempts=1,
            decoded_count=1,
            results=[
                sn_barcode.DecoderResult("fake", "S/N:2123456789AB1234", "sn", "sn"),
            ],
        )

        with mock.patch.object(scan2, "_scan_sn_barcode_report", return_value=report):
            sn, raw, source, meta = scan2.recognize_sn(
                "sn.png",
                label_id="a__label_1",
                allow_ocr=False,
            )

        self.assertEqual(sn, "")
        self.assertEqual(source, "barcode_parse_fail")
        self.assertIn("BARCODE_PARSE_FAIL", raw)
        self.assertEqual(meta["barcode_status"], "parse_failure")

    def test_parse_failure_barcode_and_ocr_agree_then_auto_learn(self):
        scan2 = _import_scan2()
        report = sn_barcode.SnBarcodeReport(
            status="parse_failure",
            raw_text="S/N:21500871474ZX1016219",
            source_region="sn",
            attempts=1,
            decoded_count=1,
            results=[
                sn_barcode.DecoderResult("fake", "S/N:21500871474ZX1016219", "sn", "sn"),
            ],
        )

        with tempfile.TemporaryDirectory() as root:
            learned_file = os.path.join(root, "sn_segments.json")
            with mock.patch.dict(os.environ, {"HUAWEIOCR_SN_SEGMENTS_FILE": learned_file}, clear=False
):
                with mock.patch.object(scan2, "_scan_sn_barcode_report", return_value=report):
                    with mock.patch.object(scan2, "load_for_ocr_color", return_value=object()):
                        with mock.patch.object(
                            scan2,
                            "ocr_text_with_details",
                            return_value=("S/N:21500871474ZX1016219", "S/N:21500871474ZX1016219", []),
                        ):
                            sn, raw, source, meta = scan2.recognize_sn(
                                "sn.png",
                                label_id="a__label_1",
                                allow_ocr=True,
                            )

                self.assertEqual(sn, "21500871474ZX1016219")
                self.assertEqual(source, "barcode_ocr_agree")
                self.assertIn("BARCODE_OCR_AGREE", raw)
                self.assertTrue(meta.get("sn_pattern_learned"))
                self.assertEqual(
                    sn_barcode.extract_sn_from_payload("S/N:21500871474ZX1016219"),
                    "21500871474ZX1016219",
                )

    def test_parse_failure_barcode_ocr_agree_does_not_crash_when_learning_fails(self):
        scan2 = _import_scan2()
        report = sn_barcode.SnBarcodeReport(
            status="parse_failure",
            raw_text="S/N:21500871474ZX1016219",
            source_region="sn",
            attempts=1,
            decoded_count=1,
            results=[
                sn_barcode.DecoderResult("fake", "S/N:21500871474ZX1016219", "sn", "sn"),
            ],
        )

        with mock.patch.object(scan2, "_scan_sn_barcode_report", return_value=report):
            with mock.patch.object(scan2, "load_for_ocr_color", return_value=object()):
                with mock.patch.object(
                    scan2,
                    "ocr_text_with_details",
                    return_value=("S/N:21500871474ZX1016219", "S/N:21500871474ZX1016219", []),
                ):
                    with mock.patch.object(scan2, "learn_sn_pattern", side_effect=OSError("read-only")
):
                        sn, raw, source, meta = scan2.recognize_sn(
                            "sn.png",
                            label_id="a__label_1",
                            allow_ocr=True,
                        )

        self.assertEqual(sn, "21500871474ZX1016219")
        self.assertEqual(source, "barcode_ocr_agree")
        self.assertIn("BARCODE_OCR_AGREE", raw)
        self.assertEqual(meta.get("sn_pattern_learn_error"), "OSError")

    def test_main_reports_barcode_hit_rate_and_ocr_recovery_separately(self):
        scan2 = _import_scan2()

        with tempfile.TemporaryDirectory() as root:
            stage2 = os.path.join(root, "stage2_fields")
            model_dir = os.path.join(stage2, "model")
            sn_dir = os.path.join(stage2, "sn")
            os.makedirs(model_dir)
            os.makedirs(sn_dir)
            sn_path = os.path.join(sn_dir, "a__label_1__sn.png")
            open(sn_path, "wb").close()
            with open(os.path.join(stage2, "manifest.jsonl"), "w", encoding="utf-8") as manifest:
                manifest.write(json.dumps({"label_id": "a__label_1", "sn_path": sn_path}) + "\n")

            meta = {
                "barcode_found": False,
                "ocr_text_found": True,
                "barcode_status": "decoder_miss",
                "barcode_attempts": 3,
                "barcode_decoded_count": 0,
            }
            with mock.patch.object(scan2, "recognize_sn", return_value=("4E25A0170000", "ocr raw", "oc
r", meta)):
                stats = scan2.main(
                    model_dir=model_dir,
                    sn_dir=sn_dir,
                    out_jsonl=os.path.join(root, "out.jsonl"),
                    debug_log=os.path.join(root, "debug.log"),
                )

        self.assertEqual(stats["sn_success"], 1)
        self.assertEqual(stats["sn_barcode_hits"], 0)
        self.assertEqual(stats["sn_ocr_recoveries"], 1)
        self.assertEqual(stats["sn_barcode_attempts"], 3)
        self.assertEqual(stats["sn_barcode_hit_rate"], 0.0)

    def test_main_counts_barcode_unmatched_as_barcode_hit(self):
        scan2 = _import_scan2()

        with tempfile.TemporaryDirectory() as root:
            stage2 = os.path.join(root, "stage2_fields")
            model_dir = os.path.join(stage2, "model")
            sn_dir = os.path.join(stage2, "sn")
            os.makedirs(model_dir)
            os.makedirs(sn_dir)
            sn_path = os.path.join(sn_dir, "a__label_1__sn.png")
            open(sn_path, "wb").close()
            with open(os.path.join(stage2, "manifest.jsonl"), "w", encoding="utf-8") as manifest:
                manifest.write(json.dumps({"label_id": "a__label_1", "sn_path": sn_path}) + "\n")

            meta = {
                "barcode_found": True,
                "ocr_text_found": False,

codex
我正在核对现有测试是否已经覆盖这几个 FAIL 点，避免引入行为偏移；接着会只改必要行并同步最小测试更新。
exec
"C:\\Program Files\\WindowsApps\\Microsoft.PowerShell_7.6.0.0_x64__8wekyb3d8bbwe\\pwsh.exe" -Command "
Get-Content -Path tests\\test_sn_barcode_scanning.py | Select-Object -Skip 420 -First 220" in D:\newhu
awei
 succeeded in 666ms:
                "barcode_status": "parse_failure",
                "barcode_attempts": 1,
                "barcode_decoded_count": 1,
            }
            with mock.patch.object(
                scan2,
                "recognize_sn",
                return_value=("9Z123456789ABCD", "raw barcode", "barcode_unmatched", meta),
            ):
                stats = scan2.main(
                    model_dir=model_dir,
                    sn_dir=sn_dir,
                    out_jsonl=os.path.join(root, "out.jsonl"),
                    debug_log=os.path.join(root, "debug.log"),
                )

        self.assertEqual(stats["sn_success"], 1)
        self.assertEqual(stats["sn_barcode_hits"], 1)
        self.assertEqual(stats["sn_barcode_parse_failures"], 1)

    def test_main_counts_barcode_ocr_agree_as_barcode_hit(self):
        scan2 = _import_scan2()

        with tempfile.TemporaryDirectory() as root:
            stage2 = os.path.join(root, "stage2_fields")
            model_dir = os.path.join(stage2, "model")
            sn_dir = os.path.join(stage2, "sn")
            os.makedirs(model_dir)
            os.makedirs(sn_dir)
            sn_path = os.path.join(sn_dir, "a__label_1__sn.png")
            open(sn_path, "wb").close()
            with open(os.path.join(stage2, "manifest.jsonl"), "w", encoding="utf-8") as manifest:
                manifest.write(json.dumps({"label_id": "a__label_1", "sn_path": sn_path}) + "\n")

            meta = {
                "barcode_found": True,
                "ocr_text_found": True,
                "barcode_status": "parse_failure",
                "barcode_attempts": 1,
                "barcode_decoded_count": 1,
            }
            with mock.patch.object(
                scan2,
                "recognize_sn",
                return_value=("21500871474ZX1016219", "raw agree", "barcode_ocr_agree", meta),
            ):
                stats = scan2.main(
                    model_dir=model_dir,
                    sn_dir=sn_dir,
                    out_jsonl=os.path.join(root, "out.jsonl"),
                    debug_log=os.path.join(root, "debug.log"),
                )

        self.assertEqual(stats["sn_success"], 1)
        self.assertEqual(stats["sn_barcode_hits"], 1)
        self.assertEqual(stats["sn_barcode_parse_failures"], 1)


class ValidationCommandTest(unittest.TestCase):
    def test_template_builder_creates_manual_review_rows_without_accepting_them(self):
        with tempfile.TemporaryDirectory() as root:
            stage2 = os.path.join(root, "stage2_fields")
            label_dir = os.path.join(root, "stage1_labels")
            sn_dir = os.path.join(stage2, "sn")
            os.makedirs(label_dir)
            os.makedirs(sn_dir)
            label_path = os.path.join(label_dir, "a__label_1.png")
            sn_path = os.path.join(sn_dir, "a__label_1__sn.png")
            label_path_2 = os.path.join(label_dir, "a__label_2.png")
            open(label_path, "wb").close()
            open(sn_path, "wb").close()
            open(label_path_2, "wb").close()

            stage2_manifest = os.path.join(stage2, "manifest.jsonl")
            with open(stage2_manifest, "w", encoding="utf-8") as manifest:
                manifest.write(json.dumps({
                    "label_id": "a__label_1",
                    "label_crop": label_path,
                    "sn_path": sn_path,
                    "sn_conf": 0.91,
                }) + "\n")
                manifest.write(json.dumps({
                    "label_id": "a__label_2",
                    "label_crop": label_path_2,
                    "sn_path": None,
                }) + "\n")

            recognized_jsonl = os.path.join(stage2, "recognized.jsonl")
            with open(recognized_jsonl, "w", encoding="utf-8") as recognized:
                recognized.write(json.dumps({
                    "label_id": "a__label_1",
                    "sn": "SN:4E25A0170000",
                    "sn_src": "ocr",
                }) + "\n")

            output_path = os.path.join(root, "validation", "candidates.jsonl")
            summary = validate_sn_barcodes.build_manifest_template_from_stage2(
                stage2_manifest,
                output_path,
                recognized_jsonl=recognized_jsonl,
            )

            with open(output_path, "r", encoding="utf-8") as f:
                rows = [json.loads(line) for line in f if line.strip()]

        self.assertEqual(summary["rows_written"], 2)
        self.assertEqual(rows[0]["label_id"], "a__label_1")
        self.assertEqual(rows[0]["expected_sn"], "")
        self.assertFalse(rows[0]["barcode_present"])
        self.assertFalse(rows[0]["accepted_quality"])
        self.assertEqual(rows[0]["pipeline_candidate_sn"], "4E25A0170000")
        self.assertIn("not ground truth", rows[0]["notes"])
        self.assertNotIn("sn_path", rows[1])

    def test_validation_fails_when_accepted_sample_count_is_too_small(self):
        with tempfile.TemporaryDirectory() as root:
            image_path = os.path.join(root, "label.png")
            open(image_path, "wb").close()
            manifest_path = os.path.join(root, "manifest.jsonl")
            with open(manifest_path, "w", encoding="utf-8") as manifest:
                manifest.write(json.dumps({
                    "image_path": image_path,
                    "label_id": "a__label_1",
                    "expected_sn": "4E25A0170000",
                    "barcode_present": True,
                    "accepted_quality": True,
                    "notes": "unit test",
                }) + "\n")

            fake_report = sn_barcode.SnBarcodeReport(
                status="hit",
                sn="4E25A0170000",
                raw_text="SN:4E25A0170000",
                source="label",
                source_region="label",
                decoder_name="fake",
                attempts=1,
                decoded_count=1,
            )
            with mock.patch.object(validate_sn_barcodes, "scan_sn_barcodes", return_value=fake_report)
:
                summary = validate_sn_barcodes.evaluate_manifest(manifest_path, min_accepted=50)

        self.assertFalse(summary.passed)
        self.assertEqual(summary.exact_hits, 1)
        self.assertIn("below required minimum 50", "\n".join(summary.errors))

    def test_validation_minimum_counts_only_accepted_barcode_rows(self):
        with tempfile.TemporaryDirectory() as root:
            manifest_path = os.path.join(root, "manifest.jsonl")
            with open(manifest_path, "w", encoding="utf-8") as manifest:
                for index in range(49):
                    image_path = os.path.join(root, f"label_{index}.png")
                    open(image_path, "wb").close()
                    manifest.write(json.dumps({
                        "image_path": image_path,
                        "label_id": f"a__label_{index}",
                        "expected_sn": "4E25A0170000",
                        "barcode_present": True,
                        "accepted_quality": True,
                        "notes": "unit test",
                    }) + "\n")
                non_barcode_path = os.path.join(root, "non_barcode.png")
                open(non_barcode_path, "wb").close()
                manifest.write(json.dumps({
                    "image_path": non_barcode_path,
                    "label_id": "non_barcode",
                    "expected_sn": "",
                    "barcode_present": False,
                    "accepted_quality": True,
                    "notes": "unit test",
                }) + "\n")

            fake_report = sn_barcode.SnBarcodeReport(
                status="hit",
                sn="4E25A0170000",
                raw_text="SN:4E25A0170000",
                source="label",
                source_region="label",
                decoder_name="fake",
                attempts=1,
                decoded_count=1,
            )
            with mock.patch.object(validate_sn_barcodes, "scan_sn_barcodes", return_value=fake_report)
:
                summary = validate_sn_barcodes.evaluate_manifest(manifest_path, min_accepted=50)

        self.assertFalse(summary.passed)
        self.assertEqual(summary.accepted_quality_rows, 50)
        self.assertEqual(summary.accepted_barcode_rows, 49)
        self.assertEqual(summary.denominator, 49)
        self.assertIn("accepted-quality barcode sample count 49", "\n".join(summary.errors))

    def test_validation_reports_below_threshold_failures(self):
        with tempfile.TemporaryDirectory() as root:
            image_path = os.path.join(root, "label.png")
            open(image_path, "wb").close()
            manifest_path = os.path.join(root, "manifest.jsonl")
            with open(manifest_path, "w", encoding="utf-8") as manifest:
                manifest.write(json.dumps({
                    "image_path": image_path,
                    "label_id": "a__label_1",
                    "expected_sn": "4E25A0170000",
                    "barcode_present": True,
                    "accepted_quality": True,
                    "notes": "unit test",
                }) + "\n")

            fake_report = sn_barcode.SnBarcodeReport(status="decoder_miss", attempts=1)
            with mock.patch.object(validate_sn_barcodes, "scan_sn_barcodes", return_value=fake_report)
:
                summary = validate_sn_barcodes.evaluate_manifest(
                    manifest_path,
                    threshold=0.90,
                    min_accepted=1,
                )

        self.assertFalse(summary.passed)
        self.assertEqual(summary.failure_counts["decoder_miss"], 1)
        self.assertIn("below threshold", "\n".join(summary.errors))


if __name__ == "__main__":

exec
"C:\\Program Files\\WindowsApps\\Microsoft.PowerShell_7.6.0.0_x64__8wekyb3d8bbwe\\pwsh.exe" -Command '
Get-Content -Path ocr.py | Select-Object -First 260' in D:\newhuawei
 succeeded in 660ms:
import os
import json
import re
import sys
import warnings
import inspect

from win_subprocess import hide_subprocess_windows

hide_subprocess_windows()

import paddle
from paddleocr import PaddleOCR
from app_paths import ensure_models_installed, get_resource_path

# ================== 配置区 ==================
# 你的 sn 小图目录
IMG_DIR = os.environ.get("OCR_IMG_DIR", os.path.join("stage2_fields", "sn"))

# 输出 JSONL 文件（每行一张图片的结果）
OUT_JSONL = os.environ.get("OCR_OUT_JSONL", os.path.join(IMG_DIR, "sn_ocr_results.jsonl"))

# 置信度阈值（低于这个就当成噪声丢掉）
MIN_SCORE = 0.5

# PaddleOCR 语言：
#   - 只识别数字/大写字母为主：建议 lang="en"
#   - 如果可能有中文：设置 HUAWEIOCR_OCR_LANG=ch
OCR_LANG = os.environ.get("HUAWEIOCR_OCR_LANG", "en").strip().lower()
if OCR_LANG not in {"en", "ch"}:
    OCR_LANG = "en"

# 是否使用 GPU（你现在一般是 CPU，就 False）
USE_GPU = False
OCR_PROFILE = os.environ.get("HUAWEIOCR_OCR_PROFILE", "mobile").strip().lower()
# ===========================================


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
    if OCR_LANG == "ch":
        if OCR_PROFILE in {"accurate", "server"}:
            return ["PP-OCRv5_server_rec", "PP-OCRv5_mobile_rec", "en_PP-OCRv5_mobile_rec"]
        return ["PP-OCRv5_mobile_rec", "PP-OCRv5_server_rec", "en_PP-OCRv5_mobile_rec"]
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

    # 你之前日志里用过这个环境变量，这里顺便设一下
    os.environ["DISABLE_MODEL_SOURCE_CHECK"] = "True"
    os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

    device = "gpu" if USE_GPU else "cpu"
    paddle.set_device(device)
    print("🔧 正在初始化 PaddleOCR 引擎（lang='{}', device='{}'）...".format(
        OCR_LANG, device
    ))
    ocr = PaddleOCR(
        lang=OCR_LANG,
        **_paddleocr_model_kwargs(model_root),
    )
    print("✅ OCR 引擎初始化完成")
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

codex
已确认 `app_paths.py` 本身 API 在；失败更像是测试中残留了 fake module。我要做的最小修复是：`AppPathsInstallTest` 强制重载真实 `app_paths`
，并修正 `scan2` 统计口径与相应测试断言。
exec
"C:\\Program Files\\WindowsApps\\Microsoft.PowerShell_7.6.0.0_x64__8wekyb3d8bbwe\\pwsh.exe" -Command '
Get-Content -Path scan2.py | Select-Object -Skip 930 -First 240' in D:\newhuawei
 succeeded in 675ms:
                if not os.path.isfile(item["model_path"]):
                    raise FileNotFoundError(f"Manifest model_path is missing at {path}:{line_no}: {ite
m['model_path']}")
                record["model_path"] = item["model_path"]
            if item.get("sn_path"):
                if not os.path.isfile(item["sn_path"]):
                    raise FileNotFoundError(f"Manifest sn_path is missing at {path}:{line_no}: {item['
sn_path']}")
                record["sn_path"] = item["sn_path"]
            original_path = item.get("original_image_path") or item.get("image_path")
            if original_path:
                if not os.path.isfile(original_path):
                    raise FileNotFoundError(f"Manifest original image is missing at {path}:{line_no}: 
{original_path}")
                record["original_image_path"] = original_path
    return records


# ===================== MAIN =====================
def main(out_dir=None, model_dir=None, sn_dir=None, out_jsonl=None, debug_log=None, log_level="info"):
    set_log_level(log_level)
    mask_raw = _env_flag("SCAN2_MASK_RAW") and not _env_flag("SCAN2_UNSAFE_RAW")
    model_barcode = _env_flag_default("SCAN2_MODEL_BARCODE", True)
    configure_paths(
        out_dir=out_dir,
        model_dir=model_dir,
        sn_dir=sn_dir,
        out_jsonl=out_jsonl,
        debug_log=debug_log,
    )
    os.makedirs(os.path.dirname(os.path.abspath(OUT_JSONL)) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(DEBUG_LOG_PATH)) or ".", exist_ok=True)
    start_debug_run()
    records = _load_manifest_records()
    model_files = []
    sn_files = []
    stats = {
        "records": 0,
        "model_total": 0,
        "model_success": 0,
        "model_barcode_hits": 0,
        "model_barcode_hit_rate": 0.0,
        "model_ocr_recoveries": 0,
        "sn_total": 0,
        "sn_attempted": 0,
        "sn_success": 0,
        "sn_barcode_attempts": 0,
        "sn_barcode_hits": 0,
        "sn_barcode_hit_rate": 0.0,
        "sn_ocr_recoveries": 0,
        "sn_barcode_parse_failures": 0,
        "sn_barcode_decoder_misses": 0,
        "sn_barcode_ambiguous": 0,
        "sn_barcode_quality_rejects": 0,
        "regex_fail": 0,
        "barcode_fail": 0,
        "ocr_fail": 0,
    }

    if os.path.isdir(MODEL_CROP_DIR):
        for fname in os.listdir(MODEL_CROP_DIR):
            ext = os.path.splitext(fname)[1].lower()
            if ext not in EXTS:
                continue
            model_files.append(fname)
            key = label_key(fname)
            records.setdefault(key, {})["model_path"] = os.path.join(MODEL_CROP_DIR, fname)

    if os.path.isdir(SN_CROP_DIR):
        for fname in os.listdir(SN_CROP_DIR):
            ext = os.path.splitext(fname)[1].lower()
            if ext not in EXTS:
                continue
            sn_files.append(fname)
            key = label_key(fname)
            records.setdefault(key, {})["sn_path"] = os.path.join(SN_CROP_DIR, fname)

    append_debug(f"[SCAN2] model_files={len(model_files)} sn_files={len(sn_files)} keys={len(records)}
")
    append_debug(f"[SCAN2] model_names={sorted(model_files)}")
    append_debug(f"[SCAN2] sn_names={sorted(sn_files)}")

    with open(OUT_JSONL, "w", encoding="utf-8") as f:
        for key in sorted(records.keys()):
            item = records[key]

            model_code = model_raw = ""
            sn_code = sn_raw = ""
            model_src = "missing"
            sn_src = "missing"
            sn_meta = {
                "barcode_found": False,
                "ocr_text_found": False,
                "barcode_status": "not_attempted",
                "barcode_attempts": 0,
                "barcode_decoded_count": 0,
            }

            if "model_path" in item:
                model_code, model_raw, model_src = recognize_model(
                    item["model_path"],
                    label_id=key,
                    use_barcode=model_barcode,
                )
                stats["model_total"] += 1
                if model_code:
                    stats["model_success"] += 1
                if model_src == "barcode":
                    stats["model_barcode_hits"] += 1
                elif model_src.startswith("ocr"):
                    stats["model_ocr_recoveries"] += 1

            sn_input_available = bool(item.get("sn_path") or item.get("label_crop"))
            if sn_input_available:
                sn_code, sn_raw, sn_src, sn_meta = recognize_sn(
                    item.get("sn_path", ""),
                    label_id=key,
                    label_path=item.get("label_crop", ""),
                    original_path=item.get("original_image_path", ""),
                )

            if sn_code.startswith("4E25A017") and model_code in {"", "S380-S8P", "S380", "S380-", "S38
0-S"}:
                model_code = "S380-S8P2T"
                model_src = f"{model_src}+sn_hint" if model_src else "sn_hint"

            if sn_code.startswith("4E25B0") and model_code in {"", "S380-S8P", "S380", "S380-", "S380-
S"}:
                model_code = "S380-S8P2T"
                model_src = f"{model_src}+sn_hint" if model_src else "sn_hint"

            if model_code == "S380-S8P2T" and sn_code and not sn_code.startswith("4E25A017"):
                append_sensitive_debug(f"[WARN] {key} model= S380-S8P2T but sn={sn_code}")

            out = {
                "label_id": key,
                "model": model_code,
                "sn": sn_code,
                "model_raw": _mask_sensitive_text(model_raw) if mask_raw else model_raw,
                "sn_raw": _mask_sensitive_text(sn_raw) if mask_raw else sn_raw,
                "model_src": model_src,
                "sn_src": sn_src,
                "sn_barcode_status": sn_meta.get("barcode_status", "not_attempted"),
                "sn_barcode_attempts": sn_meta.get("barcode_attempts", 0),
                "sn_barcode_decoded_count": sn_meta.get("barcode_decoded_count", 0),
                "sn_barcode_sources": sn_meta.get("barcode_sources", []),
                "sn_barcode_source_regions": sn_meta.get("barcode_source_regions", []),
                "sn_barcode_decoder_names": sn_meta.get("barcode_decoder_names", []),
                "sn_barcode_ambiguous_sns": sn_meta.get("barcode_ambiguous_sns", []),
            }

            _log(
                f"[{key}] "
                f"MODEL={model_code} (M_SRC={model_src}) | "
                f"SN={sn_code} (SN_SRC={sn_src})",
                "info",
            )

            f.write(json.dumps(out, ensure_ascii=False) + "\n")
            stats["records"] += 1

            if sn_input_available:
                stats["sn_total"] += 1
                barcode_status = sn_meta.get("barcode_status", "not_attempted")
                stats["sn_barcode_attempts"] += int(sn_meta.get("barcode_attempts", 0) or 0)
                if sn_src in {"barcode", "barcode_unmatched", "barcode_ocr_agree"}:
                    stats["sn_barcode_hits"] += 1
                elif sn_src.startswith("ocr") and barcode_status in {
                    "decoder_miss",
                    "parse_failure",
                    "ambiguous",
                    "quality_reject",
                }:
                    stats["sn_ocr_recoveries"] += 1
                if barcode_status == "parse_failure":
                    stats["sn_barcode_parse_failures"] += 1
                elif barcode_status == "decoder_miss":
                    stats["sn_barcode_decoder_misses"] += 1
                elif barcode_status == "ambiguous":
                    stats["sn_barcode_ambiguous"] += 1
                elif barcode_status == "quality_reject":
                    stats["sn_barcode_quality_rejects"] += 1
                if sn_meta.get("barcode_found") or sn_meta.get("ocr_text_found"):
                    stats["sn_attempted"] += 1
                if sn_code:
                    stats["sn_success"] += 1
                else:
                    if sn_meta.get("barcode_found") or sn_meta.get("ocr_text_found"):
                        stats["regex_fail"] += 1
                    if not sn_meta.get("barcode_found"):
                        stats["barcode_fail"] += 1
                    if not sn_meta.get("ocr_text_found"):
                        stats["ocr_fail"] += 1

    if stats["sn_total"]:
        stats["sn_barcode_hit_rate"] = stats["sn_barcode_hits"] / float(stats["sn_total"])
    if stats["model_total"]:
        stats["model_barcode_hit_rate"] = stats["model_barcode_hits"] / float(stats["model_total"])

    return stats


if __name__ == "__main__":
    main()

exec
"C:\\Program Files\\WindowsApps\\Microsoft.PowerShell_7.6.0.0_x64__8wekyb3d8bbwe\\pwsh.exe" -Command "
rg -n \"HUAWEIOCR_OCR_LANG|_recognition_model_candidates|OCR_LANG|ch\\)|server_rec\" tests\\test_locke
d_output_dirs.py" in D:\newhuawei
 succeeded in 677ms:
294:                "PP-OCRv5_server_rec",
339:            for name in ("PP-OCRv5_server_rec", "en_PP-OCRv5_mobile_rec"):
347:    def test_server_profile_prefers_server_recognition_model(self):
374:            for name in ("PP-OCRv5_server_rec", "en_PP-OCRv5_mobile_rec"):
379:        self.assertEqual(kwargs["text_recognition_model_name"], "PP-OCRv5_server_rec")
380:        self.assertTrue(kwargs["text_recognition_model_dir"].endswith("PP-OCRv5_server_rec"))

codex
准备开始落地修改：会改 `scan2.py`（统计分离）、`sn_barcode.py`（目录保护）、`tests/test_locked_output_dirs.py`（真实 `app_paths` 重
载 + OCR_LANG 覆盖测试）和一处 `tests/test_sn_barcode_scanning.py` 断言。
apply patch
patch: completed
D:\newhuawei\sn_barcode.py
diff --git a/sn_barcode.py b/sn_barcode.py
index 17e7dc37a97a417f6a9297dec300e71c39b04874..e6467ed158fdea38a3c600301498e1cf5b99a7e1
--- a/sn_barcode.py
+++ b/sn_barcode.py
@@ -117,7 +117,9 @@
     current.add(segment)
     persisted = sorted(current - set(BASE_SN20_SEGMENTS), key=lambda s: (-len(s), s))
     path = _sn_segments_file_path()
-    os.makedirs(os.path.dirname(path), exist_ok=True)
+    dirpath = os.path.dirname(path)
+    if dirpath:
+        os.makedirs(dirpath, exist_ok=True)
     tmp = f"{path}.tmp"
     with open(tmp, "w", encoding="utf-8") as f:
         json.dump(persisted, f, ensure_ascii=False, indent=2)

diff --git a/sn_barcode.py b/sn_barcode.py
index 17e7dc37a97a417f6a9297dec300e71c39b04874..e6467ed158fdea38a3c600301498e1cf5b99a7e1
--- a/sn_barcode.py
+++ b/sn_barcode.py
@@ -117,7 +117,9 @@
     current.add(segment)
     persisted = sorted(current - set(BASE_SN20_SEGMENTS), key=lambda s: (-len(s), s))
     path = _sn_segments_file_path()
-    os.makedirs(os.path.dirname(path), exist_ok=True)
+    dirpath = os.path.dirname(path)
+    if dirpath:
+        os.makedirs(dirpath, exist_ok=True)
     tmp = f"{path}.tmp"
     with open(tmp, "w", encoding="utf-8") as f:
         json.dump(persisted, f, ensure_ascii=False, indent=2)

apply patch
patch: completed
D:\newhuawei\scan2.py
diff --git a/scan2.py b/scan2.py
index 2ad3d5da047b09345313adcb1196779692901055..bdfdbd9393b8ebb9845931ad499dc4a6d3952398
--- a/scan2.py
+++ b/scan2.py
@@ -973,6 +973,7 @@
         "sn_success": 0,
         "sn_barcode_attempts": 0,
         "sn_barcode_hits": 0,
+        "sn_barcode_ocr_agree": 0,
         "sn_barcode_hit_rate": 0.0,
         "sn_ocr_recoveries": 0,
         "sn_barcode_parse_failures": 0,
@@ -1087,8 +1088,10 @@
                 stats["sn_total"] += 1
                 barcode_status = sn_meta.get("barcode_status", "not_attempted")
                 stats["sn_barcode_attempts"] += int(sn_meta.get("barcode_attempts", 0) or 0)
-                if sn_src in {"barcode", "barcode_unmatched", "barcode_ocr_agree"}:
+                if sn_src in {"barcode", "barcode_unmatched"}:
                     stats["sn_barcode_hits"] += 1
+                elif sn_src == "barcode_ocr_agree":
+                    stats["sn_barcode_ocr_agree"] += 1
                 elif sn_src.startswith("ocr") and barcode_status in {
                     "decoder_miss",
                     "parse_failure",
diff --git a/sn_barcode.py b/sn_barcode.py
index 17e7dc37a97a417f6a9297dec300e71c39b04874..e6467ed158fdea38a3c600301498e1cf5b99a7e1
--- a/sn_barcode.py
+++ b/sn_barcode.py
@@ -117,7 +117,9 @@
     current.add(segment)
     persisted = sorted(current - set(BASE_SN20_SEGMENTS), key=lambda s: (-len(s), s))
     path = _sn_segments_file_path()
-    os.makedirs(os.path.dirname(path), exist_ok=True)
+    dirpath = os.path.dirname(path)
+    if dirpath:
+        os.makedirs(dirpath, exist_ok=True)
     tmp = f"{path}.tmp"
     with open(tmp, "w", encoding="utf-8") as f:
         json.dump(persisted, f, ensure_ascii=False, indent=2)

diff --git a/scan2.py b/scan2.py
index 2ad3d5da047b09345313adcb1196779692901055..bdfdbd9393b8ebb9845931ad499dc4a6d3952398
--- a/scan2.py
+++ b/scan2.py
@@ -973,6 +973,7 @@
         "sn_success": 0,
         "sn_barcode_attempts": 0,
         "sn_barcode_hits": 0,
+        "sn_barcode_ocr_agree": 0,
         "sn_barcode_hit_rate": 0.0,
         "sn_ocr_recoveries": 0,
         "sn_barcode_parse_failures": 0,
@@ -1087,8 +1088,10 @@
                 stats["sn_total"] += 1
                 barcode_status = sn_meta.get("barcode_status", "not_attempted")
                 stats["sn_barcode_attempts"] += int(sn_meta.get("barcode_attempts", 0) or 0)
-                if sn_src in {"barcode", "barcode_unmatched", "barcode_ocr_agree"}:
+                if sn_src in {"barcode", "barcode_unmatched"}:
                     stats["sn_barcode_hits"] += 1
+                elif sn_src == "barcode_ocr_agree":
+                    stats["sn_barcode_ocr_agree"] += 1
                 elif sn_src.startswith("ocr") and barcode_status in {
                     "decoder_miss",
                     "parse_failure",
diff --git a/sn_barcode.py b/sn_barcode.py
index 17e7dc37a97a417f6a9297dec300e71c39b04874..e6467ed158fdea38a3c600301498e1cf5b99a7e1
--- a/sn_barcode.py
+++ b/sn_barcode.py
@@ -117,7 +117,9 @@
     current.add(segment)
     persisted = sorted(current - set(BASE_SN20_SEGMENTS), key=lambda s: (-len(s), s))
     path = _sn_segments_file_path()
-    os.makedirs(os.path.dirname(path), exist_ok=True)
+    dirpath = os.path.dirname(path)
+    if dirpath:
+        os.makedirs(dirpath, exist_ok=True)
     tmp = f"{path}.tmp"
     with open(tmp, "w", encoding="utf-8") as f:
         json.dump(persisted, f, ensure_ascii=False, indent=2)

apply patch
patch: completed
D:\newhuawei\tests\test_locked_output_dirs.py
diff --git a/scan2.py b/scan2.py
index 2ad3d5da047b09345313adcb1196779692901055..bdfdbd9393b8ebb9845931ad499dc4a6d3952398
--- a/scan2.py
+++ b/scan2.py
@@ -973,6 +973,7 @@
         "sn_success": 0,
         "sn_barcode_attempts": 0,
         "sn_barcode_hits": 0,
+        "sn_barcode_ocr_agree": 0,
         "sn_barcode_hit_rate": 0.0,
         "sn_ocr_recoveries": 0,
         "sn_barcode_parse_failures": 0,
@@ -1087,8 +1088,10 @@
                 stats["sn_total"] += 1
                 barcode_status = sn_meta.get("barcode_status", "not_attempted")
                 stats["sn_barcode_attempts"] += int(sn_meta.get("barcode_attempts", 0) or 0)
-                if sn_src in {"barcode", "barcode_unmatched", "barcode_ocr_agree"}:
+                if sn_src in {"barcode", "barcode_unmatched"}:
                     stats["sn_barcode_hits"] += 1
+                elif sn_src == "barcode_ocr_agree":
+                    stats["sn_barcode_ocr_agree"] += 1
                 elif sn_src.startswith("ocr") and barcode_status in {
                     "decoder_miss",
                     "parse_failure",
diff --git a/sn_barcode.py b/sn_barcode.py
index 17e7dc37a97a417f6a9297dec300e71c39b04874..e6467ed158fdea38a3c600301498e1cf5b99a7e1
--- a/sn_barcode.py
+++ b/sn_barcode.py
@@ -117,7 +117,9 @@
     current.add(segment)
     persisted = sorted(current - set(BASE_SN20_SEGMENTS), key=lambda s: (-len(s), s))
     path = _sn_segments_file_path()
-    os.makedirs(os.path.dirname(path), exist_ok=True)
+    dirpath = os.path.dirname(path)
+    if dirpath:
+        os.makedirs(dirpath, exist_ok=True)
     tmp = f"{path}.tmp"
     with open(tmp, "w", encoding="utf-8") as f:
         json.dump(persisted, f, ensure_ascii=False, indent=2)
diff --git a/tests/test_locked_output_dirs.py b/tests/test_locked_output_dirs.py
index 4e70450a73b518f89c56bf823b62eec445f92920..2818b374b864519c0343f7b9b5b8c43c856d55d3
--- a/tests/test_locked_output_dirs.py
+++ b/tests/test_locked_output_dirs.py
@@ -252,6 +252,28 @@
 
 
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
@@ -404,7 +426,31 @@
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
 
+    def test_invalid_lang_falls_back_to_en(self):
+        ocr = self._import_ocr_with_env({"HUAWEIOCR_OCR_LANG": "invalid", "HUAWEIOCR_OCR_PROFILE": "s
erver"})
+        self.assertEqual(ocr.OCR_LANG, "en")
+        self.assertEqual(
+            ocr._recognition_model_candidates(),
+            ["PP-OCRv5_server_rec", "en_PP-OCRv5_mobile_rec", "PP-OCRv5_mobile_rec"],
+        )
+
+
 class Scan2ManifestTest(unittest.TestCase):
     def test_main_signature_keeps_legacy_arguments(self):
         scan2 = _import_scan2()
@@ -1096,6 +1142,10 @@
 
 
 class AppPathsInstallTest(unittest.TestCase):
+    def setUp(self):
+        # Some earlier tests install a stub app_paths module; force real module import.
+        sys.modules.pop("app_paths", None)
+
     def test_incomplete_model_dir_is_replaced(self):
         import app_paths
 

diff --git a/scan2.py b/scan2.py
index 2ad3d5da047b09345313adcb1196779692901055..bdfdbd9393b8ebb9845931ad499dc4a6d3952398
--- a/scan2.py
+++ b/scan2.py
@@ -973,6 +973,7 @@
         "sn_success": 0,
         "sn_barcode_attempts": 0,
         "sn_barcode_hits": 0,
+        "sn_barcode_ocr_agree": 0,
         "sn_barcode_hit_rate": 0.0,
         "sn_ocr_recoveries": 0,
         "sn_barcode_parse_failures": 0,
@@ -1087,8 +1088,10 @@
                 stats["sn_total"] += 1
                 barcode_status = sn_meta.get("barcode_status", "not_attempted")
                 stats["sn_barcode_attempts"] += int(sn_meta.get("barcode_attempts", 0) or 0)
-                if sn_src in {"barcode", "barcode_unmatched", "barcode_ocr_agree"}:
+                if sn_src in {"barcode", "barcode_unmatched"}:
                     stats["sn_barcode_hits"] += 1
+                elif sn_src == "barcode_ocr_agree":
+                    stats["sn_barcode_ocr_agree"] += 1
                 elif sn_src.startswith("ocr") and barcode_status in {
                     "decoder_miss",
                     "parse_failure",
diff --git a/sn_barcode.py b/sn_barcode.py
index 17e7dc37a97a417f6a9297dec300e71c39b04874..e6467ed158fdea38a3c600301498e1cf5b99a7e1
--- a/sn_barcode.py
+++ b/sn_barcode.py
@@ -117,7 +117,9 @@
     current.add(segment)
     persisted = sorted(current - set(BASE_SN20_SEGMENTS), key=lambda s: (-len(s), s))
     path = _sn_segments_file_path()
-    os.makedirs(os.path.dirname(path), exist_ok=True)
+    dirpath = os.path.dirname(path)
+    if dirpath:
+        os.makedirs(dirpath, exist_ok=True)
     tmp = f"{path}.tmp"
     with open(tmp, "w", encoding="utf-8") as f:
         json.dump(persisted, f, ensure_ascii=False, indent=2)
diff --git a/tests/test_locked_output_dirs.py b/tests/test_locked_output_dirs.py
index 4e70450a73b518f89c56bf823b62eec445f92920..2818b374b864519c0343f7b9b5b8c43c856d55d3
--- a/tests/test_locked_output_dirs.py
+++ b/tests/test_locked_output_dirs.py
@@ -252,6 +252,28 @@
 
 
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
@@ -404,7 +426,31 @@
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
 
+    def test_invalid_lang_falls_back_to_en(self):
+        ocr = self._import_ocr_with_env({"HUAWEIOCR_OCR_LANG": "invalid", "HUAWEIOCR_OCR_PROFILE": "s
erver"})
+        self.assertEqual(ocr.OCR_LANG, "en")
+        self.assertEqual(
+            ocr._recognition_model_candidates(),
+            ["PP-OCRv5_server_rec", "en_PP-OCRv5_mobile_rec", "PP-OCRv5_mobile_rec"],
+        )
+
+
 class Scan2ManifestTest(unittest.TestCase):
     def test_main_signature_keeps_legacy_arguments(self):
         scan2 = _import_scan2()
@@ -1096,6 +1142,10 @@
 
 
 class AppPathsInstallTest(unittest.TestCase):
+    def setUp(self):
+        # Some earlier tests install a stub app_paths module; force real module import.
+        sys.modules.pop("app_paths", None)
+
     def test_incomplete_model_dir_is_replaced(self):
         import app_paths
 

apply patch
patch: completed
D:\newhuawei\tests\test_sn_barcode_scanning.py
diff --git a/scan2.py b/scan2.py
index 2ad3d5da047b09345313adcb1196779692901055..bdfdbd9393b8ebb9845931ad499dc4a6d3952398
--- a/scan2.py
+++ b/scan2.py
@@ -973,6 +973,7 @@
         "sn_success": 0,
         "sn_barcode_attempts": 0,
         "sn_barcode_hits": 0,
+        "sn_barcode_ocr_agree": 0,
         "sn_barcode_hit_rate": 0.0,
         "sn_ocr_recoveries": 0,
         "sn_barcode_parse_failures": 0,
@@ -1087,8 +1088,10 @@
                 stats["sn_total"] += 1
                 barcode_status = sn_meta.get("barcode_status", "not_attempted")
                 stats["sn_barcode_attempts"] += int(sn_meta.get("barcode_attempts", 0) or 0)
-                if sn_src in {"barcode", "barcode_unmatched", "barcode_ocr_agree"}:
+                if sn_src in {"barcode", "barcode_unmatched"}:
                     stats["sn_barcode_hits"] += 1
+                elif sn_src == "barcode_ocr_agree":
+                    stats["sn_barcode_ocr_agree"] += 1
                 elif sn_src.startswith("ocr") and barcode_status in {
                     "decoder_miss",
                     "parse_failure",
diff --git a/sn_barcode.py b/sn_barcode.py
index 17e7dc37a97a417f6a9297dec300e71c39b04874..e6467ed158fdea38a3c600301498e1cf5b99a7e1
--- a/sn_barcode.py
+++ b/sn_barcode.py
@@ -117,7 +117,9 @@
     current.add(segment)
     persisted = sorted(current - set(BASE_SN20_SEGMENTS), key=lambda s: (-len(s), s))
     path = _sn_segments_file_path()
-    os.makedirs(os.path.dirname(path), exist_ok=True)
+    dirpath = os.path.dirname(path)
+    if dirpath:
+        os.makedirs(dirpath, exist_ok=True)
     tmp = f"{path}.tmp"
     with open(tmp, "w", encoding="utf-8") as f:
         json.dump(persisted, f, ensure_ascii=False, indent=2)
diff --git a/tests/test_locked_output_dirs.py b/tests/test_locked_output_dirs.py
index 4e70450a73b518f89c56bf823b62eec445f92920..2818b374b864519c0343f7b9b5b8c43c856d55d3
--- a/tests/test_locked_output_dirs.py
+++ b/tests/test_locked_output_dirs.py
@@ -252,6 +252,28 @@
 
 
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
@@ -404,7 +426,31 @@
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
 
+    def test_invalid_lang_falls_back_to_en(self):
+        ocr = self._import_ocr_with_env({"HUAWEIOCR_OCR_LANG": "invalid", "HUAWEIOCR_OCR_PROFILE": "s
erver"})
+        self.assertEqual(ocr.OCR_LANG, "en")
+        self.assertEqual(
+            ocr._recognition_model_candidates(),
+            ["PP-OCRv5_server_rec", "en_PP-OCRv5_mobile_rec", "PP-OCRv5_mobile_rec"],
+        )
+
+
 class Scan2ManifestTest(unittest.TestCase):
     def test_main_signature_keeps_legacy_arguments(self):
         scan2 = _import_scan2()
@@ -1096,6 +1142,10 @@
 
 
 class AppPathsInstallTest(unittest.TestCase):
+    def setUp(self):
+        # Some earlier tests install a stub app_paths module; force real module import.
+        sys.modules.pop("app_paths", None)
+
     def test_incomplete_model_dir_is_replaced(self):
         import app_paths
 
diff --git a/tests/test_sn_barcode_scanning.py b/tests/test_sn_barcode_scanning.py
index 8a90ae5cc5def4460179cf3f15ace24802c9f3d4..22c6af09c7aeb796617f5534929e85225fac9ccd
--- a/tests/test_sn_barcode_scanning.py
+++ b/tests/test_sn_barcode_scanning.py
@@ -438,7 +438,7 @@
         self.assertEqual(stats["sn_barcode_hits"], 1)
         self.assertEqual(stats["sn_barcode_parse_failures"], 1)
 
-    def test_main_counts_barcode_ocr_agree_as_barcode_hit(self):
+    def test_main_tracks_barcode_ocr_agree_separately_from_barcode_hits(self):
         scan2 = _import_scan2()
 
         with tempfile.TemporaryDirectory() as root:
@@ -472,7 +472,8 @@
                 )
 
         self.assertEqual(stats["sn_success"], 1)
-        self.assertEqual(stats["sn_barcode_hits"], 1)
+        self.assertEqual(stats["sn_barcode_hits"], 0)
+        self.assertEqual(stats["sn_barcode_ocr_agree"], 1)
         self.assertEqual(stats["sn_barcode_parse_failures"], 1)
 
 

diff --git a/scan2.py b/scan2.py
index 2ad3d5da047b09345313adcb1196779692901055..bdfdbd9393b8ebb9845931ad499dc4a6d3952398
--- a/scan2.py
+++ b/scan2.py
@@ -973,6 +973,7 @@
         "sn_success": 0,
         "sn_barcode_attempts": 0,
         "sn_barcode_hits": 0,
+        "sn_barcode_ocr_agree": 0,
         "sn_barcode_hit_rate": 0.0,
         "sn_ocr_recoveries": 0,
         "sn_barcode_parse_failures": 0,
@@ -1087,8 +1088,10 @@
                 stats["sn_total"] += 1
                 barcode_status = sn_meta.get("barcode_status", "not_attempted")
                 stats["sn_barcode_attempts"] += int(sn_meta.get("barcode_attempts", 0) or 0)
-                if sn_src in {"barcode", "barcode_unmatched", "barcode_ocr_agree"}:
+                if sn_src in {"barcode", "barcode_unmatched"}:
                     stats["sn_barcode_hits"] += 1
+                elif sn_src == "barcode_ocr_agree":
+                    stats["sn_barcode_ocr_agree"] += 1
                 elif sn_src.startswith("ocr") and barcode_status in {
                     "decoder_miss",
                     "parse_failure",
diff --git a/sn_barcode.py b/sn_barcode.py
index 17e7dc37a97a417f6a9297dec300e71c39b04874..e6467ed158fdea38a3c600301498e1cf5b99a7e1
--- a/sn_barcode.py
+++ b/sn_barcode.py
@@ -117,7 +117,9 @@
     current.add(segment)
     persisted = sorted(current - set(BASE_SN20_SEGMENTS), key=lambda s: (-len(s), s))
     path = _sn_segments_file_path()
-    os.makedirs(os.path.dirname(path), exist_ok=True)
+    dirpath = os.path.dirname(path)
+    if dirpath:
+        os.makedirs(dirpath, exist_ok=True)
     tmp = f"{path}.tmp"
     with open(tmp, "w", encoding="utf-8") as f:
         json.dump(persisted, f, ensure_ascii=False, indent=2)
diff --git a/tests/test_locked_output_dirs.py b/tests/test_locked_output_dirs.py
index 4e70450a73b518f89c56bf823b62eec445f92920..2818b374b864519c0343f7b9b5b8c43c856d55d3
--- a/tests/test_locked_output_dirs.py
+++ b/tests/test_locked_output_dirs.py
@@ -252,6 +252,28 @@
 
 
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
@@ -404,7 +426,31 @@
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
 
+    def test_invalid_lang_falls_back_to_en(self):
+        ocr = self._import_ocr_with_env({"HUAWEIOCR_OCR_LANG": "invalid", "HUAWEIOCR_OCR_PROFILE": "s
erver"})
+        self.assertEqual(ocr.OCR_LANG, "en")
+        self.assertEqual(
+            ocr._recognition_model_candidates(),
+            ["PP-OCRv5_server_rec", "en_PP-OCRv5_mobile_rec", "PP-OCRv5_mobile_rec"],
+        )
+
+
 class Scan2ManifestTest(unittest.TestCase):
     def test_main_signature_keeps_legacy_arguments(self):
         scan2 = _import_scan2()
@@ -1096,6 +1142,10 @@
 
 
 class AppPathsInstallTest(unittest.TestCase):
+    def setUp(self):
+        # Some earlier tests install a stub app_paths module; force real module import.
+        sys.modules.pop("app_paths", None)
+
     def test_incomplete_model_dir_is_replaced(self):
         import app_paths
 
diff --git a/tests/test_sn_barcode_scanning.py b/tests/test_sn_barcode_scanning.py
index 8a90ae5cc5def4460179cf3f15ace24802c9f3d4..22c6af09c7aeb796617f5534929e85225fac9ccd
--- a/tests/test_sn_barcode_scanning.py
+++ b/tests/test_sn_barcode_scanning.py
@@ -438,7 +438,7 @@
         self.assertEqual(stats["sn_barcode_hits"], 1)
         self.assertEqual(stats["sn_barcode_parse_failures"], 1)
 
-    def test_main_counts_barcode_ocr_agree_as_barcode_hit(self):
+    def test_main_tracks_barcode_ocr_agree_separately_from_barcode_hits(self):
         scan2 = _import_scan2()
 
         with tempfile.TemporaryDirectory() as root:
@@ -472,7 +472,8 @@
                 )
 
         self.assertEqual(stats["sn_success"], 1)
-        self.assertEqual(stats["sn_barcode_hits"], 1)
+        self.assertEqual(stats["sn_barcode_hits"], 0)
+        self.assertEqual(stats["sn_barcode_ocr_agree"], 1)
         self.assertEqual(stats["sn_barcode_parse_failures"], 1)
 
 

codex
修改已完成，正在运行与本轮 FAIL 直接相关的回归验证：先跑 `tests/test_sn_barcode_scanning.py` 和 `tests/test_locked_output_dirs.p
y`。
exec
"C:\\Program Files\\WindowsApps\\Microsoft.PowerShell_7.6.0.0_x64__8wekyb3d8bbwe\\pwsh.exe" -Command '
pytest tests/test_sn_barcode_scanning.py tests/test_locked_output_dirs.py -q' in D:\newhuawei
 succeeded in 2956ms:
.......................................................................  [100%]
============================== warnings summary ===============================
tests/test_locked_output_dirs.py::GuiPipelineTest::test_gui_display_label_keeps_uniqueness_for_duplica
te_source_names
  C:\Users\55093\AppData\Local\Programs\Python\Python312\Lib\site-packages\openpyxl\compat\numbers.py:
9: UserWarning: The NumPy module was reloaded (imported a second time). This can in some cases result 
in small but subtle issues and is discouraged.
    import numpy

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
71 passed, 1 warning in 0.61s

diff --git a/scan2.py b/scan2.py
index 2ad3d5da047b09345313adcb1196779692901055..bdfdbd9393b8ebb9845931ad499dc4a6d3952398
--- a/scan2.py
+++ b/scan2.py
@@ -973,6 +973,7 @@
         "sn_success": 0,
         "sn_barcode_attempts": 0,
         "sn_barcode_hits": 0,
+        "sn_barcode_ocr_agree": 0,
         "sn_barcode_hit_rate": 0.0,
         "sn_ocr_recoveries": 0,
         "sn_barcode_parse_failures": 0,
@@ -1087,8 +1088,10 @@
                 stats["sn_total"] += 1
                 barcode_status = sn_meta.get("barcode_status", "not_attempted")
                 stats["sn_barcode_attempts"] += int(sn_meta.get("barcode_attempts", 0) or 0)
-                if sn_src in {"barcode", "barcode_unmatched", "barcode_ocr_agree"}:
+                if sn_src in {"barcode", "barcode_unmatched"}:
                     stats["sn_barcode_hits"] += 1
+                elif sn_src == "barcode_ocr_agree":
+                    stats["sn_barcode_ocr_agree"] += 1
                 elif sn_src.startswith("ocr") and barcode_status in {
                     "decoder_miss",
                     "parse_failure",
diff --git a/sn_barcode.py b/sn_barcode.py
index 17e7dc37a97a417f6a9297dec300e71c39b04874..e6467ed158fdea38a3c600301498e1cf5b99a7e1
--- a/sn_barcode.py
+++ b/sn_barcode.py
@@ -117,7 +117,9 @@
     current.add(segment)
     persisted = sorted(current - set(BASE_SN20_SEGMENTS), key=lambda s: (-len(s), s))
     path = _sn_segments_file_path()
-    os.makedirs(os.path.dirname(path), exist_ok=True)
+    dirpath = os.path.dirname(path)
+    if dirpath:
+        os.makedirs(dirpath, exist_ok=True)
     tmp = f"{path}.tmp"
     with open(tmp, "w", encoding="utf-8") as f:
         json.dump(persisted, f, ensure_ascii=False, indent=2)
diff --git a/tests/test_locked_output_dirs.py b/tests/test_locked_output_dirs.py
index 4e70450a73b518f89c56bf823b62eec445f92920..2818b374b864519c0343f7b9b5b8c43c856d55d3
--- a/tests/test_locked_output_dirs.py
+++ b/tests/test_locked_output_dirs.py
@@ -252,6 +252,28 @@
 
 
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
@@ -404,7 +426,31 @@
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
 
+    def test_invalid_lang_falls_back_to_en(self):
+        ocr = self._import_ocr_with_env({"HUAWEIOCR_OCR_LANG": "invalid", "HUAWEIOCR_OCR_PROFILE": "s
erver"})
+        self.assertEqual(ocr.OCR_LANG, "en")
+        self.assertEqual(
+            ocr._recognition_model_candidates(),
+            ["PP-OCRv5_server_rec", "en_PP-OCRv5_mobile_rec", "PP-OCRv5_mobile_rec"],
+        )
+
+
 class Scan2ManifestTest(unittest.TestCase):
     def test_main_signature_keeps_legacy_arguments(self):
         scan2 = _import_scan2()
@@ -1096,6 +1142,10 @@
 
 
 class AppPathsInstallTest(unittest.TestCase):
+    def setUp(self):
+        # Some earlier tests install a stub app_paths module; force real module import.
+        sys.modules.pop("app_paths", None)
+
     def test_incomplete_model_dir_is_replaced(self):
         import app_paths
 
diff --git a/tests/test_sn_barcode_scanning.py b/tests/test_sn_barcode_scanning.py
index 8a90ae5cc5def4460179cf3f15ace24802c9f3d4..22c6af09c7aeb796617f5534929e85225fac9ccd
--- a/tests/test_sn_barcode_scanning.py
+++ b/tests/test_sn_barcode_scanning.py
@@ -438,7 +438,7 @@
         self.assertEqual(stats["sn_barcode_hits"], 1)
         self.assertEqual(stats["sn_barcode_parse_failures"], 1)
 
-    def test_main_counts_barcode_ocr_agree_as_barcode_hit(self):
+    def test_main_tracks_barcode_ocr_agree_separately_from_barcode_hits(self):
         scan2 = _import_scan2()
 
         with tempfile.TemporaryDirectory() as root:
@@ -472,7 +472,8 @@
                 )
 
         self.assertEqual(stats["sn_success"], 1)
-        self.assertEqual(stats["sn_barcode_hits"], 1)
+        self.assertEqual(stats["sn_barcode_hits"], 0)
+        self.assertEqual(stats["sn_barcode_ocr_agree"], 1)
         self.assertEqual(stats["sn_barcode_parse_failures"], 1)
 
 

exec
"C:\\Program Files\\WindowsApps\\Microsoft.PowerShell_7.6.0.0_x64__8wekyb3d8bbwe\\pwsh.exe" -Command '
git diff --name-only' in D:\newhuawei
 succeeded in 664ms:
gui_app.py
gui_pipeline.py
ocr.py
scan2.py
sn_barcode.py
tests/test_locked_output_dirs.py
tests/test_sn_barcode_scanning.py
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

diff --git a/scan2.py b/scan2.py
index 2ad3d5da047b09345313adcb1196779692901055..bdfdbd9393b8ebb9845931ad499dc4a6d3952398
--- a/scan2.py
+++ b/scan2.py
@@ -973,6 +973,7 @@
         "sn_success": 0,
         "sn_barcode_attempts": 0,
         "sn_barcode_hits": 0,
+        "sn_barcode_ocr_agree": 0,
         "sn_barcode_hit_rate": 0.0,
         "sn_ocr_recoveries": 0,
         "sn_barcode_parse_failures": 0,
@@ -1087,8 +1088,10 @@
                 stats["sn_total"] += 1
                 barcode_status = sn_meta.get("barcode_status", "not_attempted")
                 stats["sn_barcode_attempts"] += int(sn_meta.get("barcode_attempts", 0) or 0)
-                if sn_src in {"barcode", "barcode_unmatched", "barcode_ocr_agree"}:
+                if sn_src in {"barcode", "barcode_unmatched"}:
                     stats["sn_barcode_hits"] += 1
+                elif sn_src == "barcode_ocr_agree":
+                    stats["sn_barcode_ocr_agree"] += 1
                 elif sn_src.startswith("ocr") and barcode_status in {
                     "decoder_miss",
                     "parse_failure",
diff --git a/sn_barcode.py b/sn_barcode.py
index 17e7dc37a97a417f6a9297dec300e71c39b04874..e6467ed158fdea38a3c600301498e1cf5b99a7e1
--- a/sn_barcode.py
+++ b/sn_barcode.py
@@ -117,7 +117,9 @@
     current.add(segment)
     persisted = sorted(current - set(BASE_SN20_SEGMENTS), key=lambda s: (-len(s), s))
     path = _sn_segments_file_path()
-    os.makedirs(os.path.dirname(path), exist_ok=True)
+    dirpath = os.path.dirname(path)
+    if dirpath:
+        os.makedirs(dirpath, exist_ok=True)
     tmp = f"{path}.tmp"
     with open(tmp, "w", encoding="utf-8") as f:
         json.dump(persisted, f, ensure_ascii=False, indent=2)
diff --git a/tests/test_locked_output_dirs.py b/tests/test_locked_output_dirs.py
index 4e70450a73b518f89c56bf823b62eec445f92920..2818b374b864519c0343f7b9b5b8c43c856d55d3
--- a/tests/test_locked_output_dirs.py
+++ b/tests/test_locked_output_dirs.py
@@ -252,6 +252,28 @@
 
 
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
@@ -404,7 +426,31 @@
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
 
+    def test_invalid_lang_falls_back_to_en(self):
+        ocr = self._import_ocr_with_env({"HUAWEIOCR_OCR_LANG": "invalid", "HUAWEIOCR_OCR_PROFILE": "s
erver"})
+        self.assertEqual(ocr.OCR_LANG, "en")
+        self.assertEqual(
+            ocr._recognition_model_candidates(),
+            ["PP-OCRv5_server_rec", "en_PP-OCRv5_mobile_rec", "PP-OCRv5_mobile_rec"],
+        )
+
+
 class Scan2ManifestTest(unittest.TestCase):
     def test_main_signature_keeps_legacy_arguments(self):
         scan2 = _import_scan2()
@@ -1096,6 +1142,10 @@
 
 
 class AppPathsInstallTest(unittest.TestCase):
+    def setUp(self):
+        # Some earlier tests install a stub app_paths module; force real module import.
+        sys.modules.pop("app_paths", None)
+
     def test_incomplete_model_dir_is_replaced(self):
         import app_paths
 
diff --git a/tests/test_sn_barcode_scanning.py b/tests/test_sn_barcode_scanning.py
index 8a90ae5cc5def4460179cf3f15ace24802c9f3d4..22c6af09c7aeb796617f5534929e85225fac9ccd
--- a/tests/test_sn_barcode_scanning.py
+++ b/tests/test_sn_barcode_scanning.py
@@ -438,7 +438,7 @@
         self.assertEqual(stats["sn_barcode_hits"], 1)
         self.assertEqual(stats["sn_barcode_parse_failures"], 1)
 
-    def test_main_counts_barcode_ocr_agree_as_barcode_hit(self):
+    def test_main_tracks_barcode_ocr_agree_separately_from_barcode_hits(self):
         scan2 = _import_scan2()
 
         with tempfile.TemporaryDirectory() as root:
@@ -472,7 +472,8 @@
                 )
 
         self.assertEqual(stats["sn_success"], 1)
-        self.assertEqual(stats["sn_barcode_hits"], 1)
+        self.assertEqual(stats["sn_barcode_hits"], 0)
+        self.assertEqual(stats["sn_barcode_ocr_agree"], 1)
         self.assertEqual(stats["sn_barcode_parse_failures"], 1)
 
 

codex
FIX_STATUS: APPLIED

Changed files:
- [scan2.py](D:\newhuawei\scan2.py): stopped counting `barcode_ocr_agree` as pure barcode hit; added s
eparate `sn_barcode_ocr_agree` counter.
- [sn_barcode.py](D:\newhuawei\sn_barcode.py): guarded `os.makedirs(...)` in `learn_sn_pattern()` when
 env path has no directory component.
- [tests/test_locked_output_dirs.py](D:\newhuawei\tests\test_locked_output_dirs.py): added OCR languag
e selection tests (`en`/`ch`/invalid fallback) and forced real `app_paths` import in `AppPathsInstallT
est.setUp`.
- [tests/test_sn_barcode_scanning.py](D:\newhuawei\tests\test_sn_barcode_scanning.py): updated asserti
on to require `barcode_ocr_agree` be tracked separately, not counted in `sn_barcode_hits`.

Validation run:
- `pytest tests/test_sn_barcode_scanning.py tests/test_locked_output_dirs.py -q`
- Result: `71 passed, 1 warning in 0.61s`


diff --git a/scan2.py b/scan2.py
index 2ad3d5da047b09345313adcb1196779692901055..bdfdbd9393b8ebb9845931ad499dc4a6d3952398
--- a/scan2.py
+++ b/scan2.py
@@ -973,6 +973,7 @@
         "sn_success": 0,
         "sn_barcode_attempts": 0,
         "sn_barcode_hits": 0,
+        "sn_barcode_ocr_agree": 0,
         "sn_barcode_hit_rate": 0.0,
         "sn_ocr_recoveries": 0,
         "sn_barcode_parse_failures": 0,
@@ -1087,8 +1088,10 @@
                 stats["sn_total"] += 1
                 barcode_status = sn_meta.get("barcode_status", "not_attempted")
                 stats["sn_barcode_attempts"] += int(sn_meta.get("barcode_attempts", 0) or 0)
-                if sn_src in {"barcode", "barcode_unmatched", "barcode_ocr_agree"}:
+                if sn_src in {"barcode", "barcode_unmatched"}:
                     stats["sn_barcode_hits"] += 1
+                elif sn_src == "barcode_ocr_agree":
+                    stats["sn_barcode_ocr_agree"] += 1
                 elif sn_src.startswith("ocr") and barcode_status in {
                     "decoder_miss",
                     "parse_failure",
diff --git a/sn_barcode.py b/sn_barcode.py
index 17e7dc37a97a417f6a9297dec300e71c39b04874..e6467ed158fdea38a3c600301498e1cf5b99a7e1
--- a/sn_barcode.py
+++ b/sn_barcode.py
@@ -117,7 +117,9 @@
     current.add(segment)
     persisted = sorted(current - set(BASE_SN20_SEGMENTS), key=lambda s: (-len(s), s))
     path = _sn_segments_file_path()
-    os.makedirs(os.path.dirname(path), exist_ok=True)
+    dirpath = os.path.dirname(path)
+    if dirpath:
+        os.makedirs(dirpath, exist_ok=True)
     tmp = f"{path}.tmp"
     with open(tmp, "w", encoding="utf-8") as f:
         json.dump(persisted, f, ensure_ascii=False, indent=2)
diff --git a/tests/test_locked_output_dirs.py b/tests/test_locked_output_dirs.py
index 4e70450a73b518f89c56bf823b62eec445f92920..2818b374b864519c0343f7b9b5b8c43c856d55d3
--- a/tests/test_locked_output_dirs.py
+++ b/tests/test_locked_output_dirs.py
@@ -252,6 +252,28 @@
 
 
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
@@ -404,7 +426,31 @@
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
 
+    def test_invalid_lang_falls_back_to_en(self):
+        ocr = self._import_ocr_with_env({"HUAWEIOCR_OCR_LANG": "invalid", "HUAWEIOCR_OCR_PROFILE": "s
erver"})
+        self.assertEqual(ocr.OCR_LANG, "en")
+        self.assertEqual(
+            ocr._recognition_model_candidates(),
+            ["PP-OCRv5_server_rec", "en_PP-OCRv5_mobile_rec", "PP-OCRv5_mobile_rec"],
+        )
+
+
 class Scan2ManifestTest(unittest.TestCase):
     def test_main_signature_keeps_legacy_arguments(self):
         scan2 = _import_scan2()
@@ -1096,6 +1142,10 @@
 
 
 class AppPathsInstallTest(unittest.TestCase):
+    def setUp(self):
+        # Some earlier tests install a stub app_paths module; force real module import.
+        sys.modules.pop("app_paths", None)
+
     def test_incomplete_model_dir_is_replaced(self):
         import app_paths
 
diff --git a/tests/test_sn_barcode_scanning.py b/tests/test_sn_barcode_scanning.py
index 8a90ae5cc5def4460179cf3f15ace24802c9f3d4..22c6af09c7aeb796617f5534929e85225fac9ccd
--- a/tests/test_sn_barcode_scanning.py
+++ b/tests/test_sn_barcode_scanning.py
@@ -438,7 +438,7 @@
         self.assertEqual(stats["sn_barcode_hits"], 1)
         self.assertEqual(stats["sn_barcode_parse_failures"], 1)
 
-    def test_main_counts_barcode_ocr_agree_as_barcode_hit(self):
+    def test_main_tracks_barcode_ocr_agree_separately_from_barcode_hits(self):
         scan2 = _import_scan2()
 
         with tempfile.TemporaryDirectory() as root:
@@ -472,7 +472,8 @@
                 )
 
         self.assertEqual(stats["sn_success"], 1)
-        self.assertEqual(stats["sn_barcode_hits"], 1)
+        self.assertEqual(stats["sn_barcode_hits"], 0)
+        self.assertEqual(stats["sn_barcode_ocr_agree"], 1)
         self.assertEqual(stats["sn_barcode_parse_failures"], 1)
 
 

hook: Stop
hook: Stop Completed
tokens used
52,403
FIX_STATUS: APPLIED

Changed files:
- [scan2.py](D:\newhuawei\scan2.py): stopped counting `barcode_ocr_agree` as pure barcode hit; added separate `sn_barcode_ocr_agree` counter.
- [sn_barcode.py](D:\newhuawei\sn_barcode.py): guarded `os.makedirs(...)` in `learn_sn_pattern()` when env path has no directory component.
- [tests/test_locked_output_dirs.py](D:\newhuawei\tests\test_locked_output_dirs.py): added OCR language selection tests (`en`/`ch`/invalid fallback) and forced real `app_paths` import in `AppPathsInstallTest.setUp`.
- [tests/test_sn_barcode_scanning.py](D:\newhuawei\tests\test_sn_barcode_scanning.py): updated assertion to require `barcode_ocr_agree` be tracked separately, not counted in `sn_barcode_hits`.

Validation run:
- `pytest tests/test_sn_barcode_scanning.py tests/test_locked_output_dirs.py -q`
- Result: `71 passed, 1 warning in 0.61s`



