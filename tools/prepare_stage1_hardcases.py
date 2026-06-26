from __future__ import annotations

import csv
import shutil
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = REPO_ROOT / "datasets" / "huawei_stage1_hardcases_20260622"


@dataclass(frozen=True)
class HardCase:
    filename: str
    source_path: str
    current_stage1_count: int
    target_label_count: int
    issue_type: str
    note: str


CASES = [
    HardCase(
        filename="惠州.jpg",
        source_path=r"F:\wechat\xwechat_files\wxid_br2nkrs4dgri12_68ec\msg\file\2026-06\华为出库图\惠州.jpg",
        current_stage1_count=1,
        target_label_count=1,
        issue_type="hard_negative_shipping_note",
        note="单产品标签场景；快递单很近但不应被识别为 huawei_label。",
    ),
    HardCase(
        filename="汕头.jpg",
        source_path=r"F:\wechat\xwechat_files\wxid_br2nkrs4dgri12_68ec\msg\file\2026-06\华为出库图\汕头.jpg",
        current_stage1_count=1,
        target_label_count=1,
        issue_type="hard_negative_shipping_note",
        note="单产品标签场景；快递单与产品标签平行摆放，保留为负样本干扰。",
    ),
    HardCase(
        filename="宿迁.jpg",
        source_path=r"F:\wechat\xwechat_files\wxid_br2nkrs4dgri12_68ec\msg\file\2026-06\华为出库图\宿迁.jpg",
        current_stage1_count=1,
        target_label_count=1,
        issue_type="shipping_note_overlap",
        note="标准单标签；快递单靠近但不遮挡产品标签主体。",
    ),
    HardCase(
        filename="张家港.jpg",
        source_path=r"F:\wechat\xwechat_files\wxid_br2nkrs4dgri12_68ec\msg\file\2026-06\华为出库图\张家港.jpg",
        current_stage1_count=1,
        target_label_count=1,
        issue_type="shipping_note_overlap",
        note="标准单标签；快递单前景明显但仍应只保留产品标签框。",
    ),
    HardCase(
        filename="菏泽.jpg",
        source_path=r"F:\wechat\xwechat_files\wxid_br2nkrs4dgri12_68ec\msg\file\2026-06\华为出库图\菏泽.jpg",
        current_stage1_count=7,
        target_label_count=7,
        issue_type="stacked_scene_regression_guard",
        note="多箱堆叠且快递单前景遮挡；作为多目标场景回归保护样本。",
    ),
    HardCase(
        filename="海南.jpg",
        source_path=r"F:\wechat\xwechat_files\wxid_br2nkrs4dgri12_68ec\msg\file\2026-06\华为出库图\海南.jpg",
        current_stage1_count=4,
        target_label_count=5,
        issue_type="missed_occluded_label",
        note="应标 5 个产品标签；当前 Stage1 少 1 个，右下区域受快递单遮挡。",
    ),
    HardCase(
        filename="常州.jpg",
        source_path=r"F:\wechat\xwechat_files\wxid_br2nkrs4dgri12_68ec\msg\file\2026-06\华为出库图\常州.jpg",
        current_stage1_count=3,
        target_label_count=4,
        issue_type="missed_occluded_label",
        note="应标 4 个产品标签；右上 AP162E 被快递单干扰，当前漏检。",
    ),
    HardCase(
        filename="保定.jpg",
        source_path=r"F:\wechat\xwechat_files\wxid_br2nkrs4dgri12_68ec\msg\file\2026-06\华为出库图\保定.jpg",
        current_stage1_count=4,
        target_label_count=4,
        issue_type="stacked_scene_regression_guard",
        note="4 标签场景已基本正确，用于防止重训后回退到误识别快递单。",
    ),
    HardCase(
        filename="佳木斯.jpg",
        source_path=r"F:\wechat\xwechat_files\wxid_br2nkrs4dgri12_68ec\msg\file\2026-06\华为出库图\佳木斯.jpg",
        current_stage1_count=2,
        target_label_count=3,
        issue_type="detector_recall_gap",
        note="应标 3 个产品标签；右侧 AP162E 当前 detector 本身召回不足。",
    ),
    HardCase(
        filename="嘉兴.jpg",
        source_path=r"F:\wechat\xwechat_files\wxid_br2nkrs4dgri12_68ec\msg\file\2026-06\华为出库图\嘉兴.jpg",
        current_stage1_count=6,
        target_label_count=6,
        issue_type="stacked_scene_regression_guard",
        note="6 标签场景已正确；保留作多目标和快递单负样本保护。",
    ),
    HardCase(
        filename="上海.jpg",
        source_path=r"F:\wechat\xwechat_files\wxid_br2nkrs4dgri12_68ec\msg\file\2026-06\华为出库图\上海.jpg",
        current_stage1_count=4,
        target_label_count=5,
        issue_type="missed_occluded_label",
        note="应标 5 个产品标签；当前右下小标签误成窄框并被过滤。",
    ),
    HardCase(
        filename="上海嘉定.jpg",
        source_path=r"F:\wechat\xwechat_files\wxid_br2nkrs4dgri12_68ec\msg\file\2026-06\华为出库图\上海嘉定.jpg",
        current_stage1_count=5,
        target_label_count=6,
        issue_type="missed_occluded_label",
        note="应标 6 个产品标签；顶部快递单附近标签需要保留。",
    ),
    HardCase(
        filename="成都.jpg",
        source_path=r"F:\wechat\xwechat_files\wxid_br2nkrs4dgri12_68ec\msg\file\2026-06\华为出库图\成都.jpg",
        current_stage1_count=4,
        target_label_count=5,
        issue_type="missed_occluded_label",
        note="应标 5 个产品标签；右下区域受快递单影响少 1 个。",
    ),
    HardCase(
        filename="松原.jpg",
        source_path=r"F:\wechat\xwechat_files\wxid_br2nkrs4dgri12_68ec\msg\file\2026-06\华为出库图\松原.jpg",
        current_stage1_count=4,
        target_label_count=5,
        issue_type="missed_occluded_label",
        note="应标 5 个产品标签；快递单覆盖右下，Stage1 当前少 1 个。",
    ),
    HardCase(
        filename="苏州.jpg",
        source_path=r"F:\wechat\xwechat_files\wxid_br2nkrs4dgri12_68ec\msg\file\2026-06\华为出库图\苏州.jpg",
        current_stage1_count=6,
        target_label_count=6,
        issue_type="stacked_scene_regression_guard",
        note="6 标签场景已正确；用于保护多标签召回不回退。",
    ),
    HardCase(
        filename="苏州吴江.jpg",
        source_path=r"F:\wechat\xwechat_files\wxid_br2nkrs4dgri12_68ec\msg\file\2026-06\华为出库图\苏州吴江.jpg",
        current_stage1_count=6,
        target_label_count=6,
        issue_type="stacked_scene_regression_guard",
        note="6 标签场景已正确；快递单与右上产品标签距离近。",
    ),
    HardCase(
        filename="太原.jpg",
        source_path=r"F:\wechat\xwechat_files\wxid_br2nkrs4dgri12_68ec\msg\file\2026-06\华为出库图\太原.jpg",
        current_stage1_count=6,
        target_label_count=6,
        issue_type="stacked_scene_regression_guard",
        note="6 标签场景已正确；作为快递单斜放场景回归保护。",
    ),
    HardCase(
        filename="文昌.jpg",
        source_path=r"F:\wechat\xwechat_files\wxid_br2nkrs4dgri12_68ec\msg\file\2026-06\华为出库图\文昌.jpg",
        current_stage1_count=4,
        target_label_count=5,
        issue_type="missed_occluded_label",
        note="应标 5 个产品标签；右下快递单附近漏 1 个。",
    ),
    HardCase(
        filename="武汉.jpg",
        source_path=r"F:\wechat\xwechat_files\wxid_br2nkrs4dgri12_68ec\msg\file\2026-06\华为出库图\武汉.jpg",
        current_stage1_count=3,
        target_label_count=4,
        issue_type="missed_occluded_label",
        note="应标 4 个产品标签；右上 AP162E 当前被 Stage1 过滤掉。",
    ),
    HardCase(
        filename="西安.jpg",
        source_path=r"F:\wechat\xwechat_files\wxid_br2nkrs4dgri12_68ec\msg\file\2026-06\华为出库图\西安.jpg",
        current_stage1_count=4,
        target_label_count=5,
        issue_type="missed_occluded_label",
        note="应标 5 个产品标签；右下靠快递单区域少 1 个。",
    ),
    HardCase(
        filename="新乡.jpg",
        source_path=r"F:\wechat\xwechat_files\wxid_br2nkrs4dgri12_68ec\msg\file\2026-06\华为出库图\新乡.jpg",
        current_stage1_count=6,
        target_label_count=6,
        issue_type="stacked_scene_regression_guard",
        note="6 标签场景已正确；快递单斜放，适合作为 hard negative 保护。",
    ),
    HardCase(
        filename="徐州.jpg",
        source_path=r"F:\wechat\xwechat_files\wxid_br2nkrs4dgri12_68ec\msg\file\2026-06\华为出库图\徐州.jpg",
        current_stage1_count=6,
        target_label_count=6,
        issue_type="stacked_scene_regression_guard",
        note="6 标签场景已正确；右上快递单不应触发额外框。",
    ),
    HardCase(
        filename="珠海.jpg",
        source_path=r"F:\wechat\xwechat_files\wxid_br2nkrs4dgri12_68ec\msg\file\2026-06\华为出库图\珠海.jpg",
        current_stage1_count=4,
        target_label_count=5,
        issue_type="missed_occluded_label",
        note="应标 5 个产品标签；右下快递单近旁漏 1 个。",
    ),
    HardCase(
        filename="遵义.jpg",
        source_path=r"F:\wechat\xwechat_files\wxid_br2nkrs4dgri12_68ec\msg\file\2026-06\华为出库图\遵义.jpg",
        current_stage1_count=6,
        target_label_count=6,
        issue_type="stacked_scene_regression_guard",
        note="6 标签场景已正确；作为多标签与快递单共存场景保护样本。",
    ),
]


def write_manifest(out_dir: Path) -> None:
    manifest_path = out_dir / "manifest.csv"
    with manifest_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "filename",
                "source_path",
                "current_stage1_count",
                "target_label_count",
                "issue_type",
                "note",
            ]
        )
        for case in CASES:
            writer.writerow(
                [
                    case.filename,
                    case.source_path,
                    case.current_stage1_count,
                    case.target_label_count,
                    case.issue_type,
                    case.note,
                ]
            )


def write_readme(out_dir: Path) -> None:
    readme = out_dir / "README.txt"
    text = """Stage1 hard cases for Huawei label detector retraining.

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
"""
    readme.write_text(text, encoding="utf-8")


def stage_images(out_dir: Path) -> None:
    image_dir = out_dir / "raw_images"
    label_dir = out_dir / "labels_pending"
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    for case in CASES:
        src = Path(case.source_path)
        if not src.is_file():
            raise FileNotFoundError(f"Missing source image: {src}")
        dst = image_dir / case.filename
        if not dst.exists():
            shutil.copy2(src, dst)
        label_stub = label_dir / (Path(case.filename).stem + ".txt")
        label_stub.touch(exist_ok=True)


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    stage_images(out_dir)
    write_manifest(out_dir)
    write_readme(out_dir)
    print(f"Prepared {len(CASES)} hard cases in: {out_dir}")


if __name__ == "__main__":
    main()
