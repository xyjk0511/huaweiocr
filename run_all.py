# run_all.py
import crop
import scan2

if __name__ == "__main__":
    print("===== [1/2] 裁剪标签 & model/sn 小图（Roboflow） =====")
    crop.main()

    print("\n===== [2/2] 条码 + OCR 识别 MODEL/SN =====")
    scan2.main()

    print("\n✅ 全部流程结束，可以查看：")
    print("  - stage1_labels/         (裁出来的 label 小图)")
    print("  - stage2_fields/model/   (裁出来的 model 小图)")
    print("  - stage2_fields/sn/      (裁出来的 SN 小图)")
    print("  - scan_results_ocr.jsonl (最终 model/SN 识别结果)")
    input("\n按回车退出窗口...")
