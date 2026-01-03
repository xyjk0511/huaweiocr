import os
import cv2
import numpy as np
import warnings

# 过滤无关警告，只看报错
warnings.filterwarnings("ignore")

print("--- 1. 正在初始化 PaddleOCR ---")
try:
    from paddleocr import PaddleOCR

    # 尝试初始化
    ocr = PaddleOCR(use_angle_cls=True, lang="ch", show_log=False)
    print("✅ PaddleOCR 初始化成功")
except Exception as e:
    print(f"❌ PaddleOCR 初始化挂了: {e}")
    ocr = None

print("\n--- 2. 正在生成测试图片 ---")
# 生成一张白底黑字的纯测试图
img = np.ones((100, 300, 3), dtype=np.uint8) * 255
cv2.putText(img, "Model: AR180", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
# 随便找个地方存一下，方便查看
cv2.imwrite("test_debug.png", img)
print("✅ 测试图已生成: test_debug.png")

print("\n--- 3. 正在尝试识别 (关键步骤) ---")
if ocr:
    try:
        # 【关键】这里不加 try-except，让它直接报错
        result = ocr.ocr(img, cls=True)
        print(f"✅ 识别结果: {result}")

        # 提取文字
        txts = [line[1][0] for line in result[0]]
        print(f"✅ 提取文字: {txts}")
    except Exception as e:
        print("\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("❌ 捕获到 Paddle 运行时错误:")
        print(e)
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")
        print("💡 常见原因分析:")
        if "numpy" in str(e) and "int" in str(e):
            print(">> 这是 Numpy 版本过高导致的。请执行: pip install \"numpy<1.24\"")
        elif "shapely" in str(e):
            print(">> 缺少 Shapely 库。Windows 请搜索 'shapely whl' 手动安装或检查环境。")
else:
    print("⚠️ 跳过识别测试，因为初始化已失败。")

print("\n--- 测试结束 ---")