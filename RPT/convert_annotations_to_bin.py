import os
from PIL import Image
import numpy as np

# 输入原始标签目录
label_dir = r"C:\Users\28274\Desktop\train_data2\annotations"
# 输出新目录（不会覆盖原始文件）
out_dir = r"C:\Users\28274\Desktop\train_data2\annotations_bin"

os.makedirs(out_dir, exist_ok=True)

count = 0
for root, _, files in os.walk(label_dir):
    for fn in files:
        if fn.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
            path = os.path.join(root, fn)
            try:
                # 打开并转灰度
                img = Image.open(path).convert("L")
                arr = np.array(img)

                # 二值化：>127 为前景 255，<=127 为背景 0
                arr = np.where(arr > 127, 255, 0).astype("uint8")

                # 输出路径（保持相对层级）
                rel_path = os.path.relpath(path, label_dir)
                save_path = os.path.join(out_dir, rel_path)
                os.makedirs(os.path.dirname(save_path), exist_ok=True)

                # 保存为灰度单通道 PNG
                Image.fromarray(arr, mode="L").save(save_path)

                count += 1
                print(f"Processed: {save_path} shape={arr.shape} unique={np.unique(arr)}")
            except Exception as e:
                print(f"Error processing {path}: {e}")

print(f"\nDone! Processed {count} label images. Results saved in {out_dir}")
