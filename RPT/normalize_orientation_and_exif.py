import os
import sys
from PIL import Image, ImageOps, PngImagePlugin

# === 修改为你的根目录（会递归处理 jpg/jpeg/png）===
ROOT_DIR = r"C:\Users\28274\Desktop\train_data1\images"

# 是否尽量“保留其他 EXIF 字段”
# - True: 需要安装 piexif（pip install piexif），会把 Orientation 设为 1 并尽量保留其它 EXIF
# - False: 不依赖 piexif，JPG 会去除全部 EXIF（最稳），PNG 本来也很少用 EXIF，统一移除
PRESERVE_OTHER_EXIF = True

# JPG 保存参数：尽量不劣化（依赖 Pillow 版本支持 keep；不支持时可改为具体数值如 quality=95）
JPEG_SAVE_KW = dict(quality="keep", subsampling="keep", progressive="keep")

def iter_images(root):
    exts = {".jpg", ".jpeg", ".png"}
    for r, _, fs in os.walk(root):
        for f in fs:
            ext = os.path.splitext(f.lower())[1]
            if ext in exts:
                yield os.path.join(r, f), ext

def fix_jpeg(path):
    img = Image.open(path)
    # 将 EXIF 方向应用到像素
    img = ImageOps.exif_transpose(img).convert("RGB")

    exif_bytes_to_write = None

    if PRESERVE_OTHER_EXIF:
        try:
            import piexif
            # 读取原 EXIF
            orig_exif = img.info.get("exif")
            if orig_exif:
                exif_dict = piexif.load(orig_exif)
            else:
                exif_dict = {"0th": {}, "Exif": {}, "GPS": {}, "1st": {}, "thumbnail": None}

            # Orientation 标签到 1（已正向）
            exif_dict["0th"][piexif.ImageIFD.Orientation] = 1
            exif_bytes_to_write = piexif.dump(exif_dict)
        except Exception as e:
            # 没装 piexif 或解析失败：退化成移除 EXIF
            exif_bytes_to_write = None

    # 保存为 jpg，不改变扩展名
    save_kw = {}
    # 合并保存参数（Pillow 新版支持 keep；如果你的 Pillow 不支持，可改为具体数值）
    save_kw.update(JPEG_SAVE_KW)
    if exif_bytes_to_write is not None:
        save_kw["exif"] = exif_bytes_to_write

    img.save(path, format="JPEG", **save_kw)

def fix_png(path):
    img = Image.open(path)
    # 将 EXIF 方向应用到像素
    img = ImageOps.exif_transpose(img)

    # PNG 不稳定支持 EXIF，最稳：把方向写进像素并去掉 EXIF/eXIf
    # 构造空的 pnginfo，避免复制原有的 eXIf/text 块
    pnginfo = PngImagePlugin.PngInfo()
    # 如需保留某些文本元数据，可在这里 pnginfo.add_text(key, value)

    # PNG 保存时不传 exif，这样不会写入 eXIf 块（Pillow 只有在提供 exif 参数时才写 eXIf）
    # 统一转成“合适模式”：若是标签图像也可直接用当前模式保存
    img.save(path, format="PNG", pnginfo=pnginfo, optimize=True)

def main():
    target = ROOT_DIR if len(sys.argv) < 2 else sys.argv[1]
    if not os.path.isdir(target):
        print(f"路径不存在或不是文件夹：{target}")
        sys.exit(1)

    print(f"Normalizing orientation & EXIF under: {target}")
    total = ok = 0
    failed = []

    for path, ext in iter_images(target):
        total += 1
        try:
            if ext in (".jpg", ".jpeg"):
                fix_jpeg(path)
            else:  # .png
                fix_png(path)
            ok += 1
            print(f"✅ OK  {path}")
        except Exception as e:
            failed.append((path, str(e)))
            print(f"❌ ERR {path} -> {e}")

    print("\n=== Summary ===")
    print(f"发现文件：{total}")
    print(f"成功处理：{ok}")
    if failed:
        print(f"失败：{len(failed)}")
        for p, e in failed[:10]:
            print(f" - {p} : {e}")
        if len(failed) > 10:
            print(f"   ... 其余 {len(failed)-10} 条省略")

    print("\n说明：")
    print("• 已将方向写入像素，防止查看器二次自动旋转。")
    print("• JPG：若安装 piexif 则保留其它 EXIF 并将 Orientation=1；否则移除全部 EXIF（更稳）。")
    print("• PNG：像素已转正，EXIF/eXIf 不再保留（PNG 对 EXIF 支持不一致）。")
    print("• 文件类型与扩展名均保持不变。")

if __name__ == "__main__":
    main()
