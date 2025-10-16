import os
from PIL import Image
import numpy as np
from collections import defaultdict

# === Config: change these two lines to your paths ===
IMAGES_DIR = r"C:\Users\28274\Desktop\train_data2\images"
ANN_DIR    = r"C:\Users\28274\Desktop\train_data2\annotations"

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
LBL_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}  # labels are recommended to use .png

def list_files(root, exts):
    files = []
    for r, _, fs in os.walk(root):
        for f in fs:
            if os.path.splitext(f.lower())[1] in exts:
                files.append(os.path.join(r, f))
    return sorted(files)

def stem(path):
    b = os.path.basename(path)
    s, _ = os.path.splitext(b)
    return s

def rel(root, path):
    return os.path.relpath(path, root)

def main():
    print("Scanning dataset...")
    imgs = list_files(IMAGES_DIR, IMG_EXTS)
    lbls = list_files(ANN_DIR, LBL_EXTS)

    print(f"Found images: {len(imgs)}")
    print(f"Found labels: {len(lbls)}")

    img_by_stem = defaultdict(list)
    for p in imgs:
        img_by_stem[stem(p)].append(p)

    lbl_by_stem = defaultdict(list)
    for p in lbls:
        lbl_by_stem[stem(p)].append(p)

    common = sorted(set(img_by_stem.keys()) & set(lbl_by_stem.keys()))
    only_img = sorted(set(img_by_stem.keys()) - set(lbl_by_stem.keys()))
    only_lbl = sorted(set(lbl_by_stem.keys()) - set(img_by_stem.keys()))

    problems = []
    checked = 0
    ok_pairs = 0

    if only_img:
        problems.append(f"[Missing labels] {len(only_img)} images have no corresponding labels (matched by filename): examples: {only_img[:5]}")
    if only_lbl:
        problems.append(f"[Missing images] {len(only_lbl)} labels have no corresponding images (matched by filename): examples: {only_lbl[:5]}")

    def check_pair(img_path, lbl_path):
        nonlocal ok_pairs, checked
        checked += 1
        # read image
        try:
            img = Image.open(img_path)
            arr_img = np.array(img)
        except Exception as e:
            problems.append(f"[Unreadable image] {rel(IMAGES_DIR, img_path)} error: {e}")
            return

        # read label
        try:
            lab = Image.open(lbl_path)
            arr_lab = np.array(lab)
        except Exception as e:
            problems.append(f"[Unreadable label] {rel(ANN_DIR, lbl_path)} error: {e}")
            return

        # size check
        if arr_img.shape[0] != arr_lab.shape[0] or arr_img.shape[1] != arr_lab.shape[1]:
            problems.append(f"[Size mismatch] {rel(IMAGES_DIR, img_path)} vs {rel(ANN_DIR, lbl_path)} "
                            f"image={arr_img.shape[:2]} label={arr_lab.shape[:2]}")
            return

        # label channel check
        if arr_lab.ndim == 3:
            problems.append(f"[Multi-channel label] {rel(ANN_DIR, lbl_path)} shape={arr_lab.shape} "
                            f"→ expected single-channel (H,W) index map (grayscale)")
            return
        elif arr_lab.ndim != 2:
            problems.append(f"[Label dimension error] {rel(ANN_DIR, lbl_path)} shape={arr_lab.shape} "
                            f"→ expected (H,W)")
            return

        # label values check
        uniq = np.unique(arr_lab)
        # allowed {0,1} or {0,255} or extreme single-value cases
        allowed = (
            set(uniq.tolist()) <= {0, 1} or
            set(uniq.tolist()) <= {0, 255}
        )
        if not allowed:
            problems.append(f"[Invalid label values] {rel(ANN_DIR, lbl_path)} unique={uniq[:20]} "
                            f"→ expected only {0,1} or {0,255}; please binarize")
            return

        ok_pairs += 1

    # check each pair
    for k in common:
        # if multiple files with the same name exist, only take the first one to avoid misjudgment
        img_path = img_by_stem[k][0]
        lbl_path = lbl_by_stem[k][0]
        check_pair(img_path, lbl_path)

    # summary
    print("\n=== Summary ===")
    print(f"Paired files (matched by filename): {len(common)}")
    print(f"Valid samples (size/label format OK): {ok_pairs}/{checked}")
    if problems:
        print(f"\nFound {len(problems)} issues:")
        for i, p in enumerate(problems, 1):
            print(f"{i:02d}. {p}")
    else:
        print("✅ All good: ready for training!")

    # friendly suggestions
    if problems:
        print("\n=== Suggestions ===")
        print("1) For [Multi-channel label] or [Invalid label values], batch-binarize to 0/255 first.")
        print("   → Refer to your previous convert_annotations_to_bin.py; or I can provide you with an in-place overwrite version.")
        print("2) Ensure images and annotations have identical names (without extension) and matching sizes.")
        print("3) Labels are recommended to be .png single-channel grayscale.")

if __name__ == "__main__":
    main()
