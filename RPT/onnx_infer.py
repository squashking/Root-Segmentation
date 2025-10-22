import math
from typing import List, Tuple, Dict, Any, Iterable, Union
import cv2
import numpy as np
import onnxruntime as ort

# ------------------------- utils -------------------------
def _softmax_nchw(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax on (N, C, H, W)."""
    z = x - x.max(axis=1, keepdims=True)
    e = np.exp(z, dtype=np.float32)
    return e / np.clip(e.sum(axis=1, keepdims=True), 1e-12, None)

def _flip_nchw(x: np.ndarray, fh: bool, fv: bool) -> np.ndarray:
    """Flip (N, C, H, W) along width (fh) and/or height (fv)."""
    if fh: x = x[..., :, ::-1]
    if fv: x = x[..., ::-1, :]
    return x

# ------------------------- reverse transform builders -------------------------
def get_reverse_list_np(
    ori_hw: Tuple[int, int],
    transforms: List[Dict[str, Any]],
) -> List[Tuple[str, Tuple[int, int]]]:
    """
    Build a reverse-operation list from a transforms config (mirrors the official pipeline).
    Each item is ("resize" or "padding", (h, w)) describing the pre-op size to restore to.
    """
    h, w = ori_hw
    reverse: List[Tuple[str, Tuple[int, int]]] = []

    for op in transforms or []:
        t = op.get("type", "")
        if t == "Resize":
            reverse.append(("resize", (h, w)))
            ts = op.get("target_size")
            if ts is not None:
                w, h = int(ts[0]), int(ts[1])

        elif t == "ResizeByShort":
            reverse.append(("resize", (h, w)))
            short = op["short_size"]
            long_edge = max(h, w)
            short_edge = min(h, w)
            long_edge = int(round(long_edge * short / short_edge))
            if h > w: h, w = long_edge, short
            else:     w, h = long_edge, short

        elif t == "ResizeByLong":
            reverse.append(("resize", (h, w)))
            long = op["long_size"]
            if h > w:
                w = int(round(w * long / h)); h = long
            else:
                h = int(round(h * long / w)); w = long

        elif t == "LimitLong":
            long_edge = max(h, w)
            short_edge = min(h, w)
            max_long = op.get("max_long")
            min_long = op.get("min_long")
            if (max_long is not None) and (long_edge > max_long):
                reverse.append(("resize", (h, w)))
                short_edge = int(round(short_edge * max_long / long_edge))
                long_edge = max_long
            elif (min_long is not None) and (long_edge < min_long):
                reverse.append(("resize", (h, w)))
                short_edge = int(round(short_edge * min_long / long_edge))
                long_edge = min_long
            if h > w: h, w = long_edge, short_edge
            else:     w, h = long_edge, short_edge

        elif t == "Padding":
            reverse.append(("padding", (h, w)))
            ts = op.get("target_size")
            if ts is not None:
                w, h = int(ts[0]), int(ts[1])

        elif t == "PaddingByAspectRatio":
            reverse.append(("padding", (h, w)))
            ar = float(op["aspect_ratio"])
            ratio = w / h
            if ratio > ar:
                h = int(w / ar)
            elif ratio < ar:
                w = int(h * ar)

    return reverse

def reverse_transform_np(
    pred_nchw: np.ndarray,
    ori_hw: Tuple[int, int],
    transforms: List[Dict[str, Any]],
) -> np.ndarray:
    """
    Restore (N, C, h, w) logits/probs back to the original image size by
    applying inverse transforms in reverse order (bilinear for logits/probs).
    """
    reverse = get_reverse_list_np(ori_hw, transforms)

    for item, (h, w) in reversed(reverse):
        if item == "resize":
            hwc = np.transpose(pred_nchw[0], (1, 2, 0))
            hwc = cv2.resize(hwc, (w, h), interpolation=cv2.INTER_LINEAR)
            pred_nchw = np.transpose(hwc, (2, 0, 1))[None]
        elif item == "padding":
            pred_nchw = pred_nchw[:, :, :h, :w]
        else:
            raise ValueError(f"Unexpected reverse op: {item}")

    # Ensure exact original spatial size
    H0, W0 = ori_hw
    if pred_nchw.shape[-2:] != (H0, W0):
        hwc = np.transpose(pred_nchw[0], (1, 2, 0))
        hwc = cv2.resize(hwc, (W0, H0), interpolation=cv2.INTER_LINEAR)
        pred_nchw = np.transpose(hwc, (2, 0, 1))[None]

    return pred_nchw

# ------------------------- visualization helpers -------------------------
def labels_to_mask_255(labels_hw: np.ndarray, fg_class: Union[int, Iterable[int]] = 1) -> np.ndarray:
    """
    Convert a label map (H, W, values 0..C-1) to a 0/255 binary mask.
    `fg_class` can be a single int or an iterable of ints (union).
    """
    if isinstance(fg_class, (list, tuple, set)):
        fg = np.isin(labels_hw, list(fg_class))
    else:
        fg = (labels_hw == int(fg_class))
    return (fg.astype(np.uint8) * 255)

def colorize_labels(labels_hw: np.ndarray, num_classes: int) -> np.ndarray:
    """
    Simple colorization for multi-class labels (H, W) -> (H, W, 3).
    For binary tasks, prefer labels_to_mask_255.
    """
    palette = np.array([
        [0, 0, 0], [255, 0, 0], [0, 255, 0], [0, 0, 255],
        [255, 255, 0], [255, 0, 255], [0, 255, 255], [128, 128, 128],
        [128, 0, 0], [0, 128, 0], [0, 0, 128]
    ], dtype=np.uint8)
    h, w = labels_hw.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    for k in range(num_classes):
        out[labels_hw == k] = palette[k % len(palette)]
    return out

def overlay_mask(bgr: np.ndarray, mask_255: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """Overlay a white mask onto BGR image with blending."""
    overlay = bgr.copy()
    idx = mask_255 > 0
    overlay[idx] = (overlay[idx] * (1 - alpha) + np.array([255, 255, 255]) * alpha).astype(np.uint8)
    return overlay

# ------------------------- main predictor -------------------------
class OnnxSegOfficial:
    """
    Official-semantics ONNX segmentation predictor.
    `output` selects what to return:
      - "label": class indices (uint8)
      - "mask":  binary 0/255 mask for `fg_class`
      - "color": colorized labels
    """

    def __init__(
        self,
        onnx_path: str,
        mean: Tuple[float, float, float] = (0.5, 0.5, 0.5),
        std: Tuple[float, float, float] = (0.5, 0.5, 0.5),
        use_cpu: bool = False,
        ort_disable_opt: bool = False,
    ):
        self.mean = np.array(mean, np.float32)
        self.std = np.array(std, np.float32)

        so = ort.SessionOptions()
        if ort_disable_opt:
            so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
            so.add_session_config_entry("session.disable_prepacking", "1")

        providers = ['CPUExecutionProvider'] if use_cpu else ['CUDAExecutionProvider', 'CPUExecutionProvider']
        self.sess = ort.InferenceSession(onnx_path, sess_options=so, providers=providers)
        self.inp = self.sess.get_inputs()[0].name
        self.out = self.sess.get_outputs()[0].name

    # ---------- preprocessing ----------
    def _pre(self, bgr: np.ndarray) -> np.ndarray:
        """BGR -> RGB -> float32 [0,1] -> normalize -> CHW"""
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        rgb = (rgb - self.mean) / self.std
        x = np.transpose(rgb, (2, 0, 1))[None].copy()
        return x

    def _forward(self, x: np.ndarray) -> np.ndarray:
        """Single forward pass: returns logits (1, C, h, w)."""
        return self.sess.run([self.out], {self.inp: x})[0]

    # ---------- sliding window (sum logits, average by coverage) ----------
    def _slide(self, x: np.ndarray, crop_size: Tuple[int, int], stride: Tuple[int, int]) -> np.ndarray:
        _, _, H, W = x.shape
        w_crop, h_crop = crop_size
        w_strd, h_strd = stride

        rows = 1 if H <= h_crop else int(math.ceil((H - h_crop) / h_strd)) + 1
        cols = 1 if W <= w_crop else int(math.ceil((W - w_crop) / w_strd)) + 1

        acc = None
        cnt = np.zeros((1, 1, H, W), dtype=np.float32)

        for r in range(rows):
            for c in range(cols):
                h1 = max(min(r * h_strd, H - h_crop), 0)
                w1 = max(min(c * w_strd, W - w_crop), 0)
                h2 = min(h1 + h_crop, H)
                w2 = min(w1 + w_crop, W)

                logit = self._forward(x[:, :, h1:h2, w1:w2])[0]  # (C, h, w)
                if acc is None:
                    acc = np.zeros((1, logit.shape[0], H, W), np.float32)

                acc[:, :, h1:h2, w1:w2] += logit[:, :h2 - h1, :w2 - w1]
                cnt[:, :, h1:h2, w1:w2] += 1.0

        return acc / np.clip(cnt, 1e-6, None)

    # ---------- post-conversion to desired output ----------
    @staticmethod
    def _format_output(
        labels_hw: np.ndarray,
        output: str,
        fg_class: Union[int, Iterable[int]],
        num_classes: int,
    ) -> np.ndarray:
        """
        Convert label map to chosen visualization:
          - "label": (H, W) uint8 labels
          - "mask" : (H, W) 0/255 uint8 mask for fg_class
          - "color": (H, W, 3) colorized labels
        """
        if output == "label":
            return labels_hw.astype(np.uint8)
        elif output == "mask":
            return labels_to_mask_255(labels_hw, fg_class=fg_class)
        elif output == "color":
            return colorize_labels(labels_hw.astype(np.uint8), num_classes=num_classes)
        else:
            raise ValueError("Unsupported output='{}'".format(output))

    # ---------- single-scale predict ----------
    def predict(
        self,
        bgr: np.ndarray,
        transforms_cfg: List[Dict[str, Any]] = None,
        is_slide: bool = False,
        stride: Tuple[int, int] = (448, 448),
        crop_size: Tuple[int, int] = (512, 512),
        output: str = "label",                         # "label" | "mask" | "color"
        fg_class: Union[int, Iterable[int]] = 1,       # used when output="mask"
    ) -> np.ndarray:
        x = self._pre(bgr)
        logit = self._slide(x, crop_size, stride) if is_slide else self._forward(x)  # (1, C, h, w)

        num_classes = int(logit.shape[1])  # remember C for colorization

        # Restore to original image (bilinear for logits/probs)
        logit = reverse_transform_np(logit, bgr.shape[:2], transforms_cfg or [])

        # Argmax to labels (H, W)
        labels = np.argmax(logit, axis=1, keepdims=False)[0]
        return self._format_output(labels, output=output, fg_class=fg_class, num_classes=num_classes)

    # ---------- multi-scale + flip predict ----------
    def predict_aug(
        self,
        bgr: np.ndarray,
        transforms_cfg: List[Dict[str, Any]] = None,
        scales: Union[Iterable[float], float] = (1.0,),
        flip_h: bool = False,
        flip_v: bool = False,
        is_slide: bool = False,
        stride: Tuple[int, int] = (448, 448),
        crop_size: Tuple[int, int] = (512, 512),
        output: str = "label",
        fg_class: Union[int, Iterable[int]] = 1,
    ) -> np.ndarray:
        x0 = self._pre(bgr)
        h_in, w_in = x0.shape[-2:]

        flips = [(False, False)]
        if flip_h: flips.append((True, False))
        if flip_v: flips.append((False, True))
        if flip_h and flip_v: flips.append((True, True))

        acc_prob_input = 0.0
        num_classes = None

        scale_list = list(scales) if not isinstance(scales, (list, tuple)) else list(scales)
        for s in scale_list:
            h = int(h_in * s + 0.5)
            w = int(w_in * s + 0.5)

            x_s = np.ascontiguousarray(cv2.resize(x0[0].transpose(1, 2, 0), (w, h), interpolation=cv2.INTER_LINEAR))
            x_s = np.transpose(x_s, (2, 0, 1))[None]

            for fh, fv in flips:
                x_flip = _flip_nchw(x_s, fh, fv)
                logit = self._slide(x_flip, crop_size, stride) if is_slide else self._forward(x_flip)
                if num_classes is None:
                    num_classes = int(logit.shape[1])
                logit = _flip_nchw(logit, fh, fv)  # flip back

                # Resize logits back to input size, softmax, accumulate
                hwc = np.transpose(logit[0], (1, 2, 0))
                hwc = cv2.resize(hwc, (w_in, h_in), interpolation=cv2.INTER_LINEAR)
                logit_in = np.transpose(hwc, (2, 0, 1))[None]
                acc_prob_input += _softmax_nchw(logit_in)

        # Reverse to original image size, then argmax to labels
        prob = reverse_transform_np(acc_prob_input, bgr.shape[:2], transforms_cfg or [])
        labels = np.argmax(prob, axis=1, keepdims=False)[0]
        return self._format_output(labels, output=output, fg_class=fg_class, num_classes=num_classes or 2)



seg = OnnxSegOfficial(
    "./output/esegformer_b0/best_model/model.onnx",
    mean=(0.5, 0.5, 0.5),
    std=(0.5, 0.5, 0.5),
    use_cpu=False,
)

img = cv2.imread("test2.jpg")

# 1) Binary mask (0/255) for foreground class 1 (common for 2-class models)
mask_255 = seg.predict(
    img,
    transforms_cfg=[],           # or your real transforms if you apply them
    is_slide=True,
    stride=(448, 448),
    crop_size=(512, 512),
    output="mask",
    fg_class=1,
)
cv2.imwrite("predict_mask.png", mask_255)

# 2) Colorized labels for multi-class visualization
color = seg.predict(
    img,
    transforms_cfg=[],
    is_slide=True,
    stride=(448, 448),
    crop_size=(512, 512),
    output="color",
)
cv2.imwrite("predict_color.png", color)

# 3) If you want an overlay:
overlay = overlay_mask(img, mask_255, alpha=0.4)
cv2.imwrite("predict_overlay.png", overlay)