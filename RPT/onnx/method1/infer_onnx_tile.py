# infer_onnx_tile.py
import os, math, cv2, numpy as np, onnxruntime as ort

# 固定路径（你之前给的）
MODEL  = r"C:\Users\28274\PycharmProjects\RPT\onnx_test\model.onnx"
IMAGE  = r"C:\Users\28274\PycharmProjects\RPT\onnx_test\test2.jpg"
OUTPUT = r"C:\Users\28274\PycharmProjects\RPT\onnx_test\mask_onnx2.png"

# 训练/预测一致的归一化
MEAN = (0.5, 0.5, 0.5)
STD  = (0.5, 0.5, 0.5)

# 滑窗与后处理
TILE = 512       # 每块尺寸；内存紧可降到 384/320
OVERLAP = 64     # 重叠越大越平滑（32~128）
THRESH = 0.5     # 概率阈值（导出的 onnx 已输出前景概率）
GPU_ID = 0

def preprocess(bgr):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    m = np.array(MEAN, np.float32).reshape(1,1,3)
    s = np.array(STD,  np.float32).reshape(1,1,3)
    x = (rgb - m) / s
    return np.transpose(x, (2,0,1))[None].astype(np.float32)  # (1,3,H,W)

def make_session(model_path):
    so = ort.SessionOptions()
    # 关闭高等级优化与预打包，降低峰值内存
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    so.add_session_config_entry("session.disable_prepacking", "1")
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    provider_options = [{"device_id": GPU_ID}, {}]
    try:
        sess = ort.InferenceSession(model_path, sess_options=so,
                                    providers=providers, provider_options=provider_options)
    except Exception:
        sess = ort.InferenceSession(model_path, sess_options=so, providers=["CPUExecutionProvider"])
    print("Using providers:", sess.get_providers())
    return sess

def weight_window(h, w, overlap):
    # 余弦边缘权重，减轻接缝
    y = np.linspace(0, 1, h, dtype=np.float32)
    x = np.linspace(0, 1, w, dtype=np.float32)
    xv, yv = np.meshgrid(x, y)
    edge = np.minimum.reduce([xv, 1-xv, yv, 1-yv])
    ramp = np.clip(edge / (max(1, overlap//2) / min(h, w)), 0, 1)
    win = 0.5 - 0.5 * np.cos(np.pi * ramp)
    return np.clip(win, 1e-3, 1.0)

def main():
    # 读图
    bgr = cv2.imread(IMAGE, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(IMAGE)
    H, W = bgr.shape[:2]

    sess = make_session(MODEL)
    inp = sess.get_inputs()[0].name

    tile, overlap = TILE, OVERLAP
    stride = tile - overlap
    ny = max(1, math.ceil((H - overlap) / stride))
    nx = max(1, math.ceil((W - overlap) / stride))

    acc  = np.zeros((H, W), np.float32)    # 概率累加
    wsum = np.zeros((H, W), np.float32)
    wwin = weight_window(tile, tile, overlap)

    for iy in range(ny):
        sy = iy * stride
        ey = min(sy + tile, H); sy = max(0, ey - tile)
        for ix in range(nx):
            sx = ix * stride
            ex = min(sx + tile, W); sx = max(0, ex - tile)

            patch = bgr[sy:ey, sx:ex]
            x = preprocess(patch)
            y = sess.run(None, {inp: x})[0]   # 期望 (1,1,h,w) 概率
            if y.ndim == 4: y = y[0]
            prob = y[0] if y.ndim == 3 else y  # (h,w)

            # 边缘块可能小于 TILE，匹配权重窗口
            wh = wwin[:(ey-sy), :(ex-sx)]
            acc[sy:ey, sx:ex]  += prob * wh
            wsum[sy:ey, sx:ex] += wh

    fused = acc / np.clip(wsum, 1e-6, None)
    mask  = (fused >= THRESH).astype(np.uint8) * 255

    os.makedirs(os.path.dirname(OUTPUT) or ".", exist_ok=True)
    cv2.imwrite(OUTPUT, mask)
    print("✅ 保存：", OUTPUT)

if __name__ == "__main__":
    main()
