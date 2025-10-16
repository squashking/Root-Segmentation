# export_esegformer_to_onnx.py
import os
import sys
import inspect
import shutil
import subprocess

import paddle
from paddle.static import InputSpec
import paddle.nn as nn
import paddle2onnx

# ====== 按你的工程实际导入 ======
from paddleseg.models.esegformer import (
    ESegFormer_B0, ESegFormer_B1, ESegFormer_B2,
    ESegFormer_B3, ESegFormer_B4, ESegFormer_B5
)

# ========= 配置 =========
MODEL_VERSION = "B0"  # 你的模型是 B0
NUM_CLASSES   = 2     # 你的任务类别数
PDPARAMS_PATH = r"C:\Users\28274\PycharmProjects\RPT\new_onnx_test\model.pdparams"
# 直接导出到你要用的 onnx 路径：
ONNX_PATH     = r"/output/onnx/b0/model1.onnx"
OPSET         = 13
# =======================

class WrapperWithActivation(nn.Layer):

    def __init__(self, core):
        super().__init__()
        self.core = core

    def forward(self, x):
        out = self.core(x)[0]  # (N,C,H,W)
        if out.shape[1] == 1:
            out = nn.functional.sigmoid(out)  # (N,1,H,W) 概率
        else:
            out = nn.functional.softmax(out, axis=1)[:, 1:2, :, :]  # 取前景通道 (N,1,H,W)
        return out

def build_model(version: str):
    v = version.upper()
    if v == "B0": return ESegFormer_B0(num_classes=NUM_CLASSES)
    if v == "B1": return ESegFormer_B1(num_classes=NUM_CLASSES)
    if v == "B2": return ESegFormer_B2(num_classes=NUM_CLASSES)
    if v == "B3": return ESegFormer_B3(num_classes=NUM_CLASSES)
    if v == "B4": return ESegFormer_B4(num_classes=NUM_CLASSES)
    if v == "B5": return ESegFormer_B5(num_classes=NUM_CLASSES)
    raise ValueError(f"未知版本: {version}")

def main():
    save_dir   = os.path.dirname(ONNX_PATH) or "."
    save_name  = os.path.splitext(os.path.basename(ONNX_PATH))[0]
    os.makedirs(save_dir, exist_ok=True)


    core = build_model(MODEL_VERSION)
    print(f"-> Loading: {PDPARAMS_PATH}")
    state = paddle.load(PDPARAMS_PATH)
    core.set_state_dict(state)
    core.eval()

    model = WrapperWithActivation(core)
    model.eval()

    # 2) 保存静态图（动态H/W）
    input_spec = [InputSpec(shape=[None, 3, None, None], dtype="float32", name="image")]
    jit_prefix = os.path.join(save_dir, save_name)
    paddle.jit.save(model, jit_prefix, input_spec=input_spec)

    pdmodel  = jit_prefix + ".pdmodel"
    pdiparams= jit_prefix + ".pdiparams"

    if not (os.path.exists(pdmodel) and os.path.exists(pdiparams)):
        raise FileNotFoundError("静态图导出失败：未找到 .pdmodel / .pdiparams")

    # 3) 导出 ONNX（兼容新/旧接口/CLI）
    def export_onnx():
        fn = getattr(paddle2onnx, "export_model", None)
        if callable(fn):
            return fn(model_file=pdmodel, params_file=pdiparams,
                      save_file=ONNX_PATH, opset_version=OPSET,
                      enable_onnx_checker=True)

        if hasattr(paddle2onnx, "export") and callable(paddle2onnx.export):
            sig = inspect.signature(paddle2onnx.export)
            keys = set(sig.parameters.keys())
            if {"model_dir", "model_filename", "params_filename"}.issubset(keys):
                return paddle2onnx.export(
                    model_dir=save_dir,
                    model_filename=os.path.basename(pdmodel),
                    params_filename=os.path.basename(pdiparams),
                    save_file=ONNX_PATH, opset_version=OPSET,
                    enable_onnx_checker=True)
            if {"model_file", "params_file"}.issubset(keys):
                return paddle2onnx.export(
                    model_file=pdmodel, params_file=pdiparams,
                    save_file=ONNX_PATH, opset_version=OPSET,
                    enable_onnx_checker=True)

        exe = shutil.which("paddle2onnx")
        if exe is None:
            raise RuntimeError("未找到 paddle2onnx，可 `pip install paddle2onnx`")
        cmd = [
            exe, "--model_dir", save_dir,
            "--model_filename", os.path.basename(pdmodel),
            "--params_filename", os.path.basename(pdiparams),
            "--save_file", ONNX_PATH, "--opset_version", str(OPSET),
            "--enable_onnx_checker", "True"
        ]
        print("ℹ️ 使用 CLI：", " ".join(cmd))
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            print(r.stdout)
            print(r.stderr, file=sys.stderr)
            raise RuntimeError("paddle2onnx CLI 导出失败")

    export_onnx()
    print(f" 导出成功：{ONNX_PATH}")

    try:
        import onnx, onnxruntime as ort, numpy as np
        onnx.checker.check_model(onnx.load(ONNX_PATH))
        print("✅ ONNX 检查通过")
        sess = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
        dummy = np.random.rand(1,3,256,256).astype("float32")
        out = sess.run(None, {sess.get_inputs()[0].name: dummy})[0]
        print("✅ ORT 试推理 OK, 输出形状:", out.shape, "值域: [", float(out.min()), ",", float(out.max()), "]")
    except Exception as e:
        print("⚠️ 自检提示：", e)

if __name__ == "__main__":
    main()
