import paddle, paddle2onnx
from paddleseg.cvlibs import manager
from paddle.static import InputSpec

model = manager.MODELS['ESegFormer_B2'](num_classes=2)

model.set_state_dict(paddle.load(r"C:\Users\28274\PycharmProjects\RPT\new_onnx_test\model.pdparams"))
model.eval()  # 关键：切到推理态

# 先导出 Paddle 静态图（可用动态尺寸）
paddle.jit.save(model, './tmp/eseg_b0', input_spec=[InputSpec([1, 3, 512, 512], 'float32')])

# 再转 ONNX（常用 opset 13）
paddle2onnx.export(
    model_filename='./tmp/eseg_b0.pdmodel',
    params_filename='./tmp/eseg_b0.pdiparams',
    save_file=r"C:\Users\28274\PycharmProjects\RPT\new_onnx_test\model.onnx",
    opset_version=13,
    enable_onnx_checker=True
)
