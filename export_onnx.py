# export_onnx.py
import torch
from train import CRNN, IMG_H, IMG_W

def main():
    model = CRNN()
    model.load_state_dict(torch.load('crnn_digits.pt', map_location='cpu'))
    model.eval()

    dummy = torch.randn(1, 1, IMG_H, IMG_W)
    with torch.no_grad():
        pt_out = model(dummy)
    print("PyTorch output shape:", tuple(pt_out.shape))

    torch.onnx.export(
        model, dummy, 'crnn_digits.onnx',
        input_names=['input'],
        output_names=['logits'],
        opset_version=13,
        do_constant_folding=True,
        dynamo=False,
    )
    print("✓ Saved crnn_digits.onnx")

    # صحت‌سنجی ساده بدون onnxruntime
    import onnx
    onnx_model = onnx.load('crnn_digits.onnx')
    onnx.checker.check_model(onnx_model)
    print("✓ ONNX model is valid")

if __name__ == '__main__':
    main()
    