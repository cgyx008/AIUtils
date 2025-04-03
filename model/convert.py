from pathlib import Path

import numpy as np
import onnx
import onnxsim
import torch


def load_model(pt_path):
    """
    Load PyTorch model.
    Args:
        pt_path (str | Path): pt path

    Returns:
        (torch.nn.Module): model
    """
    pt = torch.load(pt_path, map_location='cpu')
    pt_model = pt['model'] if isinstance(pt, dict) and 'model' in pt else pt
    return pt_model


def pt2onnx(pt_path, onnx_path='', simplify=True, **kwargs):
    """Convert PyTorch model to simplified ONNX model"""
    # .pt to .onnx
    pt_model = load_model(pt_path)
    dummy_input = torch.randn(1, 3, 224, 224)
    onnx_path = onnx_path or Path(pt_path).with_suffix('.onnx')
    torch.onnx.export(pt_model, dummy_input, onnx_path, **kwargs)
    print(torch.onnx.export_to_pretty_string(pt_model, dummy_input))
    print('=' * 80)

    # Checks
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    # Print a human-readable representation of the graph
    print(onnx.helper.printable_graph(onnx_model.graph))

    # Simplify
    if simplify:
        onnx_model, check = onnxsim.simplify(onnx_model)
        assert check, 'assert check failed'
        onnx.save(onnx_model, onnx_path)


def count_parameters(onnx_path):
    """
    计算 ONNX 模型的参数量。

    参数:
        onnx_path (str): ONNX 模型路径

    返回:
        int: 模型参数的数量。
    """
    model = onnx.load(onnx_path)

    total_params = 0
    for initializer in model.graph.initializer:
        # 计算张量的元素数量
        tensor_shape = initializer.dims
        tensor_elements = np.prod(tensor_shape)
        total_params += tensor_elements
    return total_params


def main():
    print(count_parameters(r'F:\GH_Novaic\model\yolov8x-worldv2.onnx'))


if __name__ == '__main__':
    main()
