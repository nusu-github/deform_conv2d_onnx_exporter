[![PyPI version](https://badge.fury.io/py/deform-conv2d-onnx-exporter.svg)](https://badge.fury.io/py/deform-conv2d-onnx-exporter) [![Test and Release](https://github.com/nusu-github/deform_conv2d_onnx_exporter/actions/workflows/ci.yml/badge.svg)](https://github.com/nusu-github/deform_conv2d_onnx_exporter/actions/workflows/ci.yml)

# deform\_conv2d\_onnx\_exporter

## Overview

This module enables you to export `deform_conv2d` to ONNX in PyTorch.

PyTorch does not natively support exporting `deform_conv2d` into ONNX, so this module provides a custom ONNX operator implementation.

This module implements Deformable Convolution v2, described in a paper, `Deformable ConvNets v2: More Deformable, Better Results <https://arxiv.org/abs/1811.11168>`, using ONNX operators.
The implementation uses ONNXScript and PyTorch's dynamo-based ONNX exporter for better compatibility with PyTorch 2.6+.

## Installation

```sh
pip install deform_conv2d_onnx_exporter
```

## Requirements

- Python 3.9 or later
- PyTorch 2.6.0 or later
- torchvision 0.21.0 or later
- onnxscript 0.5.6 or later

## Usage

```python
import torch
from torchvision.ops.deform_conv import DeformConv2d
import deform_conv2d_onnx_exporter

# Create custom ONNX operator
custom_op = deform_conv2d_onnx_exporter.create_deform_conv2d_custom_op()

# Create model and input tensors
model = DeformConv2d(
    in_channels=4,
    out_channels=8,
    kernel_size=(3, 3),
    stride=(1, 1),
    padding=(1, 1),
)
model.eval()

input = torch.randn(2, 4, 10, 10)
offset = torch.randn(2, 18, 10, 10)
mask = torch.rand(2, 9, 10, 10)

# Export to ONNX using dynamo-based exporter
onnx_program = torch.onnx.export(
    model,
    (input, offset, mask),
    "output.onnx",
    dynamo=True,
    custom_translation_table={
        torch.ops.torchvision.deform_conv2d.default: custom_op,
    },
)
```

**Note**: This module requires PyTorch 2.6+ with the dynamo-based ONNX exporter and uses ONNX Opset 18.

## Tests

Run the test suite using:

```sh
python -m unittest discover -s tests
```

Or with uv:

```sh
uv run python -m unittest discover -s tests
```

## Development notes

### Options for `deform_conv2d_onnx_exporter.create_deform_conv2d_custom_op()`

You can specify 2 options for this function:

- `use_gathernd` (default: `True`):
  If `True`, use `GatherND` operator. Otherwise, use `GatherElements` operator.
- `enable_openvino_patch` (default: `False`):
  If `True`, enable patch for OpenVINO compatibility.

### Referenced paper

This module implements Deformable Convolution v2, described in a paper, "[Deformable ConvNets v2: More Deformable, Better Results](https://arxiv.org/abs/1811.11168)", using ONNX operators.

Some of the variable names in the module, such as `p` and `p_0`, are based on the paper.

### Memory layout of `offset`

The detail of `deform_conv2d` implementation in PyTorch is not fully documented.
Therefore, I investigated the [implementation](https://github.com/pytorch/vision/blob/19ad0bbc5e26504a501b9be3f0345381d6ba1efc/torchvision/csrc/ops/cpu/deform_conv2d_kernel.cpp) to understand memory layout of some variables, such as `offset`.

- `offset`
  The shape is `(batch, 2 * group * kernel_h * kernel_w, out_h, out_w)` according to the [reference](https://pytorch.org/vision/stable/ops.html#torchvision.ops.deform_conv2d).
  The internal memory layout of `2 * group * kernel_h * kernel_w` is not clear.
  According to the source code, it seems to be `(batch, group, kernel_h, kernel_w, 2, out_h, out_w)`.
  The size `2` means "y-coords and x-coords".

### Padding of `input`

Even if `padding` is set to `0`, this module adds at least 1 padding internally.
This is necessary to handle out-of-bounds `offset` appropriately.

### Performance

When this module was originally created, ONNX did not have a native `deform_conv2d` operator.
Since then, ONNX has added a DeformConv operator (Opset 19, later enhanced in Opset 22),
but it is still not practical to use because:

- PyTorch's ONNX exporter does not support exporting to the official ONNX DeformConv operator
  - See: [pytorch/vision#2066](https://github.com/pytorch/vision/issues/2066)
- ONNX Runtime has not yet implemented the DeformConv operator
  - See: [microsoft/onnxruntime#22060](https://github.com/microsoft/onnxruntime/issues/22060)

Therefore, this module continues to implement deformable convolution using standard ONNX operators
such as `GatherND`, `Clip`, `Cast`, and others.

**Note**: While TensorRT provides plugin support for DeformConv, and you can convert ONNX models to TensorRT engines, this module targets general ONNX Runtime compatibility.

The implementation has been carefully optimized to:

- Minimize unnecessary or duplicated calculations
- Use efficient ONNX operators where possible
- Support dynamic shapes through runtime shape computation

While not as efficient as native implementations, the performance is acceptable for most use cases.

### Opset version

This module uses **Opset 18** and leverages the following operators:

- `Clip`: For clamping coordinate values within valid ranges
- `GatherND`: For gathering elements at specified coordinates (with `batch_dims` attribute)
- `Shape`, `Gather`, `Squeeze`: For dynamic shape computation
- `Cast`, `Reshape`, `Transpose`: For tensor manipulation
- `Constant`: For creating constant tensors with various data types

## Acknowledgments

This project was originally created by [Masamitsu MURASE](https://github.com/masamitsu-murase) and provided an invaluable foundation for ONNX export of deformable convolution operations.

The current version has been modernized to support PyTorch 2.6+ and the new dynamo-based ONNX exporter with ONNXScript, while preserving the core algorithm and design principles from the original implementation.

We extend our deepest gratitude to Masamitsu MURASE for his excellent work and contributions to the community.

## License

You can use this module under the MIT License.

Copyright 2021 Masamitsu MURASE (Original implementation)
Copyright 2025 nusu-github (Modernization to PyTorch 2.6+/ONNXScript)

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
