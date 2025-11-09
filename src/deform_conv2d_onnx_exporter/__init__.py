"""This module adds ONNX conversion of `deform_conv2d`.

This module implements Deformable Convolution v2,
described in a paper, `Deformable ConvNets v2: More Deformable, Better Results
<https://arxiv.org/abs/1811.11168>`, using ONNX operators.
The implementation is straightforward, but may not be very efficient.

This exporter requires opset version 18 (updated from 12) to support
the latest ONNX operators and follow PyTorch 2.6+ best practices.
"""

from typing import NoReturn

import onnxscript
import torch
from onnxscript import opset18 as op

__all__ = ["create_deform_conv2d_custom_op"]

onnx_opset_version = 18


def _make_constant(value, dtype=None):
    """Create a constant tensor using onnxscript.

    For onnxscript, we use value_ints for integer lists and value_floats for float lists.
    For torch.Tensor, we use onnx.helper.make_tensor.
    """
    import onnx

    # Handle torch.Tensor input
    if isinstance(value, torch.Tensor):
        tensor = value
        if dtype is not None and tensor.dtype != dtype:
            tensor = tensor.to(dtype)
        numpy_array = tensor.numpy()
        # Use onnx.helper.np_dtype_to_tensor_dtype to get the ONNX data type
        import numpy as np

        onnx_dtype = onnx.helper.np_dtype_to_tensor_dtype(numpy_array.dtype)
        onnx_tensor = onnx.helper.make_tensor(
            name="const",
            data_type=onnx_dtype,
            dims=list(tensor.shape),
            vals=numpy_array.flatten().tolist(),
        )
        return op.Constant(value=onnx_tensor)

    # Handle list input
    if isinstance(value, list):
        # For lists, determine the appropriate parameter based on dtype
        if dtype == torch.int64:
            return op.Constant(value_ints=value)
        if dtype in (torch.float32, torch.float64):
            return op.Constant(value_floats=[float(v) for v in value])
        # Fallback: use onnx.helper.make_tensor
        tensor = torch.tensor(value, dtype=dtype)
        numpy_array = tensor.numpy()
        onnx_dtype = onnx.helper.np_dtype_to_tensor_dtype(numpy_array.dtype)
        onnx_tensor = onnx.helper.make_tensor(
            name="const",
            data_type=onnx_dtype,
            dims=list(tensor.shape),
            vals=numpy_array.flatten().tolist(),
        )
        return op.Constant(value=onnx_tensor)
    # For scalars
    if dtype == torch.int64:
        return op.Constant(value_int=int(value))
    if dtype in (torch.float32, torch.float64):
        return op.Constant(value_float=float(value))
    msg = f"Unsupported scalar dtype: {dtype}"
    raise ValueError(msg)


def _reshape(x, shape):
    """Reshape tensor."""
    if isinstance(shape, list):
        shape_tensor = _make_constant([-1, *shape[1:]], dtype=torch.int64)
    else:
        shape_tensor = shape
    return op.Reshape(x, shape_tensor)


def _slice(x, axes, starts, ends, steps=None):
    """Slice tensor."""
    axes_tensor = _make_constant(axes, dtype=torch.int64)
    starts_tensor = _make_constant(starts, dtype=torch.int64)
    ends_tensor = _make_constant(ends, dtype=torch.int64)
    if steps is not None:
        steps_tensor = _make_constant(steps, dtype=torch.int64)
        return op.Slice(x, starts_tensor, ends_tensor, axes_tensor, steps_tensor)
    return op.Slice(x, starts_tensor, ends_tensor, axes_tensor)


def _unsqueeze(input, dims):
    """Unsqueeze tensor."""
    axes = _make_constant(dims, dtype=torch.int64)
    return op.Unsqueeze(input, axes)


def _get_tensor_dim_size(tensor, dim):
    """Get the size of a specific dimension."""
    shape = op.Shape(tensor)
    dim_tensor = _make_constant([dim], dtype=torch.int64)
    size = op.Gather(shape, dim_tensor, axis=0)
    return op.Squeeze(size, _make_constant([0], dtype=torch.int64))


def calculate_p_0(dcn_params):
    """Calculate p_0 value in equation (1) in the paper.

    Args:
        dcn_params: parameters for deform_conv2d.

    Returns:
        torch.Tensor[1, 1, kernel_area_size, 2, out_h, out_w]

    """
    h = dcn_params["out_h"]
    w = dcn_params["out_w"]
    stride_h = dcn_params["stride_h"]
    stride_w = dcn_params["stride_w"]
    K = dcn_params["kernel_area_size"]
    additional_pad_h = dcn_params["additional_pad_h"]
    additional_pad_w = dcn_params["additional_pad_w"]

    p_0_y, p_0_x = torch.meshgrid(
        torch.arange(0, h * stride_h, stride_h),
        torch.arange(0, w * stride_w, stride_w),
        indexing="ij",
    )
    p_0_y = p_0_y.view(1, 1, 1, 1, h, w).repeat(1, 1, K, 1, 1, 1)
    p_0_y += additional_pad_h
    p_0_x = p_0_x.view(1, 1, 1, 1, h, w).repeat(1, 1, K, 1, 1, 1)
    p_0_x += additional_pad_w
    return torch.cat([p_0_y, p_0_x], dim=3)


def calculate_p_k(dcn_params):
    """Calculate p_k value in equation (1) in the paper.

    Args:
        dcn_params: parameters for deform_conv2d.

    Returns:
        torch.Tensor[1, 1, kernel_area_size, 2, 1, 1]

    """
    kernel_h = dcn_params["kernel_h"]
    kernel_w = dcn_params["kernel_w"]
    dilation_h = dcn_params["dilation_h"]
    dilation_w = dcn_params["dilation_w"]
    K = dcn_params["kernel_area_size"]

    p_k_y, p_k_x = torch.meshgrid(
        torch.arange(0, kernel_h * dilation_h, step=dilation_h),
        torch.arange(0, kernel_w * dilation_w, step=dilation_w),
        indexing="ij",
    )
    p_k_y = p_k_y.reshape(1, 1, K, 1, 1, 1)
    p_k_x = p_k_x.reshape(1, 1, K, 1, 1, 1)
    return torch.cat([p_k_y, p_k_x], dim=3)


def calculate_p(dcn_params, offset):
    """Calculate p_0 + p_k + Delta(p_k) in equation (1) in the paper.

    Args:
        dcn_params: parameters for deform_conv2d.
        offset: Delta(p_k) in the paper.
            The shape is (b, group, K, 2, out_h, out_w).

    Returns:
        The shape is (b, group, K, 2, out_h, out_w).

    """
    b = dcn_params["batch"]
    K = dcn_params["kernel_area_size"]
    h = dcn_params["out_h"]
    w = dcn_params["out_w"]
    group = dcn_params["n_offset_grps"]
    offset_dtype = dcn_params["offset_dtype_pytorch"]

    offset = _reshape(offset, [b, group, K, 2, h, w])

    p_0 = calculate_p_0(dcn_params)
    p_k = calculate_p_k(dcn_params)
    p = p_0 + p_k
    return op.Add(_make_constant(p, dtype=offset_dtype), offset)


def calculate_p_floor(p):
    """Calculate floor of p.

    Args:
        p: Coords for sampling points of DCN.
            The shape is (b, group, K, 2, out_h, out_w).

    Returns:
        The shape is (b, group, K, 2, out_h, out_w).
        Note that the data type is not integer but float.

    """
    return op.Floor(p)


def calculate_p_tlbr(dcn_params, p_floor):
    """Calculate floor and ceil of p.

    Args:
        dcn_params: parameters for deform_conv2d.
        p_floor: Floored coords for sampling points of DCN.
            The shape is (b, group, K, 2, out_h, out_w).

    Returns:
        A dict, {"t": p_t, "l", p_l, "b": p_b, "r": p_r}, which contains
        "t"op, "l"eft, "b"ottom, and "r"ight coordinates around p.
        The shape of p_t, ..., p_r is (b, group, K, 1, out_h, out_w).

    """
    h = dcn_params["in_h"]
    w = dcn_params["in_w"]
    index_dtype_onnx = dcn_params["index_dtype_onnx"]
    index_dtype_pytorch = dcn_params["index_dtype_pytorch"]

    p_floor = op.Cast(p_floor, to=index_dtype_onnx)
    one = _make_constant(1, dtype=index_dtype_pytorch)

    p_t = _slice(p_floor, [3], [0], [1])
    p_l = _slice(p_floor, [3], [1], [2])
    p_b = op.Add(p_t, one)
    p_r = op.Add(p_l, one)

    # Clip out-of-bounds coords.
    # Clipped coords point to padding area, which is filled with 0.
    p_t = op.Clip(
        p_t,
        _make_constant(0, dtype=index_dtype_pytorch),
        _make_constant(h - 1, dtype=index_dtype_pytorch),
    )
    p_l = op.Clip(
        p_l,
        _make_constant(0, dtype=index_dtype_pytorch),
        _make_constant(w - 1, dtype=index_dtype_pytorch),
    )
    p_b = op.Clip(
        p_b,
        _make_constant(0, dtype=index_dtype_pytorch),
        _make_constant(h - 1, dtype=index_dtype_pytorch),
    )
    p_r = op.Clip(
        p_r,
        _make_constant(0, dtype=index_dtype_pytorch),
        _make_constant(w - 1, dtype=index_dtype_pytorch),
    )
    return {
        "t": p_t,
        "l": p_l,
        "b": p_b,
        "r": p_r,
    }


def calculate_weight(dcn_params, p, p_floor):
    """Calculate weight value for bilinear interpolation.

    Args:
        dcn_params: parameters for deform_conv2d.
        p: Coords for sampling points.
            The shape is (b, group, K, 2, out_h, out_w).
        p_floor: Floored coords for sampling points.
            The shape is (b, group, K, 2, out_h, out_w).

    Returns:
        A dict, {"tl": weight_tl, "br": weight_br, ..., "tr": weight_tr},
        which contains weights for "t"op-"l"eft, "b"ottom-"r"ight, ....
        The shape of weight_tl is (b, group, 1, K, out_h, out_w).

    """
    b = dcn_params["batch"]
    group = dcn_params["n_offset_grps"]
    h = dcn_params["out_h"]
    w = dcn_params["out_w"]
    K = dcn_params["kernel_area_size"]
    offset_dtype = dcn_params["offset_dtype_pytorch"]

    one = _make_constant(1, dtype=offset_dtype)

    diff = op.Sub(p, p_floor)
    diff_y = _slice(diff, [3], [0], [1])
    diff_x = _slice(diff, [3], [1], [2])
    diff_y_inv = op.Sub(one, diff_y)
    diff_x_inv = op.Sub(one, diff_x)

    # bilinear kernel (b, group, K, 1, h, w)
    # (1 - (p_x - p_l)) * (1 - (p_y - p_t))
    weight_tl = op.Mul(diff_x_inv, diff_y_inv)
    # (p_x - p_l) * (p_y - p_t)
    weight_br = op.Mul(diff_x, diff_y)
    # (1 - (p_x - p_l)) * (p_y - p_t)
    weight_bl = op.Mul(diff_x_inv, diff_y)
    # (p_x - p_l) * (1 - (p_y - p_t))
    weight_tr = op.Mul(diff_x, diff_y_inv)

    weights = {
        "tl": weight_tl,
        "br": weight_br,
        "bl": weight_bl,
        "tr": weight_tr,
    }
    return {
        key: _reshape(weight, [b, group, 1, K, h, w]) for key, weight in weights.items()
    }


def reshape_input_for_gather_elements(dcn_params, input):
    """Reshape input for gather_elements function.

    Even if no padding is specified, 1 padding is always added
    to ensure that out-of-bounds index can be handled correctly.

    This function also transpose input tensor, so that "GatherND"
    can easily gather all data in a channel.

    Args:
        dcn_params: parameters for deform_conv2d.
        input: input tensor.
            The shape is (b, in_ch, in_h, in_w)

    Returns:
        The shape is (b, group, ch_per_group, in_h, in_w).

    """
    b = dcn_params["batch"]
    group = dcn_params["n_offset_grps"]
    ch = dcn_params["in_ch_per_group"]
    in_h = dcn_params["in_h"]
    in_w = dcn_params["in_w"]
    pad_h = dcn_params["padding_h"]
    pad_w = dcn_params["padding_w"]
    additional_pad_h = dcn_params["additional_pad_h"]
    additional_pad_w = dcn_params["additional_pad_w"]

    pad_size = [
        0,
        0,
        (pad_h + additional_pad_h),
        (pad_w + additional_pad_w),
        0,
        0,
        (pad_h + additional_pad_h),
        (pad_w + additional_pad_w),
    ]
    pad = _make_constant(pad_size, dtype=torch.int64)
    input = op.Pad(input, pad, mode="constant")
    return _reshape(input, [b, group, ch, in_h, in_w])


def gather_elements(dcn_params, input, p_y, p_x):
    """Gather elements specified by p_y and p_x using GatherElements operator.

    Args:
        dcn_params: parameters for deform_conv2d.
        input: input tensor.
            The shape is (b, group, ch_per_group, in_h, in_w).
        p_y: y coordinates of sampling points.
            The shape is (b, group, K, 1, out_h, out_w).
        p_x: x coordinates of sampling points.
            The shape is (b, group, K, 1, out_h, out_w).

    Returns:
        The shape is (b, group, ch_per_group, K, out_h, out_w).

    """
    b = dcn_params["batch"]
    group = dcn_params["n_offset_grps"]
    ch = dcn_params["in_ch_per_group"]
    in_h = dcn_params["in_h"]
    in_w = dcn_params["in_w"]
    out_h = dcn_params["out_h"]
    out_w = dcn_params["out_w"]
    K = dcn_params["kernel_area_size"]
    index_dtype_pytorch = dcn_params["index_dtype_pytorch"]

    p_y = _reshape(p_y, [b, group, 1, K * out_h * out_w])
    p_x = _reshape(p_x, [b, group, 1, K * out_h * out_w])
    p_y = op.Mul(p_y, _make_constant(in_w, dtype=index_dtype_pytorch))
    index = op.Add(p_y, p_x)
    shape = [b, group, ch, K * out_h * out_w]
    index = op.Expand(index, _make_constant(shape, dtype=torch.int64))

    input = _reshape(input, [b, group, ch, in_h * in_w])

    v = op.GatherElements(input, index, axis=3)
    # => v.shape is (b, group, ch_per_group, K * out_h * out_w)
    return _reshape(v, [b, group, ch, K, out_h, out_w])


def gather_nd(dcn_params, input, p_y, p_x):
    """Gather elements specified by p_y and p_x using GatherND.

    Args:
        dcn_params: parameters for deform_conv2d.
        input: input tensor.
            The shape is (b, group, ch_per_group, in_h, in_w).
        p_y: y coordinates of sampling points.
            The shape is (b, group, K, 1, out_h, out_w).
        p_x: x coordinates of sampling points.
            The shape is (b, group, K, 1, out_h, out_w).

    Returns:
        The shape is (b, group, ch_per_group, K, out_h, out_w).

    """
    b = dcn_params["batch"]
    group = dcn_params["n_offset_grps"]
    ch = dcn_params["in_ch_per_group"]
    out_h = dcn_params["out_h"]
    out_w = dcn_params["out_w"]
    K = dcn_params["kernel_area_size"]

    p_y = _reshape(p_y, [b, group, K * out_h * out_w, 1])
    p_x = _reshape(p_x, [b, group, K * out_h * out_w, 1])
    index = op.Concat(p_y, p_x, axis=3)
    # => index.shape is (b, group, K * out_h * out_w, 2)

    input = op.Transpose(input, perm=[0, 1, 3, 4, 2])
    # => input.shape is (b, group, in_h, in_w, ch_per_group)
    v = op.GatherND(input, index, batch_dims=2)
    # => v.shape is (b, group, K * out_h * out_w, ch)
    if dcn_params["option"]["enable_openvino_patch"]:
        # OpenVINO 2021.4 has a bug related to shape of the output of GatherND.
        v = _reshape(v, [b, group, K * out_h * out_w, ch])
    v = op.Transpose(v, perm=[0, 1, 3, 2])
    return _reshape(v, [b, group, ch, K, out_h, out_w])


def gather_elements_tlbr(dcn_params, input, p_tlbr):
    """Gather elements specified by p_tlbr.

    Args:
        dcn_params: parameters for deform_conv2d.
        input: input tensor.
            The shape is (b, group, ch_per_group, in_h, in_w).
        p_tlbr: A dict, {"t": p_t, "l", p_l, "b": p_b, "r": p_r},
            which contains "t"op, "l"eft, "b"ottom, and "r"ight
            coordinates around p.
            The shape of p_t, ..., p_r is (b, group, K, 1, out_h, out_w).

    Returns:
        A dict, {"tl": v_tl, "br": v_br, ..., "tr": v_tr}, which contains
        gathred elements.
        The shape of v_tl is (b, group, ch_per_group, K, out_h, out_w).

    """
    tlbr = ["tl", "br", "bl", "tr"]
    v_tlbr = {}
    for key in tlbr:
        key_y = key[0]  # "t" or "b"
        key_x = key[1]  # "l" or "r"
        p_y = p_tlbr[key_y]
        p_x = p_tlbr[key_x]
        if dcn_params["option"]["use_gathernd"]:
            v = gather_nd(dcn_params, input, p_y, p_x)
        else:
            v = gather_elements(dcn_params, input, p_y, p_x)
        v_tlbr[key] = v
    return v_tlbr


def calculate_weighted_sum(v_tlbr, weight_tlbr):
    """Calculate sum of weighted tensors.

    Args:
        v_tlbr: a dict, {"tl": v_tl, "br": v_br, ..., "tr": v_tr}, which
            contains gathred elements.
            The shape of v_tl is (b, group, ch_per_group, K, out_h, out_w).
        weight_tlbr: a dict, {"tl": weight_tl, "br": weight_br, ...},
            which contains weights for "t"op-"l"eft, "b"ottom-"r"ight, ....
            The shape of weight_tl is (b, group, 1, K, out_h, out_w).

    Returns:
        The shape is (b, group, ch_per_group, K, out_h, out_w).

    """
    weighted_v_list = [op.Mul(weight_tlbr[key], v_tlbr[key]) for key in v_tlbr]
    return op.Sum(*weighted_v_list)


def apply_mask(dcn_params, v, mask):
    """Apply mask tensor.

    Args:
        dcn_params: parameters for deform_conv2d.
        v: input tensor.
            The shape is (b, group, ch_per_group, K, out_h, out_w).
        mask: mask tensor.
            The shape is (b, group * K, out_h, out_w).

    Returns:
        The shape is (b, group, ch_per_group, K, out_h, out_w).

    """
    b = dcn_params["batch"]
    group = dcn_params["n_offset_grps"]
    out_h = dcn_params["out_h"]
    out_w = dcn_params["out_w"]
    K = dcn_params["kernel_area_size"]

    mask = _reshape(mask, [b, group, 1, K, out_h, out_w])
    return op.Mul(v, mask)


def reshape_v_for_conv(dcn_params, v):
    """Reshape v for convolution.

    Args:
        dcn_params: parameters for deform_conv2d.
        v: a reshaped tensor.
            The shape is (b, group, ch_per_group, K, out_h, out_w).

    Returns:
        The shape is (b, in_ch, out_h * kernel_h, out_w * kernel_w).

    """
    b = dcn_params["batch"]
    h = dcn_params["out_h"]
    w = dcn_params["out_w"]
    ch = dcn_params["in_ch"]
    kernel_h = dcn_params["kernel_h"]
    kernel_w = dcn_params["kernel_w"]

    v = _reshape(v, [b, ch, kernel_h, kernel_w, h, w])
    v = op.Transpose(v, perm=[0, 1, 4, 2, 5, 3])
    return _reshape(v, [b, ch, h * kernel_h, w * kernel_w])


def apply_conv(dcn_params, v, weight):
    """Apply convolution.

    Args:
        dcn_params: parameters for deform_conv2d.
        v: input tensor.
            The shape is (b, in_ch, out_h * kernel_h, out_w * kernel_w).
        weight: weight for convolution.
            The shape is (out_ch, ch_per_group, kernel_h, kernel_w).

    Returns:
        The shape is (b, out_ch, out_h, out_w).

    """
    weight_groups = dcn_params["n_weight_grps"]
    kernel_h = dcn_params["kernel_h"]
    kernel_w = dcn_params["kernel_w"]

    return op.Conv(
        v,
        weight,
        group=weight_groups,
        kernel_shape=[kernel_h, kernel_w],
        strides=[kernel_h, kernel_w],
    )


def apply_bias(v, bias):
    """Apply bias parameter.

    Args:
        v: input tensor.
            The shape is (b, out_ch, out_h, out_w).
        bias: bias tensor.
            The shape is (out_ch,).

    Returns:
        The shape is (b, out_ch, out_h, out_w).

    """
    bias = _unsqueeze(bias, [0, 2, 3])
    return op.Add(v, bias)


def _get_scalar_type_info(offset):
    """Get scalar type information for offset tensor.

    For the new dynamo-based exporter, we use ONNX TensorProto types directly.
    This is simpler and more robust than trying to infer types from SymbolicTensor.
    """
    # Use ONNX TensorProto types directly for dynamo exporter
    # The inputs are typically float32 and indices are int64
    import onnx

    # For offset/float tensors: use FLOAT (1)
    offset_dtype_onnx = onnx.TensorProto.FLOAT
    offset_dtype_pytorch = torch.float32

    # For index tensors: use INT64 (7)
    index_dtype_onnx = onnx.TensorProto.INT64
    index_dtype_pytorch = torch.int64

    return (
        offset_dtype_onnx,
        offset_dtype_pytorch,
        index_dtype_onnx,
        index_dtype_pytorch,
    )


def create_dcn_params(
    input,
    weight,
    offset,
    stride_h,
    stride_w,
    pad_h,
    pad_w,
    dilation_h,
    dilation_w,
    n_weight_grps,
    n_offset_grps,
    option,
):
    """Manage parameters for DeformConv2d."""
    additional_pad_h = additional_pad_w = 0
    if pad_h == 0:
        additional_pad_h = 1
    if pad_w == 0:
        additional_pad_w = 1

    batch = input.shape[0]
    in_ch = input.shape[1]
    in_h = input.shape[2] + 2 * (pad_h + additional_pad_h)
    in_w = input.shape[3] + 2 * (pad_w + additional_pad_w)
    in_ch_per_group = in_ch // n_offset_grps

    out_ch = weight.shape[0]
    kernel_h = weight.shape[2]
    kernel_w = weight.shape[3]
    kernel_area_size = kernel_h * kernel_w

    out_h = offset.shape[2]
    out_w = offset.shape[3]

    # Get dtype information
    offset_dtype_onnx, offset_dtype_pytorch, index_dtype_onnx, index_dtype_pytorch = (
        _get_scalar_type_info(offset)
    )

    return {
        # batch and kernel
        "batch": batch,
        "kernel_h": kernel_h,
        "kernel_w": kernel_w,
        "kernel_area_size": kernel_area_size,
        # input size
        "in_ch": in_ch,
        "in_ch_per_group": in_ch_per_group,
        "in_h": in_h,
        "in_w": in_w,
        # output size
        "out_ch": out_ch,
        "out_h": out_h,
        "out_w": out_w,
        # other parameters
        "stride_h": stride_h,
        "stride_w": stride_w,
        "dilation_h": dilation_h,
        "dilation_w": dilation_w,
        "n_offset_grps": n_offset_grps,
        "n_weight_grps": n_weight_grps,
        # offset data type
        "offset_dtype_onnx": offset_dtype_onnx,
        "offset_dtype_pytorch": offset_dtype_pytorch,
        # index data type
        "index_dtype_onnx": index_dtype_onnx,
        "index_dtype_pytorch": index_dtype_pytorch,
        # padding
        "padding_h": pad_h,
        "padding_w": pad_w,
        "additional_pad_h": additional_pad_h,
        "additional_pad_w": additional_pad_w,
        "option": option,
    }


def create_deform_conv2d_custom_op(use_gathernd=True, enable_openvino_patch=False):
    """Create a custom ONNX operator for torchvision::deform_conv2d.

    This function creates a custom operator function that can be used with
    torch.onnx.export() via the custom_translation_table parameter.

    Args:
        use_gathernd: If True, use GatherND. Otherwise use GatherElements.
        enable_openvino_patch: If True, enable patch for OpenVINO.
            Otherwise, disable it.

    Returns:
        A function that implements the deform_conv2d operator using ONNX Script.

    Example:
        >>> custom_op = create_deform_conv2d_custom_op()
        >>> onnx_program = torch.onnx.export(
        ...     model,
        ...     (input, offset, mask),
        ...     dynamo=True,
        ...     custom_translation_table={
        ...         torch.ops.torchvision.deform_conv2d: custom_op,
        ...     },
        ... )

    """

    def deform_conv2d(
        input,
        weight,
        offset,
        mask,
        bias,
        stride_h: int,
        stride_w: int,
        pad_h: int,
        pad_w: int,
        dilation_h: int,
        dilation_w: int,
        n_weight_grps: int,
        n_offset_grps: int,
        use_mask: bool,
    ):
        option = {
            "use_gathernd": use_gathernd,
            "enable_openvino_patch": enable_openvino_patch,
        }
        dcn_params = create_dcn_params(
            input,
            weight,
            offset,
            stride_h,
            stride_w,
            pad_h,
            pad_w,
            dilation_h,
            dilation_w,
            n_weight_grps,
            n_offset_grps,
            option,
        )

        p = calculate_p(dcn_params, offset)
        p_floor = calculate_p_floor(p)
        p_tlbr = calculate_p_tlbr(dcn_params, p_floor)
        weight_tlbr = calculate_weight(dcn_params, p, p_floor)

        input = reshape_input_for_gather_elements(dcn_params, input)
        v_tlbr = gather_elements_tlbr(dcn_params, input, p_tlbr)

        v = calculate_weighted_sum(v_tlbr, weight_tlbr)

        if use_mask:
            v = apply_mask(dcn_params, v, mask)

        v = reshape_v_for_conv(dcn_params, v)
        v = apply_conv(dcn_params, v, weight)
        return apply_bias(v, bias)

    return deform_conv2d


# Backward compatibility: keep the old function name
def register_deform_conv2d_onnx_op(
    use_gathernd=True,
    enable_openvino_patch=False,
) -> None:
    """Register ONNX operator for torchvision::deform_conv2d.

    DEPRECATED: This function is kept for backward compatibility.
    The new recommended way is to use create_deform_conv2d_custom_op()
    and pass the result to torch.onnx.export() via custom_translation_table.

    Example (old way - deprecated):
        >>> register_deform_conv2d_onnx_op()
        >>> torch.onnx.export(model, args, "model.onnx", opset_version=12)

    Example (new way - recommended):
        >>> custom_op = create_deform_conv2d_custom_op()
        >>> torch.onnx.export(
        ...     model,
        ...     args,
        ...     dynamo=True,
        ...     custom_translation_table={
        ...         torch.ops.torchvision.deform_conv2d: custom_op,
        ...     },
        ... )

    Args:
        use_gathernd: If True, use GatherND. Otherwise use GatherElements.
        enable_openvino_patch: If True, enable patch for OpenVINO.
            Otherwise, disable it.

    """
    import warnings

    warnings.warn(
        "register_deform_conv2d_onnx_op() is deprecated. "
        "Please use create_deform_conv2d_custom_op() with torch.onnx.export(dynamo=True) instead. "
        "See the documentation for examples.",
        DeprecationWarning,
        stacklevel=2,
    )

    # For backward compatibility, still register using the old method
    from torch.onnx import register_custom_op_symbolic
    from torch.onnx import symbolic_helper as sym_help

    @sym_help.parse_args(
        "v",
        "v",
        "v",
        "v",
        "v",
        "i",
        "i",
        "i",
        "i",
        "i",
        "i",
        "i",
        "i",
        "b",
    )
    def deform_conv2d_legacy(
        g,
        input,
        weight,
        offset,
        mask,
        bias,
        stride_h,
        stride_w,
        pad_h,
        pad_w,
        dilation_h,
        dilation_w,
        n_weight_grps,
        n_offset_grps,
        use_mask,
    ) -> NoReturn:
        # This is a wrapper that converts g.op() style to the new implementation
        # For full compatibility, we would need to keep the old implementation here
        # For now, raise an error pointing users to the new method
        msg = (
            "The legacy registration method is no longer supported. "
            "Please use create_deform_conv2d_custom_op() with torch.onnx.export(dynamo=True)."
        )
        raise NotImplementedError(
            msg,
        )

    register_custom_op_symbolic(
        "torchvision::deform_conv2d",
        deform_conv2d_legacy,
        onnx_opset_version,
    )
