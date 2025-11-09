import io
import math
import random
import unittest

import numpy as np
import onnx
import onnx.checker
import onnxruntime
import torch
import torch.onnx
from torchvision.ops.deform_conv import DeformConv2d

import deform_conv2d_onnx_exporter


def tonumpy(tensor):
    return tensor.to("cpu").detach().numpy().copy()


class DeformConv2dOpTestCase(unittest.TestCase):
    OPSET_VERSION = 18  # Updated from 12 to 18

    def create_input_params(self, dcn_params):
        size = [
            dcn_params["batch"],
            dcn_params["input_ch"],
            dcn_params["input_h"],
            dcn_params["input_w"],
        ]
        input = torch.rand(size, dtype=torch.float)

        size = [
            dcn_params["batch"],
            2
            * dcn_params["n_offset_grps"]
            * dcn_params["kernel_h"]
            * dcn_params["kernel_w"],
            dcn_params["output_h"],
            dcn_params["output_w"],
        ]
        offset = torch.randn(size, dtype=torch.float) * dcn_params["kernel_w"]

        size = [
            dcn_params["batch"],
            dcn_params["n_offset_grps"]
            * dcn_params["kernel_h"]
            * dcn_params["kernel_w"],
            dcn_params["output_h"],
            dcn_params["output_w"],
        ]
        mask = torch.rand(size, dtype=torch.float)

        return input, offset, mask

    def create_pytorch_model(self, dcn_params):
        return DeformConv2d(
            in_channels=dcn_params["input_ch"],
            out_channels=dcn_params["output_ch"],
            kernel_size=(dcn_params["kernel_h"], dcn_params["kernel_w"]),
            stride=(dcn_params["stride_h"], dcn_params["stride_w"]),
            padding=(dcn_params["padding_h"], dcn_params["padding_w"]),
            dilation=(dcn_params["dilation_h"], dcn_params["dilation_w"]),
            groups=dcn_params["groups"],
            bias=dcn_params["bias"],
        )

    def check_onnx_model(self, model_data) -> None:
        try:
            onnx.checker.check_model(model_data, full_check=True)
            assert "Valid ONNX model"
        except onnx.checker.ValidationError as e:
            self.fail(f"Invalid ONNX model: {e}")
        except Exception as e:
            self.fail(f"Unknown exception: {e}")

    def convert_to_onnx_model(
        self,
        pytorch_model,
        input,
        offset,
        mask=None,
        use_gathernd=True,
        enable_openvino_patch=False,
    ):
        """Convert PyTorch model to ONNX using the new dynamo-based exporter.

        Args:
            pytorch_model: PyTorch model to convert
            input: Input tensor
            offset: Offset tensor
            mask: Optional mask tensor
            use_gathernd: If True, use GatherND. Otherwise use GatherElements.
            enable_openvino_patch: If True, enable patch for OpenVINO.

        Returns:
            ONNX Runtime InferenceSession

        """
        input_params = (input, offset, mask) if mask is not None else (input, offset)

        # Create custom operator
        custom_op = deform_conv2d_onnx_exporter.create_deform_conv2d_custom_op(
            use_gathernd=use_gathernd,
            enable_openvino_patch=enable_openvino_patch,
        )

        # Export using the new dynamo-based exporter
        onnx_program = torch.onnx.export(
            pytorch_model,
            input_params,
            dynamo=True,
            verbose=False,  # Disable verbose to avoid Unicode encoding issues on Windows
            custom_translation_table={
                torch.ops.torchvision.deform_conv2d.default: custom_op,
            },
        )

        # Get ONNX model bytes
        buffer = io.BytesIO()
        onnx_program.save(buffer)
        onnx_model_data = buffer.getvalue()

        self.check_onnx_model(onnx_model_data)

        return onnxruntime.InferenceSession(onnx_model_data)

    def run_pytorch_model(self, model, input, offset, mask=None):
        model.eval()
        return model(input, offset, mask)

    def run_onnx_model(self, model, input, offset, mask=None):
        # Get actual output names from the model
        output_names = [out.name for out in model.get_outputs()]

        input_params = {
            "input": tonumpy(input),
            "offset": tonumpy(offset),
        }
        if mask is not None:
            input_params["mask"] = tonumpy(mask)
        return model.run(output_names, input_params)[0]

    def run_with_dcn_params(
        self, dcn_params, message="", use_gathernd=True, enable_openvino_patch=False,
    ):
        input, offset, mask = self.create_input_params(dcn_params)
        pytorch_model = self.create_pytorch_model(dcn_params)
        if not dcn_params["use_mask"]:
            mask = None
        pytorch_output = self.run_pytorch_model(pytorch_model, input, offset, mask)
        onnx_model = self.convert_to_onnx_model(
            pytorch_model,
            input,
            offset,
            mask,
            use_gathernd=use_gathernd,
            enable_openvino_patch=enable_openvino_patch,
        )
        onnx_output = self.run_onnx_model(onnx_model, input, offset, mask)

        return pytorch_output, onnx_output

    def assert_result(self, pytorch_result, onnx_result, message="") -> None:
        pytorch_result = tonumpy(pytorch_result)
        assert np.allclose(pytorch_result, onnx_result, rtol=0.001, atol=1e-05), message

    def generate_dcn_parameters(self, base_dcn_params=None):
        if base_dcn_params is None:
            base_dcn_params = {}
        dcn_params = {
            "batch": random.randrange(1, 6),
            # "input_ch": 0,
            "input_h": random.randrange(100, 201),
            "input_w": random.randrange(100, 201),
            # "output_ch": 0,
            # "output_h": 0,
            # "output_w": 0,
            "kernel_h": random.randrange(1, 8),
            "kernel_w": random.randrange(1, 8),
            "stride_h": random.randrange(1, 5),
            "stride_w": random.randrange(1, 5),
            "padding_h": random.randrange(0, 5),
            "padding_w": random.randrange(0, 5),
            "dilation_h": random.randrange(1, 4),
            "dilation_w": random.randrange(1, 4),
            "groups": random.randrange(1, 4),
            "n_offset_grps": random.randrange(1, 4),
            "bias": random.choice([True, False]),
            "use_mask": random.choice([True, False]),
        }
        dcn_params.update(base_dcn_params)

        if "input_ch" not in dcn_params:
            lcm = (
                dcn_params["groups"]
                * dcn_params["n_offset_grps"]
                // math.gcd(dcn_params["groups"], dcn_params["n_offset_grps"])
            )
            dcn_params["input_ch"] = lcm * random.randrange(1, 17)
        if "output_ch" not in dcn_params:
            dcn_params["output_ch"] = dcn_params["groups"] * random.randrange(1, 17)
        ker_h = dcn_params["dilation_h"] * (dcn_params["kernel_h"] - 1) + 1
        if "output_h" not in dcn_params:
            dcn_params["output_h"] = (
                (dcn_params["input_h"] + 2 * dcn_params["padding_h"] - ker_h)
                // dcn_params["stride_h"]
            ) + 1
        ker_w = dcn_params["dilation_w"] * (dcn_params["kernel_w"] - 1) + 1
        if "output_w" not in dcn_params:
            dcn_params["output_w"] = (
                (dcn_params["input_w"] + 2 * dcn_params["padding_w"] - ker_w)
                // dcn_params["stride_w"]
            ) + 1
        return dcn_params

    def test_no_padding(self) -> None:
        dcn_params = {"padding_h": 0, "padding_w": 2}
        dcn_params = self.generate_dcn_parameters(dcn_params)
        pytorch_result, onnx_result = self.run_with_dcn_params(dcn_params)
        self.assert_result(pytorch_result, onnx_result, f"no padding_h: {dcn_params}")

        dcn_params = {"padding_h": 1, "padding_w": 0}
        dcn_params = self.generate_dcn_parameters(dcn_params)
        pytorch_result, onnx_result = self.run_with_dcn_params(dcn_params)
        self.assert_result(pytorch_result, onnx_result, f"no padding_w: {dcn_params}")

        dcn_params = {"padding_h": 0, "padding_w": 0}
        dcn_params = self.generate_dcn_parameters(dcn_params)
        pytorch_result, onnx_result = self.run_with_dcn_params(dcn_params)
        self.assert_result(pytorch_result, onnx_result, f"no paddings: {dcn_params}")

    def test_full_parameters(self) -> None:
        dcn_params = {
            "batch": 8,
            "input_ch": 64,
            "input_h": 300,
            "input_w": 200,
            "output_w": 66,
            "kernel_h": 3,
            "kernel_w": 4,
            "stride_h": 2,
            "stride_w": 3,
            "padding_h": 0,
            "padding_w": 2,
            "dilation_h": 1,
            "dilation_w": 2,
            "groups": 2,
            "n_offset_grps": 2,
            "bias": True,
            "use_mask": True,
        }
        dcn_params = self.generate_dcn_parameters(dcn_params)
        pytorch_result, onnx_result = self.run_with_dcn_params(dcn_params)
        self.assert_result(pytorch_result, onnx_result, f"full test: {dcn_params}")

    def test_random_parameters(self) -> None:
        test_count = 10
        for _ in range(test_count):
            dcn_params = self.generate_dcn_parameters()
            pytorch_result, onnx_result = self.run_with_dcn_params(dcn_params)
            self.assert_result(
                pytorch_result,
                onnx_result,
                f"random parameters: {dcn_params}",
            )

    def test_options_for_create_deform_conv2d_custom_op(self) -> None:
        """Test different options for create_deform_conv2d_custom_op."""
        option_patterns = [
            (False, False),
            (True, False),
            (False, True),
            (True, True),
        ]
        for use_gathernd, enable_openvino_patch in option_patterns:
            dcn_params = self.generate_dcn_parameters()
            pytorch_result, onnx_result = self.run_with_dcn_params(
                dcn_params,
                use_gathernd=use_gathernd,
                enable_openvino_patch=enable_openvino_patch,
            )
            self.assert_result(
                pytorch_result,
                onnx_result,
                f"parameters: {dcn_params}, "
                f"use_gathernd={use_gathernd}, "
                f"enable_openvino_patch={enable_openvino_patch}",
            )

    def test_backward_compatibility(self) -> None:
        """Test that the old API still exists (even if deprecated)."""
        # Just check that the function exists and raises a deprecation warning
        with self.assertWarns(DeprecationWarning):
            deform_conv2d_onnx_exporter.register_deform_conv2d_onnx_op()
