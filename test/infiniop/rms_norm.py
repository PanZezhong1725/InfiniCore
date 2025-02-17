from ctypes import POINTER, Structure, c_int32, c_uint64, c_void_p, c_float
import ctypes
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from libinfiniop import (
    infiniopHandle_t, infiniopTensorDescriptor_t, open_lib, to_tensor, get_test_devices,
    check_error, create_workspace, test_operator, get_args,
    debug, get_tolerance, profile_operation,
)

import torch

class RMSNormDescriptor(Structure):
    _fields_ = [("device", c_int32)]


infiniopRMSNormDescriptor_t = POINTER(RMSNormDescriptor)

def rms_norm(x, w, eps):
    input_dtype = x.dtype
    hidden_states = x.to(torch.float32)
    variance = hidden_states.pow(2).mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + eps)
    return w * hidden_states.to(input_dtype)


def test(
    lib,
    handle,
    torch_device,
    y_shape,
    x_shape,
    w_shape,
    dtype=torch.float16,
    w_dtype=torch.float16
):
    print(f"Testing RMS_Norm on {torch_device} with y_shape:{y_shape} x_shape:{x_shape} w_shape:{w_shape}"
        f" dtype:{dtype} w_dtype:{w_dtype}")

    y = torch.zeros(y_shape, dtype=dtype).to(torch_device)
    x = torch.rand(x_shape, dtype=dtype).to(torch_device)
    w = torch.ones(w_shape, dtype=w_dtype).to(torch_device)

    eps = 1e-5
    ans = rms_norm(x, w, eps)

    y_tensor = to_tensor(y, lib)
    x_tensor = to_tensor(x, lib)
    w_tensor = to_tensor(w, lib)

    descriptor = infiniopRMSNormDescriptor_t()

    check_error(
        lib.infiniopCreateRMSNormDescriptor(
            handle,
            ctypes.byref(descriptor),
            y_tensor.descriptor,
            x_tensor.descriptor,
            w_tensor.descriptor,
            eps
        )
    )

    # Invalidate the shape and strides in the descriptor to prevent them from being directly used by the kernel
    x_tensor.descriptor.contents.invalidate()
    y_tensor.descriptor.contents.invalidate()
    w_tensor.descriptor.contents.invalidate()

    workspace_size = c_uint64(0)
    check_error(
        lib.infiniopGetRMSNormWorkspaceSize(
            descriptor, ctypes.byref(workspace_size)
        )
    )
    workspace = create_workspace(workspace_size.value, y.device)
    check_error(
        lib.infiniopRMSNorm(
            descriptor,
            workspace.data_ptr() if workspace is not None else None,
            workspace_size.value,
            y_tensor.data,
            x_tensor.data,
            w_tensor.data,
            None,
        )
    )

    assert torch.allclose(y.to(dtype), ans.to(dtype), atol=1e-3, rtol=1e-3)
    check_error(lib.infiniopDestroyRMSNormDescriptor(descriptor))

# _TEST_CASES = [
#     # y_shape, x_shape, w_shape, dtype, w_dtype
#     ((16, 2048), (16, 2048), (2048,), torch.float16, torch.float16),
#     ((16, 2048), (16, 2048), (2048,), torch.float16, torch.float32),
# ]

_TEST_CASES = [
    # y_shape, x_shape, w_shape, dtype, w_dtype
    ((16, 2048), (16, 2048), (2048,)),
    ((16, 2048), (16, 2048), (2048,)),
]

_TENSOR_DTYPES = [torch.float16]
# ==============================================================================
#  Main Execution
# ==============================================================================
if __name__ == "__main__":

    args = get_args()
    lib = open_lib()
    lib.infiniopCreateRMSNormDescriptor.restype = c_int32
    lib.infiniopCreateRMSNormDescriptor.argtypes = [
        infiniopHandle_t,
        POINTER(infiniopRMSNormDescriptor_t),
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
        c_float,
    ]

    lib.infiniopGetRMSNormWorkspaceSize.restype = c_int32
    lib.infiniopGetRMSNormWorkspaceSize.argtypes = [
        infiniopRMSNormDescriptor_t,
        POINTER(c_uint64),
    ]

    lib.infiniopRMSNorm.restypes = c_int32
    lib.infiniopRMSNorm.argtypes = [
        infiniopRMSNormDescriptor_t,
        c_void_p,
        c_uint64,
        c_void_p,
        c_void_p,
        c_void_p,
        c_void_p,
    ]
    lib.infiniopDestroyRMSNormDescriptor.restype = c_int32
    lib.infiniopDestroyRMSNormDescriptor.argtypes = [
        infiniopRMSNormDescriptor_t,
    ]

    # Configure testing options
    DEBUG = args.debug
    PROFILE = args.profile
    NUM_PRERUN = args.num_prerun
    NUM_ITERATIONS = args.num_iterations

    # Execute tests
    for device in get_test_devices(args):
        test_operator(lib, device, test, _TEST_CASES, _TENSOR_DTYPES)
    print("\033[92mTest passed!\033[0m")