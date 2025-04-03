from ast import List
import numpy as np
import gguf
from typing import Optional

from .. import InfiniopTestWriter, InfiniopTestCase, np_dtype_to_ggml, gguf_strides

def random_tensor(shape: tuple, dtype: np.dtype) -> np.ndarray:
    return np.random.uniform(-1, 1, size=shape).astype(dtype)

def rms_norm(input: np.ndarray, weight: np.ndarray, epsilon: float) -> np.ndarray:
    """
    使用numpy计算rms_norm结果
    Args:
        input:  输入张量, 维度为2, 形状为 [..., hidden_size]
        weight: 缩放权重, 形状为 [hidden_size]
        epsilon: 避免除零的小常数
    Returns:
        输出张量, 形状与 input 相同
    """
    squared = input ** 2
    mean = np.mean(squared, axis=-1, keepdims=True)
    rms = np.sqrt(mean + epsilon)
    
    normalized = input / rms
    return normalized * weight

class RMSNormTestCase(InfiniopTestCase):
    def __init__(
        self,
        input_shape: tuple,
        weight_shape: tuple,
        dtype: np.dtype,
        epsilon: float = 1e-5,
    ):
        super().__init__("rms_norm")
        self.input = random_tensor(input_shape, dtype)
        self.weight = random_tensor(weight_shape, dtype)
        self.epsilon = epsilon
        self.ans = rms_norm(self.input, self.weight, self.epsilon)

    def write_test(self, test_writer: "InfiniopTestWriter"):
        super().write_test(test_writer)
        test_writer.add_float32(test_writer.gguf_key("epsilon"), self.epsilon)
        test_writer.add_tensor(
            test_writer.gguf_key("input"),
            self.input,
            raw_dtype=np_dtype_to_ggml(self.input.dtype),
        )
        test_writer.add_tensor(
            test_writer.gguf_key("weight"),
            self.weight,
            raw_dtype=np_dtype_to_ggml(self.weight.dtype),
        )
        test_writer.add_tensor(
            test_writer.gguf_key("ans"),
            self.ans,
            raw_dtype=np_dtype_to_ggml(self.ans.dtype),
        )
        result = np.zeros_like(self.ans)
        test_writer.add_tensor(
            test_writer.gguf_key("result"),
            result,
            raw_dtype=np_dtype_to_ggml(result.dtype),
        )

if __name__ == "__main__":
    test_writer = InfiniopTestWriter("rms_norm.gguf")
    
    test_cases = [
        RMSNormTestCase(
            input_shape=(2, 256), 
            weight_shape=(256,),
            dtype=np.float32,
            epsilon=1e-5,
        ),
        RMSNormTestCase(
            input_shape=(4, 512),
            weight_shape=(512,),
            dtype=np.float32,
            epsilon=1e-6,
        ),
        RMSNormTestCase(
            input_shape=(8, 1024),
            weight_shape=(1024,),
            dtype=np.float32,
        ),
        RMSNormTestCase(
            input_shape=(1, 768),
            weight_shape=(768,),
            dtype=np.float32,
        ),
        RMSNormTestCase(
            input_shape=(8, 256), 
            weight_shape=(256,),
            dtype=np.float32,
            epsilon=1e-3,
        ),
        RMSNormTestCase(
            input_shape=(2, 256),
            weight_shape=(256,),
            dtype=np.float16,
            epsilon=1e-3,
        ),
        RMSNormTestCase(
            input_shape=(4, 512),
            weight_shape=(512,),
            dtype=np.float16,
            epsilon=1e-3,
        ),
        RMSNormTestCase(
            input_shape=(2, 256),
            weight_shape=(256,),
            dtype=np.float32,
            epsilon=1e-12,
        ),
    ]
    
    test_writer.add_tests(test_cases)
    test_writer.save()