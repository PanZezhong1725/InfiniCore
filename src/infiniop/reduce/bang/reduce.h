#include "bang.h"
// 注意__mlu_func__修饰的函数默认是内联函数
template <typename T>
__mlu_func__ void ComputeInternal(T *tensor_nram, T *temp_nram, int num)
{
    int length = 128 / sizeof(T);
    __bang_sumpool(
        temp_nram,
        tensor_nram,
        /*channel*/ length,
        /*height*/ 1,
        /*width*/ num / length,
        /*kernel_height*/ 1,
        /*kernel_width*/ num / length,
        /*stride_height*/ 1,
        /*stride_width*/ 1);
    __bang_reduce_sum(temp_nram, temp_nram, length); // 这个函数把长度为num的向量做了一个norm求和
} // temp_nram是一个字节数为128的数组

__mlu_func__ void ComputeSum(float *tensor_nram, float *temp_nram, int num)
{
    ComputeInternal<float>(tensor_nram, temp_nram, num);
}

__mlu_func__ void ComputeSum(half *tensor_nram, float *temp_nram, int num)
{
    __bang_half2float(
        (float *)tensor_nram, tensor_nram + num, num); // tensor_nram[2 * num]
    ComputeInternal<float>((float *)tensor_nram, temp_nram, num);
}

__mlu_func__ void ComputeSum(bfloat16_t *tensor_nram, float *temp_nram,
                             int num)
{
    __bang_bfloat162float((float *)tensor_nram, tensor_nram + num, num); // tensor_nram[2 * num]
    ComputeInternal<float>((float *)tensor_nram, temp_nram, num);
}
