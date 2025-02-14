#ifndef ___INFINIOP_RMS_NORM_CUDA_H__
#define ___INFINIOP_RMS_NORM_CUDA_H__

#include "rms_norm_cuda_api.h"
#include "../../../devices/cuda/common_cuda.cuh"

struct InfiniopRMSNormCudaDescriptor {
    infiniDevice_t device;
    int device_id;
    infiniDtype_t dtype;
    uint64_t n;
    uint64_t d;
    int64_t stride_y;
    int64_t stride_x;
    infiniDtype_t w_datatype;
    float epsilon;
};

#endif// ___INFINIOP_RMS_NORM_CUDA_H__