#include "rms_norm_cpu.h"
#include "../../../devices/cpu/common_cpu.h"
#include "../../utils.h"
#include <cmath>

infiniopStatus_t cpuCreateRMSNormDescriptor(infiniopHandle_t, infiniopRMSNormCpuDescriptor_t *desc_ptr,
                                            infiniopTensorDescriptor_t y_desc, infiniopTensorDescriptor_t x_desc, infiniopTensorDescriptor_t w_desc, float epsilon) {
    if (y_desc->ndim != 2 || x_desc->ndim != 2 || w_desc->ndim != 1) {
        return INFINIOP_STATUS_BAD_TENSOR_SHAPE;
    }

    auto n = y_desc->shape[0],
         d = y_desc->shape[1];

    if (x_desc->shape[0] != n || x_desc->shape[1] != d || w_desc->shape[0] != d) {
        return INFINIOP_STATUS_BAD_TENSOR_SHAPE;
    }

    uint64_t stride_y = y_desc->strides[0];
    uint64_t stride_x = y_desc->strides[0];
    auto w_datatype = w_desc->dtype;

    *desc_ptr = new InfiniRMSNormCpuDescriptor{
        INFINI_DEVICE_CPU,
        y_desc->dtype,
        n,
        d,
        stride_y,
        stride_x,
        w_datatype,
        epsilon};

    return INFINIOP_STATUS_SUCCESS;
}

infiniopStatus_t cpuGetRMSNormWorkspaceSize(infiniopRMSNormCpuDescriptor_t desc, uint64_t *size) {
    *size = 0;
    return INFINIOP_STATUS_SUCCESS;
}

infiniopStatus_t cpuDestroyRMSNormDescriptor(infiniopRMSNormCpuDescriptor_t desc) {
    delete desc;
    return INFINIOP_STATUS_SUCCESS;
}

void rms_norm_cpu_f16(infiniopRMSNormCpuDescriptor_t desc, void *y, void const *x, void const *w) {
    auto n = desc->n, d = desc->d;
    auto stride_y = desc->stride_y;
    auto stride_x = desc->stride_x;
    auto epsilon = desc->epsilon;

    auto y_ptr = reinterpret_cast<uint16_t *>(y);
    auto x_ptr = reinterpret_cast<uint16_t const *>(x);
    void const *w_ptr = w;
    void const *w_ = nullptr;
    auto w_datatype = desc->w_datatype;
    if (w_datatype == INFINI_DTYPE_F16) {
        w_ = reinterpret_cast<uint16_t const *>(w_ptr);
    } else {
        w_ = reinterpret_cast<float const *>(w_ptr);
    }

    for (size_t i = 0; i < n; ++i) {
        auto y_ = reinterpret_cast<uint16_t *>(y_ptr + i * stride_y);
        auto x_ = reinterpret_cast<uint16_t const *>(x_ptr + i * stride_x);

        auto sum_sq = 0.0f;
        for (size_t j = 0; j < d; ++j) {
            auto x__ = f16_to_f32(x_[j]);
            sum_sq += x__ * x__;
        }

        auto k = std::pow(sum_sq / d + epsilon, -.5);
        for (size_t j = 0; j < d; ++j) {
            auto x__ = f16_to_f32(x_[j]);
            float w__ = 0.0f;
            if (w_datatype == INFINI_DTYPE_F16) {
                w__ = f16_to_f32(static_cast<uint16_t const *>(w_)[j]);
            } else {
                w__ = static_cast<float const *>(w_)[j];
            }

            y_[j] = f32_to_f16(k * x__ * w__);
        }
    }
}

infiniopStatus_t cpuRMSNorm(infiniopRMSNormCpuDescriptor_t desc,
                            void *workspace,
                            uint64_t workspace_size,
                            void *y, void const *x, void const *w,
                            void *stream) {
    if (desc->dtype == INFINI_DTYPE_F16) {
        rms_norm_cpu_f16(desc, y, x, w);
        return INFINIOP_STATUS_SUCCESS;
    }

    return INFINIOP_STATUS_BAD_TENSOR_DTYPE;
}
