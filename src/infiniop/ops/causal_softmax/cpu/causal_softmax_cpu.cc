#include "causal_softmax_cpu.h"
#include "../../../devices/cpu/common_cpu.h"
#include "../../../reduce/cpu/reduce.h"

namespace op::causal_softmax::cpu {
Descriptor::~Descriptor() {}

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t y_desc) {
    CausalSoftmaxInfo info;
    CHECK_STATUS(createCausalSoftmaxInfo(&info, y_desc));
    *desc_ptr = new Descriptor(nullptr, info, 0, handle->device, handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t causal_softmaxF16(const CausalSoftmaxInfo *info, fp16_t *data) {
#pragma omp parallel for
    for (size_t index = 0; index < info->batch_size * info->seq_len; index++) {
        size_t ind = index;
        size_t offset = 0;
        size_t i = (ind % info->seq_len);
        offset += (ind % info->seq_len) * info->stride_i;
        ind /= info->seq_len;
        offset += (ind % info->batch_size) * info->stride_b;
        float val = utils::cast<float>(data[offset]);
        for (size_t j = 1; j < info->total_seq_len; j++) {
            if (j <= info->total_seq_len - info->seq_len + i) {
                val = std::max(val, utils::cast<float>(data[offset + j * info->stride_j]));
            } else {
                data[offset + j * info->stride_j] = utils::cast<fp16_t>(0.0f);
            }
        }
        for (size_t j = 0; j <= info->total_seq_len - info->seq_len + i; j++) {
            data[offset + j * info->stride_j] = utils::cast<fp16_t>(std::exp(utils::cast<float>(data[offset + j * info->stride_j]) - val));
        }
        float sum = op::common_cpu::reduce_op::sum(&data[offset], info->total_seq_len - info->seq_len + i + 1, info->stride_j);
        for (size_t j = 0; j <= info->total_seq_len - info->seq_len + i; j++) {
            data[offset + j * info->stride_j] = utils::cast<fp16_t>(utils::cast<float>(data[offset + j * info->stride_j]) / sum);
        }
    }

    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t causal_softmaxF32(const CausalSoftmaxInfo *info, float *data) {
#pragma omp parallel for
    for (size_t index = 0; index < info->batch_size * info->seq_len; index++) {
        size_t ind = index;
        size_t offset = 0;
        size_t i = (ind % info->seq_len);
        offset += (ind % info->seq_len) * info->stride_i;
        ind /= info->seq_len;
        offset += (ind % info->batch_size) * info->stride_b;
        float val = data[offset];
        for (size_t j = 1; j < info->total_seq_len; j++) {
            if (j <= info->total_seq_len - info->seq_len + i) {
                val = std::max(val, utils::cast<float>(data[offset + j * info->stride_j]));
            } else {
                data[offset + j * info->stride_j] = 0.0f;
            }
        }
        for (size_t j = 0; j <= info->total_seq_len - info->seq_len + i; j++) {
            data[offset + j * info->stride_j] = std::exp(data[offset + j * info->stride_j] - val);
        }
        float sum = op::common_cpu::reduce_op::sum(&data[offset], info->total_seq_len - info->seq_len + i + 1, info->stride_j);
        for (size_t j = 0; j <= info->total_seq_len - info->seq_len + i; j++) {
            data[offset + j * info->stride_j] = data[offset + j * info->stride_j] / sum;
        }
    }

    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(void *workspace, size_t workspace_size,
                                     void *data,
                                     void *stream) {
    if (_info.dtype == INFINI_DTYPE_F16) {
        CHECK_STATUS(causal_softmaxF16(&_info, (fp16_t *)data));
    } else if (_info.dtype == INFINI_DTYPE_F32) {
        CHECK_STATUS(causal_softmaxF32(&_info, (float *)data));
    } else {
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    return INFINI_STATUS_SUCCESS;
}
} // namespace op::causal_softmax::cpu
