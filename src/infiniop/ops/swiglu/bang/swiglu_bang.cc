#include "swiglu_bang.h"
#include "../../utils.h"
#include "swiglu_bang_api.h"

infiniopStatus_t bangCreateSwiGLUDescriptor(infiniopBangHandle_t handle,
                                            infiniopSwiGLUBangDescriptor_t *desc_ptr,
                                            infiniopTensorDescriptor_t c_desc,
                                            infiniopTensorDescriptor_t a_desc,
                                            infiniopTensorDescriptor_t b_desc) {
    if (c_desc->ndim != 2 || a_desc->ndim != 2 || b_desc->ndim != 2) {
        return INFINIOP_STATUS_BAD_TENSOR_SHAPE;
    }

    infiniDtype_t dtype = c_desc->dtype;

    if (dtype != INFINI_DTYPE_F16 && dtype != INFINI_DTYPE_F32) {
        return INFINIOP_STATUS_BAD_TENSOR_DTYPE;
    }

    if (a_desc->strides[1] != 1 || b_desc->strides[1] != 1 || c_desc->strides[1] != 1) {
        return INFINIOP_STATUS_BAD_TENSOR_STRIDES;
    }

    uint64_t seq_len = c_desc->shape[0],
             di = c_desc->shape[1];

    uint64_t stride_a = a_desc->strides[0],
             stride_b = b_desc->strides[0],
             stride_c = c_desc->strides[0];


    if (a_desc->shape[0] != seq_len || a_desc->shape[1] != di || a_desc->dtype != dtype ||
        b_desc->shape[0] != seq_len || b_desc->shape[1] != di || b_desc->dtype != dtype) {
        return INFINIOP_STATUS_BAD_PARAM;
    }

    *desc_ptr = new InfiniopSwiGLUBangDescriptor{handle->device,
                                                 handle->device_id,
                                                 dtype,
                                                 seq_len,
                                                 di,
                                                 stride_a,
                                                 stride_b,
                                                 stride_c};
    return INFINIOP_STATUS_SUCCESS;
}

infiniopStatus_t bangDestroySwiGLUDescriptor(infiniopSwiGLUBangDescriptor_t desc) {
    delete desc;
    return INFINIOP_STATUS_SUCCESS;
}
