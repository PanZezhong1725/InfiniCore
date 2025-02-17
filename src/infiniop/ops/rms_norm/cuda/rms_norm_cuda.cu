#include "./rms_norm_cuda.cuh"
#include "../../utils.h"

infiniopStatus_t cudaCreateRMSNormDescriptor(infiniopCudaHandle_t handle, infiniopRMSNormCudaDescriptor_t *desc_ptr,
                                             infiniopTensorDescriptor_t y_desc,
                                             infiniopTensorDescriptor_t x_desc,
                                             infiniopTensorDescriptor_t w_desc,
                                             float epsilon) {
    if (y_desc->ndim != 2 || x_desc->ndim != 2 || w_desc->ndim != 1) {
        return INFINIOP_STATUS_BAD_TENSOR_SHAPE;
    }

    auto n = y_desc->shape[0],
         d = y_desc->shape[1];

    if (x_desc->shape[0] != n || x_desc->shape[1] != d || w_desc->shape[0] != d) {
        return INFINIOP_STATUS_BAD_TENSOR_SHAPE;
    }

    int64_t stride_y = y_desc->strides[0];
    int64_t stride_x = x_desc->strides[0];
    auto w_datatype = w_desc->dtype;
    *desc_ptr = new InfiniopRMSNormCudaDescriptor{
        handle->device,
        handle->device_id,
        y_desc->dtype,
        n,
        d,
        stride_y,
        stride_x,
        w_datatype,
        epsilon};

    return INFINIOP_STATUS_SUCCESS;
}

infiniopStatus_t cudaGetRMSNormWorkspaceSize(infiniopRMSNormCudaDescriptor_t desc, uint64_t *size) {
    *size = 0;
    return INFINIOP_STATUS_SUCCESS;
}

infiniopStatus_t cudaDestroyRMSNormDescriptor(infiniopRMSNormCudaDescriptor_t desc) {
    delete desc;
    return INFINIOP_STATUS_SUCCESS;
}
