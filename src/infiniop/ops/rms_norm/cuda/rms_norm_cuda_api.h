#ifndef __INFINIOP_RMS_NORM_CUDA_API_H__
#define __INFINIOP_RMS_NORM_CUDA_API_H__

#include "../../../devices/cuda/cuda_handle.h"
#include "infiniop/operator.h"

struct InfiniopRMSNormCudaDescriptor;
typedef struct InfiniopRMSNormCudaDescriptor *infiniopRMSNormCudaDescriptor_t;

infiniopStatus_t cudaCreateRMSNormDescriptor(infiniopCudaHandle_t handle,
                                             infiniopRMSNormCudaDescriptor_t *desc_ptr,
                                             infiniopTensorDescriptor_t y_desc,
                                             infiniopTensorDescriptor_t x_desc,
                                             infiniopTensorDescriptor_t w_desc,
                                             float epsilon);

infiniopStatus_t cudaGetRMSNormWorkspaceSize(infiniopRMSNormCudaDescriptor_t desc, uint64_t *size);

infiniopStatus_t cudaRMSNorm(infiniopRMSNormCudaDescriptor_t desc,
                             void *workspace,
                             uint64_t workspace_size,
                             void *y, void const *x, void const *w,
                             void *stream);

infiniopStatus_t cudaDestroyRMSNormDescriptor(infiniopRMSNormCudaDescriptor_t desc);

void rms_norm_nv_gpu_f16(infiniopRMSNormCudaDescriptor_t desc, void *y, void const *x, void const *w,
                            float epsilon, void *stream);

#endif// __INFINIOP_RMS_NORM_CUDA_API_H__
