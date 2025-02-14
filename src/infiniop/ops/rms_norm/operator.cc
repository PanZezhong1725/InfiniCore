#include "infiniop/ops/rms_norm.h"

#ifdef ENABLE_CPU_API
#include "cpu/rms_norm_cpu_api.h"
#endif
#ifdef ENABLE_CUDA_API
#include "cuda/rms_norm_cuda_api.h"
#endif
#ifdef ENABLE_CAMBRICON_API
#include "bang/rms_norm_cnnl_api.h"
#endif
#ifdef ENABLE_ASCEND_API
#include "ascend/rms_norm_aclnn_api.h"
#endif

__C infiniopStatus_t infiniopCreateRMSNormDescriptor(
    infiniopHandle_t handle,
    infiniopRMSNormDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t y_desc,
    infiniopTensorDescriptor_t x_desc,
    infiniopTensorDescriptor_t w_desc,
    float epsilon) {
    switch (handle->device) {
#ifdef ENABLE_CPU_API
        case INFINI_DEVICE_CPU:
            return cpuCreateRMSNormDescriptor(handle, (infiniopRMSNormCpuDescriptor_t *) desc_ptr, y_desc, x_desc, w_desc, epsilon);
#endif
#ifdef ENABLE_NV_GPU
        case DevNvGpu: {
            return cudaCreateRMSNormDescriptor((CudaHandle_t) handle, (RMSNormCudaDescriptor_t *) desc_ptr, y_desc, x_desc, w_desc, epsilon);
        }
#endif
#ifdef ENABLE_CAMBRICON_MLU
        case DevCambriconMlu: {
            return bangCreateRMSNormDescriptor((BangHandle_t) handle, (RMSNormBangDescriptor_t *) desc_ptr, y_desc, x_desc, w_desc, epsilon);
        }
#endif
#ifdef ENABLE_ASCEND_NPU
        case DevAscendNpu: {
            return aclnnCreateRMSNormDescriptor((AscendHandle_t) handle,
                                                (RMSNormAclnnDescriptor_t *) desc_ptr,
                                                y_desc,
                                                x_desc,
                                                w_desc,
                                                epsilon);
        }
#endif
#ifdef ENABLE_METAX_GPU
        case DevMetaxGpu: {
            return macaCreateRMSNormDescriptor((MacaHandle_t) handle, (RMSNormMacaDescriptor_t *) desc_ptr, y_desc, x_desc, w_desc, epsilon);
        }
#endif
#ifdef ENABLE_MTHREADS_GPU
        case DevMthreadsGpu: {
            return musaCreateRMSNormDescriptor((MusaHandle_t) handle, (RMSNormMusaDescriptor_t *) desc_ptr, y_desc, x_desc, w_desc, epsilon);
        }
#endif
    }
    return INFINIOP_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
}

__C infiniopStatus_t infiniopGetRMSNormWorkspaceSize(infiniopRMSNormDescriptor_t desc, uint64_t *size) {
    switch (desc->device) {
#ifdef ENABLE_CPU_API
        case INFINI_DEVICE_CPU:
            return cpuGetRMSNormWorkspaceSize((infiniopRMSNormCpuDescriptor_t) desc, size);
#endif
#ifdef ENABLE_NV_GPU
        case DevNvGpu: {
            return cudaGetRMSNormWorkspaceSize((RMSNormCudaDescriptor_t) desc, size);
        }

#endif
#ifdef ENABLE_CAMBRICON_MLU
        case DevCambriconMlu: {
            return bangGetRMSNormWorkspaceSize((RMSNormBangDescriptor_t) desc, size);
        }
#endif
#ifdef ENABLE_ASCEND_NPU
        case DevAscendNpu: {
            return aclnnGetRMSNormWorkspaceSize((RMSNormAclnnDescriptor_t) desc,
                                                size);
        }
#endif
#ifdef ENABLE_METAX_GPU
        case DevMetaxGpu: {
            return macaGetRMSNormWorkspaceSize((RMSNormMacaDescriptor_t) desc, size);
        }
#endif
#ifdef ENABLE_MTHREADS_GPU
        case DevMthreadsGpu: {
            return musaGetRMSNormWorkspaceSize((RMSNormMusaDescriptor_t) desc, size);
        }
#endif
    }
    return INFINIOP_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
}

__C infiniopStatus_t infiniopRMSNorm(infiniopRMSNormDescriptor_t desc, void *workspace, uint64_t workspace_size,
                                     void *y, void const *x, void const *w, void *stream) {
    switch (desc->device) {
#ifdef ENABLE_CPU_API
        case INFINI_DEVICE_CPU:
            return cpuRMSNorm((infiniopRMSNormCpuDescriptor_t) desc, workspace, workspace_size, y, x, w, stream);
#endif
#ifdef ENABLE_NV_GPU
        case DevNvGpu: {
            return cudaRMSNorm((RMSNormCudaDescriptor_t) desc, workspace, workspace_size, y, x, w, stream);
        }

#endif
#ifdef ENABLE_CAMBRICON_MLU
        case DevCambriconMlu: {
            return bangRMSNorm((RMSNormBangDescriptor_t) desc, workspace, workspace_size, y, x, w, stream);
        }
#endif
#ifdef ENABLE_ASCEND_NPU
        case DevAscendNpu: {
            return aclnnRMSNorm((RMSNormAclnnDescriptor_t) desc,
                                workspace,
                                workspace_size,
                                y,
                                x,
                                w,
                                stream);
        }
#endif
#ifdef ENABLE_METAX_GPU
        case DevMetaxGpu: {
            return macaRMSNorm((RMSNormMacaDescriptor_t) desc, workspace, workspace_size, y, x, w, stream);
        }
#endif
#ifdef ENABLE_MTHREADS_GPU
        case DevMthreadsGpu: {
            return musaRMSNorm((RMSNormMusaDescriptor_t) desc, workspace, workspace_size, y, x, w, stream);
        }
#endif
    }
    return INFINIOP_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
}

__C infiniopStatus_t infiniopDestroyRMSNormDescriptor(infiniopRMSNormDescriptor_t desc) {
    switch (desc->device) {
#ifdef ENABLE_CPU_API
        case INFINI_DEVICE_CPU:
            return cpuDestroyRMSNormDescriptor((infiniopRMSNormCpuDescriptor_t) desc);
#endif
#ifdef ENABLE_NV_GPU
        case DevNvGpu: {
            return cudaDestroyRMSNormDescriptor((RMSNormCudaDescriptor_t) desc);
        }

#endif
#ifdef ENABLE_CAMBRICON_MLU
        case DevCambriconMlu: {
            return bangDestroyRMSNormDescriptor((RMSNormBangDescriptor_t) desc);
        }
#endif
#ifdef ENABLE_ASCEND_NPU
        case DevAscendNpu: {
            return aclnnDestroyRMSNormDescriptor((RMSNormAclnnDescriptor_t) desc);
        }
#endif
#ifdef ENABLE_METAX_GPU
        case DevMetaxGpu: {
            return macaDestroyRMSNormDescriptor((RMSNormMacaDescriptor_t) desc);
        }
#endif
#ifdef ENABLE_MTHREADS_GPU
        case DevMthreadsGpu: {
            return musaDestroyRMSNormDescriptor((RMSNormMusaDescriptor_t) desc);
        }
#endif
    }
    return INFINIOP_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
}
