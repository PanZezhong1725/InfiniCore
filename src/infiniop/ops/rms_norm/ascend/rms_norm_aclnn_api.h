#ifndef __INFINIOP_RMS_NORM_ACLNN_API_H__
#define __INFINIOP_RMS_NORM_ACLNN_API_H__
#include "../../../devices/ascend/ascend_handle.h"
#include "infiniop/operator.h"

struct InfiniopRMSNormAclnnDescriptor;
typedef struct InfiniopRMSNormAclnnDescriptor *RMSNormAclnnDescriptor_t;

infiniopStatus_t aclnnCreateRMSNormDescriptor(infiniopAscendHandle_t handle,
                                              RMSNormAclnnDescriptor_t *desc_ptr,
                                              infiniopTensorDescriptor_t y_desc,
                                              infiniopTensorDescriptor_t x_desc,
                                              infiniopTensorDescriptor_t w_desc,
                                              float epsilon);

infiniopStatus_t aclnnGetRMSNormWorkspaceSize(RMSNormAclnnDescriptor_t desc,
                                              size_t *size);

infiniopStatus_t aclnnRMSNorm(RMSNormAclnnDescriptor_t desc, void *workspace,
                              size_t workspace_size, void *y, const void *x,
                              const void *w, void *stream);

infiniopStatus_t aclnnDestroyRMSNormDescriptor(RMSNormAclnnDescriptor_t desc);
#endif //__INFINIOP_RMS_NORM_ACLNN_API_H__
