#ifndef __INFINIOP_CAUSAL_SOFTMAX_ACLNN_API_H__
#define __INFINIOP_CAUSAL_SOFTMAX_ACLNN_API_H__
#include "../../../devices/ascend/ascend_handle.h"
#include "infiniop/operator.h"

struct InfiniopCausalSoftmaxAclnnDescriptor;
typedef struct InfiniopCausalSoftmaxAclnnDescriptor *CausalSoftmaxAclnnDescriptor_t;

infiniopStatus_t aclnnCreateCausalSoftmaxDescriptor(infiniopAscendHandle_t handle,
                                                    CausalSoftmaxAclnnDescriptor_t *desc_ptr,
                                                    infiniopTensorDescriptor_t y_desc);

infiniopStatus_t aclnnGetCausalSoftmaxWorkspaceSize(CausalSoftmaxAclnnDescriptor_t desc,
                                                    size_t *size);

infiniopStatus_t aclnnCausalSoftmax(CausalSoftmaxAclnnDescriptor_t desc, void *workspace,
                                    size_t workspace_size, void *data, void *stream);

infiniopStatus_t aclnnDestroyCausalSoftmaxDescriptor(CausalSoftmaxAclnnDescriptor_t desc);
#endif // __INFINIOP_CAUSAL_SOFTMAX_ACLNN_API_H__
