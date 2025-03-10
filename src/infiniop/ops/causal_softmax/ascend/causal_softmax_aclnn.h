#ifndef __ACLNN_CAUSAL_SOFTMAX_H__
#define __ACLNN_CAUSAL_SOFTMAX_H__

#include "../../../devices/ascend/tensor_aclnn.h"
#include "../../utils.h"
#include "causal_softmax_aclnn_api.h"
#include <acl/acl_base.h>
#include <aclnn/acl_meta.h>
#include <aclnnop/aclnn_masked_softmax_with_rel_pos_bias.h>

struct InfiniopCausalSoftmaxAclnnDescriptor {
    infiniDevice_t device;
    int device_id;
    aclOpExecutor *executor;
    aclnnTensorDescriptor_t aDesc, maskDesc, outDesc;
    size_t workspaceSize;
    void *maskAddr;

    InfiniopCausalSoftmaxAclnnDescriptor(infiniDevice_t _device);
};

#endif
