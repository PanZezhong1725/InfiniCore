#ifndef __ACLNN_RMS_NORM_H__
#define __ACLNN_RMS_NORM_H__

#include "../../../devices/ascend/tensor_aclnn.h"
#include "../../utils.h"
#include "rms_norm_aclnn_api.h"
#include <acl/acl_base.h>
#include <aclnn/acl_meta.h>
#include <aclnnop/aclnn_cast.h>
#include <aclnnop/aclnn_rms_norm.h>

struct InfiniopRMSNormAclnnDescriptor {
    infiniDevice_t device;
    int device_id;
    aclOpExecutor *executor;
    aclOpExecutor *castExecutor;
    aclnnTensorDescriptor_t yDesc, xDesc, wDesc, rstdDesc, castDesc;
    size_t workspaceSize;
    size_t castWorkspaceSize;
    double epsilon;

    InfiniopRMSNormAclnnDescriptor(infiniDevice_t _device);
};

#endif
