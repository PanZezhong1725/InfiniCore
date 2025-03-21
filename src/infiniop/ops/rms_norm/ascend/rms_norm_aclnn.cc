#include "rms_norm_aclnn.h"

InfiniopRMSNormAclnnDescriptor::InfiniopRMSNormAclnnDescriptor(infiniDevice_t _device) {
    device = _device;
    device_id = 0;
    executor = nullptr;
    castExecutor = nullptr;
    workspaceSize = 0;
    castWorkspaceSize = 0;
    yDesc = new aclnnTensorDescriptor();
    xDesc = new aclnnTensorDescriptor();
    wDesc = new aclnnTensorDescriptor();
    rstdDesc = new aclnnTensorDescriptor();
    castDesc = nullptr;
    epsilon = 1e-5;
}

infiniopStatus_t aclnnCreateRMSNormDescriptor(infiniopAscendHandle_t handle,
                                              RMSNormAclnnDescriptor_t *desc_ptr,
                                              infiniopTensorDescriptor_t y_desc,
                                              infiniopTensorDescriptor_t x_desc,
                                              infiniopTensorDescriptor_t w_desc,
                                              float epsilon) {
    *desc_ptr = new InfiniopRMSNormAclnnDescriptor(handle->device);
    (*desc_ptr)->device_id = handle->device_id;
    (*desc_ptr)->epsilon = static_cast<double>(epsilon);

    auto &yDesc = (*desc_ptr)->yDesc;
    auto &xDesc = (*desc_ptr)->xDesc;
    auto &wDesc = (*desc_ptr)->wDesc;
    auto &castDesc = (*desc_ptr)->castDesc;
    auto &rstdDesc = (*desc_ptr)->rstdDesc;

    CHECK_STATUS(yDesc->fromInfiniOpTensorDescriptor(y_desc), INFINIOP_STATUS_SUCCESS);
    CHECK_STATUS(xDesc->fromInfiniOpTensorDescriptor(x_desc), INFINIOP_STATUS_SUCCESS);
    CHECK_STATUS(wDesc->fromInfiniOpTensorDescriptor(w_desc), INFINIOP_STATUS_SUCCESS);

    // Set rstdDesc
    // See: https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/80RC3alpha002/apiref/appdevgapi/context/aclnnRmsNorm.md
    // rstdTensor cannot set nullptr in aclnn
    int64_t wsize = 1;
    for (uint64_t i = 0; i < wDesc->ndim; ++i) {
        wsize *= (wDesc->shape)[i];
    }
    int64_t xsize = 1;
    uint64_t rstd_dim = xDesc->ndim - 1;
    for (int64_t i = xDesc->ndim - 1; i >= 0; --i) {
        xsize *= (xDesc->shape)[i];
        rstd_dim = static_cast<uint64_t>(i);
        if (xsize == wsize) {
            break;
        }
    }

    auto rstd_shape = std::vector<int64_t>(xDesc->ndim, 1);
    auto rstd_strides = std::vector<int64_t>(xDesc->ndim, 1);

    for (uint64_t i = 0; i < rstd_dim; ++i) {
        rstd_shape[i] = (xDesc->shape)[i];
    }
    for (int64_t i = xDesc->ndim - 2; i >= 0; --i) {
        rstd_strides[i] = rstd_strides[i + 1] * rstd_shape[i + 1];
    }
    CHECK_STATUS(rstdDesc->setDescriptor(toAclDataType(INFINI_DTYPE_F32), rstd_shape, rstd_strides), INFINIOP_STATUS_SUCCESS);

    if (wDesc->dataType != xDesc->dataType) {
        castDesc = new aclnnTensorDescriptor();
        CHECK_STATUS(castDesc->fromInfiniOpTensorDescriptor(w_desc), INFINIOP_STATUS_SUCCESS);
        castDesc->dataType = xDesc->dataType;
        CHECK_STATUS(castDesc->createTensor(), INFINIOP_STATUS_SUCCESS);
    }

    CHECK_STATUS(yDesc->createTensor(), INFINIOP_STATUS_SUCCESS);
    CHECK_STATUS(xDesc->createTensor(), INFINIOP_STATUS_SUCCESS);
    CHECK_STATUS(wDesc->createTensor(), INFINIOP_STATUS_SUCCESS);
    CHECK_STATUS(rstdDesc->createTensor(), INFINIOP_STATUS_SUCCESS);

    // Get Tensor
    aclTensor *ty = yDesc->t;
    aclTensor *tx = xDesc->t;
    aclTensor *tw = wDesc->t;
    aclTensor *trstd = rstdDesc->t;

    // Get workspaceSize and set executor
    auto &workspaceSize = (*desc_ptr)->workspaceSize;
    auto &executor = (*desc_ptr)->executor;
    auto ret = aclnnRmsNormGetWorkspaceSize(tx,
                                            castDesc == nullptr ? tw
                                                                : castDesc->t,
                                            (*desc_ptr)->epsilon,
                                            ty,
                                            trstd,
                                            &workspaceSize,
                                            &executor);
    aclSetAclOpExecutorRepeatable(executor);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("aclnnRmsNormGetWorkspaceSize failed. ERROR: %d\n", ret);
              return INFINIOP_STATUS_INTERNAL_ERROR);

    // Get Cast workspaceSize and set castExecutor
    if (castDesc != nullptr) {
        auto &castExecutor = (*desc_ptr)->castExecutor;
        auto &castWorkspaceSize = (*desc_ptr)->castWorkspaceSize;
        aclTensor *tcast = castDesc->t;
        ret = aclnnCastGetWorkspaceSize(tw,
                                        castDesc->dataType,
                                        tcast,
                                        &castWorkspaceSize,
                                        &castExecutor);
        aclSetAclOpExecutorRepeatable(castExecutor);
        CHECK_RET(ret == ACL_SUCCESS,
                  LOG_PRINT("aclnnCastGetWorkspaceSize failed. ERROR: %d\n", ret);
                  return INFINIOP_STATUS_INTERNAL_ERROR);
    }

    return INFINIOP_STATUS_SUCCESS;
}

infiniopStatus_t aclnnGetRMSNormWorkspaceSize(RMSNormAclnnDescriptor_t desc,
                                              size_t *size) {
    auto &rstdDesc = desc->rstdDesc;
    auto &castDesc = desc->castDesc;

    *size = desc->workspaceSize + numElements(rstdDesc->shape.data(), rstdDesc->ndim) * aclDataTypeSize(rstdDesc->dataType);

    if (castDesc != nullptr) {
        *size += desc->castWorkspaceSize;
        *size += numElements(castDesc->shape.data(), castDesc->ndim) * aclDataTypeSize(castDesc->dataType);
    }

    return INFINIOP_STATUS_SUCCESS;
}

infiniopStatus_t aclnnRMSNorm(RMSNormAclnnDescriptor_t desc, void *workspace,
                              size_t workspace_size, void *y, const void *x,
                              const void *w, void *stream) {
    auto &yDesc = desc->yDesc;
    auto &xDesc = desc->xDesc;
    auto &wDesc = desc->wDesc;
    auto &rstdDesc = desc->rstdDesc;
    auto &castDesc = desc->castDesc;

    // Get Tensor
    aclTensor *ty = yDesc->t;
    aclTensor *tx = xDesc->t;
    aclTensor *tw = wDesc->t;
    aclTensor *trstd = rstdDesc->t;

    auto &executor = desc->executor;
    auto &castExecutor = desc->castExecutor;
    auto &workspaceSize = desc->workspaceSize;
    auto &castWorkspaceSize = desc->castWorkspaceSize;

    auto rstd = (void *)((uint8_t *)workspace + workspaceSize);

    // Set device
    aclrtSetDevice(desc->device_id);
    aclnnStatus ret;

    void *castPtr = nullptr;

    // Cast w
    if (castDesc != nullptr) {
        aclTensor *tcast = castDesc->t;
        castPtr = (void *)((float *)rstd + numElements(rstdDesc->shape.data(), rstdDesc->ndim));

        AclSetTensorAddr(castExecutor, 0, tw, (void *)w);
        AclSetTensorAddr(castExecutor, 1, tcast, castPtr);
        ret = aclnnCast(nullptr, castWorkspaceSize, castExecutor, stream);
        CHECK_RET(ret == ACL_SUCCESS,
                  LOG_PRINT("aclnnCast failed. ERROR: %d\n", ret);
                  return INFINIOP_STATUS_INTERNAL_ERROR);
    }

    // Do RmsNorm calc
    AclSetTensorAddr(executor, 0, tx, (void *)x);
    if (castDesc != nullptr) {
        AclSetTensorAddr(executor, 1, castDesc->t, castPtr);
    } else {
        AclSetTensorAddr(executor, 1, tw, (void *)w);
    }
    AclSetTensorAddr(executor, 2, ty, y);
    AclSetTensorAddr(executor, 3, trstd, rstd);

    ret = aclnnRmsNorm(workspace,
                       workspaceSize,
                       executor,
                       stream);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("aclnnRmsNorm failed. ERROR: %d\n", ret);
              return INFINIOP_STATUS_INTERNAL_ERROR);

    return INFINIOP_STATUS_SUCCESS;
}

infiniopStatus_t aclnnDestroyRMSNormDescriptor(RMSNormAclnnDescriptor_t desc) {
    delete desc->yDesc;
    delete desc->wDesc;
    delete desc->xDesc;
    delete desc->rstdDesc;
    aclDestroyAclOpExecutor(desc->executor);
    if (desc->castDesc != nullptr) {
        delete desc->castDesc;
        aclDestroyAclOpExecutor(desc->castExecutor);
    }
    delete desc;

    return INFINIOP_STATUS_SUCCESS;
}