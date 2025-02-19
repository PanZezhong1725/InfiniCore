#include "causal_softmax_aclnn.h"
#include "../../../devices/ascend/common_ascend.h"
#include <aclnnop/aclnn_masked_softmax_with_rel_pos_bias.h>

namespace op::causal_softmax::ascend {

struct Descriptor::Opaque {
    mutable aclOpExecutor *executor;
    aclnnTensorDescriptor_t x;
    aclnnTensorDescriptor_t mask;
    aclnnTensorDescriptor_t y;
    void *maskAddr;

    ~Opaque() {
        delete x;
        delete mask;
        delete y;

        aclDestroyAclOpExecutor(executor);
    }
};

Descriptor::~Descriptor() {
    delete _opaque;
}

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t y_desc) {
    if (!y_desc->isContiguous()) {
        return INFINI_STATUS_INTERNAL_ERROR;
    }
    CausalSoftmaxInfo info;
    auto handle_ascend = reinterpret_cast<device::ascend::Handle *>(handle);
    CHECK_STATUS(createCausalSoftmaxInfo(&info, y_desc));

    aclOpExecutor *executor = nullptr;
    aclnnTensorDescriptor_t y = nullptr;
    aclnnTensorDescriptor_t mask = nullptr;
    aclnnTensorDescriptor_t x = nullptr;
    void *maskAddr = nullptr;

    // Create Aclnn Tensor Descriptors for input , mask and output
    std::vector<int64_t> shape = {1, static_cast<int64_t>(info.batch_size), static_cast<int64_t>(info.seq_len), static_cast<int64_t>(info.total_seq_len)};
    std::vector<int64_t> strides = {0, static_cast<int64_t>(info.stride_b), static_cast<int64_t>(info.stride_i), static_cast<int64_t>(info.stride_j)};
    y = new aclnnTensorDescriptor(toAclDataType(info.dtype), shape, strides);
    x = new aclnnTensorDescriptor(toAclDataType(info.dtype), shape, strides);
    mask = new aclnnTensorDescriptor(toAclDataType(info.dtype), shape, strides);

    std::cout << y->toString() << std::endl;
    std::cout << x->toString() << std::endl;
    std::cout << mask->toString() << std::endl;

    // Get the workspace size for the op
    aclTensor *tx = x->tensor;
    aclTensor *ty = y->tensor;
    aclTensor *tmask = mask->tensor;

    size_t workspaceSize = 0;
    CHECK_ACL(aclnnMaskedSoftmaxWithRelPosBiasGetWorkspaceSize(tx, nullptr, tmask, 1.0, 0, ty, &workspaceSize, &executor));
    aclSetAclOpExecutorRepeatable(executor);

    // Fill Mask Tensor up Matrix
    std::vector<std::vector<std::vector<uint16_t>>> mask_matrix(info.batch_size, std::vector<std::vector<uint16_t>>(info.seq_len, std::vector<uint16_t>(info.total_seq_len, 0)));

    for (size_t i = 0; i < info.batch_size; ++i) {
        for (size_t j = 0; j < info.seq_len; ++j) {
            for (size_t k = info.total_seq_len - info.seq_len + j + 1; k < info.total_seq_len; ++k) {
                mask_matrix[i][j][k] = 0xF880;
            }
        }
    }
    // std::cout << "Mask Matrix:" << std::endl;
    // for (size_t i = 0; i < info.batch_size; ++i) {
    //     for (size_t j = 0; j < info.seq_len; ++j) {
    //         for (size_t k = 0; k < info.total_seq_len; ++k) {
    //             std::cout << mask_matrix[i][j][k] << " ";
    //         }
    //         std::cout << std::endl;
    //     }
    // }

    auto mask_size = mask->numel() * aclDataTypeSize(mask->dataType);
    CHECK_ACL(aclrtMalloc(&maskAddr, mask_size, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMemcpy(maskAddr, mask_size, mask_matrix.data(), mask_size, ACL_MEMCPY_HOST_TO_DEVICE));

    *desc_ptr = new Descriptor(new Opaque{executor, x, mask, y, maskAddr}, info, workspaceSize, handle_ascend->device, handle_ascend->device_id);

    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(void *workspace, size_t workspace_size, void *data, void *stream) {
    if (workspace_size < workspaceSize()) {
        return INFINI_STATUS_INSUFFICIENT_WORKSPACE;
    }
    auto tx = _opaque->x->tensor;
    auto ty = _opaque->y->tensor;
    auto tmask = _opaque->mask->tensor;
    auto mask_addr = _opaque->maskAddr;
    auto executor = _opaque->executor;

    AclSetTensorAddr(executor, 0, tx, data);
    AclSetTensorAddr(executor, 2, tmask, mask_addr);
    AclSetTensorAddr(executor, 3, ty, data);

    CHECK_ACL(aclnnMaskedSoftmaxWithRelPosBias(workspace, workspaceSize(), executor, stream));

    return INFINI_STATUS_SUCCESS;
}

} // namespace op::causal_softmax::ascend