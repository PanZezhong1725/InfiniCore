#include "rms_norm_bang.h"
#include "../../../devices/bang/common_bang.h"

namespace op::rms_norm::bang {
Descriptor::~Descriptor() {}

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t y_desc,
    infiniopTensorDescriptor_t x_desc,
    infiniopTensorDescriptor_t w_desc,
    float epsilon) {
    auto handle = reinterpret_cast<device::bang::cambricon::Handle *>(handle_);
    RMSNormInfo info;
    CHECK_STATUS(createRMSNormInfo(&info, y_desc, x_desc, w_desc, epsilon));
    size_t workspace_size = info.ndim() * (sizeof(size_t) + 2 * sizeof(ptrdiff_t));//用来存储一个shape和两个stride数组
    *desc_ptr = new Descriptor(nullptr, info, workspace_size, handle->device, handle->device_id);
    return INFINI_STATUS_SUCCESS;
}
} // namespace op::rms_norm::bang
