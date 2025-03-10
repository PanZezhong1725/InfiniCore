#include "ascend_handle.h"

namespace device::ascend {

Handle::Handle() : InfiniopHandle{INFINI_DEVICE_ASCEND, 0} {}

infiniStatus_t Handle::create(InfiniopHandle **Handle_ptr, int) {
    *Handle_ptr = new Handle();
    return INFINI_STATUS_SUCCESS;
}

} // namespace device::ascend
