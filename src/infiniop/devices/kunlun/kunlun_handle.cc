#include "kunlun_handle.h"
#include "common_kunlun.h"

infiniopStatus_t createKunlunHandle(infiniopKunlunHandle_t *handle_ptr,
                                    int device_id) {
    int device_count;
    xpu_device_count(&device_count);
    if (device_id >= device_count) {
        return INFINIOP_STATUS_BAD_DEVICE;
    }

    auto pool = std::make_shared<Pool<xdnnHandle_t>>();
    if (xpu_set_device(device_id) != XPU_SUCCESS) {
        return INFINIOP_STATUS_BAD_DEVICE;
    }
    xdnnHandle_t handle = xdnn::create_context();
    pool->push(std::move(handle));

    *handle_ptr = new InfiniopKunlunHandle{
        INFINI_DEVICE_KUNLUN,
        device_id,
        std::move(pool),
    };

    return INFINIOP_STATUS_SUCCESS;
}

infiniopStatus_t deleteKunlunHandle(infiniopKunlunHandle_t handle) {
    handle->xdnn_handles_t = nullptr;
    delete handle;
    return INFINIOP_STATUS_SUCCESS;
}
