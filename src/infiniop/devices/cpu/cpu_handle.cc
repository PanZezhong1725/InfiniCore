#include "./cpu_handle.h"

infiniopStatus_t createCpuHandle(infiniopCpuHandle_t *handle_ptr) {
    *handle_ptr = new InfiniopHandle{INFINI_DEVICE_CPU, 0};
    return INFINIOP_STATUS_SUCCESS;
}

infiniopStatus_t destroyCpuHandle(infiniopCpuHandle_t handle) {
    delete handle;
    return INFINIOP_STATUS_SUCCESS;
}
