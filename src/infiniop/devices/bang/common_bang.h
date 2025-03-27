#ifndef __COMMON_BANG_H__
#define __COMMON_BANG_H__

#include "../../../utils.h"
#include "../../tensor.h"
#include "../pool.h"
#include "bang_handle.h"
#include "cnnl.h"
#include "cnrt.h"
#include "infinicore.h"
#include <functional>

#define CHECK_BANG(API) CHECK_INTERNAL(API, CNNL_STATUS_SUCCESS)

#define NRAM_MAX_SIZE_372 786432 // 1024 * 768
#define NRAM_MAX_SIZE_592 524288 // 1024 * 512,实际测试申请内存的时候char nram_buffer[x], x必须小于1024 * 512

#define GDRAM_MAX_SIZE_372 23934976 * 1024 // 1024 * 23374 * 1024
#define GDRAM_MAX_SIZE_592 40715264 * 1024 // 1024 * 39761 * 1024

namespace device::bang {

class Handle::Internal {
    Pool<cnnlHandle_t> cnnl_handles;

    int _NRAM_MAX_SIZE;  // 单位byte
    int _GDRAM_MAX_SIZE; // 单位byte

    template <typename T>
    using Fn = std::function<infiniStatus_t(T)>;

public:
    Internal(int);

    infiniStatus_t useCnnl(cnrtQueue_t queue, const Fn<cnnlHandle_t> &f) const;

    int getNramSize() const;
    int getGdramSize() const;
};

cnnlDataType_t getCnnlDtype(infiniDtype_t dt);

// set cnnl tensor descriptor without strides
infiniStatus_t setCnnlTensor(cnnlTensorDescriptor_t desc,
                             const InfiniopTensorDescriptor *layout);

// set cnnl tensor descriptor with strides
infiniStatus_t setCnnlTensorEx(cnnlTensorDescriptor_t desc,
                               const InfiniopTensorDescriptor *layout);

} // namespace device::bang

#endif // __COMMON_BANG_H__
