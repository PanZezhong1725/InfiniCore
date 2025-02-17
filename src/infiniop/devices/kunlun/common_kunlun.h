#ifndef __INFINIOP_COMMON_KUNLUN_H__
#define __INFINIOP_COMMON_KUNLUN_H__

#include "xpu/runtime.h"
#include "xpu/runtime_ex.h"
#include "xpu/xdnn.h"

namespace xdnn = baidu::xpu::api;
typedef xdnn::Context *xdnnHandle_t;

#define checkKUNLUNError(call)                                         \
    {                                                                  \
        auto err = call;                                               \
        if (XPU_SUCCESS != err) {                                      \
            fprintf(stderr, "KUNLUN error in %s:%i : %s.\n", __FILE__, \
                    __LINE__, xpu_strerror(err));                      \
            exit(EXIT_FAILURE);                                        \
        }                                                              \
    }

#include "../pool.h"
#include "infinicore.h"
#include "kunlun_handle.h"
#include <memory>

struct InfiniopKunlunHandle {
    infiniDevice_t device;
    int device_id;
    std::shared_ptr<Pool<xdnnHandle_t>> xdnn_handles_t;
};

template<typename T>
void use_xdnn(std::shared_ptr<Pool<xdnnHandle_t>> xdnn_handles_t,
              XPUStream stream, T const &f) {
    auto handle = xdnn_handles_t->pop();
    if (!handle) {
        *handle = xdnn::create_context();
    }
    (*handle)->set_stream(stream);
    f(*handle);
    xdnn_handles_t->push(std::move(*handle));
}

#endif
