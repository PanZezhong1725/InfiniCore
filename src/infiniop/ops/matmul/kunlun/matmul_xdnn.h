#ifndef __MATMUL_XDNN_H__
#define __MATMUL_XDNN_H__

#include "../../../devices/kunlun/common_kunlun.h"
#include "../../utils.h"
#include "../blas.h"
#include "matmul_xdnn_api.h"

struct MatmulKunlunDescriptor {
    infiniDevice_t device;
    infiniDtype_t dtype;
    int device_id;
    MatmulInfo info;
    std::shared_ptr<Pool<xdnnHandle_t>> xdnn_handles_t;
};

#endif
