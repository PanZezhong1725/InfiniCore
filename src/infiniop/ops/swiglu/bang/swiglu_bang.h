#ifndef __INFINIOP_SWIGLU_BANG_H__
#define __INFINIOP_SWIGLU_BANG_H__

#include "../../../devices/bang/common_bang.h"

struct InfiniopSwiGLUBangDescriptor {
    infiniDevice_t device;
    int device_id;
    infiniDtype_t dtype;
    uint64_t seq_len;
    uint64_t di;
    uint64_t stride_a;
    uint64_t stride_b;
    uint64_t stride_c;
};


#endif// __INFINIOP_SWIGLU_BANG_H__
