#ifndef __INFINIOP_SWIGLU_BANG_H__
#define __INFINIOP_SWIGLU_BANG_H__

#include "../../../devices/bang/common_bang.h"

struct InfiniopSwiGLUBangDescriptor {
    infiniDevice_t device;
    int device_id;
    infiniDtype_t dtype;
    int ndim;
    int *shape;
    int64_t *strides_a;
    int64_t *strides_b;
    int64_t *strides_c;
};


#endif// __INFINIOP_SWIGLU_BANG_H__
