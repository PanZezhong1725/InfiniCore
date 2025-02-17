#ifndef __INFINIOP_ROPE_CPU_H__
#define __INFINIOP_ROPE_CPU_H__

#include "./rope_cpu_api.h"

typedef struct RoPECpuDescriptor {
    infiniDevice_t device;
    infiniDtype_t ty_t, ty_pos;
    size_t nt, nh, dh, nsin, ncos;
    ptrdiff_t
        s_nt,
        s_nh,
        s_np,
        s_nsin, s_ncos,
        s_dsin, s_dcos;
} RoPECpuDescriptor;

#endif // __INFINIOP_ROPE_CPU_H__
