#ifndef __INFINIOP_RANDOM_SAMPLE_CPU_H__
#define __INFINIOP_RANDOM_SAMPLE_CPU_H__

#include "random_sample_cpu_api.h"

typedef struct RandomSampleCpuDescriptor {
    infiniDevice_t device;
    infiniDtype_t ty_i, ty_p;
    size_t n;
    ptrdiff_t s;
} RandomSampleCpuDescriptor;

#endif // __INFINIOP_RANDOM_SAMPLE_CPU_H__
