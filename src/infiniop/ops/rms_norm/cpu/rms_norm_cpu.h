#ifndef __INFINIOP_RMS_NORM_CPU_H__
#define __INFINIOP_RMS_NORM_CPU_H__

#include "./rms_norm_cpu_api.h"

struct InfiniRMSNormCpuDescriptor {
    infiniDevice_t device;
    infiniDtype_t dtype;
    uint64_t n;
    uint64_t d;
    uint64_t stride_y;
    uint64_t stride_x;
    infiniDtype_t w_datatype;
    float epsilon;
};

#endif // __INFINIOP_RMS_NORM_CPU_H__
