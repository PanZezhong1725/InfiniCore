#ifndef __INFINIOP_RANDOM_SAMPLE_CPU_API_H__
#define __INFINIOP_RANDOM_SAMPLE_CPU_API_H__

#include "../../../devices/cpu/cpu_handle.h"
#include "infiniop/operator.h"

struct RandomSampleCpuDescriptor;

typedef struct RandomSampleCpuDescriptor *infiniopRandomSampleCpuDescriptor_t;

infiniopStatus_t cpuCreateRandomSampleDescriptor(
    infiniopCpuHandle_t handle,
    infiniopRandomSampleCpuDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t result_desc,
    infiniopTensorDescriptor_t probs_desc);

infiniopStatus_t cpuRandomSample(
    infiniopRandomSampleCpuDescriptor_t desc,
    void *result, void const *probs,
    float random_val,
    float topp, int topk, float temperature);

infiniopStatus_t cpuDestroyRandomSampleDescriptor(
    infiniopRandomSampleCpuDescriptor_t desc);

#endif // __INFINIOP_RANDOM_SAMPLE_CPU_API_H__
