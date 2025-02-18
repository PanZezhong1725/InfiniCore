#ifndef __INFINIOP_ROPE_CPU_API_H__
#define __INFINIOP_ROPE_CPU_API_H__

#include "../../../devices/cpu/cpu_handle.h"
#include "infiniop/operator.h"

struct RoPECpuDescriptor;

typedef struct RoPECpuDescriptor *infiniopRoPECpuDescriptor_t;

infiniopStatus_t cpuCreateRoPEDescriptor(
    infiniopCpuHandle_t handle,
    infiniopRoPECpuDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t t_desc,
    infiniopTensorDescriptor_t pos_desc,
    infiniopTensorDescriptor_t sin_desc,
    infiniopTensorDescriptor_t cos_desc);

infiniopStatus_t cpuRoPE(
    infiniopRoPECpuDescriptor_t desc,
    void *t,
    void const *pos,
    void const *sin,
    void const *cos);

infiniopStatus_t cpuDestroyRoPEDescriptor(
    infiniopRoPECpuDescriptor_t desc);

#endif // __INFINIOP_ROPE_CPU_API_H__
