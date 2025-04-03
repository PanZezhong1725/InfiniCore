#ifndef __INFINIOP_ROTARY_EMBEDDING_API_H__
#define __INFINIOP_ROTARY_EMBEDDING_API_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopRoPEDescriptor_t;

__C __export infiniStatus_t infiniopCreateRoPEDescriptor(
    infiniopHandle_t handle,
    infiniopRoPEDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t t,
    infiniopTensorDescriptor_t pos_ids,
    infiniopTensorDescriptor_t sin_table,
    infiniopTensorDescriptor_t cos_table);

__C __export infiniStatus_t infiniopGetRoPEWorkspaceSize(infiniopRoPEDescriptor_t desc, size_t *size);

__C __export infiniStatus_t infiniopRoPE(
    infiniopRoPEDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *t,
    const void *pos_ids,
    const float *sin_table,
    const float *cos_table,
    void *stream);

__C __export infiniStatus_t infiniopDestroyRoPEDescriptor(infiniopRoPEDescriptor_t desc);

#endif
