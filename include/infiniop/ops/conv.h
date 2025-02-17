#ifndef __INFINIOP_CONV_H__
#define __INFINIOP_CONV_H__

#include "../operator.h"

typedef InfiniopDescriptor *infiniopConvDescriptor_t;

__C __export infiniopStatus_t infiniopCreateConvDescriptor(infiniopHandle_t handle,
                                                           infiniopConvDescriptor_t *desc_ptr,
                                                           infiniopTensorDescriptor_t y,
                                                           infiniopTensorDescriptor_t x,
                                                           infiniopTensorDescriptor_t w,
                                                           void *pads,
                                                           void *strides,
                                                           void *dilations,
                                                           size_t n);

__C __export infiniopStatus_t infiniopGetConvWorkspaceSize(infiniopConvDescriptor_t desc, size_t *size);

__C __export infiniopStatus_t infiniopConv(infiniopConvDescriptor_t desc, void *workspace, size_t workspace_size, void *y, void const *x, void const *w, void *stream);

__C __export infiniopStatus_t infiniopDestroyConvDescriptor(infiniopConvDescriptor_t desc);

#endif
