#ifndef __MATMUL_XDNN_API_H__
#define __MATMUL_XDNN_API_H__

#include "../../../devices/kunlun/kunlun_handle.h"
#include "infiniop/operator.h"

struct InfiniopMatmulKunlunDescriptor;
typedef struct InfiniopMatmulKunlunDescriptor *infiniopMatmulKunlunDescriptor_t;

infiniopStatus_t kunlunCreateMatmulDescriptor(infiniopKunlunHandle_t handle,
                                              infiniopMatmulKunlunDescriptor_t *desc_ptr,
                                              infiniopTensorDescriptor_t c_desc,
                                              infiniopTensorDescriptor_t a_desc,
                                              infiniopTensorDescriptor_t b_desc);

infiniopStatus_t kunlunGetMatmulWorkspaceSize(infiniopMatmulKunlunDescriptor_t desc,
                                              uint64_t *size);

infiniopStatus_t kunlunMatmul(infiniopMatmulKunlunDescriptor_t desc,
                              void *workspace,
                              uint64_t workspace_size,
                              void *c,
                              void const *a,
                              void const *b,
                              float alpha,
                              float beta,
                              void *stream);

infiniopStatus_t kunlunDestroyMatmulDescriptor(infiniopMatmulKunlunDescriptor_t desc);

#endif
