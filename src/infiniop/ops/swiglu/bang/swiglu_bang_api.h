#ifndef __INFINIOP_SWIGLU_BANG_API_H__
#define __INFINIOP_SWIGLU_BANG_API_H__

#include "../../../devices/bang/bang_handle.h"
#include "infiniop/operator.h"

struct InfiniopSwiGLUBangDescriptor;
typedef struct InfiniopSwiGLUBangDescriptor *infiniopSwiGLUBangDescriptor_t;

infiniopStatus_t bangCreateSwiGLUDescriptor(infiniopBangHandle_t handle,
                                            infiniopSwiGLUBangDescriptor_t *desc_ptr,
                                            infiniopTensorDescriptor_t c_dec,
                                            infiniopTensorDescriptor_t a_desc,
                                            infiniopTensorDescriptor_t b_desc);

infiniopStatus_t bangSwiGLU(infiniopSwiGLUBangDescriptor_t desc,
                            void *c,
                            void const *a,
                            void const *b,
                            void *stream);

infiniopStatus_t bangDestroySwiGLUDescriptor(infiniopSwiGLUBangDescriptor_t desc);

#endif
