#include "infiniop/ops/swiglu.h"


#ifdef ENABLE_CAMBRICON_API
#include "bang/swiglu_bang_api.h"
#endif

__C infiniopStatus_t infiniopCreateSwiGLUDescriptor(
    infiniopHandle_t handle, infiniopSwiGLUDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t c_desc, infiniopTensorDescriptor_t a_desc,
    infiniopTensorDescriptor_t b_desc) {
    switch (handle->device) {
#ifdef ENABLE_CAMBRICON_API
        case INFINI_DEVICE_CAMBRICON: {
            return bangCreateSwiGLUDescriptor((infiniopBangHandle_t) handle,
                                              (infiniopSwiGLUBangDescriptor_t *) desc_ptr,
                                              c_desc, a_desc, b_desc);
        }
#endif
    }
    return INFINIOP_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
};

__C infiniopStatus_t infiniopSwiGLU(infiniopSwiGLUDescriptor_t desc, void *c,
                                    void const *a, void const *b,
                                    void *stream) {
    switch (desc->device) {
#ifdef ENABLE_CAMBRICON_API
        case INFINI_DEVICE_CAMBRICON: {
            return bangSwiGLU((infiniopSwiGLUBangDescriptor_t) desc, c, a, b, stream);
        }
#endif
    }
    return INFINIOP_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
}

__C infiniopStatus_t
infiniopDestroySwiGLUDescriptor(infiniopSwiGLUDescriptor_t desc) {
    switch (desc->device) {
#ifdef ENABLE_CAMBRICON_API
        case INFINI_DEVICE_CAMBRICON: {
            return bangDestroySwiGLUDescriptor((infiniopSwiGLUBangDescriptor_t) desc);
        }
#endif
    }
    return INFINIOP_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
}
