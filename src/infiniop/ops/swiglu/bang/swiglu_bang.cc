#include "swiglu_bang.h"
#include "../../utils.h"
#include "swiglu_bang_api.h"

infiniopStatus_t bangCreateSwiGLUDescriptor(infiniopBangHandle_t handle,
                                            infiniopSwiGLUBangDescriptor_t *desc_ptr,
                                            infiniopTensorDescriptor_t c_desc,
                                            infiniopTensorDescriptor_t a_desc,
                                            infiniopTensorDescriptor_t b_desc) {
    if ((c_desc->ndim != a_desc->ndim) && (c_desc->ndim != b_desc->ndim)) {
        return INFINIOP_STATUS_BAD_TENSOR_SHAPE;
    }
    int ndim = static_cast<int>(c_desc->ndim);
    infiniDtype_t dtype = c_desc->dtype;

    if (dtype != INFINI_DTYPE_F16 && dtype != INFINI_DTYPE_F32) {
        return INFINIOP_STATUS_BAD_TENSOR_DTYPE;
    }

    if (a_desc->strides[ndim - 1] != 1 || b_desc->strides[ndim - 1] != 1 || c_desc->strides[ndim - 1] != 1) {
        return INFINIOP_STATUS_BAD_TENSOR_STRIDES;
    }
    for (size_t i = 0; i < c_desc->ndim; i++) {
        if (c_desc->shape[i] != a_desc->shape[i] || c_desc->shape[i] != b_desc->shape[i]) {
            return INFINIOP_STATUS_BAD_TENSOR_SHAPE;
        }
    }


    if (a_desc->dtype != dtype || b_desc->dtype != dtype) {
        return INFINIOP_STATUS_BAD_PARAM;
    }
    int *shape = new int[ndim];
    int64_t *strides_a = new int64_t[ndim];
    int64_t *strides_b = new int64_t[ndim];
    int64_t *strides_c = new int64_t[ndim];

    for (int i = 0; i < ndim; i++) {
        shape[i] = static_cast<int>(c_desc->shape[i]);
    }
    for (int i = 0; i < ndim; i++) {
        strides_a[i] = static_cast<int64_t>(a_desc->strides[i]);
    }
    for (int i = 0; i < ndim; i++) {
        strides_b[i] = static_cast<int64_t>(b_desc->strides[i]);
    }
    for (int i = 0; i < ndim; i++) {
        strides_c[i] = static_cast<int64_t>(c_desc->strides[i]);
    }
    char *tmpDevice;
    CNRT_CHECK(cnrtMalloc((void **) &tmpDevice, ndim * sizeof(int) + 3 * ndim * sizeof(int64_t)));
    char *tmpStrides = tmpDevice + ndim * sizeof(int);
    int *mlu_shape = (int *) tmpDevice;
    int64_t *mlu_strides_a = (int64_t *) tmpStrides;
    int64_t *mlu_strides_b = mlu_strides_a + ndim;
    int64_t *mlu_strides_c = mlu_strides_a + 2 * ndim;
    CNRT_CHECK(cnrtMemcpy(mlu_shape, shape, ndim * sizeof(int), cnrtMemcpyHostToDev));
    CNRT_CHECK(cnrtMemcpy(mlu_strides_a, strides_a, ndim * sizeof(int64_t), cnrtMemcpyHostToDev));
    CNRT_CHECK(cnrtMemcpy(mlu_strides_b, strides_b, ndim * sizeof(int64_t), cnrtMemcpyHostToDev));
    CNRT_CHECK(cnrtMemcpy(mlu_strides_c, strides_c, ndim * sizeof(int64_t), cnrtMemcpyHostToDev));

    *desc_ptr = new InfiniopSwiGLUBangDescriptor{handle->device,
                                                 handle->device_id,
                                                 dtype,
                                                 ndim,
                                                 mlu_shape,
                                                 mlu_strides_a,
                                                 mlu_strides_b,
                                                 mlu_strides_c};
    return INFINIOP_STATUS_SUCCESS;
}

infiniopStatus_t bangDestroySwiGLUDescriptor(infiniopSwiGLUBangDescriptor_t desc) {
    cnrtFree(desc->shape);
    delete desc;
    return INFINIOP_STATUS_SUCCESS;
}
