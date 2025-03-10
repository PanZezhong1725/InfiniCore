#ifndef __MATMUL_H__
#define __MATMUL_H__

#include "../../../utils.h"
#include "../../operator.h"

#define DESCRIPTOR(NAMESPACE)                             \
                                                          \
    namespace op::rearrange::NAMESPACE {                  \
    class Descriptor final : public InfiniopDescriptor {  \
        struct Opaque;                                    \
        Opaque *_opaque;                                  \
        utils::RearrangeMeta _meta;                       \
                                                          \
        Descriptor(                                       \
            utils::RearrangeMeta meta,                    \
            Opaque *opaque,                               \
            infiniDevice_t device_type,                   \
            int device_id)                                \
            : InfiniopDescriptor{device_type, device_id}, \
              _opaque(opaque),                            \
              _meta(meta) {}                              \
                                                          \
    public:                                               \
        ~Descriptor();                                    \
                                                          \
        static infiniStatus_t create(                     \
            infiniopHandle_t handle,                      \
            Descriptor **desc_ptr,                        \
            infiniopTensorDescriptor_t y_desc,            \
            infiniopTensorDescriptor_t x_desc);           \
                                                          \
        infiniStatus_t calculate(                         \
            void *y,                                      \
            const void *x,                                \
            void *stream) const;                          \
    };                                                    \
    }

#endif // __MATMUL_H__
