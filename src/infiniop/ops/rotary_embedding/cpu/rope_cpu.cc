#include "rope_cpu.h"
#include "../../../devices/cpu/common_cpu.h"
#include <cmath>
#include <cstdlib>

infiniopStatus_t cpuCreateRoPEDescriptor(
    infiniopCpuHandle_t handle,
    infiniopRoPECpuDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t t_desc,
    infiniopTensorDescriptor_t pos_desc,
    infiniopTensorDescriptor_t sin_desc,
    infiniopTensorDescriptor_t cos_desc) {
    auto const ty_t = t_desc->dtype,
               ty_pos = pos_desc->dtype;

    // Check dtypes

    constexpr infiniDtype_t SUPPORTED_TY_T[] = {
        INFINI_DTYPE_F16,
        INFINI_DTYPE_F32,
        INFINI_DTYPE_F64,
    };
    constexpr infiniDtype_t SUPPORTED_TY_POS[] = {
        INFINI_DTYPE_U8,
        INFINI_DTYPE_U16,
        INFINI_DTYPE_U32,
        INFINI_DTYPE_U64,
    };
    auto supported_t = false,
         supported_pos = false;
    for (auto supported_dtype : SUPPORTED_TY_T) {
        if (ty_t == supported_dtype) {
            supported_t = true;
            break;
        }
    }
    for (auto supported_dtype : SUPPORTED_TY_POS) {
        if (ty_pos == supported_dtype) {
            supported_pos = true;
            break;
        }
    }
    if (!supported_t || !supported_pos || sin_desc->dtype != ty_t || cos_desc->dtype != ty_t) {
        return INFINIOP_STATUS_BAD_TENSOR_DTYPE;
    }

    // Check shapes

    if (t_desc->ndim != 3 || pos_desc->ndim != 1 || sin_desc->ndim != 2 || cos_desc->ndim != 2) {
        return INFINIOP_STATUS_BAD_TENSOR_SHAPE;
    }
    auto const nt = t_desc->shape[0],
               nh = t_desc->shape[1],
               dh = t_desc->shape[2],
               np = pos_desc->shape[0],
               nsin = sin_desc->shape[0],
               dh_sin = sin_desc->shape[1],
               ncos = cos_desc->shape[0],
               dh_cos = cos_desc->shape[1];
    if (nt != np || dh_sin != dh || dh_cos != dh || dh % 2 != 0) {
        // rope 的最后一维要视作 [T;2] 处理
        return INFINIOP_STATUS_BAD_TENSOR_SHAPE;
    }
    if (t_desc->strides[2] != 1) {
        // hidden state 最后一维必须连续才能视作 [T;2] 处理
        return INFINIOP_STATUS_BAD_TENSOR_STRIDES;
    }

    // Create descriptor

    *desc_ptr = new RoPECpuDescriptor{
        INFINI_DEVICE_CPU,
        ty_t,
        ty_pos,
        nt,
        nh,
        dh,
        nsin,
        ncos,
        t_desc->strides[0],
        t_desc->strides[1],
        pos_desc->strides[0],
        sin_desc->strides[0],
        cos_desc->strides[0],
        sin_desc->strides[1],
        cos_desc->strides[1],
    };
    return INFINIOP_STATUS_SUCCESS;
}

size_t read_pos(infiniDtype_t ty, uint8_t const *pos) {
    switch (ty) {
    case INFINI_DTYPE_U8:
        return *pos;
    case INFINI_DTYPE_U16:
        return *reinterpret_cast<uint16_t const *>(pos);
    case INFINI_DTYPE_U32:
        return *reinterpret_cast<uint32_t const *>(pos);
    case INFINI_DTYPE_U64:
        return *reinterpret_cast<uint64_t const *>(pos);
    default:
        // unreachable
        std::abort();
    }
}

template <class T>
void rope_ptr(uint8_t *t, uint8_t const *sin, uint8_t const *cos) {
    auto &a = reinterpret_cast<T *>(t)[0],
         &b = reinterpret_cast<T *>(t)[1];
    auto sin_ = *reinterpret_cast<T const *>(sin),
         cos_ = *reinterpret_cast<T const *>(cos);
    auto a_ = a,
         b_ = b;
    a = a_ * cos_ - b_ * sin_;
    b = a_ * sin_ + b_ * cos_;
}

template <>
void rope_ptr<uint16_t>(uint8_t *t, uint8_t const *sin, uint8_t const *cos) {
    auto &a = reinterpret_cast<uint16_t *>(t)[0],
         &b = reinterpret_cast<uint16_t *>(t)[1];
    auto sin_ = f16_to_f32(*reinterpret_cast<uint16_t const *>(sin)),
         cos_ = f16_to_f32(*reinterpret_cast<uint16_t const *>(cos));
    auto a_ = f16_to_f32(a),
         b_ = f16_to_f32(b);
    a = f32_to_f16(a_ * cos_ - b_ * sin_);
    b = f32_to_f16(a_ * sin_ + b_ * cos_);
}

infiniopStatus_t cpuRoPE(
    infiniopRoPECpuDescriptor_t desc,
    void *t,
    void const *pos,
    void const *sin,
    void const *cos) {

    auto const t_ = reinterpret_cast<uint8_t *>(t);
    auto const pos_ = reinterpret_cast<uint8_t const *>(pos);
    auto const sin_ = reinterpret_cast<uint8_t const *>(sin);
    auto const cos_ = reinterpret_cast<uint8_t const *>(cos);
    auto const unit_t = infiniSizeof(desc->ty_t),
               unit_pos = infiniSizeof(desc->ty_pos);

    for (size_t i = 0; i < desc->nt; ++i) {
        auto const t__ = t_ + i * desc->s_nt * unit_t;
        auto const pos__ = read_pos(desc->ty_pos, pos_ + i * desc->s_np * unit_pos);
        auto const sin__ = sin_ + pos__ * desc->s_nsin * unit_t;
        auto const cos__ = cos_ + pos__ * desc->s_ncos * unit_t;
        if (pos__ >= desc->nsin || pos__ >= desc->ncos) {
            // sin cos 表容量不足
            return INFINIOP_STATUS_BAD_TENSOR_SHAPE;
        }

        for (size_t j = 0; j < desc->nh; ++j) {
            for (size_t k = 0; k < desc->dh / 2; ++k) {
                auto const t___ = t__ + (j * desc->s_np + k * 2) * unit_t;
                auto const sin___ = sin__ + k * desc->s_dsin * unit_t;
                auto const cos___ = cos__ + k * desc->s_dcos * unit_t;
                switch (desc->ty_t) {
                case INFINI_DTYPE_F16:
                    rope_ptr<uint16_t>(t___, sin___, cos___);
                    break;
                case INFINI_DTYPE_F32:
                    rope_ptr<float>(t___, sin___, cos___);
                    break;
                case INFINI_DTYPE_F64:
                    rope_ptr<double>(t___, sin___, cos___);
                    break;
                default:
                    // unreachable
                    std::abort();
                }
            }
        }
    }
    return INFINIOP_STATUS_SUCCESS;
}

infiniopStatus_t cpuDestroyRoPEDescriptor(
    infiniopRoPECpuDescriptor_t desc) {
    delete desc;
    return INFINIOP_STATUS_SUCCESS;
}
