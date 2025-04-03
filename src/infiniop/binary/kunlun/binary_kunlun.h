#ifndef __INFINIOP_BINARY_KUNLUN_H__
#define __INFINIOP_BINARY_KUNLUN_H__

#include "../../devices/kunlun/common_kunlun.h"
#include "../binary.h"
#include "xpu/kernel/xtdk.h"
// #include "xpu/kernel/xtdk_math.h"
#include <utility>

namespace op::common_kunlun {

namespace binary_op {

// Perform binary computation when inputs and the output can have different dtypes
template <typename Tc, typename Ta, typename Tb, typename BinaryOp, typename... Args>
void calculate(op::binary::BinaryInfo info, void *c, const void *a, const void *b, Args &&...args) {
    auto a_ = reinterpret_cast<const Ta *>(a);
    auto b_ = reinterpret_cast<const Tb *>(b);
    auto c_ = reinterpret_cast<Tc *>(c);
    ptrdiff_t data_size = info.c_data_size;

    constexpr int byte_size = std::max(sizeof(Tc), std::max(sizeof(Ta), sizeof(Tb)));
    constexpr int buf_size = 4 * 1024 / byte_size; // 保证所有内存加起来不超过16kB
    __local__ Ta a_local[buf_size];
    __local__ Tb b_local[buf_size];
    __local__ Tc c_local[buf_size];

    int remain = data_size % buf_size;
    int repeat = (data_size - remain) / buf_size;

    for (int r = 0; r < repeat + (remain > 0 ? 1 : 0); r++) {
        int read_len = (r < repeat ? buf_size : remain);
        if (info.contiguous) {
            GM2LM(a_ + r * buf_size, a_local, read_len * sizeof(Ta));
            GM2LM(b_ + r * buf_size, b_local, read_len * sizeof(Tb));
            GM2LM(c_ + r * buf_size, c_local, read_len * sizeof(Tc));

            for (int i = 0; i < read_len; i++) {
                c_local[i] = BinaryOp{}(a_local[i], b_local[i], std::forward<Args>(args)...);
            }
            mfence();
            LM2GM(c_local, c_ + r * buf_size, read_len * sizeof(Tc));
        } else {
            for (int i = r * buf_size; i < r * buf_size + read_len; i++) {
                int a_index = info.broadcasted ? op::common_kunlun::indexToReducedOffset(i, info.ndim, info.c_strides.data(), info.a_strides.data()) : op::common_kunlun::indexToOffset(i, info.ndim, info.a_shape.data(), info.a_strides.data());
                int b_index = info.broadcasted ? op::common_kunlun::indexToReducedOffset(i, info.ndim, info.c_strides.data(), info.b_strides.data()) : op::common_kunlun::indexToOffset(i, info.ndim, info.b_shape.data(), info.b_strides.data());
                int c_index = op::common_kunlun::indexToOffset(i, info.ndim, info.c_shape.data(), info.c_strides.data());

                c_[c_index] = BinaryOp{}(a_[a_index], b_[b_index], std::forward<Args>(args)...);
            }
        }
    }
}

// Perform binary computation when all inputs and the output share the same dtype
template <typename Tdata, typename BinaryOp, typename... Args>
void calculate(op::binary::BinaryInfo info, void *c, const void *a, const void *b, Args &&...args) {
    auto a_ = reinterpret_cast<const Tdata *>(a);
    auto b_ = reinterpret_cast<const Tdata *>(b);
    auto c_ = reinterpret_cast<Tdata *>(c);
    ptrdiff_t data_size = info.c_data_size;

    constexpr int buf_size = 4 * 1024 / sizeof(Tdata); // 保证所有内存加起来不超过16kB
    __local__ Tdata a_local[buf_size];
    __local__ Tdata b_local[buf_size];
    __local__ Tdata c_local[buf_size];

    int remain = data_size % buf_size;
    int repeat = (data_size - remain) / buf_size;

    for (int r = 0; r < repeat + (remain > 0 ? 1 : 0); r++) {
        int read_len = (r < repeat ? buf_size : remain);
        if (info.contiguous) {
            GM2LM(a_ + r * buf_size, a_local, read_len * sizeof(Tdata));
            GM2LM(b_ + r * buf_size, b_local, read_len * sizeof(Tdata));
            GM2LM(c_ + r * buf_size, c_local, read_len * sizeof(Tdata));

            for (int i = 0; i < read_len; i++) {
                c_local[i] = BinaryOp{}(a_local[i], b_local[i], std::forward<Args>(args)...);
            }
            mfence();
            LM2GM(c_local, c_ + r * buf_size, read_len * sizeof(Tdata));
        } else {
            for (int i = r * buf_size; i < r * buf_size + read_len; i++) {
                int a_index = info.broadcasted ? op::common_kunlun::indexToReducedOffset(i, info.ndim, info.c_strides.data(), info.a_strides.data()) : op::common_kunlun::indexToOffset(i, info.ndim, info.a_shape.data(), info.a_strides.data());
                int b_index = info.broadcasted ? op::common_kunlun::indexToReducedOffset(i, info.ndim, info.c_strides.data(), info.b_strides.data()) : op::common_kunlun::indexToOffset(i, info.ndim, info.b_shape.data(), info.b_strides.data());
                int c_index = op::common_kunlun::indexToOffset(i, info.ndim, info.c_shape.data(), info.c_strides.data());

                c_[c_index] = BinaryOp{}(a_[a_index], b_[b_index], std::forward<Args>(args)...);
            }
        }
    }
}

} // namespace binary_op
} // namespace op::common_kunlun

#endif // __INFINIOP_BINARY_KUNLUN_H__
