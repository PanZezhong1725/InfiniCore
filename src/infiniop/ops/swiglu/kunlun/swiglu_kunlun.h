#ifndef __SWIGLU_KUNLUN_H__
#define __SWIGLU_KUNLUN_H__

#include "../../../binary/kunlun/binary_kunlun.h"

BINARY_DESCRIPTOR(swiglu, kunlun)

struct SwiGLUOp {
private:
    template <typename T>
    T sigmoid(const T &x) const {
        return 1 / (1 + std::exp(-x));
    }

public:
    template <typename T>
    T operator()(const T &up, const T &gate) const {
        return gate * sigmoid(gate) * up;
    }
};

#endif // __SWIGLU_KUNLUN_H__
