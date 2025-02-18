#include "random_sample_cpu.h"
#include "../../../devices/cpu/common_cpu.h"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <vector>

infiniopStatus_t cpuCreateRandomSampleDescriptor(
    infiniopCpuHandle_t handle,
    infiniopRandomSampleCpuDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t result_desc,
    infiniopTensorDescriptor_t probs_desc) {
    auto const ty_i = result_desc->dtype,
               ty_p = probs_desc->dtype;

    // Check dtypes

    constexpr infiniDtype_t SUPPORTED_TY_I[] = {
        INFINI_DTYPE_U8,
        INFINI_DTYPE_U16,
        INFINI_DTYPE_U32,
        INFINI_DTYPE_U64,
    };
    constexpr infiniDtype_t SUPPORTED_TY_P[] = {
        INFINI_DTYPE_F16,
        INFINI_DTYPE_F32,
        INFINI_DTYPE_F64,
    };
    auto supported_i = false,
         supported_p = false;
    for (auto supported_dtype : SUPPORTED_TY_I) {
        if (ty_i == supported_dtype) {
            supported_i = true;
            break;
        }
    }
    for (auto supported_dtype : SUPPORTED_TY_P) {
        if (ty_p == supported_dtype) {
            supported_p = true;
            break;
        }
    }
    if (!supported_i || !supported_p) {
        return INFINIOP_STATUS_BAD_TENSOR_DTYPE;
    }

    // Check shapes

    if (result_desc->ndim != 0 || probs_desc->ndim != 1) {
        return INFINIOP_STATUS_BAD_TENSOR_SHAPE;
    }
    auto const n = probs_desc->shape[0];
    auto const s = probs_desc->strides[0];

    // Create descriptor

    *desc_ptr = new RandomSampleCpuDescriptor{
        INFINI_DEVICE_CPU,
        ty_i,
        ty_p,
        n,
        s,
    };
    return INFINIOP_STATUS_SUCCESS;
}

template <class Tidx, class Tval>
void argmax(void *result, void const *probs, uint64_t n, int64_t s) {
    auto ptr = reinterpret_cast<Tval const *>(probs);
    auto idx = reinterpret_cast<Tidx *>(result);
    *idx = 0;

    auto max_val = ptr[0];
    for (size_t i = 0; i < n; i++) {
        if (auto val = ptr[i * s]; val > max_val) {
            max_val = val;
            *idx = static_cast<Tidx>(i);
        }
    }
}

struct KVPair {
    uint32_t idx;
    float val;

    bool operator<(const KVPair &other) const {
        return val < other.val;
    }
};

template <class Tval>
float read_float(Tval const *ptr, size_t i, int64_t s) {
    return static_cast<float>(ptr[i * s]);
}

template <>
float read_float<uint16_t>(uint16_t const *ptr, size_t i, int64_t s) {
    return f16_to_f32(ptr[i * s]);
}

template <class Tidx, class Tval>
void random(void *result, void const *probs, uint64_t n, int64_t s,
            float random_val, float topp, int topk, float temperature) {
    auto ptr = reinterpret_cast<Tval const *>(probs);
    auto idx = reinterpret_cast<Tidx *>(result);
    // build & sort
    std::vector<KVPair> pairs(n);
    for (size_t i = 0; i < n; i++) {
        pairs[i] = {static_cast<uint32_t>(i), read_float<Tval>(ptr, i, s)};
    }
    std::sort(pairs.begin(), pairs.end());
    // softmax & sum
    auto const max_val = pairs[0].val;
    pairs[0].val = 1;
    for (size_t i = 1; i < n; i++) {
        pairs[i].val = pairs[i - 1].val + std::exp((pairs[i].val - max_val) / temperature);
    }
    // topk & topp & limit
    auto const pk = pairs[std::min(static_cast<uint64_t>(topk), n)].val,
               pp = pairs[n - 1].val * topp,
               plimit = random_val * std::min(pk, pp);
    // sample
    for (size_t i = 0; i < n; i++) {
        if (pairs[i].val >= plimit) {
            *idx = pairs[i].idx;
            break;
        }
    }
}

template <class Tidx, class Tval>
void switch_f(
    infiniopRandomSampleCpuDescriptor_t desc,
    void *result, void const *probs,
    float random_val, float topp, int topk, float temperature) {
    if (random_val == 0 || topp == 0 || topk == 1 || temperature == 0) {
        argmax<Tidx, Tval>(result, probs, desc->n, desc->s);
    } else {
        random<Tidx, Tval>(result, probs, desc->n, desc->s, random_val, topp, topk, temperature);
    }
}

template <class Tidx>
void switch_val(
    infiniopRandomSampleCpuDescriptor_t desc,
    void *result, void const *probs,
    float random_val, float topp, int topk, float temperature) {
    switch (desc->ty_p) {
    case INFINI_DTYPE_F16:
        switch_f<Tidx, uint16_t>(desc, result, probs, random_val, topp, topk, temperature);
        break;
    case INFINI_DTYPE_F32:
        switch_f<Tidx, float>(desc, result, probs, random_val, topp, topk, temperature);
        break;
    case INFINI_DTYPE_F64:
        switch_f<Tidx, double>(desc, result, probs, random_val, topp, topk, temperature);
        break;
    default:
        // unreachable
        std::abort();
    }
}

void switch_idx(
    infiniopRandomSampleCpuDescriptor_t desc,
    void *result, void const *probs,
    float random_val, float topp, int topk, float temperature) {
    switch (desc->ty_i) {
    case INFINI_DTYPE_U8:
        switch_val<uint8_t>(desc, result, probs, random_val, topp, topk, temperature);
        break;
    case INFINI_DTYPE_U16:
        switch_val<uint16_t>(desc, result, probs, random_val, topp, topk, temperature);
        break;
    case INFINI_DTYPE_U32:
        switch_val<uint32_t>(desc, result, probs, random_val, topp, topk, temperature);
        break;
    case INFINI_DTYPE_U64:
        switch_val<uint64_t>(desc, result, probs, random_val, topp, topk, temperature);
        break;
    default:
        // unreachable
        std::abort();
    }
}

infiniopStatus_t cpuRandomSample(
    infiniopRandomSampleCpuDescriptor_t desc,
    void *result, void const *probs,
    float random_val,
    float topp, int topk, float temperature) {
    switch_idx(desc, result, probs, random_val, topp, topk, temperature);
    return INFINIOP_STATUS_SUCCESS;
}

infiniopStatus_t cpuDestroyRandomSampleDescriptor(
    infiniopRandomSampleCpuDescriptor_t desc) {
    delete desc;
    return INFINIOP_STATUS_SUCCESS;
}
