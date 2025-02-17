#include "../../utils.h"
#include "./rms_norm_cuda.cuh"
#include <cub/block/block_load.cuh>
#include <cub/block/block_reduce.cuh>

// assert BLOCK_SIZE >= blockDim.x
template<unsigned int BLOCK_SIZE, class Tdata, class Wdata>
static __global__ void rms_norm_padding(
    Tdata *__restrict__ o_,
    unsigned int const stride_y,
    Tdata const *__restrict__ x_,
    unsigned int const stride_x,
    Wdata const *__restrict__ w_,
    float const epsilon) {
    auto y = o_ + blockIdx.x * stride_y + threadIdx.x;
    auto x = x_[blockIdx.x * stride_x + threadIdx.x];
    auto w = w_[threadIdx.x];

    using BlockOp = cub::BlockReduce<float, BLOCK_SIZE>;
    __shared__ typename BlockOp::TempStorage temp_storage;
    auto acc = BlockOp(temp_storage).Reduce(x * x, cub::Sum());

    __shared__ Tdata rms;
    if (threadIdx.x == 0) {
        rms = Tdata(rsqrtf(acc / float(blockDim.x) + epsilon));
    }
    __syncthreads();

    *y = rms * x * (Tdata) w;
}

template<unsigned int BLOCK_SIZE, unsigned int ITEMS_PER_THREAD, class Tdata, class Wdata>
static __global__ void rms_norm_folding(
    Tdata *__restrict__ y,
    unsigned int const stride_y,
    Tdata const *__restrict__ x,
    unsigned int const stride_x,
    Wdata const *__restrict__ w,
    float const epsilon,
    unsigned int const items_size) {
    y += blockIdx.x * stride_y;
    x += blockIdx.x * stride_x;

    float thread_data[ITEMS_PER_THREAD];
    {
        using BlockOp = cub::BlockLoad<float, BLOCK_SIZE, ITEMS_PER_THREAD>;
        __shared__ typename BlockOp::TempStorage temp_storage;
        BlockOp(temp_storage).Load(x, thread_data, items_size, 0.f);
    }

    float squared[ITEMS_PER_THREAD];
#pragma unroll
    for (unsigned int i = 0; i < ITEMS_PER_THREAD; ++i) {
        squared[i] = thread_data[i] * thread_data[i];
    }

    float acc;
    {
        using BlockOp = cub::BlockReduce<float, BLOCK_SIZE>;
        __shared__ typename BlockOp::TempStorage temp_storage;
        acc = BlockOp(temp_storage).Reduce(squared, cub::Sum());
    }

    __shared__ Tdata rms;
    if (threadIdx.x == 0) {
        rms = Tdata(rsqrtf(acc / float(items_size) + epsilon));
    }
    __syncthreads();

#pragma unroll
    for (unsigned int i = 0; i < ITEMS_PER_THREAD; ++i) {
        if (auto j = i + threadIdx.x * ITEMS_PER_THREAD; j < items_size) {
            y[j] = Tdata(float(rms) * float(thread_data[i]) * float(w[j]));
        }
    }
}

template<unsigned int BLOCK_SIZE, class Tdata, class Wdata>
static __global__ void rms_norm_standard(
    Tdata *__restrict__ y_,
    unsigned int const stride_y,
    Tdata const *__restrict__ x_,
    unsigned int const stride_x,
    Wdata const *__restrict__ w,
    float const epsilon,
    unsigned int const d) {
    auto y = y_ + blockIdx.x * stride_y;
    auto x = x_ + blockIdx.x * stride_x;

    __shared__ float partial_sum[BLOCK_SIZE];

    float sum = 0.0f;
    for (int i = threadIdx.x; i < d; i += BLOCK_SIZE) {
        sum += float(x[i]) * float(x[i]);
    }

    partial_sum[threadIdx.x] = sum;
    __syncthreads();
    for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            partial_sum[threadIdx.x] += partial_sum[threadIdx.x + stride];
        }
        __syncthreads();
    }

    __shared__ Tdata rms;
    if (threadIdx.x == 0) {
        float row_sum = partial_sum[0];
        rms = Tdata(rsqrtf(row_sum / float(d) + epsilon));
    }
    __syncthreads();

    for (int i = threadIdx.x; i < d; i += BLOCK_SIZE) {
        y[i] = rms * x[i] * (Tdata) w[i];
    }
}

void rms_norm_nv_gpu_f16(infiniopRMSNormCudaDescriptor_t desc, void *y, void const *x, void const *w, void *stream) {
    auto n = desc->n, d = desc->d;
    auto y_ = reinterpret_cast<half *>(y);
    auto x_ = reinterpret_cast<half const *>(x);
    auto epsilon = desc->epsilon;

    // Get strides in terms of elements
    auto stride_y = desc->stride_y;
    auto stride_x = desc->stride_x;

    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    unsigned int items_per_thread = ROUND_UP_DIV(d, MAX_THREADS_PER_BLOCK);
    auto w_datatype = desc->w_datatype;
    if (w_datatype == INFINI_DTYPE_F16) {
        auto w_ = reinterpret_cast<half const *>(w);
        if (items_per_thread == 1) {
            rms_norm_padding<MAX_THREADS_PER_BLOCK, half, half>
                <<<n, d, 0, cuda_stream>>>(y_, stride_y, x_, stride_x, w_, epsilon);
        } else if (items_per_thread <= 16) {
            rms_norm_folding<MAX_THREADS_PER_BLOCK, 16, half, half>
                <<<n, MAX_THREADS_PER_BLOCK, 0, cuda_stream>>>(y_, stride_y, x_, stride_x, w_, epsilon, d);
        } else {
            rms_norm_standard<MAX_THREADS_PER_BLOCK, half, half>
                <<<n, MAX_THREADS_PER_BLOCK, 0, cuda_stream>>>(y_, stride_y, x_, stride_x, w_, epsilon, d);
        }
    } else {
        auto w_ = reinterpret_cast<float const *>(w);
        if (items_per_thread == 1) {
            rms_norm_padding<MAX_THREADS_PER_BLOCK, half, float>
                <<<n, d, 0, cuda_stream>>>(y_, stride_y, x_, stride_x, w_, epsilon);
        } else if (items_per_thread <= 16) {
            rms_norm_folding<MAX_THREADS_PER_BLOCK, 16, half, float>
                <<<n, MAX_THREADS_PER_BLOCK, 0, cuda_stream>>>(y_, stride_y, x_, stride_x, w_, epsilon, d);
        } else {
            rms_norm_standard<MAX_THREADS_PER_BLOCK, half, float>
                <<<n, MAX_THREADS_PER_BLOCK, 0, cuda_stream>>>(y_, stride_y, x_, stride_x, w_, epsilon, d);
        }
    }
}

infiniopStatus_t cudaRMSNorm(infiniopRMSNormCudaDescriptor_t desc,
                             void *workspace,
                             uint64_t workspace_size,
                             void *y, void const *x, void const *w,
                             void *stream) {
    if (cudaSetDevice(desc->device_id) != cudaSuccess) {
        return INFINIOP_STATUS_BAD_DEVICE;
    }
    if (desc->dtype == INFINI_DTYPE_F16) {
        rms_norm_nv_gpu_f16(desc, y, x, w, stream);
        return INFINIOP_STATUS_SUCCESS;
    }

    return INFINIOP_STATUS_BAD_TENSOR_DTYPE;
}
