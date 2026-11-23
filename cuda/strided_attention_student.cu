#include <cassert>
#include <cmath>
#include <torch/extension.h>
#include <cuda_fp16.h>

#include <c10/cuda/CUDAException.h>

// CUDA kernel for the forward pass of strided attention
//
// T: data type (e.g., float, double, half)
//
// Computes: output = Softmax((Q * K_strided^T) / sqrt(head_dim)) * V_strided
//
// Grid/Block Dimensions:
// - gridDim.x: batch_size * num_heads
// - gridDim.y: seq_len (for queries)
// - blockDim.x: 256. Represents threads per query token.
//

template <typename T>
__device__ __forceinline__ T warp_max(T val) {
    using AccT = typename std::conditional<std::is_same<T, double>::value, double, float>::type;
    AccT val_cpy = static_cast<AccT>(val);
    for (int offset = 16; offset > 0; offset >>= 1) {
        AccT other = __shfl_down_sync(0xffffffff, val_cpy, offset);
        val_cpy = max(other, val_cpy); 
    }
    return static_cast<T>(val_cpy);
}

template <typename T>
__device__ __forceinline__ T warp_sum(T val) {
    using AccT = typename std::conditional<std::is_same<T, double>::value, double, float>::type;
    AccT val_cpy = static_cast<AccT>(val);
    for (int offset = 16; offset > 0; offset >>= 1) {
        val_cpy += __shfl_down_sync(0xffffffff, val_cpy, offset);
    }
    return static_cast<T>(val_cpy);
}

template <typename T>
__global__ void strided_attention_forward_kernel(
    const T* __restrict__ q_ptr,
    const T* __restrict__ k_ptr,
    const T* __restrict__ v_ptr,
    T* __restrict__ output_ptr,
    const int batch_size,
    const int num_heads,
    const int seq_len,
    const int head_dim,
    const int stride) {

    // =================================================================
    // TODO: IMPLEMENT YOUR STRIDED ATTENTION KERNEL HERE
    // =================================================================
    //
    // Hints:
    // 1. Calculate thread/block indices to figure out which output element
    //    this thread block is responsible for.
    //    - `blockIdx.x` can map to (batch, head).
    //    - `blockIdx.y` can map to the query token index (`i`).
    //    - `threadIdx.x` is the thread within the block.

    // 2. Load the query vector `q_i` for the current query token `i`.
    //    Since all threads in the block need it, you can load it into shared memory.

    // 3. Compute dot products `Q_i * K_j^T` for all `j` in the stride.
    //    - Parallelize the dot product calculation across threads in the block.
    //    - Each thread can compute a partial dot product and then you can use
    //      a parallel reduction (e.g., using shared memory and `__syncthreads()`)
    //      to get the final score.

    // 4. Compute Softmax.
    //    - Find the maximum score among the strided scores (for numerical stability). This is another parallel reduction.
    //    - Compute the exponential of each score (subtracting the max).
    //    - Compute the sum of the exponentials (another parallel reduction).
    //    - Normalize to get the final attention probabilities.

    // 5. Compute the weighted sum of value vectors `V_j`.
    //    - Use the attention probabilities to weight the `v_j` vectors.
    //    - This is another parallel dot-product-like operation. Each thread can
    //      compute a part of the final output vector component.

    // 6. Write the final output vector to `output_ptr`.

    // Example of getting indices (you'll need to expand on this):
    int batch_idx = blockIdx.x / num_heads;
    int head_idx = blockIdx.x % num_heads;
    int query_idx = blockIdx.y; // This is 'i' in the outer loop
    int idx = threadIdx.x; // thread index in a block

    // Warp
    int warp_id = idx / warpSize;
    int lane_id = idx % warpSize;
    int num_warps = blockDim.x / warpSize;

    // Placeholder: just copy query to output to have a compilable stub
    long bh_offset = (long)batch_idx * num_heads * seq_len * head_dim +
                    (long)head_idx * seq_len * head_dim;
    long q_offset = bh_offset + (long)query_idx * head_dim;
    
    long output_offset = q_offset;

    __shared__ T q[256];
    __shared__ T scores[256];
    __shared__ T reduced_max[32];
    __shared__ T reduced_sum[32];

    // assert(blockDim.x <= 256);
    // assert(32 * warpSize > blockDim.x);

    using AccT = typename std::conditional<std::is_same<T, double>::value, double, float>::type;

    const int strided_seq_len = (seq_len + stride - 1) / stride;
    const T scale = static_cast<T>(1.0 / sqrt(static_cast<double>(head_dim)));

    // load q_i
    for (int i = idx; i < head_dim; i += blockDim.x) {
        q[i] = q_ptr[q_offset + i];
    }

    __syncthreads();
    
    // scores (strided) + warp
    for (int i = warp_id; i < strided_seq_len; i += num_warps) {
        T sum = static_cast<T>(0.0);
        long k_offset = bh_offset + (long)i * stride * head_dim;

        for (int j = lane_id; j < head_dim; j += warpSize) {
            sum += q[j] * k_ptr[k_offset + j];
        }

        sum = warp_sum(sum);
        if (lane_id == 0) {
            scores[i] = sum * scale;
        }
    }

    __syncthreads();

    // max_scores
    {
        T val = idx < strided_seq_len ? scores[idx] : static_cast<T>(-INFINITY);
        val = warp_max(val);
        if (lane_id == 0) {
            reduced_max[warp_id] = val;
        }
    }

    __syncthreads();

    // max_scores continue
    if (warp_id == 0) {
        T val = lane_id < num_warps ? reduced_max[lane_id] : static_cast<T>(-INFINITY);
        val = warp_max(val);
        if (lane_id == 0) {
            reduced_max[warp_id] = val; // reduced_max[0] is maxima
        }
    }
    // T max_val = reduced_max[0];

    __syncthreads();

    // scores -> shifted -> exps
    for (int i = idx; i < strided_seq_len; i += blockDim.x) {
        scores[i] = static_cast<T>(exp(static_cast<AccT>(scores[i] - reduced_max[0])));
    }

    __syncthreads();

    // sums
    {
        T val = idx < strided_seq_len ? scores[idx] : static_cast<T>(0.0);
        val = warp_sum(val);
        if (lane_id == 0) {
            reduced_sum[warp_id] = val;
        }
    }

    __syncthreads();

    // sums continue
    if (warp_id == 0) {
        T val = lane_id < num_warps ? reduced_sum[lane_id] : static_cast<T>(0.0);
        val = warp_sum(val);
        if (lane_id == 0) {
            reduced_sum[0] = val + static_cast<T>(1e-6);
        }
    }

    __syncthreads();

    // weights = exps / sums
    for (int i = idx; i < strided_seq_len; i += blockDim.x) {
        scores[i] /= reduced_sum[0];
    } 

    __syncthreads();

    // write to output
    for (int i = idx; i < head_dim; i += blockDim.x) {
        T output = static_cast<T>(0.0);
        for (int j = 0; j < strided_seq_len; ++j) {
            long v_offset = bh_offset + (long)j * stride * head_dim;
            output += scores[j] * v_ptr[v_offset + i];
        }
        output_ptr[output_offset + i] = output;
    }

    // warp version, even slower, because stride >> 1
    // for (int d = warp_id; d < head_dim; d += num_warps) {
    //     T sum = 0.0;
    //     for (int j = lane_id; j < strided_seq_len; j += warpSize) {
    //         long v_offset = bh_offset + (long)j * stride * head_dim;
    //         sum += scores[j] * v_ptr[v_offset + d];
    //     }
    //     sum = warp_sum(sum);
    //     if (lane_id == 0) {
    //         output_ptr[output_offset + d] = sum;
    //     }
    // }
}


// C++ function that dispatches the CUDA kernel
torch::Tensor strided_attention_forward_cuda(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    int stride) {

    // Validate inputs
    TORCH_CHECK(q.is_cuda(), "Input tensor Q must be on a CUDA device");
    TORCH_CHECK(q.is_contiguous(), "Input tensor Q must be contiguous");
    TORCH_CHECK(k.is_contiguous(), "Input tensor K must be contiguous");
    TORCH_CHECK(v.is_contiguous(), "Input tensor V must be contiguous");

    const auto batch_size = q.size(0);
    const auto num_heads = q.size(1);
    const auto seq_len = q.size(2);
    const auto head_dim = q.size(3);

    // Create an output tensor of the same shape as Q
    auto output = torch::empty_like(q);

    // Define grid and block dimensions
    // Grid: One block per (batch, head, query_token)
    dim3 gridDim(batch_size * num_heads, seq_len);
    // Block: Threads to parallelize the work for a single query token
    dim3 blockDim(256);

    // Dispatch the kernel based on the data type
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(q.scalar_type(), "strided_attention_forward", ([&] {
        strided_attention_forward_kernel<scalar_t><<<gridDim, blockDim>>>(
            q.data_ptr<scalar_t>(),
            k.data_ptr<scalar_t>(),
            v.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            batch_size,
            num_heads,
            seq_len,
            head_dim,
            stride
        );
    }));

    // Check for any CUDA errors
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return output;
}


// Bind the C++ function to a Python module
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("strided_attention_forward", &strided_attention_forward_cuda, "Strided Attention Forward (CUDA)");
}
