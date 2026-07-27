/*
 * Fused LayerNorm + residual + dropout.
 *
 * Replaces 3 separate kernels: LayerNorm -> Dropout -> Add.
 */

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

namespace qec {
    namespace {
        constexpr int kMaxThreads{256};
        constexpr int kMinThreads{32};
        constexpr float kEps{1e-5f};

        // ------------------------------------------------------------------
        // Warp-level reduction (inlined, no shared memory)
        // ------------------------------------------------------------------

        __device__ __forceinline__ float warp_reduce_sum(float val) {
            #pragma unroll
            for (int offset{16}; offset > 0; offset >>= 1) {
                val += __shfl_xor_sync(0xFFFFFFFF, val, offset);
            }
            return val;
        }

        // ------------------------------------------------------------------
        // Block-level dual reduction: reduces (sum, sum_sq) simultaneously.
        // Uses dynamic shared memory: first 32 floats for sum, next 32 for sum_sq.
        // ------------------------------------------------------------------

        __device__ __forceinline__ void block_reduce_dual(
            float& sum, float& sum_sq, float* smem
        ) {
            const int tid{static_cast<int>(threadIdx.x)};
            const int lane{tid % 32};
            const int warp_id{tid / 32};
            const int num_warps{static_cast<int>((blockDim.x + 31) / 32)};

            float* smem_sum{smem};
            float* smem_sq{smem + 32};

            sum = warp_reduce_sum(sum);
            sum_sq = warp_reduce_sum(sum_sq);

            if (lane == 0) {
                smem_sum[warp_id] = sum;
                smem_sq[warp_id] = sum_sq;
            }
            __syncthreads();

            if (warp_id == 0) {
                sum = (lane < num_warps) ? smem_sum[lane] : 0.0f;
                sum_sq = (lane < num_warps) ? smem_sq[lane] : 0.0f;
                sum = warp_reduce_sum(sum);
                sum_sq = warp_reduce_sum(sum_sq);
            }
        }

        // ------------------------------------------------------------------
        // Kernel: templated on training mode
        // ------------------------------------------------------------------

        // Dynamic shared memory layout:
        //   [0..63]:        block_reduce_dual scratch (2 × 32 floats)
        //   [64..64+H-1]:  gamma cache
        //   [64+H..64+2H-1]: beta cache
        //   [64+2H]:       mean
        //   [64+2H+1]:     inv_std

        template <bool Training>
        __global__ void __launch_bounds__(kMaxThreads)
        fused_layernorm_residual_dropout_kernel(
            const float* __restrict__ input,
            const float* __restrict__ residual,
            const float* __restrict__ gamma,
            const float* __restrict__ beta,
            float* __restrict__ output,
            const int N,
            const int H,
            const float dropout_p,
            const unsigned long long seed
        ) {
            const int row{static_cast<int>(blockIdx.x)};
            if (row >= N) return;

            extern __shared__ float smem[];
            float* reduce_scratch{smem};                    // 64 floats
            float* s_gamma{smem + 64};                      // H floats
            float* s_beta{smem + 64 + H};                   // H floats
            float* s_stats{smem + 64 + 2 * H};              // 2 floats: mean, inv_std

            const float* __restrict__ inp{input + row * H};
            const float* __restrict__ res{residual + row * H};
            float* __restrict__ out{output + row * H};

            const int tid{static_cast<int>(threadIdx.x)};
            const int stride{static_cast<int>(blockDim.x)};

            // --- Cooperative load of gamma/beta into shared memory ---
            for (int h{tid}; h < H; h += stride) {
                s_gamma[h] = gamma[h];
                s_beta[h] = beta[h];
            }

            // --- Pass 1: sum and sum_sq in a single read of the input row ---
            float local_sum{0.0f};
            float local_sum_sq{0.0f};
            for (int h{tid}; h < H; h += stride) {
                const float val{inp[h]};
                local_sum += val;
                local_sum_sq += val * val;
            }

            block_reduce_dual(local_sum, local_sum_sq, reduce_scratch);

            if (tid == 0) {
                const float mean{local_sum / static_cast<float>(H)};
                const float var{local_sum_sq / static_cast<float>(H) - mean * mean};
                s_stats[0] = mean;
                s_stats[1] = rsqrtf(var + kEps);
            }
            __syncthreads();

            const float mean{s_stats[0]};
            const float inv_std{s_stats[1]};

            // --- Pass 2: normalize, scale, (dropout if training), residual add ---
            [[maybe_unused]] curandStatePhilox4_32_10_t rng_state;
            [[maybe_unused]] float scale;

            if constexpr (Training) {
                if (dropout_p > 0.0f) {
                    curand_init(
                        seed,
                        static_cast<unsigned long long>(row) * H + tid,
                        0, &rng_state
                    );
                    scale = 1.0f / (1.0f - dropout_p);
                }
            }

            for (int h{tid}; h < H; h += stride) {
                float normalized{(inp[h] - mean) * inv_std};
                float scaled{normalized * s_gamma[h] + s_beta[h]};

                if constexpr (Training) {
                    if (dropout_p > 0.0f) {
                        const float r{curand_uniform(&rng_state)};
                        scaled = (r >= dropout_p) ? scaled * scale : 0.0f;
                    }
                }

                out[h] = scaled + res[h];
            }
        }

        // ------------------------------------------------------------------
        // Thread count and shared memory sizing
        // ------------------------------------------------------------------

        inline int select_threads(int H) noexcept {
            return std::max(kMinThreads, std::min(kMaxThreads, H));
        }

        inline size_t shared_mem_bytes(int H) noexcept {
            // 64 (reduce scratch) + H (gamma) + H (beta) + 2 (mean, inv_std)
            return static_cast<size_t>(64 + 2 * H + 2) * sizeof(float);
        }
    }  // anonymous namespace

    // ------------------------------------------------------------------
    // Host entry point
    // ------------------------------------------------------------------

    torch::Tensor fused_norm_residual_dropout(
        torch::Tensor input,
        torch::Tensor residual,
        torch::Tensor gamma,
        torch::Tensor beta,
        float dropout_p,
        bool training
    ) {
        TORCH_CHECK(
            input.is_cuda() && input.is_contiguous(),
            "input must be a contiguous CUDA tensor"
        );
        TORCH_CHECK(
            residual.is_cuda() && residual.is_contiguous(),
            "residual must be a contiguous CUDA tensor"
        );
        TORCH_CHECK(
            gamma.is_cuda() && gamma.is_contiguous(),
            "gamma must be a contiguous CUDA tensor"
        );
        TORCH_CHECK(
            beta.is_cuda() && beta.is_contiguous(),
            "beta must be a contiguous CUDA tensor"
        );

        const int N{static_cast<int>(input.size(0))};
        const int H{static_cast<int>(input.size(1))};

        auto output{torch::empty_like(input)};
        if (N == 0) return output;

        const int threads{select_threads(H)};
        const size_t smem{shared_mem_bytes(H)};
        const auto stream{at::cuda::getCurrentCUDAStream()};

        // Draw seed from PyTorch's default CUDA generator for reproducibility.
        unsigned long long rng_seed{0ULL};
        if (training && dropout_p > 0.0f) {
            auto gen{at::cuda::detail::getDefaultCUDAGenerator()};
            std::lock_guard<std::mutex> lock(gen.mutex());
            rng_seed = gen.current_seed();
        }

        if (training) {
            fused_layernorm_residual_dropout_kernel<true>
                <<<N, threads, smem, stream>>>(
                    input.data_ptr<float>(),
                    residual.data_ptr<float>(),
                    gamma.data_ptr<float>(),
                    beta.data_ptr<float>(),
                    output.data_ptr<float>(),
                    N, H,
                    dropout_p, rng_seed
                );
        } else {
            fused_layernorm_residual_dropout_kernel<false>
                <<<N, threads, smem, stream>>>(
                    input.data_ptr<float>(),
                    residual.data_ptr<float>(),
                    gamma.data_ptr<float>(),
                    beta.data_ptr<float>(),
                    output.data_ptr<float>(),
                    N, H,
                    dropout_p, rng_seed
                );
        }
        C10_CUDA_KERNEL_LAUNCH_CHECK();

        return output;
    }
}  // namespace qec
