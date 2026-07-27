/*
 * Fused edge update for GNN message passing.
 *
 * Replaces the materialise-then-GEMM pair
 *
 *     concat = [h_src + h_dst | |h_src - h_dst| | e]      (E, 3H)
 *     out    = concat @ W^T + bias                        (E, H_out)
 *
 * with a single launch that never forms `concat`.  Two algebraic facts drive
 * the shape of this kernel:
 *
 *   1. W·[a|b|c] = W_sum·a + W_diff·b + W_edge·c, where W_sum, W_diff and
 *      W_edge are column blocks of the trained weight.  Exact identity.
 *   2. a = h_src + h_dst is *linear*, so W_sum·a = (W_sum·h)[src] +
 *      (W_sum·h)[dst].  That block moves off the edges and onto the nodes.
 *      Complete graphs have E = N(N-1), so at d=7 this is ~44x less work for
 *      that third of the GEMM, and the caller computes it with one cuBLAS
 *      call on (N, H) before entering here.
 *
 * What remains is a GEMM with K = 2H rather than 3H, whose A operand is
 * produced in registers: columns [0, H) are |h_src - h_dst| gathered per
 * edge, columns [H, 2H) are the edge embedding read straight through.  The
 * node-side term enters in the epilogue.
 *
 *     out[e] = p[src] + p[dst] + [ |h_src - h_dst| | e ] @ Wr^T + bias
 *
 * with Wr = [W_diff | W_edge] laid out (H_out, 2H) by the caller, once.
 *
 * Arithmetic is float32 throughout, operands and accumulation alike.  A
 * reduced-precision tensor-core path was built and measured: it bought 1.4%
 * at batch 1 and 5.3% at batch 128 while making p99 at batch 1 worse (2.735
 * ms against 1.887 ms), because this workload is launch- and memory-bound
 * rather than compute-bound.  It was retired — see the cuda-fast-path
 * decision `encoder-gemms-run-bf16-with-fp32-accumulation-under-a`.
 *
 * Target: sm_80+ (Ampere, Ada Lovelace).
 */

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>

namespace qec {
    namespace {
        // One block computes BLOCK_M edges x BLOCK_N outputs, marching over K
        // in BLOCK_K slices, with each thread owning a 4 x 4 register patch of
        // the output tile.
        constexpr int kBlockM{64};
        constexpr int kBlockN{128};
        constexpr int kBlockK{32};
        constexpr int kThreads{256};

        // Shared-memory row padding: the inner loop walks the A tile down a
        // column (one k, many m), so an unpadded power-of-two row stride would
        // put every one of those rows in the same bank.
        constexpr int kSkew{8};

        __host__ __device__ __forceinline__ int ceil_div(int a, int b) {
            return (a + b - 1) / b;
        }

        // ------------------------------------------------------------------
        // A-operand construction
        //
        // Row m of the A tile is edge (tile_m0 + m); column k is
        //   k <  H : |h[src][k] - h[dst][k]|
        //   k >= H : e[edge][k - H]
        // Out-of-range rows are zeroed so the tail tile contributes nothing.
        // ------------------------------------------------------------------

        __device__ __forceinline__ float a_element(
            const float* __restrict__ h,
            const float* __restrict__ e,
            const int64_t* __restrict__ src_idx,
            const int64_t* __restrict__ dst_idx,
            const int edge,
            const int k,
            const int hidden
        ) {
            if (k < hidden) {
                const int s{static_cast<int>(src_idx[edge])};
                const int d{static_cast<int>(dst_idx[edge])};
                return fabsf(h[s * hidden + k] - h[d * hidden + k]);
            }
            return e[edge * hidden + (k - hidden)];
        }

        // ------------------------------------------------------------------
        // float32 shared-memory tiling with plain FMA.
        //
        // Any deviation from eager PyTorch beyond atol 1e-5 is a structural
        // defect here — wrong weight block, transposed operand, bad gather,
        // off-by-one edge index — with no precision noise to hide behind.
        // ------------------------------------------------------------------

        __global__ void __launch_bounds__(kThreads)
        fused_edge_update_kernel(
            const float* __restrict__ p,        // (N, H_out) node-side term
            const float* __restrict__ h,        // (N, H)
            const float* __restrict__ e,        // (E, H)
            const float* __restrict__ weight,   // (H_out, 2H) row-major
            const float* __restrict__ bias,     // (H_out,) or null
            const int64_t* __restrict__ src_idx,
            const int64_t* __restrict__ dst_idx,
            float* __restrict__ out,            // (E, H_out)
            const int num_edges,
            const int hidden,
            const int out_dim
        ) {
            __shared__ float a_tile[kBlockM][kBlockK + kSkew];
            __shared__ float b_tile[kBlockK][kBlockN + kSkew];

            const int tile_m0{static_cast<int>(blockIdx.x) * kBlockM};
            const int tile_n0{static_cast<int>(blockIdx.y) * kBlockN};
            const int tid{static_cast<int>(threadIdx.x)};
            const int k_total{2 * hidden};

            // Each thread owns a 4 x 4 patch of the 64 x 128 output tile.
            constexpr int kRegM{kBlockM * kBlockN / kThreads / 4};  // 4
            constexpr int kRegN{4};
            float acc[kRegM][kRegN];
            #pragma unroll
            for (int i{0}; i < kRegM; ++i) {
                #pragma unroll
                for (int j{0}; j < kRegN; ++j) acc[i][j] = 0.0f;
            }

            const int thread_row{(tid / (kBlockN / kRegN)) * kRegM};
            const int thread_col{(tid % (kBlockN / kRegN)) * kRegN};

            for (int k0{0}; k0 < k_total; k0 += kBlockK) {
                for (int idx{tid}; idx < kBlockM * kBlockK; idx += kThreads) {
                    const int m{idx / kBlockK};
                    const int k{idx % kBlockK};
                    const int edge{tile_m0 + m};
                    a_tile[m][k] =
                        (edge < num_edges && (k0 + k) < k_total)
                            ? a_element(h, e, src_idx, dst_idx, edge, k0 + k, hidden)
                            : 0.0f;
                }
                for (int idx{tid}; idx < kBlockK * kBlockN; idx += kThreads) {
                    const int k{idx / kBlockN};
                    const int n{idx % kBlockN};
                    const int col{tile_n0 + n};
                    b_tile[k][n] =
                        (col < out_dim && (k0 + k) < k_total)
                            ? weight[col * k_total + (k0 + k)]
                            : 0.0f;
                }
                __syncthreads();

                #pragma unroll
                for (int k{0}; k < kBlockK; ++k) {
                    #pragma unroll
                    for (int i{0}; i < kRegM; ++i) {
                        const float av{a_tile[thread_row + i][k]};
                        #pragma unroll
                        for (int j{0}; j < kRegN; ++j) {
                            acc[i][j] = fmaf(av, b_tile[k][thread_col + j], acc[i][j]);
                        }
                    }
                }
                __syncthreads();
            }

            #pragma unroll
            for (int i{0}; i < kRegM; ++i) {
                const int edge{tile_m0 + thread_row + i};
                if (edge >= num_edges) continue;
                const int s{static_cast<int>(src_idx[edge])};
                const int d{static_cast<int>(dst_idx[edge])};
                #pragma unroll
                for (int j{0}; j < kRegN; ++j) {
                    const int col{tile_n0 + thread_col + j};
                    if (col >= out_dim) continue;
                    float v{acc[i][j]};
                    v += p[s * out_dim + col] + p[d * out_dim + col];
                    if (bias != nullptr) v += bias[col];
                    out[edge * out_dim + col] = v;
                }
            }
        }

        inline void check_matrix(
            const torch::Tensor& t,
            const char* name,
            int64_t rows,
            int64_t cols,
            torch::ScalarType dtype
        ) {
            TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
            TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
            TORCH_CHECK(
                t.scalar_type() == dtype,
                name, " must be ", dtype, ", got ", t.scalar_type()
            );
            TORCH_CHECK(
                t.dim() == 2 && t.size(0) == rows && t.size(1) == cols,
                name, " must have shape (", rows, ", ", cols, "), got ", t.sizes()
            );
        }
    }  // anonymous namespace

    // ------------------------------------------------------------------
    // Host entry point
    // ------------------------------------------------------------------

    torch::Tensor fused_edge_update(
        torch::Tensor node_term,   // (N, H_out) = h @ W_sum^T, precomputed
        torch::Tensor node_h,      // (N, H)
        torch::Tensor edge_h,      // (E, H)
        torch::Tensor weight,      // (H_out, 2H) = [W_diff | W_edge]
        c10::optional<torch::Tensor> bias,
        torch::Tensor edge_index   // (2, E)
    ) {
        TORCH_CHECK(node_h.is_cuda() && node_h.is_contiguous(),
                    "node_h must be a contiguous CUDA tensor");
        TORCH_CHECK(node_h.dim() == 2, "node_h must have shape (N, H), got ",
                    node_h.sizes());
        TORCH_CHECK(node_h.scalar_type() == torch::kFloat32,
                    "node_h must be float32");

        const int64_t num_nodes{node_h.size(0)};
        const int64_t hidden{node_h.size(1)};
        const int64_t out_dim{weight.size(0)};

        TORCH_CHECK(edge_index.is_cuda() && edge_index.is_contiguous(),
                    "edge_index must be a contiguous CUDA tensor");
        TORCH_CHECK(edge_index.scalar_type() == torch::kInt64,
                    "edge_index must be int64");
        TORCH_CHECK(edge_index.dim() == 2 && edge_index.size(0) == 2,
                    "edge_index must have shape (2, E), got ", edge_index.sizes());

        const int64_t num_edges{edge_index.size(1)};

        check_matrix(node_term, "node_term", num_nodes, out_dim, torch::kFloat32);
        check_matrix(edge_h, "edge_h", num_edges, hidden, torch::kFloat32);
        check_matrix(weight, "weight", out_dim, 2 * hidden, torch::kFloat32);

        const float* bias_ptr{nullptr};
        if (bias.has_value()) {
            const torch::Tensor& b{bias.value()};
            TORCH_CHECK(b.is_cuda() && b.is_contiguous(),
                        "bias must be a contiguous CUDA tensor");
            TORCH_CHECK(b.scalar_type() == torch::kFloat32, "bias must be float32");
            TORCH_CHECK(b.dim() == 1 && b.size(0) == out_dim,
                        "bias must have shape (", out_dim, "), got ", b.sizes());
            bias_ptr = b.data_ptr<float>();
        }

        auto out{torch::empty({num_edges, out_dim}, node_h.options())};
        if (num_edges == 0) return out;

        const dim3 grid(
            static_cast<unsigned>(ceil_div(static_cast<int>(num_edges), kBlockM)),
            static_cast<unsigned>(ceil_div(static_cast<int>(out_dim), kBlockN))
        );
        const auto stream{at::cuda::getCurrentCUDAStream()};

        const int64_t* src{edge_index.data_ptr<int64_t>()};
        const int64_t* dst{src + num_edges};

        fused_edge_update_kernel<<<grid, kThreads, 0, stream>>>(
            node_term.data_ptr<float>(),
            node_h.data_ptr<float>(),
            edge_h.data_ptr<float>(),
            weight.data_ptr<float>(),
            bias_ptr, src, dst,
            out.data_ptr<float>(),
            static_cast<int>(num_edges),
            static_cast<int>(hidden),
            static_cast<int>(out_dim)
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();

        return out;
    }
}  // namespace qec
