/*
 * Batched fired-detector graph construction (v2 representation).
 *
 * Two kernels reproduce `sampling.graph.build_fired_detector_graph` for a
 * whole batch of syndromes without leaving the device:
 *
 *   fired_detector_node_features  compacted fired detectors -> (N, 6) float32
 *   fired_detector_edges          node features -> all-pairs (2, E) + (E, 6)
 *
 * Both write into caller-owned output buffers, addressed through per-shot
 * base offsets.  With compact offsets the result is the PyG-batched layout;
 * with strided offsets the same kernels fill fixed-shape bucket buffers.
 *
 * Numerics: the node kernel mirrors numpy exactly by normalising in float64
 * and rounding once on store; the edge kernel then reads those already
 * rounded float32 values.  Its float arithmetic uses explicit round-to-nearest
 * intrinsics because nvcc contracts `a*a + b*b` into an FMA by default, which
 * rounds differently and would break bit-identity with the reference.
 *
 * Target: sm_80+ (Ampere, Ada Lovelace).
 */

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>

namespace qec {
    namespace {
        constexpr int kBlockSize{256};
        constexpr int kNodeDim{6};
        constexpr int kEdgeDim{6};
        constexpr int kCoordDim{3};

        // ------------------------------------------------------------------
        // Node features: [x_norm, y_norm, t_norm, d_x, d_y, basis]
        // ------------------------------------------------------------------

        __global__ void __launch_bounds__(kBlockSize)
        node_features_kernel(
            const double* __restrict__ coords,     // (D, 3)
            const int64_t* __restrict__ detector,  // (N,) fired detector id
            const int64_t* __restrict__ slot,      // (N,) destination row
            float* __restrict__ out,               // (M, 6)
            const int64_t num_nodes,
            const double spatial_scale,
            const double temporal_scale,
            const double distance
        ) {
            const int64_t i{blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x};
            if (i >= num_nodes) return;

            const double* __restrict__ c{coords + detector[i] * kCoordDim};
            const double x{c[0]};
            const double y{c[1]};
            const double t{c[2]};

            // numpy: (((x + y) / 2).astype(np.intp) % 2) — truncation toward
            // zero, then a Python-signed modulo (always non-negative for 2).
            const long long half{static_cast<long long>((x + y) / 2.0)};

            float* __restrict__ o{out + slot[i] * kNodeDim};
            o[0] = static_cast<float>(x / spatial_scale);
            o[1] = static_cast<float>(y / spatial_scale);
            o[2] = static_cast<float>(t / temporal_scale);
            o[3] = static_cast<float>((x - distance) / distance);
            o[4] = static_cast<float>((y - distance) / distance);
            o[5] = static_cast<float>(((half % 2) + 2) % 2);
        }

        // ------------------------------------------------------------------
        // Edge features: [dx, dy, dt, euclidean, chebyshev, dem_weight]
        // ------------------------------------------------------------------

        // Largest b in [0, num_shots) with prefix[b] <= e.  `prefix` is the
        // ascending cumulative edge count, so this is the shot owning edge e.
        __device__ __forceinline__ int64_t owning_shot(
            const int64_t* __restrict__ prefix,
            const int64_t num_shots,
            const int64_t e
        ) {
            int64_t lo{0};
            int64_t hi{num_shots - 1};
            while (lo < hi) {
                const int64_t mid{lo + (hi - lo + 1) / 2};
                if (prefix[mid] <= e) {
                    lo = mid;
                } else {
                    hi = mid - 1;
                }
            }
            return lo;
        }

        __global__ void __launch_bounds__(kBlockSize)
        edge_kernel(
            const float* __restrict__ node_features,  // (M, 6)
            const int64_t* __restrict__ edge_prefix,  // (B + 1,) cumulative
            const int64_t* __restrict__ node_base,    // (B,) first node slot
            const int64_t* __restrict__ node_count,   // (B,) fired detectors
            const int64_t* __restrict__ edge_base,    // (B,) first edge slot
            const float* __restrict__ dem_weights,    // (D, D) DEM pair probs
            const int64_t* __restrict__ detector_ids, // (M,) original det id
            const int64_t num_detectors,              // D
            int64_t* __restrict__ edge_index,         // (2, Me)
            float* __restrict__ edge_attr,            // (Me, 6)
            const int64_t num_edges,
            const int64_t edge_stride,                // row stride of edge_index
            const int64_t num_shots
        ) {
            const int64_t e{blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x};
            if (e >= num_edges) return;

            const int64_t shot{owning_shot(edge_prefix, num_shots, e)};
            const int64_t local{e - edge_prefix[shot]};

            // Source-major enumeration of the complete graph with the diagonal
            // skipped, matching the numpy builder's repeat/tile + mask order.
            const int64_t span{node_count[shot] - 1};
            const int64_t src_local{local / span};
            const int64_t k{local - src_local * span};
            const int64_t dst_local{k + (k >= src_local ? 1 : 0)};

            const int64_t base{node_base[shot]};
            const int64_t src{base + src_local};
            const int64_t dst{base + dst_local};

            const int64_t out{edge_base[shot] + local};
            edge_index[out] = src;
            edge_index[edge_stride + out] = dst;

            const float* __restrict__ s{node_features + src * kNodeDim};
            const float* __restrict__ d{node_features + dst * kNodeDim};

            const float dx{__fsub_rn(d[0], s[0])};
            const float dy{__fsub_rn(d[1], s[1])};
            const float dt{__fsub_rn(d[2], s[2])};

            // ((dx*dx + dy*dy) + dt*dt), unfused, in numpy's reduction order.
            const float sq{
                __fadd_rn(
                    __fadd_rn(__fmul_rn(dx, dx), __fmul_rn(dy, dy)),
                    __fmul_rn(dt, dt)
                )
            };

            const int64_t det_src{detector_ids[src]};
            const int64_t det_dst{detector_ids[dst]};

            float* __restrict__ o{edge_attr + out * kEdgeDim};
            o[0] = dx;
            o[1] = dy;
            o[2] = dt;
            o[3] = __fsqrt_rn(sq);
            o[4] = fmaxf(fmaxf(fabsf(dx), fabsf(dy)), fabsf(dt));
            o[5] = dem_weights[det_src * num_detectors + det_dst];
        }

        inline int64_t grid_for(int64_t n) noexcept {
            return (n + kBlockSize - 1) / kBlockSize;
        }

        inline void check_index_vector(
            const torch::Tensor& t, const char* name, int64_t expected
        ) {
            TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
            TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
            TORCH_CHECK(
                t.scalar_type() == torch::kInt64, name, " must be int64"
            );
            TORCH_CHECK(
                t.dim() == 1 && t.size(0) == expected,
                name, " must have shape (", expected, "), got ", t.sizes()
            );
        }
    }  // anonymous namespace

    // ------------------------------------------------------------------
    // Host entry points
    // ------------------------------------------------------------------

    void fired_detector_node_features(
        torch::Tensor coords,
        torch::Tensor detector,
        torch::Tensor slot,
        torch::Tensor out,
        int64_t distance,
        int64_t rounds
    ) {
        TORCH_CHECK(coords.is_cuda() && coords.is_contiguous(),
                    "coords must be a contiguous CUDA tensor");
        TORCH_CHECK(coords.scalar_type() == torch::kFloat64,
                    "coords must be float64 to match the numpy reference");
        TORCH_CHECK(coords.dim() == 2 && coords.size(1) == kCoordDim,
                    "coords must have shape (D, 3), got ", coords.sizes());
        TORCH_CHECK(out.is_cuda() && out.is_contiguous(),
                    "out must be a contiguous CUDA tensor");
        TORCH_CHECK(out.scalar_type() == torch::kFloat32, "out must be float32");
        TORCH_CHECK(out.dim() == 2 && out.size(1) == kNodeDim,
                    "out must have shape (M, 6), got ", out.sizes());
        TORCH_CHECK(distance >= 1, "distance must be >= 1, got ", distance);
        TORCH_CHECK(rounds >= 1, "rounds must be >= 1, got ", rounds);

        const int64_t num_nodes{detector.size(0)};
        check_index_vector(detector, "detector", num_nodes);
        check_index_vector(slot, "slot", num_nodes);
        if (num_nodes == 0) return;

        const auto stream{at::cuda::getCurrentCUDAStream()};
        node_features_kernel<<<grid_for(num_nodes), kBlockSize, 0, stream>>>(
            coords.data_ptr<double>(),
            detector.data_ptr<int64_t>(),
            slot.data_ptr<int64_t>(),
            out.data_ptr<float>(),
            num_nodes,
            2.0 * static_cast<double>(distance),
            static_cast<double>(rounds),
            static_cast<double>(distance)
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    void fired_detector_edges(
        torch::Tensor node_features,
        torch::Tensor edge_prefix,
        torch::Tensor node_base,
        torch::Tensor node_count,
        torch::Tensor edge_base,
        torch::Tensor dem_weights,
        torch::Tensor detector_ids,
        torch::Tensor edge_index,
        torch::Tensor edge_attr,
        int64_t num_edges
    ) {
        TORCH_CHECK(node_features.is_cuda() && node_features.is_contiguous(),
                    "node_features must be a contiguous CUDA tensor");
        TORCH_CHECK(node_features.scalar_type() == torch::kFloat32,
                    "node_features must be float32");
        TORCH_CHECK(node_features.dim() == 2 && node_features.size(1) == kNodeDim,
                    "node_features must have shape (M, 6), got ",
                    node_features.sizes());
        TORCH_CHECK(dem_weights.is_cuda() && dem_weights.is_contiguous(),
                    "dem_weights must be a contiguous CUDA tensor");
        TORCH_CHECK(dem_weights.scalar_type() == torch::kFloat32,
                    "dem_weights must be float32");
        TORCH_CHECK(dem_weights.dim() == 2
                    && dem_weights.size(0) == dem_weights.size(1),
                    "dem_weights must have shape (D, D), got ",
                    dem_weights.sizes());
        check_index_vector(detector_ids, "detector_ids",
                           node_features.size(0));
        TORCH_CHECK(edge_index.is_cuda() && edge_index.is_contiguous(),
                    "edge_index must be a contiguous CUDA tensor");
        TORCH_CHECK(edge_index.scalar_type() == torch::kInt64,
                    "edge_index must be int64");
        TORCH_CHECK(edge_index.dim() == 2 && edge_index.size(0) == 2,
                    "edge_index must have shape (2, Me), got ", edge_index.sizes());
        TORCH_CHECK(edge_attr.is_cuda() && edge_attr.is_contiguous(),
                    "edge_attr must be a contiguous CUDA tensor");
        TORCH_CHECK(edge_attr.scalar_type() == torch::kFloat32,
                    "edge_attr must be float32");
        TORCH_CHECK(edge_attr.dim() == 2 && edge_attr.size(1) == kEdgeDim,
                    "edge_attr must have shape (Me, 6), got ", edge_attr.sizes());
        TORCH_CHECK(edge_attr.size(0) == edge_index.size(1),
                    "edge_attr rows ", edge_attr.size(0),
                    " != edge_index columns ", edge_index.size(1));
        TORCH_CHECK(num_edges >= 0 && num_edges <= edge_index.size(1),
                    "num_edges ", num_edges, " exceeds buffer capacity ",
                    edge_index.size(1));

        const int64_t num_shots{node_base.size(0)};
        check_index_vector(edge_prefix, "edge_prefix", num_shots + 1);
        check_index_vector(node_base, "node_base", num_shots);
        check_index_vector(node_count, "node_count", num_shots);
        check_index_vector(edge_base, "edge_base", num_shots);
        if (num_edges == 0) return;

        const auto stream{at::cuda::getCurrentCUDAStream()};
        edge_kernel<<<grid_for(num_edges), kBlockSize, 0, stream>>>(
            node_features.data_ptr<float>(),
            edge_prefix.data_ptr<int64_t>(),
            node_base.data_ptr<int64_t>(),
            node_count.data_ptr<int64_t>(),
            edge_base.data_ptr<int64_t>(),
            dem_weights.data_ptr<float>(),
            detector_ids.data_ptr<int64_t>(),
            dem_weights.size(0),
            edge_index.data_ptr<int64_t>(),
            edge_attr.data_ptr<float>(),
            num_edges,
            edge_index.size(1),
            num_shots
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
}  // namespace qec
