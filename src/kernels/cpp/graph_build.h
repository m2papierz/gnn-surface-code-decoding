/*
 * Host entry points of graph_build.cu.
 *
 * The one declaration of each, included by its definition and by the Python
 * bindings.  The definitions are written as qualified `qec::` names, which
 * C++ accepts only for a function already declared in the namespace, so a
 * definition that drifts from this signature fails to compile instead of
 * leaving the binding pointing at a symbol nothing defines.
 */

#pragma once

#include <torch/types.h>

namespace qec {
    // Fired-detector node features into rows `slot` of a caller-owned (M, 6)
    // float32 buffer.
    void fired_detector_node_features(
        torch::Tensor coords,     // (D, 3) float64 detector coordinates
        torch::Tensor detector,   // (N,) int64 fired detector id
        torch::Tensor slot,       // (N,) int64 destination row
        torch::Tensor out,        // (M, 6) float32
        int64_t distance,
        int64_t rounds
    );

    // All-pairs edge index and features of each shot's complete graph into
    // caller-owned buffers, addressed through per-shot base offsets.
    void fired_detector_edges(
        torch::Tensor node_features,  // (M, 6) float32
        torch::Tensor edge_prefix,    // (B + 1,) int64 cumulative edge count
        torch::Tensor node_base,      // (B,) int64 first node slot
        torch::Tensor node_count,     // (B,) int64 fired detectors
        torch::Tensor edge_base,      // (B,) int64 first edge slot
        torch::Tensor dem_weights,    // (D, D) float32 DEM pair probabilities
        torch::Tensor detector_ids,   // (M,) int64 detector id of each node
        torch::Tensor edge_index,     // (2, Me) int64
        torch::Tensor edge_attr,      // (Me, 6) float32
        int64_t num_edges
    );
}  // namespace qec
