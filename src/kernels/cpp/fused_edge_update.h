/*
 * Host entry point of fused_edge_update.cu.
 *
 * The one declaration, included by its definition and by the Python
 * bindings; see graph_build.h for why the definition is a qualified name.
 */

#pragma once

#include <torch/types.h>

namespace qec {
    // [|h_src - h_dst| | e] @ weight^T + node_term[src] + node_term[dst] + bias
    torch::Tensor fused_edge_update(
        torch::Tensor node_term,   // (N, H_out) = h @ W_sum^T, precomputed
        torch::Tensor node_h,      // (N, H)
        torch::Tensor edge_h,      // (E, H)
        torch::Tensor weight,      // (H_out, 2H) = [W_diff | W_edge]
        c10::optional<torch::Tensor> bias,  // (H_out,)
        torch::Tensor edge_index   // (2, E)
    );
}  // namespace qec
