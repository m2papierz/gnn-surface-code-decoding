/*
 * Host entry point of fused_norm_residual.cu.
 *
 * The one declaration, included by its definition and by the Python
 * bindings; see graph_build.h for why the definition is a qualified name.
 */

#pragma once

#include <torch/types.h>

namespace qec {
    // dropout(layer_norm(input) * gamma + beta) + residual, row-wise.
    torch::Tensor fused_norm_residual_dropout(
        torch::Tensor input,      // (N, H) float32
        torch::Tensor residual,   // (N, H) float32
        torch::Tensor gamma,      // (H,) float32
        torch::Tensor beta,       // (H,) float32
        float dropout_p,
        bool training
    );
}  // namespace qec
