#include <torch/extension.h>

namespace qec {
    torch::Tensor fused_edge_update(
        torch::Tensor node_term,
        torch::Tensor node_h,
        torch::Tensor edge_h,
        torch::Tensor weight,
        c10::optional<torch::Tensor> bias,
        torch::Tensor edge_index
    );

    torch::Tensor fused_norm_residual_dropout(
        torch::Tensor input,
        torch::Tensor residual,
        torch::Tensor gamma,
        torch::Tensor beta,
        float dropout_p,
        bool training
    );

    void fired_detector_node_features(
        torch::Tensor coords,
        torch::Tensor detector,
        torch::Tensor slot,
        torch::Tensor out,
        int64_t distance,
        int64_t rounds
    );

    void fired_detector_edges(
        torch::Tensor node_features,
        torch::Tensor edge_prefix,
        torch::Tensor node_base,
        torch::Tensor node_count,
        torch::Tensor edge_base,
        torch::Tensor edge_index,
        torch::Tensor edge_attr,
        int64_t num_edges
    );
}  // namespace qec

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "fused_edge_update",
        &qec::fused_edge_update,
        "Fused gather + symmetric combine + GEMM edge update (CUDA)",
        pybind11::arg("node_term"),
        pybind11::arg("node_h"),
        pybind11::arg("edge_h"),
        pybind11::arg("weight"),
        pybind11::arg("bias"),
        pybind11::arg("edge_index")
    );
    m.def(
        "fused_norm_residual_dropout",
        &qec::fused_norm_residual_dropout,
        "Fused row-wise LayerNorm + residual + dropout (CUDA)"
    );
    m.def(
        "fired_detector_node_features",
        &qec::fired_detector_node_features,
        "Fired-detector node features into a caller-owned buffer (CUDA)"
    );
    m.def(
        "fired_detector_edges",
        &qec::fired_detector_edges,
        "All-pairs edge index and features into caller-owned buffers (CUDA)"
    );
}
