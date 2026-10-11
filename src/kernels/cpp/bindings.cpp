#include "fused_edge_update.h"
#include "fused_norm_residual.h"
#include "graph_build.h"

#include <torch/extension.h>

// The build passes kernels.extension.build_sha256() as a bare hex token, and
// kernels.extension.load() refuses a binary whose stamp differs from the tree.
#ifndef QEC_BUILD_SHA256
#error "QEC_BUILD_SHA256 is set by the build; build with `make kernels`"
#endif
#define QEC_STRINGIFY_TOKEN(token) #token
#define QEC_STRINGIFY(macro) QEC_STRINGIFY_TOKEN(macro)

// Every argument is named, so the Python wrappers call by keyword and a
// reordered parameter fails with a TypeError naming it rather than binding a
// tensor to the wrong role.
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.attr("build_sha256") = QEC_STRINGIFY(QEC_BUILD_SHA256);
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
        "Fused row-wise LayerNorm + residual + dropout (CUDA)",
        pybind11::arg("input"),
        pybind11::arg("residual"),
        pybind11::arg("gamma"),
        pybind11::arg("beta"),
        pybind11::arg("dropout_p"),
        pybind11::arg("training")
    );
    m.def(
        "fired_detector_node_features",
        &qec::fired_detector_node_features,
        "Fired-detector node features into a caller-owned buffer (CUDA)",
        pybind11::arg("coords"),
        pybind11::arg("detector"),
        pybind11::arg("slot"),
        pybind11::arg("out"),
        pybind11::arg("distance"),
        pybind11::arg("rounds")
    );
    m.def(
        "fired_detector_edges",
        &qec::fired_detector_edges,
        "All-pairs edge index and features into caller-owned buffers (CUDA)",
        pybind11::arg("node_features"),
        pybind11::arg("edge_prefix"),
        pybind11::arg("node_base"),
        pybind11::arg("node_count"),
        pybind11::arg("edge_base"),
        pybind11::arg("dem_weights"),
        pybind11::arg("detector_ids"),
        pybind11::arg("edge_index"),
        pybind11::arg("edge_attr"),
        pybind11::arg("num_edges")
    );
}
