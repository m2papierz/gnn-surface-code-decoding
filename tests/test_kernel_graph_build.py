"""GPU graph construction vs the numpy reference builder.

Acceptance is bit-identity, not a tolerance - the kernel and
``build_fired_detector_graph`` must define the same representation, and a
tolerance would hide exactly the feature-definition drift this guards against.
"""

from __future__ import annotations

import numpy as np
import pytest
import stim
import torch

from sampling.graph import (
    CircuitMetadata,
    build_fired_detector_graph,
    extract_circuit_metadata,
)


CIRCUITS = {
    3: "data/circuits/memory/d3_r3_p0_01.stim",
    5: "data/circuits/memory/d5_r5_p0_01.stim",
    7: "data/circuits/memory/d7_r7_p0_01.stim",
}


def _kernels_available() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        import kernels
    except ImportError:
        return False
    return kernels.AVAILABLE


pytestmark = pytest.mark.skipif(
    not _kernels_available(), reason="CUDA kernels not built or no GPU"
)


def _metadata(distance: int) -> CircuitMetadata:
    circuit = stim.Circuit.from_file(CIRCUITS[distance])
    return extract_circuit_metadata(circuit, distance=distance, rounds=distance)


def _sample(distance: int, n_shots: int, seed: int = 7) -> np.ndarray:
    circuit = stim.Circuit.from_file(CIRCUITS[distance])
    sampler = circuit.compile_detector_sampler(seed=seed)
    return sampler.sample(shots=n_shots, bit_packed=False).astype(np.uint8)


def _build_on_device(syndromes: np.ndarray, metadata: CircuitMetadata):
    from kernels.graph_build import (
        build_fired_detector_graphs,
        detector_coords_to_device,
    )

    coords = detector_coords_to_device(metadata, "cuda")
    device_syndromes = torch.as_tensor(syndromes, device="cuda")
    return build_fired_detector_graphs(
        device_syndromes,
        coords,
        distance=metadata.distance,
        rounds=metadata.rounds,
    )


def _reference_batch(syndromes: np.ndarray, metadata: CircuitMetadata):
    """Concatenate per-shot numpy graphs into the PyG-batched layout."""
    nodes, edge_src, edge_dst, edge_feats, batch, counts = [], [], [], [], [], []
    offset = 0
    for shot, syndrome in enumerate(syndromes):
        graph = build_fired_detector_graph(syndrome, metadata)
        nodes.append(graph.node_features)
        edge_src.append(graph.edge_index[0] + offset)
        edge_dst.append(graph.edge_index[1] + offset)
        edge_feats.append(graph.edge_features)
        batch.append(np.full(graph.num_fired, shot, dtype=np.int64))
        counts.append(graph.num_fired)
        offset += graph.num_fired

    return (
        np.concatenate(nodes),
        np.stack([np.concatenate(edge_src), np.concatenate(edge_dst)]),
        np.concatenate(edge_feats),
        np.concatenate(batch),
        np.asarray(counts, dtype=np.int64),
    )


def _assert_identical(built, syndromes: np.ndarray, metadata: CircuitMetadata) -> None:
    x, edge_index, edge_attr, batch, counts = _reference_batch(syndromes, metadata)

    np.testing.assert_array_equal(built.x.cpu().numpy(), x)
    np.testing.assert_array_equal(built.edge_index.cpu().numpy(), edge_index)
    np.testing.assert_array_equal(built.edge_attr.cpu().numpy(), edge_attr)
    np.testing.assert_array_equal(built.batch.cpu().numpy(), batch)
    np.testing.assert_array_equal(built.num_fired.cpu().numpy(), counts)
    assert built.num_graphs == len(syndromes)


class TestBitIdentity:
    @pytest.mark.parametrize("distance", [3, 5, 7])
    def test_matches_numpy_reference(self, distance: int) -> None:
        metadata = _metadata(distance)
        syndromes = _sample(distance, n_shots=512)
        _assert_identical(_build_on_device(syndromes, metadata), syndromes, metadata)

    @pytest.mark.parametrize("distance", [3, 5, 7])
    def test_matches_on_dense_syndromes(self, distance: int) -> None:
        """All detectors fired - the largest complete graph the code can build."""
        metadata = _metadata(distance)
        syndromes = np.ones((4, metadata.num_detectors), dtype=np.uint8)
        _assert_identical(_build_on_device(syndromes, metadata), syndromes, metadata)

    def test_accepts_bool_syndromes(self) -> None:
        metadata = _metadata(3)
        syndromes = _sample(3, n_shots=64)
        built = _build_on_device(syndromes.astype(bool), metadata)
        _assert_identical(built, syndromes, metadata)


class TestDegenerateShapes:
    def test_all_shots_empty(self) -> None:
        metadata = _metadata(3)
        syndromes = np.zeros((8, metadata.num_detectors), dtype=np.uint8)
        built = _build_on_device(syndromes, metadata)

        assert built.x.shape == (0, 6)
        assert built.edge_index.shape == (2, 0)
        assert built.edge_attr.shape == (0, 6)
        assert built.num_graphs == 8
        assert built.num_fired.sum().item() == 0

    def test_single_detector_shots_have_no_edges(self) -> None:
        metadata = _metadata(3)
        syndromes = np.zeros((3, metadata.num_detectors), dtype=np.uint8)
        syndromes[np.arange(3), [0, 5, 11]] = 1
        built = _build_on_device(syndromes, metadata)

        assert built.x.shape == (3, 6)
        assert built.edge_index.shape == (2, 0)
        _assert_identical(built, syndromes, metadata)

    def test_mixed_empty_single_and_dense(self) -> None:
        """Empty, one-node, two-node and dense shots in one batch."""
        metadata = _metadata(5)
        d = metadata.num_detectors
        syndromes = np.zeros((4, d), dtype=np.uint8)
        syndromes[1, 3] = 1
        syndromes[2, [3, 17]] = 1
        syndromes[3, :] = 1
        built = _build_on_device(syndromes, metadata)

        np.testing.assert_array_equal(
            built.num_fired.cpu().numpy(), np.array([0, 1, 2, d])
        )
        _assert_identical(built, syndromes, metadata)

    def test_single_shot_batch(self) -> None:
        metadata = _metadata(3)
        syndromes = _sample(3, n_shots=64)[:1]
        _assert_identical(_build_on_device(syndromes, metadata), syndromes, metadata)


class TestValidation:
    def test_rejects_cpu_syndromes(self) -> None:
        from kernels.graph_build import build_fired_detector_graphs

        metadata = _metadata(3)
        coords = torch.zeros(metadata.num_detectors, 3, dtype=torch.float64).cuda()
        with pytest.raises(ValueError, match="must be CUDA tensors"):
            build_fired_detector_graphs(
                torch.zeros(2, metadata.num_detectors, dtype=torch.uint8),
                coords,
                distance=3,
                rounds=3,
            )

    def test_rejects_rank_one_syndromes(self) -> None:
        from kernels.graph_build import build_fired_detector_graphs

        metadata = _metadata(3)
        coords = torch.zeros(metadata.num_detectors, 3, dtype=torch.float64).cuda()
        with pytest.raises(ValueError, match=r"shape \(B, D\)"):
            build_fired_detector_graphs(
                torch.zeros(metadata.num_detectors, dtype=torch.uint8).cuda(),
                coords,
                distance=3,
                rounds=3,
            )

    def test_rejects_detector_count_mismatch(self) -> None:
        from kernels.graph_build import build_fired_detector_graphs

        coords = torch.zeros(24, 3, dtype=torch.float64).cuda()
        with pytest.raises(ValueError, match="!= syndrome detectors"):
            build_fired_detector_graphs(
                torch.zeros(2, 25, dtype=torch.uint8).cuda(),
                coords,
                distance=3,
                rounds=3,
            )

    def test_kernel_rejects_float32_coords(self) -> None:
        from kernels._C import fired_detector_node_features

        with pytest.raises(RuntimeError, match="float64"):
            fired_detector_node_features(
                torch.zeros(4, 3, dtype=torch.float32).cuda(),
                torch.zeros(1, dtype=torch.int64).cuda(),
                torch.zeros(1, dtype=torch.int64).cuda(),
                torch.zeros(1, 6, dtype=torch.float32).cuda(),
                3,
                3,
            )


class TestDeviceGraphBatchRecord:
    def test_rejects_inconsistent_batch_vector(self) -> None:
        from kernels.graph_build import DeviceGraphBatch

        with pytest.raises(ValueError, match="batch shape"):
            DeviceGraphBatch(
                x=torch.zeros(4, 6),
                edge_index=torch.zeros(2, 0, dtype=torch.int64),
                edge_attr=torch.zeros(0, 6),
                batch=torch.zeros(3, dtype=torch.int64),
                num_fired=torch.tensor([4], dtype=torch.int64),
                num_graphs=1,
            )

    def test_rejects_num_fired_length_mismatch(self) -> None:
        from kernels.graph_build import DeviceGraphBatch

        with pytest.raises(ValueError, match="num_fired shape"):
            DeviceGraphBatch(
                x=torch.zeros(2, 6),
                edge_index=torch.zeros(2, 2, dtype=torch.int64),
                edge_attr=torch.zeros(2, 6),
                batch=torch.zeros(2, dtype=torch.int64),
                num_fired=torch.tensor([2, 0], dtype=torch.int64),
                num_graphs=1,
            )


class TestModelIntegration:
    def test_decoder_accepts_device_graph_batch(self) -> None:
        """DeviceGraphBatch satisfies the BatchedGraph protocol structurally."""
        from model.decoder import build_model

        metadata = _metadata(3)
        syndromes = _sample(3, n_shots=16)
        syndromes[0] = 0  # keep an empty shot in the batch
        built = _build_on_device(syndromes, metadata)

        model = build_model(hidden_dim=16, num_layers=2, dropout=0.0).cuda().eval()
        with torch.no_grad():
            logits = model(built)

        assert logits.shape == (16, 1)
        assert torch.isfinite(logits).all()

    def test_matches_numpy_path_through_the_model(self) -> None:
        """Same logits whether the graph came from numpy or GPU builder."""
        from torch_geometric.data import Batch, Data

        from model.decoder import build_model

        metadata = _metadata(5)
        syndromes = _sample(5, n_shots=32)
        built = _build_on_device(syndromes, metadata)

        graphs = [build_fired_detector_graph(s, metadata) for s in syndromes]
        reference = Batch.from_data_list(
            [
                Data(
                    x=torch.from_numpy(g.node_features),
                    edge_index=torch.from_numpy(g.edge_index),
                    edge_attr=torch.from_numpy(g.edge_features),
                )
                for g in graphs
            ]
        ).to("cuda")

        torch.manual_seed(0)
        model = build_model(hidden_dim=32, num_layers=3, dropout=0.0).cuda().eval()
        with torch.no_grad():
            torch.testing.assert_close(model(built), model(reference))
