"""Tests for the graph representation versioning and data contract."""

from __future__ import annotations

import pytest

from sampling.profile import resolve_profile
from sampling.representation import (
    SPATIAL,
    DataContract,
    LabelSpec,
    RepresentationDescriptor,
    resolve_builder,
)


class TestRepresentationDescriptor:
    def test_spatial_field_by_field(self) -> None:
        assert SPATIAL.version == "spatial"
        assert SPATIAL.node_dim == 6
        assert SPATIAL.edge_dim == 6
        assert SPATIAL.node_features == (
            "x_norm",
            "y_norm",
            "t_norm",
            "d_x",
            "d_y",
            "basis",
        )
        assert SPATIAL.edge_features == (
            "dx",
            "dy",
            "dt",
            "euclidean",
            "chebyshev",
            "dem_weight",
        )

    def test_node_features_length_matches_dim(self) -> None:
        assert len(SPATIAL.node_features) == SPATIAL.node_dim

    def test_edge_features_length_matches_dim(self) -> None:
        assert len(SPATIAL.edge_features) == SPATIAL.edge_dim

    def test_rejects_empty_version(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            RepresentationDescriptor(
                version="",
                node_dim=1,
                edge_dim=1,
                node_features=("a",),
                edge_features=("b",),
            )

    def test_rejects_dim_mismatch(self) -> None:
        with pytest.raises(ValueError, match="node_features length"):
            RepresentationDescriptor(
                version="bad",
                node_dim=2,
                edge_dim=1,
                node_features=("only_one",),
                edge_features=("b",),
            )

    def test_rejects_zero_dim(self) -> None:
        with pytest.raises(ValueError, match="node_dim must be >= 1"):
            RepresentationDescriptor(
                version="bad",
                node_dim=0,
                edge_dim=1,
                node_features=(),
                edge_features=("b",),
            )

    def test_frozen(self) -> None:
        with pytest.raises(AttributeError):
            SPATIAL.version = "phased"  # type: ignore[misc]


class TestLabelSpec:
    def test_memory_labels(self) -> None:
        spec = LabelSpec(num_observables=1, observable_names=("logical_observable",))
        assert spec.num_observables == 1
        assert spec.observable_names == ("logical_observable",)

    def test_rejects_count_mismatch(self) -> None:
        with pytest.raises(ValueError, match="observable_names length"):
            LabelSpec(num_observables=2, observable_names=("only_one",))

    def test_rejects_zero_observables(self) -> None:
        with pytest.raises(ValueError, match="num_observables must be >= 1"):
            LabelSpec(num_observables=0, observable_names=())


class TestDataContract:
    def test_spatial_memory_properties(self) -> None:
        assert resolve_profile("memory").data_contract.node_dim == 6
        assert resolve_profile("memory").data_contract.edge_dim == 6
        assert resolve_profile("memory").data_contract.num_observables == 1
        assert resolve_profile("memory").data_contract.version == "spatial"

    def test_roundtrip(self) -> None:
        d = resolve_profile("memory").data_contract.to_dict()
        restored = DataContract.from_dict(d)
        assert restored == resolve_profile("memory").data_contract

    def test_to_dict_contents(self) -> None:
        d = resolve_profile("memory").data_contract.to_dict()
        assert d["version"] == "spatial"
        assert d["node_dim"] == 6
        assert d["edge_dim"] == 6
        assert d["node_features"] == list(SPATIAL.node_features)
        assert d["edge_features"] == list(SPATIAL.edge_features)
        assert d["num_observables"] == 1
        assert d["observable_names"] == ["logical_observable"]

    def test_from_checkpoint_config(self) -> None:
        cfg = {
            "contract": resolve_profile("memory").data_contract.to_dict(),
            "other_stuff": 42,
        }
        restored = DataContract.from_checkpoint_config(cfg)
        assert restored == resolve_profile("memory").data_contract

    def test_from_checkpoint_config_rejects_missing_contract(self) -> None:
        with pytest.raises(KeyError):
            DataContract.from_checkpoint_config({"node_dim": 6, "edge_dim": 6})

    def test_frozen(self) -> None:
        with pytest.raises(AttributeError):
            resolve_profile("memory").data_contract.representation = SPATIAL  # type: ignore[misc]


class TestBuilderResolution:
    def test_spatial_resolves(self) -> None:
        from sampling.graph import build_fired_detector_graph

        builder = resolve_builder("spatial")
        assert builder is build_fired_detector_graph

    def test_unknown_version_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown representation version"):
            resolve_builder("v99")

    def test_builder_produces_correct_shapes(self) -> None:
        import numpy as np

        from sampling.graph import CircuitMetadata

        meta = CircuitMetadata(
            detector_coords=np.array(
                [[0, 0, 0], [2, 0, 0], [0, 2, 1]], dtype=np.float64
            ),
            distance=3,
            rounds=3,
            num_detectors=3,
            dem_edge_weights=np.zeros((3, 3), dtype=np.float64),
        )
        syndrome = np.array([1, 1, 0], dtype=np.uint8)
        builder = resolve_builder("spatial")
        graph = builder(syndrome, meta)
        assert graph.node_features.shape == (2, SPATIAL.node_dim)
        assert graph.edge_features.shape == (2, SPATIAL.edge_dim)
