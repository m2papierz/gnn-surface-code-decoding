"""Bucketed CUDA-Graphs forward: equivalence with the eager PyTorch reference.

The runner must produce logits within atol=1e-5 of eager PyTorch on any valid
batch, including degenerate cases (all-empty, single-node, mixed sizes), and
must not leak state between calls that share a bucket - the pad step resets
only the slack the previous call dirtied, so a large batch followed by a
smaller one is the case most likely to expose a stale-buffer bug.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch


def _cuda_available() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        import kernels

        return kernels.AVAILABLE
    except ImportError:
        return False


pytestmark = pytest.mark.skipif(
    not _cuda_available(), reason="CUDA kernels not built or no GPU"
)


def _make_model(hidden: int = 32, layers: int = 2):
    from model.decoder import build_model

    return build_model(hidden_dim=hidden, num_layers=layers, dropout=0.0).cuda().eval()


def _make_syndromes(distance: int, n_shots: int, seed: int = 7) -> np.ndarray:
    import stim

    path = f"data/circuits/memory/d{distance}_r{distance}_p0_01.stim"
    circuit = stim.Circuit.from_file(path)
    sampler = circuit.compile_detector_sampler(seed=seed)
    return sampler.sample(shots=n_shots, bit_packed=False).astype(np.uint8)


def _build_device_batch(syndromes: np.ndarray, distance: int):
    import stim

    from kernels.graph_build import build_fired_detector_graphs, metadata_to_device
    from sampling.graph import extract_circuit_metadata

    path = f"data/circuits/memory/d{distance}_r{distance}_p0_01.stim"
    circuit = stim.Circuit.from_file(path)
    metadata = extract_circuit_metadata(circuit, distance=distance, rounds=distance)
    dev_meta = metadata_to_device(metadata, "cuda")
    device_syndromes = torch.as_tensor(syndromes, device="cuda")
    return build_fired_detector_graphs(
        device_syndromes,
        dev_meta.coords,
        distance=distance,
        rounds=distance,
        dem_weights=dev_meta.dem_weights,
    )


class TestLadder:
    def test_exact_match(self) -> None:
        from kernels.bucketed import _rung_for

        assert _rung_for(8, (8, 16, 32), "nodes") == 8

    def test_rounds_up(self) -> None:
        from kernels.bucketed import _rung_for

        assert _rung_for(5, (8, 16, 32), "nodes") == 8

    def test_exceeds_ladder_raises(self) -> None:
        from kernels.bucketed import _rung_for

        with pytest.raises(ValueError, match="largest nodes rung"):
            _rung_for(100, (8, 16, 32), "nodes")

    def test_one_node(self) -> None:
        from kernels.bucketed import _rung_for

        assert _rung_for(1, (8, 16, 32), "nodes") == 8

    def test_geometric_ladder_is_ascending_and_aligned(self) -> None:
        from kernels.bucketed import _geometric_ladder

        ladder = _geometric_ladder(start=8, stop=1000, ratio=1.25, align=8)

        assert list(ladder) == sorted(set(ladder))
        assert all(rung % 8 == 0 for rung in ladder)
        assert ladder[0] == 8
        assert ladder[-1] >= 1000 or ladder[-1] * 1.25 > 1000

    def test_geometric_ladder_bounds_waste(self) -> None:
        """Each step is bounded by the ratio plus one alignment quantum.

        Alignment, not the ratio, dominates the low rungs - 32 to 64 is a
        factor of two - but the absolute waste there is a few dozen edges.
        The asymptotic bound is what matters, so it is asserted separately.
        """
        from kernels.bucketed import _geometric_ladder

        ladder = _geometric_ladder(start=32, stop=100_000, ratio=1.25, align=32)

        for lower, upper in zip(ladder, ladder[1:], strict=False):
            assert upper <= lower * 1.25 + 32

        large = [r for r in ladder if r >= 4096]
        for lower, upper in zip(large, large[1:], strict=False):
            assert upper / lower <= 1.25 * 1.01

    def test_default_ladders_span_the_measured_workload(self) -> None:
        """d=7 at batch 512 is the largest shape the benchmarks exercise."""
        from kernels.bucketed import DEFAULT_EDGE_BUCKETS, DEFAULT_NODE_BUCKETS

        assert DEFAULT_NODE_BUCKETS[-1] >= 23_000
        assert DEFAULT_EDGE_BUCKETS[-1] >= 509_000
        assert DEFAULT_NODE_BUCKETS[0] <= 8
        assert DEFAULT_EDGE_BUCKETS[0] <= 32


class TestBucketedRunnerEquivalence:
    @pytest.mark.parametrize("distance", [3, 5, 7])
    def test_matches_eager(self, distance: int) -> None:
        from kernels.bucketed import BucketedGraphRunner

        model = _make_model()
        syndromes = _make_syndromes(distance, n_shots=16)
        batch = _build_device_batch(syndromes, distance)

        with torch.no_grad():
            ref = model(batch)

        runner = BucketedGraphRunner(model, batch_size=16)
        out = runner.forward_from_batch(batch)

        torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)

    def test_single_shot_batch(self) -> None:
        """B=1 is the target latency point and the worst case for sink waste."""
        from kernels.bucketed import BucketedGraphRunner

        model = _make_model()
        syndromes = _make_syndromes(7, n_shots=1)
        batch = _build_device_batch(syndromes, 7)

        with torch.no_grad():
            ref = model(batch)

        runner = BucketedGraphRunner(model, batch_size=1)
        out = runner.forward_from_batch(batch)

        torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)

    def test_explicit_forward_signature(self) -> None:
        from kernels.bucketed import BucketedGraphRunner

        model = _make_model()
        syndromes = _make_syndromes(3, n_shots=16)
        batch = _build_device_batch(syndromes, 3)

        with torch.no_grad():
            ref = model(batch)

        runner = BucketedGraphRunner(model, batch_size=16)
        out = runner.forward(
            batch.x,
            batch.edge_index,
            batch.edge_attr,
            batch.batch,
            batch.num_graphs,
        )

        torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)


class TestNoStaleState:
    """The pad step clears only slack, so reuse is where leaks would appear."""

    def test_large_then_small_batch_same_bucket(self) -> None:
        from kernels.bucketed import BucketedGraphRunner

        model = _make_model()
        dense = np.zeros((8, 24), dtype=np.uint8)
        dense[:, :12] = 1  # 96 nodes total
        sparse = np.zeros((8, 24), dtype=np.uint8)
        sparse[0, 0] = 1
        sparse[1, [2, 3]] = 1  # 3 nodes total

        dense_batch = _build_device_batch(dense, 3)
        sparse_batch = _build_device_batch(sparse, 3)

        with torch.no_grad():
            sparse_ref = model(sparse_batch)

        # One bucket wide enough for both, so the second call reuses the
        # buffers the first one filled.
        runner = BucketedGraphRunner(
            model, batch_size=8, node_buckets=(128,), edge_buckets=(2048,)
        )
        runner.forward_from_batch(dense_batch)
        out = runner.forward_from_batch(sparse_batch)

        assert len(runner._buckets) == 1
        torch.testing.assert_close(out, sparse_ref, atol=1e-5, rtol=1e-5)

    def test_repeated_calls_are_stable(self) -> None:
        """Replaying the same input must not drift.

        Not bitwise: the encoder's neighbourhood aggregation and the head's
        pooling both scatter with atomic float adds, whose summation order
        varies run to run.  The observed spread is ~5e-7, so 1e-5 still fails
        loudly on a stale-buffer bug, which perturbs logits by far more.
        """
        from kernels.bucketed import BucketedGraphRunner

        model = _make_model()
        syndromes = _make_syndromes(5, n_shots=8)
        batch = _build_device_batch(syndromes, 5)

        runner = BucketedGraphRunner(model, batch_size=8)
        first = runner.forward_from_batch(batch).clone()
        for _ in range(3):
            again = runner.forward_from_batch(batch)
            torch.testing.assert_close(again, first, atol=1e-5, rtol=1e-5)

    def test_same_bucket_reused_across_batches(self) -> None:
        from kernels.bucketed import BucketedGraphRunner

        model = _make_model()
        runner = BucketedGraphRunner(
            model, batch_size=8, node_buckets=(256,), edge_buckets=(8192,)
        )

        runner.forward_from_batch(_build_device_batch(_make_syndromes(3, 8), 3))
        assert len(runner._buckets) == 1

        runner.forward_from_batch(
            _build_device_batch(_make_syndromes(3, 8, seed=99), 3)
        )
        assert len(runner._buckets) == 1


class TestPrewarm:
    """Captures must be movable off the latency path.

    Rungs are selected from batch totals, which vary shot to shot, so a live
    stream keeps reaching first-time buckets long after startup - 24 distinct
    rung pairs over 20k shots at d=7/B=1.  Each is a capture, and a capture in
    the serving stream stalls it far past the p50 this path exists to deliver.
    """

    def test_prewarmed_batch_captures_nothing_on_the_latency_path(self) -> None:
        from kernels.bucketed import BucketedGraphRunner

        model = _make_model()
        runner = BucketedGraphRunner(model, batch_size=4)
        batch = _build_device_batch(_make_syndromes(5, 4), 5)

        assert runner.prewarm([(batch.total_nodes, batch.total_edges)]) == 1

        captured = len(runner._buckets)
        runner.forward_from_batch(batch)
        assert len(runner._buckets) == captured, "forward captured despite prewarm"

    def test_prewarm_is_idempotent(self) -> None:
        from kernels.bucketed import BucketedGraphRunner

        model = _make_model()
        runner = BucketedGraphRunner(model, batch_size=4)
        batch = _build_device_batch(_make_syndromes(5, 4), 5)
        totals = [(batch.total_nodes, batch.total_edges)]

        assert runner.prewarm(totals) == 1
        assert runner.prewarm(totals) == 0

    def test_prewarm_output_matches_a_lazily_captured_run(self) -> None:
        """Prewarming must change only *when* capture happens, not the result."""
        from kernels.bucketed import BucketedGraphRunner

        model = _make_model()
        batch = _build_device_batch(_make_syndromes(5, 4), 5)

        lazy = BucketedGraphRunner(model, batch_size=4).forward_from_batch(batch)
        lazy = lazy.clone()

        warm = BucketedGraphRunner(model, batch_size=4)
        warm.prewarm([(batch.total_nodes, batch.total_edges)])

        torch.testing.assert_close(
            warm.forward_from_batch(batch), lazy, atol=1e-5, rtol=1e-5
        )

    def test_prewarm_skips_empty_batches(self) -> None:
        """A zero-node batch short-circuits, so it selects no bucket."""
        from kernels.bucketed import BucketedGraphRunner

        runner = BucketedGraphRunner(_make_model(), batch_size=4)
        assert runner.prewarm([(0, 0)]) == 0
        assert not runner._buckets

    def test_prewarm_fails_loudly_out_of_ladder(self) -> None:
        """The fail-loud rule surfaces at setup instead of mid-stream."""
        from kernels.bucketed import BucketedGraphRunner

        runner = BucketedGraphRunner(
            _make_model(), batch_size=4, node_buckets=(64,), edge_buckets=(4096,)
        )
        with pytest.raises(ValueError, match="largest nodes rung"):
            runner.prewarm([(65, 128)])


class TestDegenerateCases:
    def test_all_empty_syndromes(self) -> None:
        from kernels.bucketed import BucketedGraphRunner

        model = _make_model()
        batch = _build_device_batch(np.zeros((8, 24), dtype=np.uint8), 3)

        runner = BucketedGraphRunner(model, batch_size=8)
        out = runner.forward_from_batch(batch)

        assert out.shape == (8, 1)
        assert (out == 0).all()

    def test_mixed_empty_and_nonempty(self) -> None:
        from kernels.bucketed import BucketedGraphRunner

        model = _make_model()
        syndromes = np.zeros((8, 24), dtype=np.uint8)
        syndromes[1, 0] = 1
        syndromes[3, [0, 5, 11]] = 1
        syndromes[5, :10] = 1
        batch = _build_device_batch(syndromes, 3)

        with torch.no_grad():
            ref = model(batch)

        runner = BucketedGraphRunner(model, batch_size=8)
        out = runner.forward_from_batch(batch)

        torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)

    def test_single_detector_shot_has_no_edges(self) -> None:
        from kernels.bucketed import BucketedGraphRunner

        model = _make_model()
        syndromes = np.zeros((4, 24), dtype=np.uint8)
        syndromes[2, 7] = 1
        batch = _build_device_batch(syndromes, 3)
        assert batch.total_edges == 0

        with torch.no_grad():
            ref = model(batch)

        runner = BucketedGraphRunner(model, batch_size=4)
        out = runner.forward_from_batch(batch)

        torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)


class TestValidation:
    def test_wrong_batch_size_raises(self) -> None:
        from kernels.bucketed import BucketedGraphRunner

        model = _make_model()
        runner = BucketedGraphRunner(model, batch_size=16)

        with pytest.raises(ValueError, match="batch_size"):
            runner.forward(
                torch.zeros(1, 6).cuda(),
                torch.zeros(2, 0, dtype=torch.int64).cuda(),
                torch.zeros(0, 6).cuda(),
                torch.zeros(1, dtype=torch.int64).cuda(),
                num_graphs=8,
            )

    def test_exceeds_node_ladder_raises(self) -> None:
        from kernels.bucketed import BucketedGraphRunner

        model = _make_model()
        runner = BucketedGraphRunner(
            model, batch_size=1, node_buckets=(4,), edge_buckets=(64,)
        )

        with pytest.raises(ValueError, match="largest nodes rung"):
            runner.forward(
                torch.randn(5, 6).cuda(),
                torch.zeros(2, 20, dtype=torch.int64).cuda(),
                torch.zeros(20, 6).cuda(),
                torch.zeros(5, dtype=torch.int64).cuda(),
                num_graphs=1,
            )

    def test_exceeds_edge_ladder_raises(self) -> None:
        """The edge axis fails independently of the node axis."""
        from kernels.bucketed import BucketedGraphRunner

        model = _make_model()
        runner = BucketedGraphRunner(
            model, batch_size=1, node_buckets=(64,), edge_buckets=(16,)
        )

        with pytest.raises(ValueError, match="largest edges rung"):
            runner.forward(
                torch.randn(5, 6).cuda(),
                torch.zeros(2, 20, dtype=torch.int64).cuda(),
                torch.zeros(20, 6).cuda(),
                torch.zeros(5, dtype=torch.int64).cuda(),
                num_graphs=1,
            )

    def test_cpu_model_raises(self) -> None:
        from kernels.bucketed import BucketedGraphRunner
        from model.decoder import build_model

        model = build_model(hidden_dim=16, num_layers=1).eval()
        with pytest.raises(ValueError, match="CUDA"):
            BucketedGraphRunner(model, batch_size=1)

    def test_zero_batch_size_raises(self) -> None:
        from kernels.bucketed import BucketedGraphRunner

        model = _make_model()
        with pytest.raises(ValueError, match="batch_size must be >= 1"):
            BucketedGraphRunner(model, batch_size=0)
