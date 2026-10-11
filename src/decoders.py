"""QEC decoders behind one protocol.

Thin adapters that give the GNN and the classical baselines - MWPM and
correlated matching (PyMatching), Belief-Matching and Tesseract - a uniform
``Decoder`` interface (``decode_batch(syndromes) -> observables``). A baseline
takes only its circuit: its configuration is fixed here, so every number it
produces was decoded the same way.
"""

from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np
import pymatching
import stim
import tesseract_decoder.tesseract as tesseract
import tesseract_decoder.utils as tesseract_utils
import torch
from beliefmatching import BeliefMatching
from numpy.typing import NDArray
from torch_geometric.data import Data

from model.decoder import (
    QECDecoder,
    build_model,
    collate_graphs,
    require_model_contract,
)
from sampling.graph import CircuitMetadata
from sampling.representation import (
    DataContract,
    require_contract,
    requires_layout,
    resolve_builder,
)


@runtime_checkable
class Decoder(Protocol):
    """Protocol for decoders usable in the evaluation harness.

    Parameters
    ----------
    syndromes : ndarray, shape ``(N, num_detectors)``
        Batch of binary syndrome vectors.

    Returns
    -------
    ndarray, shape ``(N, num_observables)``
        Predicted observable flips in {0, 1}.
    """

    @property
    def name(self) -> str: ...

    def decode_batch(self, syndromes: NDArray[np.uint8]) -> NDArray[np.uint8]: ...


def _decomposed_dem(circuit_path: Path) -> stim.DetectorErrorModel:
    """The circuit's DEM with hyperedges decomposed into graph edges."""
    circuit = stim.Circuit.from_file(str(circuit_path))
    return circuit.detector_error_model(decompose_errors=True)


def _matched_observables(
    matching: pymatching.Matching,
    syndromes: NDArray[np.uint8],
    num_observables: int,
    *,
    correlated: bool,
) -> NDArray[np.uint8]:
    """Decode a batch and keep the observable columns, as uint8."""
    predictions = matching.decode_batch(syndromes, enable_correlations=correlated)
    # Without ``return_weights`` PyMatching returns the predictions alone; its
    # annotation is the union over both modes.
    assert isinstance(predictions, np.ndarray)
    return predictions[:, :num_observables].astype(np.uint8, copy=False)


class PyMatchingDecoder:
    """MWPM decoder constructed directly from a detector error model.

    Parameters
    ----------
    circuit_path : Path
        Path to the Stim circuit file.
    """

    def __init__(self, circuit_path: Path) -> None:
        dem = _decomposed_dem(circuit_path)
        self._matching = pymatching.Matching.from_detector_error_model(dem)
        self._num_obs = dem.num_observables

    @property
    def name(self) -> str:
        return "mwpm"

    def decode_batch(self, syndromes: NDArray[np.uint8]) -> NDArray[np.uint8]:
        """Decode syndromes via minimum-weight perfect matching.

        Parameters
        ----------
        syndromes : ndarray, shape ``(N, D)``
            Binary syndrome vectors.

        Returns
        -------
        ndarray, shape ``(N, num_observables)``
            Predicted observable flips.
        """
        return _matched_observables(
            self._matching, syndromes, self._num_obs, correlated=False
        )


class CorrelatedMatchingDecoder:
    """Two-pass correlated matching decoder via PyMatching 2.

    Exploits hyperedge error decompositions to improve over standard MWPM.
    The first pass runs ordinary matching; the second re-weights edges using
    the first-pass solution and re-matches, capturing correlations that plain
    MWPM misses (e.g. Y errors in the surface code).

    Parameters
    ----------
    circuit_path : Path
        Path to the Stim circuit file.

    References
    ----------
    Algorithm: Fowler, arXiv:1310.0863.
    Implementation: Higgott & Gidney, arXiv:2303.15933 (PyMatching 2).
    """

    def __init__(self, circuit_path: Path) -> None:
        dem = _decomposed_dem(circuit_path)
        self._matching = pymatching.Matching.from_detector_error_model(
            dem, enable_correlations=True
        )
        self._num_obs = dem.num_observables

    @property
    def name(self) -> str:
        return "correlated"

    def decode_batch(self, syndromes: NDArray[np.uint8]) -> NDArray[np.uint8]:
        """Decode syndromes via two-pass correlated matching.

        Parameters
        ----------
        syndromes : ndarray, shape ``(N, D)``
            Binary syndrome vectors.

        Returns
        -------
        ndarray, shape ``(N, num_observables)``
            Predicted observable flips.
        """
        return _matched_observables(
            self._matching, syndromes, self._num_obs, correlated=True
        )


class GNNDecoder:
    """GNN decoder wrapping a trained model for batch evaluation.

    Builds fired-detector graphs from syndromes, runs the model forward,
    and applies a decision threshold.

    Parameters
    ----------
    model : QECDecoder
        Trained model; put in eval mode.
    metadata : CircuitMetadata
        Circuit metadata for graph construction.
    contract : DataContract
        Contract the model was trained under; selects the graph builder and
        must match the model's input and output widths.
    threshold : float
        Decision boundary for logit => binary prediction.
    device : torch.device
        Device for inference.
    batch_size : int
        Inference batch size.

    Raises
    ------
    ValueError
        If the model's widths are not the contract's, or the contract's
        representation reads a logical-operation layout the metadata lacks.
    """

    def __init__(
        self,
        model: QECDecoder,
        metadata: CircuitMetadata,
        *,
        contract: DataContract,
        threshold: float = 0.0,
        device: torch.device | None = None,
        batch_size: int = 256,
    ) -> None:
        require_model_contract(model, contract)
        if requires_layout(contract.version) and metadata.layout is None:
            raise ValueError(
                f"Contract representation {contract.version!r} reads a "
                f"logical-operation layout, but the CircuitMetadata lacks phased "
                f"fields. The eval set must carry the metadata its representation "
                f"needs."
            )
        self._model = model
        self._metadata = metadata
        self._threshold = threshold
        self._device = device or torch.device("cpu")
        self._batch_size = batch_size
        self._contract = contract
        self._builder = resolve_builder(contract.version)
        self._model.eval()

    @property
    def name(self) -> str:
        return "gnn"

    def decode_batch(self, syndromes: NDArray[np.uint8]) -> NDArray[np.uint8]:
        """Decode syndromes via GNN inference.

        Empty syndromes (zero fired detectors) short-circuit to no-flip
        without reaching the model forward pass; every other shot is
        predicted as its :meth:`logits` above the decision threshold.

        Parameters
        ----------
        syndromes : ndarray, shape ``(N, D)``
            Binary syndrome vectors.

        Returns
        -------
        ndarray, shape ``(N, num_observables)``
            Predicted observable flips.
        """
        fired = syndromes.any(axis=1)
        predictions = np.zeros(
            (syndromes.shape[0], self._contract.num_observables), dtype=np.uint8
        )
        predictions[fired] = self.logits(syndromes[fired]) > self._threshold
        return predictions

    def logits(self, syndromes: NDArray[np.uint8]) -> NDArray[np.float32]:
        """Model logits of shots that each carry at least one detection event.

        A shot without a detection event has no graph; :meth:`decode_batch`
        decides it without the model, and a calibration excludes it.

        Parameters
        ----------
        syndromes : ndarray, shape ``(N, D)``
            Binary syndrome vectors, none of them all-zero.

        Returns
        -------
        ndarray, shape ``(N, num_observables)``, float32
            Logit of each observable flip, in shot order.

        Raises
        ------
        ValueError
            If a shot carries no detection event.
        """
        empty = np.flatnonzero(~syndromes.any(axis=1))
        if empty.size:
            raise ValueError(
                f"{empty.size} of {syndromes.shape[0]} shot(s) carry no detection "
                f"event (first at row {empty[0]}); such a shot has no graph to "
                f"decode and is decided without the model"
            )
        logits = np.empty(
            (syndromes.shape[0], self._contract.num_observables), dtype=np.float32
        )
        for start in range(0, syndromes.shape[0], self._batch_size):
            stop = start + self._batch_size
            logits[start:stop] = self._forward(syndromes[start:stop])
        return logits

    def _forward(self, syndromes: NDArray[np.uint8]) -> NDArray[np.float32]:
        """Logits ``(B, num_observables)`` of one inference batch."""
        graphs = []
        for syndrome in syndromes:
            graph = self._builder(syndrome, self._metadata)
            graphs.append(
                Data(
                    x=torch.from_numpy(graph.node_features),
                    edge_index=torch.from_numpy(graph.edge_index),
                    edge_attr=torch.from_numpy(graph.edge_features),
                )
            )
        batch = collate_graphs(graphs, self._device)
        with (
            torch.inference_mode(),
            torch.autocast(
                device_type=self._device.type,
                enabled=self._device.type == "cuda",
                dtype=torch.bfloat16,
            ),
        ):
            # bf16 widens to float32 exactly, so the threshold compares the
            # logits the model produced.
            return self._model(batch).float().cpu().numpy()

    @classmethod
    def from_metadata(
        cls,
        model: QECDecoder,
        metadata: CircuitMetadata,
        *,
        contract: DataContract,
        threshold: float = 0.0,
        device: torch.device | None = None,
        batch_size: int = 256,
    ) -> GNNDecoder:
        """Construct from an already-loaded model, moving it to *device*.

        Parameters
        ----------
        model : QECDecoder
            Model instance (will be put in eval mode).
        metadata : CircuitMetadata
            Pre-extracted circuit metadata.
        contract : DataContract
            Data contract (controls output shape and builder selection).
        threshold : float
            Decision threshold for logit => prediction.
        device : torch.device or None
            Target device.
        batch_size : int
            Inference batch size.

        Returns
        -------
        GNNDecoder
        """
        if device is None:
            device = torch.device("cpu")
        return cls(
            model.to(device),
            metadata,
            contract=contract,
            threshold=threshold,
            device=device,
            batch_size=batch_size,
        )


@dataclass(frozen=True, slots=True)
class GNNCheckpoint:
    """A trained GNN restored for inference, with what its checkpoint records.

    Parameters
    ----------
    path : Path
        Checkpoint file the model was restored from.
    sha256 : str
        Hex sha256 of the bytes the model was restored from.
    model : QECDecoder
        Restored model, in eval mode on the requested device.
    contract : DataContract
        Contract the model was trained under.
    decision_threshold : float
        Logit threshold the trainer calibrated on validation.
    samples_consumed : int
        Training samples consumed when the checkpoint was written.
    """

    path: Path
    sha256: str
    model: QECDecoder
    contract: DataContract
    decision_threshold: float
    samples_consumed: int

    def provenance(self) -> dict[str, Any]:
        """What a result scored by this model records of it, JSON-ready.

        The checkpoint is named by path and by the sha256 of the bytes the
        model was restored from, with the threshold and samples it recorded.
        """
        return {
            "path": str(self.path),
            "sha256": self.sha256,
            "samples_consumed": self.samples_consumed,
            "decision_threshold": self.decision_threshold,
        }


def load_gnn_checkpoint(
    path: Path, *, contract: DataContract, device: torch.device
) -> GNNCheckpoint:
    """Restore a trained GNN for inference under the contract it must serve.

    Parameters
    ----------
    path : Path
        Checkpoint written by the trainer.
    contract : DataContract
        The contract of the operation being decoded, i.e. its profile's
        ``data_contract``.
    device : torch.device
        Device the model is restored onto.

    Returns
    -------
    GNNCheckpoint

    Raises
    ------
    KeyError
        If the checkpoint lacks a key the trainer always writes, naming it.
    ValueError
        If the checkpoint was trained under a different contract.
    """
    # One read serves both the hash and the load, so the digest a result
    # records is that of the model it scored, whatever the path holds later.
    data = path.read_bytes()
    checkpoint = torch.load(io.BytesIO(data), weights_only=False, map_location=device)
    try:
        config = checkpoint["config"]
        saved_contract = DataContract.from_checkpoint_config(config)
        hidden_dim = config["hidden_dim"]
        num_layers = config["num_layers"]
        state_dict = checkpoint["model_state_dict"]
        decision_threshold = float(checkpoint["decision_threshold"])
        samples_consumed = int(checkpoint["samples_consumed"])
    except KeyError as exc:
        raise KeyError(f"Checkpoint {path} has no {exc.args[0]!r}") from exc
    require_contract(contract, saved_contract, source=path)

    model = build_model(
        node_dim=contract.node_dim,
        edge_dim=contract.edge_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_observables=contract.num_observables,
        dropout=0.0,
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return GNNCheckpoint(
        path=path,
        sha256=hashlib.sha256(data).hexdigest(),
        model=model,
        contract=contract,
        decision_threshold=decision_threshold,
        samples_consumed=samples_consumed,
    )


class BeliefMatchingDecoder:
    """Belief-Matching: belief propagation re-weighting minimum-weight matching.

    Product-sum BP, at most 20 iterations, runs on the DEM's hyperedges. A shot
    on which BP converges is predicted from BP's correction; any other is
    matched with edge weights from BP's posteriors. Runs shot by shot on CPU.

    Parameters
    ----------
    circuit_path : Path
        Path to the Stim circuit file.

    References
    ----------
    Higgott, Bohdanowicz, Kubica, Flammia & Campbell, arXiv:2203.04948.
    """

    def __init__(self, circuit_path: Path) -> None:
        self._decoder = BeliefMatching(
            _decomposed_dem(circuit_path), max_bp_iters=20, bp_method="product_sum"
        )

    @property
    def name(self) -> str:
        return "belief_matching"

    def decode_batch(self, syndromes: NDArray[np.uint8]) -> NDArray[np.uint8]:
        """Decode syndromes via Belief-Matching.

        Parameters
        ----------
        syndromes : ndarray, shape ``(N, D)``
            Binary syndrome vectors.

        Returns
        -------
        ndarray, shape ``(N, num_observables)``
            Predicted observable flips.
        """
        return self._decoder.decode_batch(syndromes).astype(np.uint8)


class TesseractDecoder:
    """Near-MLE decoder: beam search over explanations of the detection events.

    Searches the circuit's undecomposed DEM, whose error mechanisms it treats
    as hyperedges, with the package's default configuration - the long-beam
    setup of the Tesseract paper: beam 20 with beam climbing, 21 detector
    orders, a priority queue of at most one million states.

    Parameters
    ----------
    circuit_path : Path
        Path to the Stim circuit file.

    References
    ----------
    Aghababaie Beni, Higgott & Shutty, arXiv:2503.10988.
    """

    def __init__(self, circuit_path: Path) -> None:
        dem = stim.Circuit.from_file(str(circuit_path)).detector_error_model()
        config = tesseract.TesseractConfig(
            dem=dem,
            det_beam=20,
            beam_climbing=True,
            no_revisit_dets=True,
            merge_errors=True,
            pqlimit=1_000_000,
            det_orders=tesseract_utils.build_det_orders(
                dem=dem,
                num_det_orders=21,
                method=tesseract_utils.DetOrder.DetIndex,
                seed=2384753,
            ),
        )
        self._decoder = tesseract.TesseractDecoder(config)

    @property
    def name(self) -> str:
        return "tesseract"

    def decode_batch(self, syndromes: NDArray[np.uint8]) -> NDArray[np.uint8]:
        """Decode syndromes via near-MLE beam search.

        Parameters
        ----------
        syndromes : ndarray, shape ``(N, D)``
            Binary syndrome vectors.

        Returns
        -------
        ndarray, shape ``(N, num_observables)``
            Predicted observable flips.
        """
        # The binding annotates its argument and result ``ndarray[bool, ...]``,
        # with ``bool`` in the shape slot; both are boolean 2-D arrays.
        predictions = self._decoder.decode_batch(
            syndromes.astype(np.bool_)  # pyright: ignore[reportArgumentType]
        )
        return np.asarray(predictions, dtype=np.uint8)


__all__ = [
    "BeliefMatchingDecoder",
    "CorrelatedMatchingDecoder",
    "Decoder",
    "GNNCheckpoint",
    "GNNDecoder",
    "PyMatchingDecoder",
    "TesseractDecoder",
    "load_gnn_checkpoint",
]
