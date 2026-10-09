"""QEC decoder implementations with a shared protocol.

Thin adapters that give GNN models, PyMatching, and Belief-Matching a uniform
``Decoder`` interface (``decode_batch(syndromes) -> observables``).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np
import pymatching
import stim
import torch
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


logger = logging.getLogger(__name__)


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
        circuit = stim.Circuit.from_file(str(circuit_path))
        dem = circuit.detector_error_model(decompose_errors=True)
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
        circuit = stim.Circuit.from_file(str(circuit_path))
        dem = circuit.detector_error_model(decompose_errors=True)
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
        without reaching the model forward pass.

        Parameters
        ----------
        syndromes : ndarray, shape ``(N, D)``
            Binary syndrome vectors.

        Returns
        -------
        ndarray, shape ``(N, num_observables)``
            Predicted observable flips.
        """
        n_shots = syndromes.shape[0]
        num_obs = self._contract.num_observables
        predictions = np.zeros((n_shots, num_obs), dtype=np.uint8)
        use_amp = self._device.type == "cuda"

        for start in range(0, n_shots, self._batch_size):
            end = min(start + self._batch_size, n_shots)
            data_list: list[Data] = []
            nonempty_indices: list[int] = []

            for i in range(start, end):
                if int(syndromes[i].sum()) == 0:
                    continue

                graph = self._builder(syndromes[i], self._metadata)
                if graph.num_fired == 0:
                    continue

                data_list.append(
                    Data(
                        x=torch.from_numpy(graph.node_features),
                        edge_index=torch.from_numpy(graph.edge_index),
                        edge_attr=torch.from_numpy(graph.edge_features),
                        num_fired=torch.tensor(graph.num_fired, dtype=torch.long),
                    )
                )
                nonempty_indices.append(i - start)

            if not data_list:
                continue

            batch = collate_graphs(data_list, self._device)

            with (
                torch.no_grad(),
                torch.autocast(
                    device_type=self._device.type,
                    enabled=use_amp,
                    dtype=torch.bfloat16,
                ),
            ):
                logits = self._model(batch)

            preds = (logits > self._threshold).cpu().numpy().astype(np.uint8)
            if preds.ndim == 1:
                preds = preds[:, np.newaxis]

            for pred_idx, local_idx in enumerate(nonempty_indices):
                predictions[start + local_idx] = preds[pred_idx]

        return predictions

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
    model: QECDecoder
    contract: DataContract
    decision_threshold: float
    samples_consumed: int


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
    checkpoint = torch.load(path, weights_only=False, map_location=device)
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
        model=model,
        contract=contract,
        decision_threshold=decision_threshold,
        samples_consumed=samples_consumed,
    )


class BeliefMatchingDecoder:
    """Belief-Matching decoder combining BP soft information with matching.

    Uses the ``beliefmatching`` package (Oscar Higgott) which runs BP to
    compute soft edge weights, then feeds those into PyMatching for
    minimum-weight perfect matching. Often outperforms both plain MWPM
    and BP+OSD.

    Parameters
    ----------
    circuit_path : Path
        Path to the Stim circuit file.
    max_bp_iters : int
        Maximum belief propagation iterations.
    """

    def __init__(
        self,
        circuit_path: Path,
        *,
        max_bp_iters: int = 20,
    ) -> None:
        try:
            from beliefmatching import BeliefMatching
        except ImportError as exc:
            raise ImportError(
                "Belief-Matching decoder requires beliefmatching: "
                "pip install beliefmatching"
            ) from exc

        circuit = stim.Circuit.from_file(str(circuit_path))
        dem = circuit.detector_error_model(decompose_errors=True)
        self._num_obs = dem.num_observables
        self._decoder = BeliefMatching(dem, max_bp_iters=max_bp_iters)

        logger.debug(
            "BeliefMatchingDecoder: %d detectors, max_bp_iters=%d",
            dem.num_detectors,
            max_bp_iters,
        )

    @property
    def name(self) -> str:
        return "belief_matching"

    def decode_batch(self, syndromes: NDArray[np.uint8]) -> NDArray[np.uint8]:
        """Decode syndromes via belief-matching.

        Parameters
        ----------
        syndromes : ndarray, shape ``(N, D)``
            Binary syndrome vectors.

        Returns
        -------
        ndarray, shape ``(N, num_observables)``
            Predicted observable flips.
        """
        n = syndromes.shape[0]
        out = np.empty((n, self._num_obs), dtype=np.uint8)
        for i in range(n):
            pred = self._decoder.decode(syndromes[i])
            out[i] = pred[: self._num_obs].astype(np.uint8)
        return out


class TesseractDecoder:
    """Near-MLE decoder via beam search over the DEM hypergraph.

    Uses ``tesseract-decoder`` (quantumlib) which performs a prioritized
    search over detection-event explanations, approaching maximum-likelihood
    accuracy on surface codes.

    Parameters
    ----------
    circuit_path : Path
        Path to the Stim circuit file.
    det_beam : int
        Beam width for the detector-event search.  Larger values give higher
        accuracy at the cost of runtime.

    References
    ----------
    Fowler et al., arXiv:2503.10988.
    """

    def __init__(
        self,
        circuit_path: Path,
        *,
        det_beam: int = 50,
    ) -> None:
        try:
            from tesseract_decoder.tesseract import (
                TesseractConfig,
            )
            from tesseract_decoder.tesseract import (
                TesseractDecoder as _TesseractDecoder,
            )
        except ImportError as exc:
            raise ImportError(
                "Tesseract decoder requires tesseract-decoder: "
                "pip install tesseract-decoder"
            ) from exc

        circuit = stim.Circuit.from_file(str(circuit_path))
        dem = circuit.detector_error_model(decompose_errors=True)
        self._num_obs = dem.num_observables

        config = TesseractConfig(dem=dem, det_beam=det_beam)
        self._decoder = _TesseractDecoder(config)

        logger.debug(
            "TesseractDecoder: %d detectors, det_beam=%d",
            dem.num_detectors,
            det_beam,
        )

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
        # The binding annotates ``ndarray[bool, ...]``, with ``bool`` in the shape
        # slot; a boolean ``(N, D)`` array is what it decodes.
        predictions = self._decoder.decode_batch(
            syndromes.astype(np.bool_)  # pyright: ignore[reportArgumentType]
        )
        return predictions[:, : self._num_obs].astype(np.uint8, copy=False)


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
