"""QEC decoder implementations with a shared protocol.

Thin adapters that give GNN models, PyMatching, and Belief-Matching a uniform
``Decoder`` interface (``decode_batch(syndromes) -> observables``).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np
import pymatching
import stim
import torch
from numpy.typing import NDArray
from torch_geometric.data import Batch, Data

from sampling.graph import CircuitMetadata
from sampling.representation import SPATIAL_MEMORY, DataContract, resolve_builder


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
        predictions = self._matching.decode_batch(syndromes)
        return predictions[:, : self._num_obs].astype(np.uint8, copy=False)


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
        predictions = self._matching.decode_batch(syndromes, enable_correlations=True)
        return predictions[:, : self._num_obs].astype(np.uint8, copy=False)


class GNNDecoder:
    """GNN decoder wrapping a trained model for batch evaluation.

    Builds fired-detector graphs from syndromes, runs the model forward,
    and applies a decision threshold.

    Parameters
    ----------
    model : torch.nn.Module
        Trained QECDecoder model (eval mode).
    metadata : CircuitMetadata
        Circuit metadata for graph construction.
    threshold : float
        Decision boundary for logit => binary prediction.
    device : torch.device
        Device for inference.
    batch_size : int
        Inference batch size.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        metadata: CircuitMetadata,
        threshold: float = 0.0,
        device: torch.device | None = None,
        batch_size: int = 256,
        contract: DataContract = SPATIAL_MEMORY,
    ) -> None:
        version = contract.version
        has_phased = metadata.phase_ids is not None
        if version != "spatial" and not has_phased:
            raise ValueError(
                f"Contract requires representation {version!r} but the "
                f"CircuitMetadata lacks phased fields (phase_ids, patch_ids, "
                f"seam_mask). The eval set must carry the metadata its "
                f"representation needs."
            )
        self._model = model
        self._metadata = metadata
        self._threshold = threshold
        self._device = device or torch.device("cpu")
        self._batch_size = batch_size
        self._contract = contract
        self._builder = resolve_builder(version)
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

            batch = Batch.from_data_list(data_list).to(self._device)

            with (
                torch.no_grad(),
                torch.amp.autocast(
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
        model: torch.nn.Module,
        metadata: CircuitMetadata,
        threshold: float = 0.0,
        device: torch.device | None = None,
        batch_size: int = 256,
        contract: DataContract = SPATIAL_MEMORY,
    ) -> GNNDecoder:
        """Construct from an already-loaded model and metadata.

        Parameters
        ----------
        model : torch.nn.Module
            Model instance (will be put in eval mode).
        metadata : CircuitMetadata
            Pre-extracted circuit metadata.
        threshold : float
            Decision threshold for logit => prediction.
        device : torch.device or None
            Target device.
        batch_size : int
            Inference batch size.
        contract : DataContract
            Data contract (controls output shape and builder selection).

        Returns
        -------
        GNNDecoder
        """
        if device is None:
            device = torch.device("cpu")
        model = model.to(device)
        return cls(
            model=model,
            metadata=metadata,
            threshold=threshold,
            device=device,
            batch_size=batch_size,
            contract=contract,
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
        predictions = self._decoder.decode_batch(syndromes.astype(np.bool_))
        return predictions[:, : self._num_obs].astype(np.uint8, copy=False)


__all__ = [
    "BeliefMatchingDecoder",
    "CorrelatedMatchingDecoder",
    "Decoder",
    "GNNDecoder",
    "PyMatchingDecoder",
    "TesseractDecoder",
]
