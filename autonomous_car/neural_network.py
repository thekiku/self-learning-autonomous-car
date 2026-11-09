"""A compact NumPy multilayer perceptron used as the driving policy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

import numpy as np


@dataclass
class DenseLayer:
    """Weights and bias for one fully connected layer."""

    weights: np.ndarray
    bias: np.ndarray


class NeuralNetwork:
    """Feed-forward tanh network with a flat, GA-friendly genome."""

    def __init__(
        self,
        input_size: int,
        hidden_sizes: Sequence[int],
        output_size: int,
        rng: np.random.Generator,
        genome: np.ndarray | None = None,
    ) -> None:
        sizes = [input_size, *hidden_sizes, output_size]
        self.layers: List[DenseLayer] = []
        for fan_in, fan_out in zip(sizes[:-1], sizes[1:]):
            # Xavier uniform initialization keeps the variance of activations
            # approximately stable across layers before evolution begins.
            limit = np.sqrt(6.0 / (fan_in + fan_out))
            self.layers.append(
                DenseLayer(
                    rng.uniform(-limit, limit, (fan_out, fan_in)),
                    np.zeros(fan_out, dtype=np.float64),
                )
            )
        if genome is not None:
            self.set_genome(genome)

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        """Return tanh-bounded actions for one normalized observation."""
        activation = np.asarray(inputs, dtype=np.float64)
        for layer in self.layers:
            # Row-major matrix multiplication maps each input feature to every
            # neuron; the bias shifts each neuron's pre-activation.
            activation = layer.weights @ activation + layer.bias
            activation = np.tanh(activation)
        return activation

    def genome(self) -> np.ndarray:
        """Serialize all parameters into one contiguous vector."""
        # A flat representation lets genetic operators treat every weight and
        # bias as a gene while preserving deterministic layer reconstruction.
        return np.concatenate(
            [part.ravel() for layer in self.layers for part in (layer.weights, layer.bias)]
        )

    def set_genome(self, genome: Iterable[float]) -> None:
        """Load a genome and reject malformed parameter vectors explicitly."""
        values = np.asarray(genome, dtype=np.float64)
        expected = self.genome().size
        if values.size != expected:
            raise ValueError(f"Expected genome with {expected} values, got {values.size}.")
        offset = 0
        for layer in self.layers:
            weight_size = layer.weights.size
            layer.weights[...] = values[offset : offset + weight_size].reshape(layer.weights.shape)
            offset += weight_size
            bias_size = layer.bias.size
            layer.bias[...] = values[offset : offset + bias_size]
            offset += bias_size
