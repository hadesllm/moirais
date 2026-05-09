"""Softmax and temperature-scaled softmax."""

import numpy as np

from ._containers import DescriptiveResult


def softmax(x, temperature=1.0, axis=-1):
    """
    Numerically stable softmax with optional temperature scaling.

    :param x: Input array (any shape).
    :param temperature: Temperature parameter (lower = sharper).
    :param axis: Axis along which to compute softmax.
    :return: DescriptiveResult with probabilities, entropy.

    References
    ----------
    Goodfellow I, Bengio Y, Courville A (2016). Deep Learning. MIT Press.
    """
    x = np.asarray(x, dtype=np.float64)
    if temperature <= 0:
        raise ValueError("Temperature must be positive")
    scaled = x / temperature
    shifted = scaled - np.max(scaled, axis=axis, keepdims=True)
    exp_x = np.exp(shifted)
    probs = exp_x / np.sum(exp_x, axis=axis, keepdims=True)
    entropy = -np.sum(probs * np.log(probs + 1e-300), axis=axis)

    return DescriptiveResult(
        name="softmax",
        value=float(np.mean(entropy)),
        extra={
            "probabilities": probs,
            "entropy": entropy,
            "temperature": float(temperature),
            "max_prob": float(np.max(probs)),
        },
    )


def cheatsheet() -> str:
    return "softmax({}) -> Softmax and temperature-scaled softmax."
