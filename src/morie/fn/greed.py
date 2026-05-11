# morie.fn — function file (hadesllm/morie)
r"""Greedy decoding with temperature.

Selects highest-probability token at each step.

References
----------
Karpukhin, V., Ouz, B., Chan, S., Lewis, M., Wu, L., Wenzek, G., ... & Schwenk, H. (2020).
Dense passage retrieval for open-domain question answering.
In EMNLP (pp. 6897-6912).
"""

__all__ = ["greed"]

import numpy as np
from ._richresult import RichResult


def greed(
    initial_token,
    step_fn,
    max_length=100,
    temperature=1.0,
    seed=None,
):
    """
    Greedy decoding with temperature.

    Parameters
    ----------
    initial_token : int
        Starting token ID.
    step_fn : callable
        Function(token_ids) -> logits.
    max_length : int, optional
        Maximum length. Default 100.
    temperature : float, optional
        Softmax temperature. Default 1.0.
    seed : int, optional
        Random seed.

    Returns
    -------
    dict
        Keys: 'sequence', 'probabilities'.
    """
    rng = np.random.RandomState(seed)

    if temperature <= 0:
        raise ValueError("temperature must be positive")

    sequence = [initial_token]
    probabilities = []

    for step in range(max_length - 1):
        logits = step_fn(np.array(sequence))
        scaled_logits = logits / temperature
        probs = np.exp(scaled_logits - np.max(scaled_logits))
        probs = probs / np.sum(probs)

        next_token = np.argmax(probs)
        sequence.append(int(next_token))
        probabilities.append(float(probs[next_token]))

    return RichResult(payload={"sequence": sequence, "probabilities": probabilities})
