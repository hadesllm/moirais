r"""Top-k and top-p (nucleus) sampling.

Filters low-probability tokens before sampling.

References
----------
Holtzman, A., Buys, J., Du, L., Forbes, M., & Choi, Y. (2019).
The curious case of neural text degeneration.
arXiv preprint arXiv:1910.14599.
"""

__all__ = ["topkp"]

import numpy as np
from ._richresult import RichResult


def topkp(
    logits,
    top_k=None,
    top_p=0.9,
    temperature=1.0,
    seed=None,
):
    """
    Top-k and top-p sampling.

    Parameters
    ----------
    logits : ndarray
        Logits, shape (vocab_size,).
    top_k : int, optional
        Keep top-k tokens. If None, no filtering.
    top_p : float, optional
        Keep tokens with cumulative prob <= top_p. Default 0.9.
    temperature : float, optional
        Softmax temperature. Default 1.0.
    seed : int, optional
        Random seed.

    Returns
    -------
    dict
        Keys: 'token_id', 'probabilities'.
    """
    logits = np.asarray(logits, dtype=float)

    if temperature <= 0:
        raise ValueError("temperature must be positive")

    probs = np.exp((logits - np.max(logits)) / temperature)
    probs = probs / np.sum(probs)

    if top_k is not None:
        top_k_idx = np.argsort(probs)[-top_k:]
        probs_filtered = np.zeros_like(probs)
        probs_filtered[top_k_idx] = probs[top_k_idx]
        probs = probs_filtered / np.sum(probs_filtered)

    if top_p < 1.0:
        sorted_idx = np.argsort(probs)[::-1]
        sorted_probs = probs[sorted_idx]
        cumsum = np.cumsum(sorted_probs)
        valid_idx = np.where(cumsum <= top_p)[0]
        if len(valid_idx) == 0:
            valid_idx = np.array([0])
        keep_idx = sorted_idx[valid_idx]
        probs_filtered = np.zeros_like(probs)
        probs_filtered[keep_idx] = probs[keep_idx]
        probs = probs_filtered / np.sum(probs_filtered)

    rng = np.random.RandomState(seed)
    token_id = rng.choice(len(logits), p=probs)

    return RichResult(payload={"token_id": int(token_id), "probabilities": probs})
