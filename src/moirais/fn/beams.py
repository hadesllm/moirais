# moirais.fn — function file from book-equation translation pipeline (hadesllm/moirais)
r"""Beam search decoding.

Maintains multiple candidate hypotheses during sequence generation.

References
----------
Freitag, M., & Al-Onaizan, Y. (2017).
Beam search for neural machine translation.
arXiv preprint arXiv:1702.01806.
"""

__all__ = ["beams"]

import numpy as np
from ._richresult import RichResult


def beams(
    initial_token,
    step_fn,
    max_length=100,
    beam_width=3,
    vocab_size=1000,
    seed=None,
):
    """
    Beam search decoding.

    Parameters
    ----------
    initial_token : int
        Starting token ID.
    step_fn : callable
        Function(token_ids) -> logits. Computes next-token logits.
    max_length : int, optional
        Maximum sequence length. Default 100.
    beam_width : int, optional
        Number of beams to maintain. Default 3.
    vocab_size : int, optional
        Vocabulary size. Default 1000.
    seed : int, optional
        Random seed.

    Returns
    -------
    dict
        Keys: 'sequences', 'scores'.
    """
    rng = np.random.RandomState(seed)

    sequences = [[initial_token]] * beam_width
    scores = np.zeros(beam_width)

    for step in range(max_length - 1):
        candidates = []

        for beam_idx, seq in enumerate(sequences):
            logits = step_fn(np.array(seq))
            probs = np.exp(logits - np.max(logits))
            probs = probs / np.sum(probs)

            top_k_idx = np.argsort(probs)[-beam_width:]
            for token_idx in top_k_idx:
                new_seq = seq + [token_idx]
                new_score = scores[beam_idx] - np.log(probs[token_idx] + 1e-10)
                candidates.append((new_score, new_seq))

        candidates.sort(key=lambda x: x[0])
        scores = np.array([c[0] for c in candidates[:beam_width]])
        sequences = [c[1] for c in candidates[:beam_width]]

    return RichResult(payload={"sequences": sequences, "scores": scores})
