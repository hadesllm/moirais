# morie.fn — function file (hadesllm/morie)
"""Heaps' law estimation."""

import numpy as np

from ._containers import ESRes


def heaps_law(tokens: list[str], **kwargs) -> ESRes:
    """
    Estimate Heaps' law parameters from a token sequence.

    .. math::

        V(n) = K \\cdot n^\\beta

    where V(n) is vocabulary size after n tokens.

    :param tokens: List of string tokens.
    :return: ESRes with estimated K and beta.

    References
    ----------
    Heaps HS (1978). Information Retrieval: Computational and
    Theoretical Aspects. Academic Press.
    """
    if len(tokens) < 2:
        raise ValueError("Need at least 2 tokens.")
    n_points = min(100, len(tokens))
    step = max(1, len(tokens) // n_points)
    ns = []
    vs = []
    vocab: set[str] = set()
    for i, tok in enumerate(tokens):
        vocab.add(tok)
        if (i + 1) % step == 0 or i == len(tokens) - 1:
            ns.append(i + 1)
            vs.append(len(vocab))

    log_n = np.log(np.array(ns, dtype=np.float64))
    log_v = np.log(np.array(vs, dtype=np.float64))

    A = np.column_stack([log_n, np.ones(len(log_n))])
    coeffs = np.linalg.lstsq(A, log_v, rcond=None)[0]
    beta = float(coeffs[0])
    K = float(np.exp(coeffs[1]))

    return ESRes(
        measure="heaps_law",
        estimate=beta,
        n=len(tokens),
        extra={"K": K, "beta": beta, "vocab_size": len(vocab)},
    )


heaps = heaps_law


def cheatsheet() -> str:
    return "heaps_law(tokens) -> Heaps' law K, beta estimation."
