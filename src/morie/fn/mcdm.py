# morie.fn — function file (hadesllm/morie)
"""TOPSIS multi-criteria decision making."""

import numpy as np

from ._containers import DescriptiveResult


def topsis(decision_matrix: np.ndarray, weights: np.ndarray, impacts: list[str]) -> DescriptiveResult:
    """
    TOPSIS (Technique for Order Preference by Similarity to Ideal Solution).

    :param decision_matrix: (n_alternatives, n_criteria) performance matrix.
    :param weights: (n_criteria,) criterion weights (sum to 1).
    :param impacts: List of '+' (benefit) or '-' (cost) per criterion.
    :return: DescriptiveResult with rankings and scores.
    :raises ValueError: If dimensions mismatch.

    References
    ----------
    Hwang CL, Yoon K (1981). Multiple Attribute Decision Making.
    Springer.
    """
    D = np.asarray(decision_matrix, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    m, n = D.shape
    if len(w) != n or len(impacts) != n:
        raise ValueError("weights and impacts must match number of criteria.")
    norms = np.sqrt(np.sum(D**2, axis=0))
    norms[norms == 0] = 1
    R = D / norms
    V = R * w
    ideal_best = np.zeros(n)
    ideal_worst = np.zeros(n)
    for j in range(n):
        if impacts[j] == "+":
            ideal_best[j] = V[:, j].max()
            ideal_worst[j] = V[:, j].min()
        else:
            ideal_best[j] = V[:, j].min()
            ideal_worst[j] = V[:, j].max()
    d_best = np.sqrt(np.sum((V - ideal_best) ** 2, axis=1))
    d_worst = np.sqrt(np.sum((V - ideal_worst) ** 2, axis=1))
    scores = d_worst / (d_best + d_worst + 1e-10)
    ranking = np.argsort(-scores) + 1
    return DescriptiveResult(
        name="topsis",
        value=float(scores.max()),
        extra={"scores": scores, "ranking": ranking, "best_alternative": int(np.argmax(scores)), "m": m, "n": n},
    )


mcdm = topsis


def cheatsheet() -> str:
    return "topsis({}) -> TOPSIS multi-criteria decision making."
