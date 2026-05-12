# morie.fn -- function file (hadesllm/morie)
"""I cannot teach anybody anything. I can only make them think. -- Socrates"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ._containers import DescriptiveResult


def detect_duplicates(
    data: pd.DataFrame,
    *,
    threshold: float = 0.95,
    method: str = "cosine",
) -> DescriptiveResult:
    """Detect near-duplicate rows using pairwise similarity.

    Computes a similarity metric between all pairs of numeric rows and
    flags pairs exceeding *threshold*.

    Parameters
    ----------
    data : DataFrame
        Input data (numeric columns used).
    threshold : float
        Similarity threshold (0--1) above which a pair is flagged.
    method : str
        ``"cosine"`` or ``"jaccard"`` (binarised at median).

    Returns
    -------
    DescriptiveResult
        ``value`` is the number of near-duplicate pairs; ``extra`` has
        the pair indices and their similarities.
    """
    numeric = data.select_dtypes(include="number").dropna()
    if len(numeric) < 2:
        raise ValueError("Need at least 2 complete numeric rows")
    X = numeric.to_numpy(dtype=np.float64)
    n = X.shape[0]

    if method == "cosine":
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        Xn = X / norms
        sim_matrix = Xn @ Xn.T
    elif method == "jaccard":
        medians = np.median(X, axis=0)
        B = (medians < X).astype(float)
        sim_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                inter = float(np.sum(np.minimum(B[i], B[j])))
                union = float(np.sum(np.maximum(B[i], B[j])))
                sim_matrix[i, j] = sim_matrix[j, i] = inter / union if union > 0 else 0.0
    else:
        raise ValueError(f"method must be 'cosine' or 'jaccard', got '{method}'")

    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            if sim_matrix[i, j] >= threshold:
                pairs.append(
                    {
                        "row_i": int(numeric.index[i]),
                        "row_j": int(numeric.index[j]),
                        "similarity": float(sim_matrix[i, j]),
                    }
                )

    return DescriptiveResult(
        name="Near-Duplicate Detection",
        value=len(pairs),
        extra={
            "pairs": pairs[:100],
            "method": method,
            "threshold": threshold,
            "n_rows": n,
        },
    )


deja = detect_duplicates


def cheatsheet() -> str:
    return "detect_duplicates({}) -> Near-duplicate detection. 'A deja vu is usually a glitch in "
