# morie.fn — function file from book-equation translation pipeline (hadesllm/morie)
"""Analytic Hierarchy Process (AHP) weights."""

import numpy as np

from ._containers import DescriptiveResult


def ahp_weights(pairwise_matrix: np.ndarray) -> DescriptiveResult:
    """
    Compute AHP priority weights from a pairwise comparison matrix.

    Uses the eigenvector method: the priority vector is the normalised
    principal eigenvector of the comparison matrix.

    :param pairwise_matrix: (n, n) positive reciprocal matrix.
    :return: DescriptiveResult with weights and consistency ratio.
    :raises ValueError: If matrix not square.

    References
    ----------
    Saaty TL (1980). The Analytic Hierarchy Process. McGraw-Hill.
    """
    A = np.asarray(pairwise_matrix, dtype=np.float64)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("Matrix must be square.")
    n = A.shape[0]
    eigvals, eigvecs = np.linalg.eig(A)
    idx = np.argmax(np.real(eigvals))
    lambda_max = float(np.real(eigvals[idx]))
    w = np.real(eigvecs[:, idx])
    w = np.abs(w)
    if w.sum() < 1e-12:
        col_sums = A.sum(axis=0)
        w = (A / col_sums[np.newaxis, :]).mean(axis=1)
    else:
        w = w / w.sum()
    if np.max(np.real(eigvals)) - np.min(np.real(eigvals)) < 1e-10:
        col_sums = A.sum(axis=0)
        w = (A / col_sums[np.newaxis, :]).mean(axis=1)
    CI = (lambda_max - n) / (n - 1) if n > 1 else 0.0
    RI_table = {1: 0, 2: 0, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49}
    RI = RI_table.get(n, 1.49)
    CR = CI / RI if RI > 0 else 0.0
    consistent = CR < 0.1
    return DescriptiveResult(
        name="ahp_weights",
        value=float(CR),
        extra={"weights": w, "lambda_max": lambda_max, "CI": CI, "CR": CR, "RI": RI, "consistent": consistent, "n": n},
    )


ahp = ahp_weights


def cheatsheet() -> str:
    return "ahp_weights({}) -> Analytic Hierarchy Process (AHP) weights."
