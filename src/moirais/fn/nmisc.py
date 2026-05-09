# moirais.fn — function file (hadesllm/moirais)
"""Normalized mutual information."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def nmi(
    labels_true: np.ndarray,
    labels_pred: np.ndarray,
) -> DescriptiveResult:
    """Normalized Mutual Information between two label assignments.

    Parameters
    ----------
    labels_true : ndarray (n,)
        Ground truth labels.
    labels_pred : ndarray (n,)
        Predicted labels.

    Returns
    -------
    DescriptiveResult
        ``value`` is the NMI in [0, 1].
    """
    y = np.asarray(labels_true)
    yp = np.asarray(labels_pred)
    n = len(y)

    classes_t = np.unique(y)
    classes_p = np.unique(yp)

    contingency = np.zeros((len(classes_t), len(classes_p)))
    for i, ct in enumerate(classes_t):
        for j, cp in enumerate(classes_p):
            contingency[i, j] = np.sum((y == ct) & (yp == cp))

    pi = contingency.sum(axis=1) / n
    pj = contingency.sum(axis=0) / n
    pij = contingency / n

    H_true = -np.sum(pi[pi > 0] * np.log(pi[pi > 0]))
    H_pred = -np.sum(pj[pj > 0] * np.log(pj[pj > 0]))

    mi = 0.0
    for i in range(len(classes_t)):
        for j in range(len(classes_p)):
            if pij[i, j] > 0 and pi[i] > 0 and pj[j] > 0:
                mi += pij[i, j] * np.log(pij[i, j] / (pi[i] * pj[j]))

    denom = (H_true + H_pred) / 2
    nmi_val = mi / denom if denom > 0 else 0.0

    return DescriptiveResult(
        name="NMI",
        value=float(np.clip(nmi_val, 0, 1)),
    )


nmisc = nmi


def cheatsheet() -> str:
    return "nmi({}) -> Normalized mutual information."
