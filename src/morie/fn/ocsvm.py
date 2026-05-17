# morie.fn -- function file (hadesllm/morie)
"""OC SVM-like separating hyperplane."""

from __future__ import annotations

from ._containers import DescriptiveResult


def oc_svm_classify(X, labels) -> DescriptiveResult:
    """Find separating hyperplane via simple linear discriminant.

    .. epigraph:: Number rules the universe. -- Pythagoras
    """
    import numpy as np

    X = np.asarray(X, dtype=float)
    labels = np.asarray(labels, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    m0 = np.mean(X[labels == 0], axis=0)
    m1 = np.mean(X[labels == 1], axis=0)
    w = m1 - m0
    w_norm = w / (np.linalg.norm(w) + 1e-15)
    midpoint = (m0 + m1) / 2.0
    cutpoint = float(np.dot(w_norm, midpoint))
    proj = X @ w_norm
    predicted = (proj >= cutpoint).astype(float)
    acc = float(np.mean(predicted == labels))
    return DescriptiveResult(
        name="oc_svm_classify",
        value=acc,
        extra={
            "normal": w_norm.tolist(),
            "cutpoint": cutpoint,
            "accuracy": acc,
            "n": len(labels),
        },
    )


ocsvm = oc_svm_classify


def cheatsheet() -> str:
    return "oc_svm_classify({}) -> OC SVM-like separating hyperplane."
