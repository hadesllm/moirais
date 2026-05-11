# morie.fn — function file (hadesllm/morie)
"""K-fold cross-validation."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def kfcvl_fn(X: np.ndarray, y: np.ndarray, k: int = 5) -> DescriptiveResult:
    """Evaluate classifier accuracy using k-fold cross-validation.

    :param X: Feature matrix (samples x features).
    :param y: Class labels.
    :param k: Number of folds (default 5).
    :return: DescriptiveResult with mean accuracy and per-fold accuracies.
    """
    from morie._classify import kfold_cv

    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    mean_acc, fold_accs = kfold_cv(X, y, k=k)
    return DescriptiveResult(
        name="kfold_cv",
        value=mean_acc,
        extra={"mean_accuracy": mean_acc, "fold_accuracies": fold_accs, "k": k},
    )


kfcvl = kfcvl_fn


def cheatsheet() -> str:
    return "kfcvl_fn({}) -> K-fold cross-validation."
