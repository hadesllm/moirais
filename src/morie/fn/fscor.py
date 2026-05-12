# morie.fn — function file (hadesllm/morie)
"""F-beta score."""

import numpy as np

from ._containers import ESRes

_QUOTE = "So this is how liberty dies. With thunderous applause. -- Padme"


def f_score(y_true, y_pred, beta: float = 1.0, **kwargs) -> ESRes:
    r"""
    Compute the F_beta score for binary classification.

    .. math::

        F_{\\beta} = (1 + \\beta^2) \\cdot
        \\frac{P \\cdot R}{\\beta^2 P + R}

    :param y_true: array-like of true binary labels.
    :param y_pred: array-like of predicted binary labels.
    :param beta: Weight of recall relative to precision.
    :return: ESRes with F_beta score.

    References
    ----------
    van Rijsbergen CJ (1979). Information Retrieval, 2nd ed.
    Butterworths, London.
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have same length.")
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    b2 = beta**2
    fb = (1 + b2) * prec * rec / (b2 * prec + rec) if (b2 * prec + rec) > 0 else 0.0
    return ESRes(
        measure="f_score",
        estimate=fb,
        n=len(y_true),
        extra={"beta": beta, "precision": prec, "recall": rec},
    )


fscor = f_score


def cheatsheet() -> str:
    return "f_score({}) -> F-beta score."
