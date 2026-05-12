# morie.fn -- function file (hadesllm/morie)
"""Matthews Correlation Coefficient."""

import numpy as np

from ._containers import ESRes

_QUOTE = "The ability to speak does not make you intelligent. -- Qui-Gon"


def mcc_score(y_true, y_pred, **kwargs) -> ESRes:
    r"""
    Compute Matthews Correlation Coefficient (MCC).

    .. math::

        MCC = \\frac{TP \\cdot TN - FP \\cdot FN}
        {\\sqrt{(TP+FP)(TP+FN)(TN+FP)(TN+FN)}}

    MCC is in [-1, +1]; +1 is perfect, 0 is random, -1 is inverse.

    :param y_true: array-like of true binary labels.
    :param y_pred: array-like of predicted binary labels.
    :return: ESRes with MCC.

    References
    ----------
    Matthews BW (1975). Comparison of the predicted and observed
    secondary structure of T4 phage lysozyme. Biochimica et
    Biophysica Acta, 405(2), 442-451.
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have same length.")
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    denom = np.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    mcc = (tp * tn - fp * fn) / denom if denom > 0 else 0.0
    return ESRes(
        measure="mcc",
        estimate=float(mcc),
        n=len(y_true),
        extra={"tp": tp, "fp": fp, "fn": fn, "tn": tn},
    )


mccrr = mcc_score


def cheatsheet() -> str:
    return "mcc_score({}) -> Matthews Correlation Coefficient."
