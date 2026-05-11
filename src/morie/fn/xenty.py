r"""Cross-entropy loss (binary and multi-class).

Measures dissimilarity between predicted and true probability distributions.

References
----------
Goodfellow, I., Bengio, Y., & Courville, A. (2016).
Deep Learning. MIT press.
"""

__all__ = ["xenty"]

import numpy as np


def xenty(
    y_true,
    y_pred,
    from_logits=False,
    epsilon=1e-7,
):
    """
    Cross-entropy loss.

    Parameters
    ----------
    y_true : ndarray
        True labels. Shape (n_samples,) for binary or (n_samples, n_classes)
        for multi-class.
    y_pred : ndarray
        Predicted probabilities (or logits if from_logits=True).
        Same shape as y_true.
    from_logits : bool, optional
        If True, y_pred are logits (apply softmax). Default False.
    epsilon : float, optional
        Numerical stability clipping. Default 1e-7.

    Returns
    -------
    float
        Mean cross-entropy loss.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have same shape")

    if from_logits:
        if y_pred.ndim == 1:
            y_pred = 1.0 / (1.0 + np.exp(-y_pred))
        else:
            y_pred = np.exp(y_pred) / np.sum(
                np.exp(y_pred), axis=1, keepdims=True
            )

    y_pred = np.clip(y_pred, epsilon, 1.0 - epsilon)

    if y_pred.ndim == 1:
        loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    else:
        loss = -np.mean(np.sum(y_true * np.log(y_pred), axis=1))

    return float(loss)
