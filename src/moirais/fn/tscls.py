"""1-NN DTW time series classifier."""

import numpy as np

from ._containers import DescriptiveResult


def _dtw_dist(x, y):
    n, m = len(x), len(y)
    D = np.full((n + 1, m + 1), np.inf)
    D[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            D[i, j] = (x[i - 1] - y[j - 1]) ** 2 + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    return np.sqrt(D[n, m])


def ts_classify(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray) -> DescriptiveResult:
    """
    1-nearest-neighbor time series classifier using DTW distance.

    :param X_train: (n_train, T) training time series.
    :param y_train: (n_train,) class labels.
    :param X_test: (n_test, T) test time series.
    :return: DescriptiveResult with predicted labels.

    References
    ----------
    Xi X et al. (2006). Fast time series classification using
    numerosity reduction. ICML.
    """
    X_tr = np.asarray(X_train, dtype=np.float64)
    y_tr = np.asarray(y_train)
    X_te = np.asarray(X_test, dtype=np.float64)
    if X_tr.ndim == 1:
        X_tr = X_tr.reshape(1, -1)
    if X_te.ndim == 1:
        X_te = X_te.reshape(1, -1)
    predictions = []
    for i in range(X_te.shape[0]):
        best_dist = np.inf
        best_label = y_tr[0]
        for j in range(X_tr.shape[0]):
            d = _dtw_dist(X_te[i], X_tr[j])
            if d < best_dist:
                best_dist = d
                best_label = y_tr[j]
        predictions.append(best_label)
    preds = np.array(predictions)
    return DescriptiveResult(
        name="ts_classify",
        value=float(len(preds)),
        extra={"predictions": preds, "n_train": X_tr.shape[0], "n_test": X_te.shape[0]},
    )


tscls = ts_classify


def cheatsheet() -> str:
    return "_dtw_dist({}) -> 1-NN DTW time series classifier."
