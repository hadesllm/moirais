"""Naive Bayes text classifier. 'We are what we repeatedly do. Excellence is not an act, but a habit. -- Aristotle'"""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def text_classify(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray | None = None,
    method: str = "nb",
    alpha: float = 1.0,
) -> DescriptiveResult:
    """Classify documents using Multinomial Naive Bayes on TF-IDF features.

    Pure NumPy implementation of multinomial NB with Laplace smoothing.
    Suitable for text classification on TF-IDF or count vectors.

    Parameters
    ----------
    X_train : ndarray, shape (n_train, n_features)
        Training document-term matrix.
    y_train : ndarray, shape (n_train,)
        Training labels (integer-coded).
    X_test : ndarray or None, shape (n_test, n_features)
        Test matrix. If None, predicts on training data.
    method : str
        Classification method. Currently 'nb' (Multinomial Naive Bayes).
    alpha : float
        Laplace smoothing parameter (1.0 = standard Laplace).

    Returns
    -------
    DescriptiveResult
        name='Text Classifier', value=accuracy on test set,
        extra has 'predictions', 'log_priors', 'n_classes', 'method'.

    References
    ----------
    McCallum, A. & Nigam, K. (1998). A comparison of event models
    for Naive Bayes text classification. *AAAI Workshop on Learning
    for Text Categorization*, 41-48.
    """
    X_train = np.asarray(X_train, dtype=np.float64)
    y_train = np.asarray(y_train).ravel()

    if X_test is None:
        X_test = X_train

    X_test = np.asarray(X_test, dtype=np.float64)

    classes = np.unique(y_train)
    n_classes = len(classes)
    n_features = X_train.shape[1]

    log_priors = np.zeros(n_classes)
    log_likelihoods = np.zeros((n_classes, n_features))

    for i, c in enumerate(classes):
        X_c = X_train[y_train == c]
        log_priors[i] = np.log(X_c.shape[0] / X_train.shape[0])
        feature_counts = X_c.sum(axis=0) + alpha
        log_likelihoods[i] = np.log(feature_counts / feature_counts.sum())

    log_posteriors = X_test @ log_likelihoods.T + log_priors
    pred_indices = np.argmax(log_posteriors, axis=1)
    predictions = classes[pred_indices]

    accuracy = float(np.mean(predictions == y_train[: len(predictions)])) if X_test is X_train else None

    return DescriptiveResult(
        name="Text Classifier",
        value=accuracy,
        extra={
            "predictions": predictions,
            "log_priors": log_priors,
            "n_classes": n_classes,
            "method": method,
            "alpha": alpha,
        },
    )


def cheatsheet() -> str:
    return "text_classify({}) -> Naive Bayes text classifier. 'Do. Or do not. There is no try"
