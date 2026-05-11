# morie.fn — function file from book-equation translation pipeline (hadesllm/morie)
"""Bayesian model averaging (predictive)."""

from __future__ import annotations

__all__ = ["bayesian_model_averaging", "bmapr"]

from typing import Any, Union

import numpy as np


def bayesian_model_averaging(
    model_predictions: Union[list, np.ndarray],
    model_log_evidence: Union[list, np.ndarray],
    *,
    prior_weights: Union[list, np.ndarray, None] = None,
) -> dict[str, Any]:
    """
    Bayesian model averaging for predictive inference.

    Combines predictions from multiple models weighted by their
    posterior model probabilities:

    .. math::

        p(\\tilde{y} | y) = \\sum_{k=1}^{K} p(\\tilde{y} | M_k, y)
        \\, p(M_k | y)

    Posterior model probabilities are computed from log-evidence
    (marginal likelihood) values via Bayes' rule with optional
    prior model weights.

    Parameters
    ----------
    model_predictions : array-like
        Predictions from each model.  Shape (K, n) where K is the
        number of models and n is the number of predictions/samples.
    model_log_evidence : array-like
        Log marginal likelihood (or proxy like -0.5 * BIC) for each
        model (K,).
    prior_weights : array-like or None
        Prior model probabilities (K,).  Default is uniform.

    Returns
    -------
    dict
        averaged_predictions : ndarray (n,)
        model_weights : ndarray (K,)
        model_log_evidence : ndarray (K,)
        weighted_mean : float
        weighted_sd : float

    References
    ----------
    Hoeting, J. A., Madigan, D., Raftery, A. E., & Volinsky, C. T.
    (1999). Bayesian model averaging: A tutorial. *Statistical
    Science*, 14(4), 382--417.
    """
    preds = np.asarray(model_predictions, dtype=float)
    log_ev = np.asarray(model_log_evidence, dtype=float).ravel()
    K = len(log_ev)

    if preds.ndim == 1:
        preds = preds.reshape(K, 1)
    if preds.shape[0] != K:
        raise ValueError("First dim of model_predictions must equal len(model_log_evidence).")

    if prior_weights is None:
        log_prior = np.zeros(K)
    else:
        pw = np.asarray(prior_weights, dtype=float)
        pw = pw / np.sum(pw)
        log_prior = np.log(pw + 1e-30)

    log_post = log_ev + log_prior
    log_post -= np.max(log_post)
    weights = np.exp(log_post)
    weights /= np.sum(weights)

    averaged = weights @ preds

    weighted_mean = float(np.mean(averaged))
    weighted_sd = float(np.std(averaged, ddof=0)) if len(averaged) > 1 else 0.0

    return {
        "averaged_predictions": averaged,
        "model_weights": weights,
        "model_log_evidence": log_ev,
        "weighted_mean": weighted_mean,
        "weighted_sd": weighted_sd,
    }


bmapr = bayesian_model_averaging


def cheatsheet() -> str:
    return "bayesian_model_averaging(preds, log_evidence) -> Bayesian model averaging."
