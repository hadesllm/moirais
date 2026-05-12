# morie.fn -- function file from book-equation translation pipeline (hadesllm/morie)
"""Bayesian model averaging."""

from __future__ import annotations

from typing import Any, Union

import numpy as np


def bayesian_model_averaging(
    predictions: Union[list, np.ndarray],
    log_marginal_likelihoods: Union[list, np.ndarray],
    *,
    prior_weights: Union[list, np.ndarray, None] = None,
) -> dict[str, Any]:
    """
    Bayesian model averaging: combine predictions weighted by posterior model probabilities.

    :param predictions: Predictions from each model (K, n) or list of arrays.
    :param log_marginal_likelihoods: Log marginal likelihoods for K models.
    :param prior_weights: Prior model probabilities. Default uniform.
    :return: Dictionary with averaged predictions, model_weights, top_model.

    References
    ----------
    Hoeting, J. A., et al. (1999). *Statistical Science*, 14(4), 382--401.
    """
    preds = np.asarray(predictions, dtype=float)
    log_ml = np.asarray(log_marginal_likelihoods, dtype=float)
    K = len(log_ml)

    if prior_weights is None:
        log_prior = np.zeros(K)
    else:
        pw = np.asarray(prior_weights, dtype=float)
        pw = pw / np.sum(pw)
        log_prior = np.log(pw + 1e-30)

    log_post = log_ml + log_prior
    log_post -= np.max(log_post)
    post_weights = np.exp(log_post)
    post_weights /= np.sum(post_weights)

    avg_pred = post_weights @ preds
    top_model = int(np.argmax(post_weights))

    return {
        "averaged_predictions": avg_pred.tolist(),
        "model_weights": post_weights.tolist(),
        "top_model": top_model,
        "top_model_weight": float(post_weights[top_model]),
        "K": K,
    }


bmaxe = bayesian_model_averaging


def cheatsheet() -> str:
    return "bayesian_model_averaging({}) -> Bayesian model averaging."
