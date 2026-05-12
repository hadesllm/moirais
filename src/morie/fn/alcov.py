# morie.fn -- function file from book-equation translation pipeline (hadesllm/morie)
"""ALCOVE attention-learning covering map model."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def alcove_model(
    stimuli,
    categories,
    *,
    n_exemplars: int = 0,
    lr_attn: float = 0.01,
    lr_assoc: float = 0.03,
    specificity: float = 1.0,
    n_epochs: int = 50,
) -> DescriptiveResult:
    """ALCOVE attention-learning model (Kruschke 1992).

    An exemplar-based categorisation model with learned attention weights
    over stimulus dimensions. Used in spatial judgment contexts.

    :param stimuli: (n_stimuli x n_dims) stimulus coordinates.
    :param categories: 1D array of category labels (0-indexed).
    :param n_exemplars: Number of exemplars (0 = use all stimuli).
    :param lr_attn: Attention weight learning rate.
    :param lr_assoc: Association weight learning rate.
    :param specificity: Gaussian specificity parameter.
    :param n_epochs: Training epochs.
    :return: DescriptiveResult with attention weights and accuracy.

    References
    ----------
    Armstrong (2014), Ch 9. Kruschke (1992).

    .. epigraph:: "Your focus determines your reality." -- Qui-Gon, Star Wars
    """
    X = np.asarray(stimuli, dtype=float)
    y = np.asarray(categories, dtype=int).ravel()
    n, d = X.shape
    n_cat = int(y.max()) + 1
    if n_exemplars <= 0:
        n_exemplars = n

    rng = np.random.default_rng(42)
    exemplar_idx = rng.choice(n, size=min(n_exemplars, n), replace=False)
    exemplars = X[exemplar_idx]

    alpha = np.ones(d) / d
    W = np.zeros((len(exemplars), n_cat))

    correct_total = 0
    total = 0
    for epoch in range(n_epochs):
        order = rng.permutation(n)
        for idx in order:
            xi = X[idx]
            dists = np.sum(alpha * np.abs(exemplars - xi), axis=1)
            activations = np.exp(-specificity * dists)
            output = activations @ W
            probs = np.exp(output - output.max())
            probs /= probs.sum()

            target = np.zeros(n_cat)
            target[y[idx]] = 1.0
            error = target - probs

            pred = int(np.argmax(probs))
            if pred == y[idx]:
                correct_total += 1
            total += 1

            W += lr_assoc * np.outer(activations, error)

            for dim in range(d):
                grad = -specificity * np.sign(exemplars[:, dim] - xi[dim]) * activations
                alpha_grad = (grad @ W @ error)
                alpha[dim] += lr_attn * alpha_grad
            alpha = np.maximum(alpha, 0.0)
            a_sum = alpha.sum()
            if a_sum > 0:
                alpha /= a_sum

    accuracy = correct_total / max(total, 1)
    return DescriptiveResult(
        name="alcove_model",
        value={"attention_weights": alpha, "accuracy": accuracy},
        extra={
            "association_weights": W,
            "n_exemplars": len(exemplars),
            "n_categories": n_cat,
            "n_epochs": n_epochs,
        },
    )


alcov = alcove_model


def cheatsheet() -> str:
    return "alcove_model({}) -> ALCOVE attention-learning model."
