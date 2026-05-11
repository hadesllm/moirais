"""Semi-supervised self-training classifier."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "The belonging you seek is not behind you, it is ahead."


def semi_supervised(X_labeled, y_labeled, X_unlabeled, n_iter=10, threshold=0.9, **kwargs) -> DescriptiveResult:
    """Self-training semi-supervised classifier.

    Uses nearest-centroid as the base classifier. In each iteration,
    unlabeled samples with confidence above threshold are pseudo-labeled
    and added to the training set.

    Parameters
    ----------
    X_labeled : array-like of shape (n_l, p)
    y_labeled : array-like of shape (n_l,)
    X_unlabeled : array-like of shape (n_u, p)
    n_iter : int
        Self-training iterations (default 10).
    threshold : float
        Confidence threshold for pseudo-labeling (default 0.9).

    Returns
    -------
    DescriptiveResult
    """
    X_l = np.asarray(X_labeled, dtype=float)
    y_l = np.asarray(y_labeled).ravel()
    X_u = np.asarray(X_unlabeled, dtype=float)

    X_train = X_l.copy()
    y_train = y_l.copy()
    remaining = np.arange(len(X_u))
    n_added_total = 0

    for iteration in range(n_iter):
        if len(remaining) == 0:
            break

        classes = np.unique(y_train)
        centroids = np.array([X_train[y_train == c].mean(axis=0) for c in classes])

        X_rem = X_u[remaining]
        dists = np.array([np.sum((X_rem - c) ** 2, axis=1) for c in centroids])
        min_dist = dists.min(axis=0)
        max_dist = dists.max(axis=0)
        range_dist = max_dist - min_dist
        range_dist[range_dist == 0] = 1.0
        confidence = 1.0 - min_dist / range_dist

        confident = confidence >= threshold
        if not confident.any():
            break

        pseudo_labels = classes[np.argmin(dists[:, confident], axis=0)]
        X_train = np.vstack([X_train, X_rem[confident]])
        y_train = np.concatenate([y_train, pseudo_labels])
        n_added_total += int(confident.sum())
        remaining = remaining[~confident]

    classes = np.unique(y_train)
    centroids = np.array([X_train[y_train == c].mean(axis=0) for c in classes])
    if len(remaining) > 0:
        dists_rem = np.array([np.sum((X_u[remaining] - c) ** 2, axis=1) for c in centroids])
        final_preds_rem = classes[np.argmin(dists_rem, axis=0)]
    else:
        final_preds_rem = np.array([], dtype=y_l.dtype)

    all_preds = np.empty(len(X_u), dtype=y_l.dtype)
    labeled_mask = np.ones(len(X_u), dtype=bool)
    labeled_mask[remaining] = False
    all_preds[labeled_mask] = y_train[len(X_l) :]
    if len(remaining) > 0:
        all_preds[remaining] = final_preds_rem

    return DescriptiveResult(
        name="semi_supervised",
        value=float(n_added_total),
        extra={
            "predictions": all_preds,
            "n_pseudo_labeled": n_added_total,
            "n_remaining": len(remaining),
            "n_iterations": n_iter,
            "final_train_size": len(y_train),
        },
    )


sslcl = semi_supervised


def cheatsheet() -> str:
    return "semi_supervised({}) -> Semi-supervised self-training classifier."
