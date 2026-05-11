# morie.fn — function file (hadesllm/morie)
"""Cross-validation for genomic prediction."""

__all__ = ["cvgen"]

import numpy as np

from ._containers import GenomicsResult


def cvgen(
    y: np.ndarray,
    G: np.ndarray,
    *,
    n_folds: int = 5,
    lambda_val: float = 1.0,
    seed: int = 42,
) -> GenomicsResult:
    """K-fold cross-validation for GBLUP genomic prediction.

    Partitions individuals into k folds, trains GBLUP on k-1 folds,
    predicts the held-out fold, and reports prediction accuracy.

    Parameters
    ----------
    y : array, shape (n,)
        Phenotype vector.
    G : array, shape (n, n)
        Genomic relationship matrix.
    n_folds : int
        Number of CV folds.
    lambda_val : float
        Shrinkage parameter (var_e / var_g).
    seed : int
        Random seed for fold assignment.

    Returns
    -------
    GenomicsResult
        statistic = mean prediction accuracy (correlation),
        extra has 'fold_accuracies', 'fold_mse',
        'predictions' (full n-vector of CV predictions).

    References
    ----------
    Daetwyler, H. D., et al. (2013). Genomic prediction in animals
        and plants: simulation of data, validation, reporting,
        and benchmarking. Genetics, 193(2), 347-365.
    """
    rng = np.random.default_rng(seed)
    y = np.asarray(y, dtype=float).ravel()
    G = np.asarray(G, dtype=float)
    n = len(y)

    if G.shape != (n, n):
        raise ValueError(f"G must be ({n},{n}).")
    if n_folds < 2 or n_folds > n:
        raise ValueError(f"n_folds must be in [2, {n}].")

    idx = rng.permutation(n)
    folds = np.array_split(idx, n_folds)

    predictions = np.full(n, np.nan)
    fold_acc = []
    fold_mse = []

    for f in range(n_folds):
        test_idx = folds[f]
        train_idx = np.concatenate([folds[j] for j in range(n_folds) if j != f])

        n_tr = len(train_idx)
        y_tr = y[train_idx]

        G_tr = G[np.ix_(train_idx, train_idx)]
        G_te_tr = G[np.ix_(test_idx, train_idx)]

        X_tr = np.ones((n_tr, 1))

        lhs = np.zeros((1 + n_tr, 1 + n_tr))
        lhs[0, 0] = n_tr
        lhs[0, 1:] = 1.0
        lhs[1:, 0] = 1.0
        G_inv = np.linalg.inv(G_tr + np.eye(n_tr) * 1e-6)
        lhs[1:, 1:] = np.eye(n_tr) + G_inv * lambda_val

        rhs = np.concatenate([[np.sum(y_tr)], y_tr])
        sol = np.linalg.solve(lhs, rhs)

        mu = sol[0]
        gebv_tr = sol[1:]

        alpha = np.linalg.solve(G_tr + np.eye(n_tr) * 1e-6, gebv_tr)
        gebv_te = G_te_tr @ alpha
        y_pred = mu + gebv_te

        predictions[test_idx] = y_pred
        y_te = y[test_idx]

        corr = float(np.corrcoef(y_te, y_pred)[0, 1]) if np.std(y_pred) > 0 else 0.0
        mse = float(np.mean((y_te - y_pred) ** 2))
        fold_acc.append(corr)
        fold_mse.append(mse)

    mean_acc = float(np.mean(fold_acc))
    se_acc = float(np.std(fold_acc) / np.sqrt(n_folds))

    return GenomicsResult(
        name="CV_GBLUP",
        statistic=mean_acc,
        n=n,
        extra={
            "fold_accuracies": fold_acc,
            "fold_mse": fold_mse,
            "se_accuracy": se_acc,
            "n_folds": n_folds,
            "predictions": predictions.tolist(),
        },
    )


def cheatsheet() -> str:
    return "cvgen(y, G) -> Cross-validation for genomic prediction."
