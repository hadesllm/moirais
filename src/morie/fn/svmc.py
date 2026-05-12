"""Support vector machine classifier -- simplified SMO."""

import numpy as np

from ._containers import DescriptiveResult


def svm_classify(X, y, C=1.0, kernel="rbf", gamma=None, max_iter=500):
    """
    Binary SVM classifier via simplified SMO algorithm.

    :param X: (n, p) feature matrix.
    :param y: (n,) labels in {-1, +1}.
    :param C: Regularization parameter.
    :param kernel: 'linear' or 'rbf'.
    :param gamma: RBF bandwidth. Defaults to 1/p.
    :param max_iter: Maximum iterations.
    :return: DescriptiveResult with support vectors, accuracy.

    References
    ----------
    Platt JC (1998). Sequential Minimal Optimization: A Fast Algorithm
    for Training Support Vector Machines. Microsoft Research.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).ravel()
    n, p = X.shape
    if gamma is None:
        gamma = 1.0 / p

    if kernel == "linear":
        K = X @ X.T
    else:
        sq = np.sum(X**2, axis=1)
        K = np.exp(-gamma * (sq[:, None] + sq[None, :] - 2 * X @ X.T))

    alpha = np.zeros(n)
    b = 0.0

    for _ in range(max_iter):
        changed = 0
        for i in range(n):
            ei = np.sum(alpha * y * K[i]) + b - y[i]
            if (y[i] * ei < -0.001 and alpha[i] < C) or (y[i] * ei > 0.001 and alpha[i] > 0):
                j = np.random.randint(n)
                while j == i:
                    j = np.random.randint(n)
                ej = np.sum(alpha * y * K[j]) + b - y[j]
                ai_old, aj_old = alpha[i], alpha[j]
                if y[i] != y[j]:
                    L = max(0, alpha[j] - alpha[i])
                    H = min(C, C + alpha[j] - alpha[i])
                else:
                    L = max(0, alpha[i] + alpha[j] - C)
                    H = min(C, alpha[i] + alpha[j])
                if L >= H:
                    continue
                eta = 2 * K[i, j] - K[i, i] - K[j, j]
                if eta >= 0:
                    continue
                alpha[j] -= y[j] * (ei - ej) / eta
                alpha[j] = np.clip(alpha[j], L, H)
                if abs(alpha[j] - aj_old) < 1e-5:
                    continue
                alpha[i] += y[i] * y[j] * (aj_old - alpha[j])
                b1 = b - ei - y[i] * (alpha[i] - ai_old) * K[i, i] - y[j] * (alpha[j] - aj_old) * K[i, j]
                b2 = b - ej - y[i] * (alpha[i] - ai_old) * K[i, j] - y[j] * (alpha[j] - aj_old) * K[j, j]
                b = (b1 + b2) / 2
                changed += 1
        if changed == 0:
            break

    sv_mask = alpha > 1e-5
    preds = np.sign(np.sum(alpha * y * K, axis=1) + b)
    acc = float(np.mean(preds == y))

    return DescriptiveResult(
        name="svm_classify",
        value=acc,
        extra={
            "accuracy": acc,
            "n_support_vectors": int(sv_mask.sum()),
            "alpha": alpha[sv_mask].tolist(),
            "bias": float(b),
            "kernel": kernel,
            "C": float(C),
            "n": n,
        },
    )


def cheatsheet() -> str:
    return "svm_classify({}) -> Support vector machine classifier -- simplified SMO."
