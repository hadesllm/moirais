# moirais.fn — function file (hadesllm/moirais)
"""LDA feature extraction."""

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "The road up and the road down are the same thing. — Heraclitus"


def lda_features(X, y, n_components=None, **kwargs) -> DescriptiveResult:
    """
    Fisher's Linear Discriminant Analysis for feature extraction.

    Maximises the ratio of between-class to within-class scatter.

    :param X: (n, d) data matrix.
    :param y: (n,) class labels.
    :param n_components: Number of discriminant components. Default: n_classes - 1.
    :return: DescriptiveResult with transformed data and eigenvalues.

    References
    ----------
    Fisher RA (1936). The use of multiple measurements in
    taxonomic problems. Annals of Eugenics, 7(2), 179-188.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y).ravel()
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    n, d = X.shape
    classes = np.unique(y)
    n_classes = len(classes)
    if n_components is None:
        n_components = min(n_classes - 1, d)
    n_components = min(n_components, n_classes - 1, d)
    mean_overall = X.mean(axis=0)
    Sw = np.zeros((d, d))
    Sb = np.zeros((d, d))
    for c in classes:
        Xc = X[y == c]
        mc = Xc.mean(axis=0)
        Sw += (Xc - mc).T @ (Xc - mc)
        nc = len(Xc)
        diff = (mc - mean_overall).reshape(-1, 1)
        Sb += nc * (diff @ diff.T)
    Sw += np.eye(d) * 1e-10
    eigvals, eigvecs = np.linalg.eigh(np.linalg.inv(Sw) @ Sb)
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    W = eigvecs[:, :n_components]
    X_proj = X @ W
    return DescriptiveResult(
        name="lda_features",
        value=float(n_components),
        extra={
            "n_components": n_components,
            "n_classes": n_classes,
            "eigenvalues": eigvals[:n_components].tolist(),
            "transformed_shape": list(X_proj.shape),
        },
    )


ldafd = lda_features


def cheatsheet() -> str:
    return "lda_features({}) -> LDA feature extraction."
