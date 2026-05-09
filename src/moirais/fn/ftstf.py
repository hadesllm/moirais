# moirais.fn — function file (hadesllm/moirais)
"""ANOVA F-test per feature for selection."""

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "I cannot teach anybody anything. I can only make them think. — Socrates"


def f_test_features(X, y, **kwargs) -> DescriptiveResult:
    """
    Compute one-way ANOVA F-statistic for each feature as a
    univariate feature selection score.

    :param X: (n, d) data matrix.
    :param y: (n,) class labels.
    :return: DescriptiveResult with F-scores per feature.

    References
    ----------
    Fisher RA (1925). Statistical Methods for Research Workers.
    Oliver & Boyd, Edinburgh.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y).ravel()
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    n, d = X.shape
    classes = np.unique(y)
    k = len(classes)
    if k < 2:
        raise ValueError("Need at least 2 classes.")
    grand_mean = X.mean(axis=0)
    ss_between = np.zeros(d)
    ss_within = np.zeros(d)
    for c in classes:
        Xc = X[y == c]
        nc = len(Xc)
        mc = Xc.mean(axis=0)
        ss_between += nc * (mc - grand_mean) ** 2
        ss_within += np.sum((Xc - mc) ** 2, axis=0)
    df_between = k - 1
    df_within = n - k
    if df_within <= 0:
        raise ValueError("Not enough observations for ANOVA.")
    ms_between = ss_between / df_between
    ms_within = ss_within / df_within
    ms_within[ms_within == 0] = 1e-12
    f_scores = ms_between / ms_within
    ranking = np.argsort(f_scores)[::-1]
    return DescriptiveResult(
        name="f_test_features",
        value=float(np.max(f_scores)),
        extra={
            "f_scores": f_scores.tolist(),
            "ranking": ranking.tolist(),
            "df_between": df_between,
            "df_within": df_within,
        },
    )


ftstf = f_test_features


def cheatsheet() -> str:
    return "f_test_features({}) -> ANOVA F-test per feature for selection."
