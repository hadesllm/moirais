# morie.fn — function file (hadesllm/morie)
"""Expected Log Predictive Density. 'Pass on what you have learned.'"""

from __future__ import annotations

import numpy as np
from scipy import special

from ._containers import DescriptiveResult


def expected_log_pred(
    log_lik_matrix: np.ndarray,
) -> DescriptiveResult:
    """Expected Log Pointwise Predictive Density (ELPD).

    Computes ELPD using the log-sum-exp trick over posterior draws:
    elpd_i = log(1/S * sum_s p(y_i | theta^s))

    :param log_lik_matrix: (S, n) matrix where S = posterior draws, n = observations.
    :return: DescriptiveResult with ELPD and its standard error.
    """
    ll = np.asarray(log_lik_matrix, dtype=float)
    if ll.ndim != 2:
        raise ValueError("log_lik_matrix must be 2-D (S x n).")
    S, n = ll.shape

    elpd_i = special.logsumexp(ll, axis=0) - np.log(S)
    elpd = float(np.sum(elpd_i))
    se = float(np.sqrt(n * np.var(elpd_i, ddof=1)))

    return DescriptiveResult(
        name="elpd",
        value=elpd,
        extra={"elpd_i": elpd_i.tolist(), "se": se, "n": n, "S": S},
    )


elpd = expected_log_pred


def cheatsheet() -> str:
    return "expected_log_pred({}) -> Expected Log Predictive Density. 'Pass on what you have lear"
