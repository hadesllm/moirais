"""Watterson's theta estimator from segregating sites."""


from ._containers import GenomicsResult


def watterson_theta(S: int, n: int, L: int = 1) -> GenomicsResult:
    """
    Compute Watterson's theta from the number of segregating sites.

    .. math::

        \\hat{\\theta}_W = \\frac{S}{a_n \\cdot L}

    where :math:`a_n = \\sum_{i=1}^{n-1} 1/i` is the (n-1)th harmonic number.

    :param S: Number of segregating sites.
    :param n: Number of sequences (sample size).
    :param L: Sequence length (default 1, returns per-site theta).
    :return: GenomicsResult with theta_w as statistic.
    :raises ValueError: If n < 2 or S < 0.

    References
    ----------
    Watterson GA (1975). On the number of segregating sites in genetical
    models without recombination. Theoretical Population Biology,
    7(2), 256-276.
    """
    if n < 2:
        raise ValueError("Need at least 2 sequences.")
    if S < 0:
        raise ValueError("S must be non-negative.")
    a_n = sum(1.0 / i for i in range(1, n))
    theta_w = S / (a_n * L)
    b_n = sum(1.0 / (i * i) for i in range(1, n))
    var_theta = (S * a_n + S * S * b_n) / (a_n * a_n * L * L * (a_n * a_n + b_n))
    return GenomicsResult(
        name="watterson_theta",
        statistic=float(theta_w),
        n=n,
        extra={"S": S, "a_n": a_n, "variance": float(var_theta), "L": L},
    )


theta = watterson_theta


def cheatsheet() -> str:
    return "watterson_theta({}) -> Watterson's theta estimator from segregating sites."
