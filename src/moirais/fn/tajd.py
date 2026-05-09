"""Tajima's D statistic for testing neutrality."""

import math

from ._containers import GenomicsResult


def tajimas_d(
    S: int,
    n: int,
    pi: float,
) -> GenomicsResult:
    """Tajima's D statistic for detecting departures from neutral evolution.

    D = (pi - S/a1) / sqrt(Var)

    where a1 = sum(1/i for i=1..n-1) is the (n-1)th harmonic number,
    and the variance is computed from the Tajima (1989) formulae.

    Parameters
    ----------
    S : int
        Number of segregating sites.
    n : int
        Number of sequences sampled (>= 3).
    pi : float
        Average number of pairwise differences.

    Returns
    -------
    GenomicsResult
        name="Tajima_D", statistic=D value.

    Raises
    ------
    ValueError
        If n < 3 or S < 0 or pi < 0.

    References
    ----------
    Tajima, F. (1989). Statistical method for testing the neutral mutation
        hypothesis by DNA polymorphism. Genetics, 123(3), 585-595.
    """
    if n < 3:
        raise ValueError("Need at least 3 sequences (n >= 3).")
    if S < 0:
        raise ValueError("Segregating sites S must be >= 0.")
    if pi < 0:
        raise ValueError("Average pairwise differences pi must be >= 0.")

    a1 = sum(1.0 / i for i in range(1, n))
    a2 = sum(1.0 / (i * i) for i in range(1, n))

    b1 = (n + 1) / (3 * (n - 1))
    b2 = 2 * (n * n + n + 3) / (9 * n * (n - 1))

    c1 = b1 - 1.0 / a1
    c2 = b2 - (n + 2) / (a1 * n) + a2 / (a1 * a1)

    e1 = c1 / a1
    e2 = c2 / (a1 * a1 + a2)

    theta_w = S / a1
    d_num = pi - theta_w

    if S == 0:
        return GenomicsResult(
            name="Tajima_D",
            statistic=0.0,
            n=n,
            extra={"S": S, "pi": pi, "theta_w": theta_w},
        )

    variance = e1 * S + e2 * S * (S - 1)
    if variance <= 0:
        return GenomicsResult(
            name="Tajima_D",
            statistic=0.0,
            n=n,
            extra={"S": S, "pi": pi, "theta_w": theta_w, "var": variance},
        )

    D = d_num / math.sqrt(variance)

    return GenomicsResult(
        name="Tajima_D",
        statistic=float(D),
        n=n,
        extra={"S": S, "pi": pi, "theta_w": theta_w},
    )


tajd = tajimas_d


def cheatsheet() -> str:
    return "tajimas_d({}) -> Tajima's D statistic for testing neutrality."
