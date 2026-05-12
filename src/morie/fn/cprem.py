# morie.fn -- function file (hadesllm/morie)
"""Buhlmann credibility premium."""

from __future__ import annotations

from ._containers import DescriptiveResult


def credibility_premium(
    x_i: float,
    x_bar: float,
    n: int,
    k: float,
) -> DescriptiveResult:
    r"""Buhlmann credibility premium.

    The credibility-weighted premium blends individual experience
    with the collective mean:

    .. math::

        P_C = Z \\cdot \\bar{X}_i + (1 - Z) \\cdot \\bar{X}

    where the credibility factor is:

    .. math::

        Z = \\frac{n}{n + k}

    and :math:`k = \\text{Var}(\\mu(\\theta)) / E[\\sigma^2(\\theta)]`
    is the Buhlmann credibility parameter.

    Parameters
    ----------
    x_i : float
        Individual (policyholder) mean claim experience.
    x_bar : float
        Collective (portfolio) mean claim experience.
    n : int
        Number of exposure periods for the individual.
    k : float
        Buhlmann k-parameter (ratio of between-group variance to
        within-group variance, k > 0).

    Returns
    -------
    DescriptiveResult
        ``value`` is the credibility premium.  ``extra`` has ``Z``
        (credibility factor), ``x_i``, ``x_bar``, ``n``, ``k``.

    Raises
    ------
    ValueError
        If n < 1 or k <= 0.

    References
    ----------
    Buhlmann, H. (1967). Experience rating and credibility.
    *ASTIN Bulletin*, 4(3), 199--207.

    Buhlmann, H., & Straub, E. (1970). Glaubwurdigkeit fur
    Schadensatze. *Mitteilungen der Vereinigung Schweizerischer
    Versicherungsmathematiker*, 70, 111--133.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}.")
    if k <= 0:
        raise ValueError(f"k must be > 0, got {k}.")

    Z = n / (n + k)
    premium = Z * x_i + (1 - Z) * x_bar

    return DescriptiveResult(
        name="CredibilityPremium",
        value=float(premium),
        extra={
            "Z": float(Z),
            "x_i": x_i,
            "x_bar": x_bar,
            "n": n,
            "k": k,
        },
    )


cprem = credibility_premium


def cheatsheet() -> str:
    return "credibility_premium({}) -> Buhlmann credibility premium."
