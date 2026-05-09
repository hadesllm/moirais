# moirais.fn — function file from book-equation translation pipeline (hadesllm/moirais)
"""Present value of annuity."""

from __future__ import annotations

from ._containers import DescriptiveResult


def annuity_value(
    rate: float,
    n_periods: int,
    payment: float = 1.0,
    due: bool = False,
) -> DescriptiveResult:
    """Present value of an ordinary annuity or annuity-due.

    Ordinary annuity (payments at end of period):

    .. math::

        PV = C \\cdot \\frac{1 - (1 + r)^{-n}}{r}

    Annuity-due (payments at start of period):

    .. math::

        PV_{\\text{due}} = PV \\cdot (1 + r)

    Parameters
    ----------
    rate : float
        Interest rate per period (e.g. 0.05 for 5%).
    n_periods : int
        Number of payment periods (n >= 1).
    payment : float, default 1.0
        Payment amount per period.
    due : bool, default False
        If True, compute annuity-due (payments at start of period).

    Returns
    -------
    DescriptiveResult
        ``value`` is the present value.  ``extra`` has
        ``future_value``, ``total_payments``, ``rate``, ``n_periods``.

    Raises
    ------
    ValueError
        If rate < 0 or n_periods < 1.

    References
    ----------
    Bowers, N. L., Gerber, H. U., Hickman, J. C., Jones, D. A., &
    Nesbitt, C. J. (1997). *Actuarial Mathematics* (2nd ed.).
    Society of Actuaries.
    """
    if rate < 0:
        raise ValueError(f"rate must be >= 0, got {rate}.")
    if n_periods < 1:
        raise ValueError(f"n_periods must be >= 1, got {n_periods}.")

    if rate == 0:
        pv = payment * n_periods
    else:
        pv = payment * (1 - (1 + rate) ** (-n_periods)) / rate

    if due:
        pv *= 1 + rate

    fv = pv * (1 + rate) ** n_periods
    total = payment * n_periods

    return DescriptiveResult(
        name="AnnuityPV",
        value=float(pv),
        extra={
            "future_value": float(fv),
            "total_payments": float(total),
            "rate": rate,
            "n_periods": n_periods,
            "due": due,
        },
    )


annty = annuity_value


def cheatsheet() -> str:
    return "annuity_value({}) -> Present value of annuity."
