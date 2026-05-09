# moirais.fn — function file (hadesllm/moirais)
"""Lead blood-level → IQ loss dose-response (Lanphear 2005 pooled)."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

# Lanphear et al. 2005, Environ Health Perspect 113(7):894-899,
# pooled analysis of 7 international cohorts. The paper fits a
# log-linear concentration-response model:
#
#     ΔIQ = a * ln(BLL + 1)  [relative to BLL = 1 µg/dL]
#
# where a is the pooled coefficient. The paper's headline estimates:
#   - 1  -> 10 µg/dL: −6.2 IQ points (6.9 in subset, reported CI wide)
#   - 10 -> 20 µg/dL: −1.9 IQ points
#   - 20 -> 30 µg/dL: −1.1 IQ points
# The log-linear fit that reproduces these differences uses
# a ≈ −2.7 IQ points per natural-log unit of BLL.
#
# We use the pooled central estimate a = -2.7 for reproducibility.
# Users needing the confidence interval: Lanphear reports −6.2
# (95% CI −8.7 to −3.6) for the 1→10 contrast; this translates to
# a in (-3.78, -1.56).
_LANPHEAR_POOLED_A = -2.7
_LANPHEAR_95CI_A = (-3.78, -1.56)


def lead_iq_loss(
    bll_ug_per_dL: float | np.ndarray,
    reference_bll: float = 1.0,
    *,
    with_ci: bool = False,
) -> DescriptiveResult:
    """Estimate IQ loss from childhood blood lead level (BLL).

    Uses the Lanphear et al. 2005 pooled log-linear concentration-
    response model (7 international cohorts, 1,333 children). The
    headline result: no safe level detected — effects per µg/dL are
    *largest* at the lowest exposures (sub-linear, super-log-sub-linear
    curvature per the figure).

    .. math::

        \\Delta IQ = a \\cdot \\ln(BLL / BLL_0)

    with pooled :math:`a = -2.7` IQ points per ln-unit of BLL.

    Parameters
    ----------
    bll_ug_per_dL : float or array-like
        Blood lead level in µg/dL. Values must be > 0.
    reference_bll : float, default 1.0
        Reference BLL below which we assume no excess loss. 1.0 µg/dL
        (the US CDC BLRV detection floor) is the convention.
    with_ci : bool, default False
        If True, include 95% CI in the returned `extra`.

    Returns
    -------
    DescriptiveResult
        value = estimated IQ loss (negative number) relative to
        `reference_bll`.
        extra has per-observation loss, optional CI, and the formula
        constants used.

    Examples
    --------
    A child with BLL 10 µg/dL (well within historical "acceptable")
    still has an estimated IQ cost of ~6 points relative to 1 µg/dL:

    >>> r = lead_iq_loss(10.0)
    >>> round(r.value, 1)
    -6.2

    Notes
    -----
    The CDC's current blood lead reference value is 3.5 µg/dL (2021
    revision). A BLL above that triggers intervention. The model here
    does not claim a threshold; it only quantifies expected per-unit
    IQ cost relative to a chosen reference.

    References
    ----------
    Lanphear, B. P., Hornung, R., Khoury, J., et al. (2005). Low-level
    environmental lead exposure and children's intellectual function:
    an international pooled analysis. Environmental Health Perspectives,
    113(7), 894-899.

    CDC (2021). Blood Lead Reference Value.
    https://www.cdc.gov/nceh/lead/data/blood-lead-reference-value.htm
    """
    bll = np.atleast_1d(np.asarray(bll_ug_per_dL, dtype=float))
    if np.any(bll <= 0):
        raise ValueError("BLL values must be > 0 µg/dL.")
    if reference_bll <= 0:
        raise ValueError("reference_bll must be > 0.")

    delta = _LANPHEAR_POOLED_A * np.log(bll / reference_bll)
    val = float(delta.mean()) if delta.size > 1 else float(delta.item())

    extra: dict[str, object] = {
        "iq_loss": delta.tolist() if delta.size > 1 else float(delta.item()),
        "reference_bll_ug_per_dL": reference_bll,
        "pooled_slope": _LANPHEAR_POOLED_A,
        "source": "Lanphear et al. 2005 EHP 113(7):894-899",
    }
    if with_ci:
        lo, hi = _LANPHEAR_95CI_A
        delta_lo = hi * np.log(bll / reference_bll)   # hi is less neg -> smaller loss
        delta_hi = lo * np.log(bll / reference_bll)   # lo is more neg -> larger loss
        extra["iq_loss_lower_95"] = (delta_lo.tolist() if delta_lo.size > 1
                                     else float(delta_lo.item()))
        extra["iq_loss_upper_95"] = (delta_hi.tolist() if delta_hi.size > 1
                                     else float(delta_hi.item()))
        extra["slope_95ci"] = _LANPHEAR_95CI_A

    return DescriptiveResult(name="lead_iq_loss", value=val, extra=extra)


leadiq = lead_iq_loss


def cheatsheet() -> str:
    return "leadiq(BLL_ug_dL) -> IQ loss (Lanphear 2005 pooled log-linear)."
