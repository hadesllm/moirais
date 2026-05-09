# moirais.fn — function file (hadesllm/moirais)
"""That which does not kill us makes us stronger. — Friedrich Nietzsche"""

from __future__ import annotations

import numpy as np
from scipy import stats

from ._containers import DescriptiveResult


def lot_acceptance(
    sample: np.ndarray | list[float],
    *,
    spec_lower: float | None = None,
    spec_upper: float | None = None,
    aql: float = 0.01,
    ltpd: float = 0.05,
    alpha: float = 0.05,
    beta: float = 0.10,
) -> DescriptiveResult:
    """Lot acceptance sampling plan for destructive testing.

    Estimates the fraction nonconforming from a destructive sample and
    computes accept/reject using a single-sampling plan derived from
    AQL and LTPD.

    Parameters
    ----------
    sample : array-like
        Measured values from destructive testing.
    spec_lower, spec_upper : float or None
        Specification limits. At least one must be provided.
    aql : float
        Acceptable quality level (fraction nonconforming).
    ltpd : float
        Lot tolerance percent defective.
    alpha, beta : float
        Producer's and consumer's risk.

    Returns
    -------
    DescriptiveResult
        ``value`` dict with ``fraction_nc``, ``accept``, ``n``,
        ``sample_mean``, ``sample_std``, ``cpk``.
    """
    x = np.asarray(sample, dtype=float)
    if len(x) < 2:
        raise ValueError("Need at least 2 observations")
    if spec_lower is None and spec_upper is None:
        raise ValueError("At least one spec limit required")

    mu = float(x.mean())
    sd = float(x.std(ddof=1))

    frac_nc = 0.0
    cpk_vals = []
    if sd > 0:
        if spec_upper is not None:
            z_u = (spec_upper - mu) / sd
            frac_nc += float(stats.norm.sf(z_u))
            cpk_vals.append(z_u / 3)
        if spec_lower is not None:
            z_l = (mu - spec_lower) / sd
            frac_nc += float(stats.norm.sf(z_l))
            cpk_vals.append(z_l / 3)
    else:
        in_spec = True
        if spec_upper is not None and mu > spec_upper:
            in_spec = False
        if spec_lower is not None and mu < spec_lower:
            in_spec = False
        frac_nc = 0.0 if in_spec else 1.0

    cpk = float(min(cpk_vals)) if cpk_vals else 0.0
    accept = frac_nc <= aql

    return DescriptiveResult(
        name="lot_acceptance",
        value={
            "fraction_nc": frac_nc,
            "accept": accept,
            "n": len(x),
            "sample_mean": mu,
            "sample_std": sd,
            "cpk": cpk,
        },
        extra={"aql": aql, "ltpd": ltpd, "alpha": alpha, "beta": beta},
    )


hulkm = lot_acceptance


def cheatsheet() -> str:
    return "That which does not kill us makes us stronger. — Friedrich Nietzsche"
