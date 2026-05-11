"""Cross-jurisdiction SIU comparison."""

from __future__ import annotations

import numpy as np

from morie.fn._containers import DescriptiveResult


def siu_comparison(
    rates_dict: dict[str, float],
) -> DescriptiveResult:
    """Compare SIU case rates across jurisdictions.

    Parameters
    ----------
    rates_dict : dict
        Jurisdiction name -> case rate.

    Returns
    -------
    DescriptiveResult
    """
    if len(rates_dict) == 0:
        raise ValueError("rates_dict must be non-empty")
    vals = np.array(list(rates_dict.values()))
    mean_rate = float(np.mean(vals))
    ranked = sorted(rates_dict.items(), key=lambda x: x[1], reverse=True)
    return DescriptiveResult(
        name="siu_comparison",
        value=mean_rate,
        extra={
            "rates": rates_dict,
            "highest": ranked[0][0],
            "lowest": ranked[-1][0],
            "range": float(np.max(vals) - np.min(vals)),
            "n_jurisdictions": len(rates_dict),
        },
    )


siucmp = siu_comparison


def cheatsheet() -> str:
    return "siu_comparison({}) -> Cross-jurisdiction SIU comparison."
