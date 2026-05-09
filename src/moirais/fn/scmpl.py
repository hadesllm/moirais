# moirais.fn — function file (hadesllm/moirais)
"""Hjorth complexity parameter."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "The greatest teacher, failure is."


def complexity(x, **kwargs) -> DescriptiveResult:
    """Compute the Hjorth Complexity parameter.

    .. math::

        \\text{Complexity} = \\frac{\\text{Mobility}(x')}{\\text{Mobility}(x)}

    Parameters
    ----------
    x : array-like
        Input signal.

    Returns
    -------
    DescriptiveResult
    """
    x = np.asarray(x, dtype=float)

    def _mobility(v):
        dv = np.diff(v)
        var_v = np.var(v, ddof=0)
        var_dv = np.var(dv, ddof=0)
        return np.sqrt(var_dv / var_v) if var_v > 0 else 0.0

    mob_x = _mobility(x)
    mob_dx = _mobility(np.diff(x))
    cplx = float(mob_dx / mob_x) if mob_x > 0 else 0.0
    return DescriptiveResult(
        name="complexity",
        value=cplx,
        extra={"complexity": cplx, "mobility_x": float(mob_x), "mobility_dx": float(mob_dx), "n": len(x)},
    )


scmpl = complexity


def cheatsheet() -> str:
    return "complexity({}) -> Hjorth complexity parameter."
