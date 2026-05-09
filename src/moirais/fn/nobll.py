# moirais.fn — function file (hadesllm/moirais)
"""NOMINATE bill outcome positions."""

from __future__ import annotations

from ._containers import DescriptiveResult


def nominate_bill_params(nv, mid) -> DescriptiveResult:
    """Compute yea/nay outcome positions from normal vectors and midpoints.

    .. epigraph:: "We're done when I say we're done." -- Walter White, Breaking Bad
    """
    import numpy as np

    nv = np.asarray(nv, dtype=float)
    mid = np.asarray(mid, dtype=float)
    spread = 0.5
    yea_pos = mid + spread * nv
    nay_pos = mid - spread * nv
    return DescriptiveResult(
        name="nominate_bill_params",
        value=float(np.mean(mid)),
        extra={
            "yea_positions": yea_pos,
            "nay_positions": nay_pos,
            "n_bills": len(mid),
            "spread": spread,
        },
    )


nobll = nominate_bill_params


def cheatsheet() -> str:
    return "nominate_bill_params({}) -> NOMINATE bill outcome positions."
