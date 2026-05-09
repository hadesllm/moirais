# moirais.fn — function file (hadesllm/moirais)
"""Highest posterior density interval with R-style verbose result."""

from typing import Sequence, Union
import numpy as np


def hpdint(samples: Union[Sequence[float], np.ndarray], cred: float = 0.95):
    """HPD credible interval for posterior MCMC samples."""
    from ._richresult import RichResult
    a = np.asarray(samples, dtype=float).copy()
    a.sort()
    n = a.size
    if not 0 < cred < 1:
        raise ValueError(f"cred must be in (0,1), got {cred}.")
    k = int(np.floor(cred * n))
    if k < 1:
        raise ValueError(f"need more samples for {cred*100:.0f}% interval; got {n}.")
    widths = a[k:] - a[:n - k]
    i = int(np.argmin(widths))
    lo, hi = float(a[i]), float(a[i + k])
    return RichResult(
        title="Highest Posterior Density credible interval",
        summary_lines=[
            (f"{cred*100:.0f}% HPD interval", f"[{lo:.4g}, {hi:.4g}]"),
            ("Width", hi - lo),
            ("n samples", n),
            ("Posterior mean", float(a.mean())),
            ("Posterior median", float(np.median(a))),
        ],
        warnings=[] if n >= 1000 else
                 [f"n={n} samples is small for stable HPD; consider running "
                  "more MCMC iterations."],
        interpretation=(f"Posterior probability that the parameter lies in "
                        f"[{lo:.4g}, {hi:.4g}] is {cred*100:.0f}%."),
        payload={"value": (lo, hi), "lo": lo, "hi": hi,
                 "credible_level": cred, "width": hi - lo},
    )
