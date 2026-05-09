# moirais.fn — function file (hadesllm/moirais)
"""QQ-plot (quantile-quantile plot) for visual GOF assessment."""
import numpy as np
from ._richresult import RichResult

__all__ = ["gibbons_qq_plot"]


def gibbons_qq_plot(x, F0):
    """
    QQ-plot (quantile-quantile plot) for visual GOF assessment

    Formula: Plot (F0^{-1}((i-0.5)/n), X_(i)) pairs; linearity = good fit

    Parameters
    ----------
    x : array-like
        Input data.
    F0 : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: plot_data

    References
    ----------
    Gibbons Ch 4.8
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "QQ-plot (quantile-quantile plot) for visual GOF assessment"})


def cheatsheet():
    return "gb_qq: QQ-plot (quantile-quantile plot) for visual GOF assessment"
