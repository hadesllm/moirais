# moirais.fn — function file (hadesllm/moirais)
"""Chi-squared distribution probability density function."""

from typing import Union

import numpy as np
import scipy.stats as stats


def dchisq(x: Union[float, np.ndarray], df: float, log: bool = False) -> Union[float, np.ndarray]:
    """
    Chi-squared distribution PDF.

    Mirrors R's ``dchisq(x, df, log)``.

    :param x: Quantile(s) (>= 0).
    :param df: Degrees of freedom (> 0).
    :param log: If True return log-density. Default False.
    :return: Density value(s).
    :raises ValueError: If df <= 0.

    References
    ----------
    R Core Team (2024). dchisq {stats}. R documentation.
    """
    if df <= 0:
        raise ValueError(f"df must be > 0, got {df}.")
    dist = stats.chi2(df=df)
    return dist.logpdf(x) if log else dist.pdf(x)


dchsq = dchisq


def cheatsheet() -> str:
    return "dchisq({}) -> Chi-squared distribution probability density function."
