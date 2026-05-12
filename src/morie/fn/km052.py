r"""T5 template obj.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["kamath_ch3_t5_template_obj"]


def kamath_ch3_t5_template_obj(D_train, T, T5):
    r"""
    T5 template obj.

    Formula: \sum_{(x_{in},y)\in D_{train}} \log P_{T5}(T|T(x_{in},y))

    Parameters
    ----------
    D_train : array-like
        Input data.
    T : array-like
        Input data.
    T5 : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: result

    References
    ----------
    Kamath et al (2024), Ch 3, Eq 3.11, p. 108
    r"""
    D_train = np.atleast_1d(np.asarray(D_train, dtype=float))
    n = len(D_train)
    result = float(np.mean(D_train))
    se = float(np.std(D_train, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "T5 template obj."})


def cheatsheet():
    return "km052: T5 template obj."
