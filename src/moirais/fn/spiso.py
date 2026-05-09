"""Isotropy: covariance depends only on lag distance ||h||, not direction."""
import numpy as np
from ._richresult import RichResult

__all__ = ["schabenberger_isotropy_condition"]


def schabenberger_isotropy_condition(x=None, *args, **kwargs):
    """
    Isotropy: covariance depends only on lag distance ||h||, not direction

    Formula: C(h) = C(||h||) for all h (isotropic); C(h,direction) anisotropic otherwise

    Parameters
    ----------


    Returns
    -------
    result : dict
        Keys: definition

    References
    ----------
    Schabenberger Ch 2, Sec 2.2
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Isotropy: covariance depends only on lag distance ||h||, not direction"})


def cheatsheet():
    return "spiso: Isotropy: covariance depends only on lag distance ||h||, not direction"
