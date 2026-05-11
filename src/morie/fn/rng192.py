"""Updated SPKI rule when a QRS is detected in the search-back procedure.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["rangayyan_ch4_pan_tompkins_searchback_update"]


def rangayyan_ch4_pan_tompkins_searchback_update(PEAKI, SPKI):
    """
    Updated SPKI rule when a QRS is detected in the search-back procedure.

    Formula: SPKI = 0.25 * PEAKI + 0.75 * SPKI

    Parameters
    ----------
    PEAKI : array-like
        Input data.
    SPKI : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: value

    References
    ----------
    Rangayyan (2024), Ch 4, Eq 4.18, p. 224
    """
    PEAKI = np.atleast_1d(np.asarray(PEAKI, dtype=float))
    n = len(PEAKI)
    result = float(np.mean(PEAKI))
    se = float(np.std(PEAKI, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Updated SPKI rule when a QRS is detected in the search-back procedure."})


def cheatsheet():
    return "rng192: Updated SPKI rule when a QRS is detected in the search-back procedure."
