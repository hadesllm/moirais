"""Contact tracing yield."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["contact_tracing_yield"]


def contact_tracing_yield(contacts, detection_rate, positivity):
    """
    Contact tracing yield

    Formula: E[# contacts found] × P(positive)

    Parameters
    ----------
    contacts : array-like
        Input data.
    detection_rate : array-like
        Input data.
    positivity : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Hellewell et al (2020)
    """
    contacts = np.atleast_1d(np.asarray(contacts, dtype=float))
    n = len(contacts)
    result = float(np.mean(contacts))
    se = float(np.std(contacts, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Contact tracing yield"})


def cheatsheet():
    return "ttrace: Contact tracing yield"
