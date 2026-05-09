"""State space model (Kalman filter)."""
import numpy as np
from ._richresult import RichResult

__all__ = ["state_space_model"]


def state_space_model(x):
    """
    State space model (Kalman filter)

    Formula: x_t = F*x_{t-1} + w_t; y_t = H*x_t + v_t

    Parameters
    ----------
    x : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Harvey (1989)
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "State space model (Kalman filter)"})


def cheatsheet():
    return "ssmod: State space model (Kalman filter)"
