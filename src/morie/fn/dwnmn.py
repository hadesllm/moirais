# morie.fn — function file (hadesllm/morie)
"""Dynamic W-NOMINATE / dynamic ideal points (Armstrong Ch 6)."""
import numpy as np
from ._richresult import RichResult

__all__ = ["dynamic_wnominate", "dwnmn"]


def dynamic_wnominate(x, sigma_w: float = 0.1):
    """Dynamic ideal-point smoother: random-walk over time.

    Model: x_{i,t} = x_{i,t-1} + w_t,  w_t ~ N(0, sigma_w^2).
    Estimator: Kalman-style RTS smoother applied to a series of
    per-period MLE/raw ideal points x_t with measurement noise s2_obs
    set to the cross-sectional variance of the period.

    Parameters
    ----------
    x : (T,) or (n, T) array
        Either a single legislator's per-period ideal points (1-D) or a
        panel (rows = legislators, cols = periods).
    sigma_w : float
        Random-walk innovation SD.

    Returns
    -------
    RichResult with keys: smoothed, raw, sigma_w, n_periods
    """
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        raw = x.copy()
        T = raw.size
        if T == 0:
            return RichResult(payload={"smoothed": np.array([]),
                                       "raw": np.array([]),
                                       "sigma_w": sigma_w, "n_periods": 0,
                                       "method": "dynamic_wnominate"})
        # 1-D Kalman + RTS smoother on raw scalar
        s2_obs = float(np.var(raw) + 1e-6)
        # forward
        m = np.zeros(T); P = np.zeros(T)
        m[0] = raw[0]; P[0] = s2_obs
        for t in range(1, T):
            mp = m[t-1]
            Pp = P[t-1] + sigma_w ** 2
            K = Pp / (Pp + s2_obs)
            m[t] = mp + K * (raw[t] - mp)
            P[t] = (1 - K) * Pp
        # backward smoother
        ms = m.copy(); Ps = P.copy()
        for t in range(T - 2, -1, -1):
            Pp = P[t] + sigma_w ** 2
            J = P[t] / Pp
            ms[t] = m[t] + J * (ms[t+1] - m[t])
            Ps[t] = P[t] + J ** 2 * (Ps[t+1] - Pp)
        return RichResult(
            title="Dynamic ideal points (RTS smoother)",
            summary_lines=[("T periods", T), ("sigma_w", sigma_w)],
            payload={"smoothed": ms, "raw": raw, "P_smoothed": Ps,
                     "sigma_w": float(sigma_w), "n_periods": int(T),
                     "method": "dynamic_wnominate"},
        )
    # Panel: smooth each legislator independently
    n, T = x.shape
    out = np.empty_like(x)
    for i in range(n):
        sub = dynamic_wnominate(x[i], sigma_w=sigma_w)
        out[i] = sub["smoothed"]
    return RichResult(
        title="Dynamic ideal points (RTS smoother, panel)",
        summary_lines=[("n legislators", n), ("T periods", T),
                       ("sigma_w", sigma_w)],
        payload={"smoothed": out, "raw": x, "sigma_w": float(sigma_w),
                 "n_periods": int(T), "n_units": int(n),
                 "method": "dynamic_wnominate"},
    )


dwnmn = dynamic_wnominate


def cheatsheet():
    return "dwnmn: Dynamic W-NOMINATE — RTS smoother over random-walk ideal pts."


# CANONICAL TEST
# >>> r = dynamic_wnominate([0.0, 0.1, 0.2, 0.3, 0.4])
# >>> assert r["smoothed"].shape == (5,)
