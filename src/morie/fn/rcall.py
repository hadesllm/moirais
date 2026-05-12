# morie.fn -- function file (hadesllm/morie)
"""Roll-call matrix analysis (Armstrong Ch 2)."""
import numpy as np
from ._richresult import RichResult

__all__ = ["roll_call_analysis", "rcall"]


def roll_call_analysis(x, absent_codes=(np.nan, 9)):
    """Summarise a roll-call matrix V = {v_ij}, v_ij ∈ {yea, nay, absent}.

    Encoding (Poole-Rosenthal convention):
        yea    = {1, 2, 3}  -> 1
        nay    = {4, 5, 6}  -> 0
        absent = {0, 7, 8, 9, NaN}  -> NaN

    Accepts any of those codings; values not in {0, 1, NaN} after
    coercion are mapped via the Poole-Rosenthal rule above.

    Parameters
    ----------
    x : (n, m) array-like.

    Returns
    -------
    RichResult with keys: n, m, n_yea, n_nay, n_abs, marginal_yea,
        marginal_nay, lopsided_pct
    """
    V = np.asarray(x, dtype=float)
    if V.ndim == 1:
        V = V.reshape(-1, 1)
    n, m = V.shape
    # Apply Poole-Rosenthal style decoding if values outside {0,1,NaN}
    if np.any(~(np.isnan(V) | (V == 0) | (V == 1))):
        Vp = np.full_like(V, np.nan)
        Vp[(V == 1) | (V == 2) | (V == 3)] = 1.0
        Vp[(V == 4) | (V == 5) | (V == 6)] = 0.0
        V = Vp
    n_yea = int(np.nansum(V == 1))
    n_nay = int(np.nansum(V == 0))
    n_abs = int(np.sum(np.isnan(V)))
    # Per-roll-call marginals
    marg_yea = np.nansum(V == 1, axis=0)
    marg_nay = np.nansum(V == 0, axis=0)
    denom = marg_yea + marg_nay
    pct_yea = np.where(denom > 0, marg_yea / np.maximum(denom, 1), np.nan)
    # Poole-Rosenthal "lopsided" cutoff: >= 97.5% on the majority side
    lopsided = float(np.mean((pct_yea >= 0.975) | (pct_yea <= 0.025)))
    return RichResult(
        title="Roll-call matrix summary",
        summary_lines=[("n legislators", n), ("m roll calls", m),
                       ("Total yeas", n_yea), ("Total nays", n_nay),
                       ("Total absences", n_abs),
                       ("Lopsided (>=97.5%) share", lopsided)],
        payload={"n": int(n), "m": int(m), "n_yea": n_yea, "n_nay": n_nay,
                 "n_abs": n_abs, "marginal_yea": marg_yea,
                 "marginal_nay": marg_nay,
                 "pct_yea": pct_yea, "lopsided_pct": float(lopsided),
                 "method": "roll_call_analysis"},
    )


rcall = roll_call_analysis


def cheatsheet():
    return "rcall: Roll-call summary -- yea/nay/absent + Poole-Rosenthal lopsided."


# CANONICAL TEST
# >>> M = np.array([[1,0,1],[1,1,0],[0,1,np.nan]])
# >>> r = roll_call_analysis(M)
# >>> assert r["n_yea"] == 5 and r["n_nay"] == 3 and r["n_abs"] == 1
