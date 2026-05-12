# morie.fn — function file (hadesllm/morie)
"""Party alignment / Rice cohesion (Armstrong Ch 8)."""
import numpy as np
from ._richresult import RichResult

__all__ = ["party_alignment", "algnm"]


def party_alignment(x, party=None):
    """Rice index of party cohesion (Rice 1928).

        Rice_p = |%yea_p - %nay_p|

    For a single party (1-D x): returns the scalar Rice index. For a
    multi-party panel (2-D x = vote matrix with `party` group vector):
    returns the mean Rice index across roll calls per party.

    Parameters
    ----------
    x : array-like
        1-D: votes (0/1) of a single party on a series of roll calls.
        2-D (n × m): vote matrix; rows = legislators, cols = roll calls.
    party : array-like (n,), optional
        Party labels (required for 2-D input).

    Returns
    -------
    RichResult with keys: estimate (Rice), per_party, n
    """
    X = np.asarray(x, dtype=float)
    if X.ndim == 1:
        valid = X[~np.isnan(X)]
        if valid.size == 0:
            return RichResult(payload={"estimate": np.nan, "n": 0,
                                       "method": "rice_cohesion"})
        pct_yea = float(np.mean(valid == 1))
        pct_nay = float(np.mean(valid == 0))
        rice = abs(pct_yea - pct_nay)
        return RichResult(
            title="Rice cohesion index",
            summary_lines=[("Rice", rice), ("%yea", pct_yea),
                           ("%nay", pct_nay), ("n", int(valid.size))],
            payload={"estimate": rice, "pct_yea": pct_yea,
                     "pct_nay": pct_nay, "n": int(valid.size),
                     "method": "rice_cohesion"},
        )
    n, m = X.shape
    if party is None:
        # Treat whole chamber as one party
        per = {}
        rice = np.empty(m)
        for j in range(m):
            col = X[:, j]
            col = col[~np.isnan(col)]
            if col.size == 0:
                rice[j] = np.nan
            else:
                rice[j] = abs(np.mean(col == 1) - np.mean(col == 0))
        per["all"] = float(np.nanmean(rice))
        overall = per["all"]
    else:
        p = np.asarray(party).ravel()
        if p.size != n:
            raise ValueError("party must have length == n rows of x")
        per = {}
        for lbl in np.unique(p):
            sub = X[p == lbl]
            rice_p = np.empty(m)
            for j in range(m):
                col = sub[:, j]
                col = col[~np.isnan(col)]
                rice_p[j] = (abs(np.mean(col == 1) - np.mean(col == 0))
                             if col.size else np.nan)
            per[str(lbl)] = float(np.nanmean(rice_p))
        overall = float(np.nanmean(list(per.values())))
    return RichResult(
        title="Rice cohesion (per-party)",
        summary_lines=[("Mean Rice (all parties)", overall),
                       ("n legislators", n), ("m roll calls", m)],
        payload={"estimate": overall, "per_party": per,
                 "n": int(n), "m": int(m),
                 "method": "rice_cohesion"},
    )


algnm = party_alignment


def cheatsheet():
    return "algnm: Rice cohesion = |%yea - %nay| per party."


# CANONICAL TEST
# >>> r = party_alignment([1,1,1,1,0])
# >>> assert abs(r["estimate"] - 0.6) < 1e-9
