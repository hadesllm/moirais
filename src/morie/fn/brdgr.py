# morie.fn -- function file (hadesllm/morie)
"""Bridge observations for cross-chamber comparison (Armstrong Ch 6)."""
import numpy as np
from ._richresult import RichResult

__all__ = ["bridge_observations", "brdgr"]


def bridge_observations(x, y=None):
    """Identify and weight bridge legislators across two sessions /
    chambers. A "bridge" is any legislator who appears in both sessions
    and votes on at least one common roll call (Bailey 2007; Armstrong
    Ch 6).

    Parameters
    ----------
    x : either
        - (n, m1) roll-call matrix for session 1, OR
        - 1-D vector of session-1 legislator IDs (then `y` is session-2 IDs)
    y : (n, m2) roll-call matrix for session 2 OR list of session-2 IDs.

    Returns
    -------
    RichResult with keys: n_bridges, bridge_ids, share, n1, n2
    """
    if y is None:
        # Single-arg mode: treat x as a (n,) vector where >0 ⇒ bridge.
        x = np.asarray(x).ravel()
        n_bridges = int(np.sum(np.asarray(x, dtype=bool)))
        return RichResult(
            title="Bridge observations (single-arg shortcut)",
            summary_lines=[("n bridges", n_bridges), ("n", int(x.size))],
            payload={"n_bridges": n_bridges,
                     "bridge_ids": np.where(np.asarray(x, dtype=bool))[0],
                     "share": float(n_bridges / max(x.size, 1)),
                     "n1": int(x.size), "n2": int(x.size),
                     "method": "bridge_observations"},
        )
    # Two-arg: matrices or ID lists
    xa = np.asarray(x); ya = np.asarray(y)
    if xa.ndim == 1 and ya.ndim == 1:
        s1 = set(xa.tolist()); s2 = set(ya.tolist())
        common = sorted(s1 & s2)
        return RichResult(
            title="Bridge legislators across sessions",
            summary_lines=[("n bridges", len(common)),
                           ("n session 1", len(s1)),
                           ("n session 2", len(s2)),
                           ("share of session 1",
                            len(common) / max(len(s1), 1))],
            payload={"n_bridges": len(common),
                     "bridge_ids": np.array(common),
                     "share": float(len(common) / max(len(s1), 1)),
                     "n1": len(s1), "n2": len(s2),
                     "method": "bridge_observations"},
        )
    # Matrix mode: bridges = rows non-empty in both
    if xa.ndim != 2 or ya.ndim != 2 or xa.shape[0] != ya.shape[0]:
        raise ValueError("x and y must be 2-D matrices with the same n rows "
                         "(or 1-D ID vectors).")
    n = xa.shape[0]
    has1 = np.any(~np.isnan(xa.astype(float)), axis=1)
    has2 = np.any(~np.isnan(ya.astype(float)), axis=1)
    bridges = has1 & has2
    n_b = int(np.sum(bridges))
    return RichResult(
        title="Bridge observations (panel intersection)",
        summary_lines=[("n bridges", n_b), ("n legislators", n),
                       ("share", float(n_b / max(n, 1)))],
        payload={"n_bridges": n_b,
                 "bridge_ids": np.where(bridges)[0],
                 "share": float(n_b / max(n, 1)),
                 "n1": int(has1.sum()), "n2": int(has2.sum()),
                 "method": "bridge_observations"},
    )


brdgr = bridge_observations


def cheatsheet():
    return "brdgr: Bridge observations -- common legislators across sessions."


# CANONICAL TEST
# >>> r = bridge_observations(["A","B","C"], ["B","C","D"])
# >>> assert r["n_bridges"] == 2
