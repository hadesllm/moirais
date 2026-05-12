# morie.fn -- function file (hadesllm/morie)
"""Optimal Classification cutting-plane estimator (Poole 2000; Armstrong Ch 3)."""
import numpy as np
from ._richresult import RichResult

__all__ = ["optimal_classification", "optcl"]


def optimal_classification(x, votes=None):
    """Optimal Classification (Poole 2000): 1-D cutting-point that
    minimises classification error of a binary vote vector.

    Formula: minimise sum_i I{predict(x_i) != y_i} via a hyperplane
              x_i > c  =>  yea
    For 1-D this reduces to choosing c* among midpoints between
    successive sorted ideal points so that "correctly classified"
    is maximised over both polarities (yea-high vs yea-low).

    Parameters
    ----------
    x : array-like (n,) or (n,1)
        Legislator ideal points on a single dimension.
    votes : array-like (n,) of {0,1}, optional
        Observed votes (1 = yea). If None, the function returns the
        median split as the trivial cutting point.

    Returns
    -------
    RichResult with keys: cut, correct_class, polarity, pre, n
    """
    x = np.asarray(x, dtype=float).ravel()
    n = int(x.size)
    if n == 0:
        return RichResult(payload={"cut": np.nan, "correct_class": 0,
                                   "polarity": 1, "pre": np.nan, "n": 0,
                                   "method": "optimal_classification"})
    if votes is None:
        cut = float(np.median(x))
        return RichResult(
            title="Optimal Classification (Poole 2000)",
            summary_lines=[("Cut (median split)", cut), ("n", n)],
            payload={"cut": cut, "correct_class": n // 2 + n % 2,
                     "polarity": 1, "pre": np.nan, "n": n,
                     "method": "optimal_classification"},
        )
    y = np.asarray(votes, dtype=int).ravel()
    # Candidate cuts: midpoints between sorted unique ideal pts
    order = np.argsort(x)
    xs = x[order]
    candidates = list((xs[:-1] + xs[1:]) / 2.0)
    candidates = [xs[0] - 1.0] + candidates + [xs[-1] + 1.0]
    best_cc = -1
    best_cut = float(np.median(x))
    best_pol = 1
    for c in candidates:
        for pol in (1, -1):
            pred = ((x > c).astype(int) if pol == 1 else (x <= c).astype(int))
            cc = int(np.sum(pred == y))
            if cc > best_cc:
                best_cc, best_cut, best_pol = cc, float(c), pol
    # PRE (proportional reduction in error vs majority guess)
    p = float(np.mean(y))
    base_correct = max(p, 1 - p) * n
    pre = (best_cc - base_correct) / (n - base_correct) if n > base_correct else 0.0
    return RichResult(
        title="Optimal Classification (Poole 2000)",
        summary_lines=[
            ("Optimal cut c*", best_cut),
            ("Correctly classified", best_cc),
            ("Polarity (+1 yea-high)", best_pol),
            ("PRE", pre), ("n", n),
        ],
        interpretation=(
            f"Cut at {best_cut:.4f} correctly classifies {best_cc}/{n} "
            f"votes (PRE = {pre:.3f})."),
        payload={"cut": best_cut, "correct_class": int(best_cc),
                 "polarity": int(best_pol), "pre": float(pre), "n": n,
                 "method": "optimal_classification"},
    )


optcl = optimal_classification


def cheatsheet():
    return "optcl: Optimal classification -- Poole's cutting-plane PRE."


# CANONICAL TEST
# >>> r = optimal_classification([-2,-1,0,1,2], votes=[0,0,0,1,1])
# >>> assert r["correct_class"] == 5
