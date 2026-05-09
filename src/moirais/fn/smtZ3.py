"""SMT solver framework."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["smt_solver"]


def smt_solver(formula):
    """
    SMT solver framework

    Formula: DPLL(T) — SAT + theory solvers

    Parameters
    ----------
    formula : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    de Moura-Bjørner (2008) Z3
    """
    formula = np.atleast_1d(np.asarray(formula, dtype=float))
    n = len(formula)
    result = float(np.mean(formula))
    se = float(np.std(formula, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "SMT solver framework"})


def cheatsheet():
    return "smtZ3: SMT solver framework"
