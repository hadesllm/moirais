# morie.fn -- function file (hadesllm/morie)
"""Goldman-Hodgkin-Katz (GHK) equation for resting membrane potential."""
import numpy as np
from ._richresult import RichResult

__all__ = ["rangayyan_goldman_eqn"]


def rangayyan_goldman_eqn(T, P_K, P_Na, P_Cl, ion_concs):
    """
    Goldman-Hodgkin-Katz (GHK) equation for resting membrane potential

    Formula: V_m = (RT/F)*ln((P_K[K]_o+P_Na[Na]_o+P_Cl[Cl]_i)/(P_K[K]_i+P_Na[Na]_i+P_Cl[Cl]_o))

    Parameters
    ----------
    T : array-like
        Input data.
    P_K : array-like
        Input data.
    P_Na : array-like
        Input data.
    P_Cl : array-like
        Input data.
    ion_concs : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: V_resting

    References
    ----------
    Rangayyan Ch 1.2.1
    """
    T = np.asarray(T, dtype=float)
    n = int(T) if T.ndim == 0 else len(T)
    result = float(np.mean(T))
    se = float(np.std(T, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Goldman-Hodgkin-Katz (GHK) equation for resting membrane potential"})


def cheatsheet():
    return "rgghk: Goldman-Hodgkin-Katz (GHK) equation for resting membrane potential"
