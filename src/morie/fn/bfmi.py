# morie.fn -- function file from book-equation translation pipeline (hadesllm/morie)
"""Bayesian Fraction of Missing Information (BFMI)."""

from __future__ import annotations

__all__ = ["bayesian_fmi", "bfmi"]

from typing import Any, Union

import numpy as np


def bayesian_fmi(
    energy: Union[list, np.ndarray],
) -> dict[str, Any]:
    """
    Bayesian Fraction of Missing Information (BFMI / E-BFMI).

    Diagnostic for HMC / NUTS samplers.  Compares the variance of
    energy transitions to the marginal variance of the energy.
    Low BFMI (< 0.3) indicates the sampler has difficulty exploring
    the target distribution.

    .. math::

        E\\text{-BFMI} = \\frac{\\text{Var}(E_n - E_{n-1})}{\\text{Var}(E_n)}

    Parameters
    ----------
    energy : array-like
        Hamiltonian energy values from each HMC/NUTS iteration (n,).

    Returns
    -------
    dict
        bfmi : float
        adequate : bool (bfmi >= 0.3)
        energy_var : float
        transition_var : float

    References
    ----------
    Betancourt, M. (2017). A conceptual introduction to Hamiltonian
    Monte Carlo. arXiv:1701.02434.
    """
    e = np.asarray(energy, dtype=float).ravel()
    if len(e) < 3:
        raise ValueError("Need at least 3 energy values.")

    energy_var = float(np.var(e, ddof=1))
    transitions = np.diff(e)
    transition_var = float(np.var(transitions, ddof=1))

    if energy_var < 1e-30:
        bfmi_val = 1.0
    else:
        bfmi_val = transition_var / energy_var

    return {
        "bfmi": bfmi_val,
        "adequate": bool(bfmi_val >= 0.3),
        "energy_var": energy_var,
        "transition_var": transition_var,
    }


bfmi = bayesian_fmi


def cheatsheet() -> str:
    return "bayesian_fmi(energy) -> Bayesian Fraction of Missing Information."
