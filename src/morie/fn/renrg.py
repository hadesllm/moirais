# morie.fn -- function file (hadesllm/morie)
"""Renormalization group beta function."""

__all__ = ["renrg"]

import numpy as np


def renrg(
    coupling: float,
    n_flavors: int = 6,
    n_colors: int = 3,
    gauge_group: str = "SU3",
    loop_order: int = 1,
) -> dict:
    r"""
    Compute the renormalization group beta function for QCD.

    One-loop beta function:

    .. math::

        \\beta(g) = -\\frac{g^3}{16\\pi^2}
        \\left(\\frac{11}{3} C_A - \\frac{4}{3} T_F n_f\\right)

    For SU(N): :math:`C_A = N`, :math:`T_F = 1/2`.

    Two-loop:

    .. math::

        \\beta(g) = -b_0 g^3 - b_1 g^5

    Parameters
    ----------
    coupling : float
        Gauge coupling constant g (> 0).
    n_flavors : int
        Number of active quark flavors.
    n_colors : int
        Number of colors (3 for QCD).
    gauge_group : str
        'SU2' or 'SU3'.
    loop_order : int
        1 or 2.

    Returns
    -------
    dict
        Keys: beta, b0, b1 (if 2-loop), asymptotic_freedom (bool),
        alpha_s (strong coupling alpha = g^2/(4*pi)).
    """
    if coupling <= 0:
        raise ValueError("Coupling must be > 0.")

    N = n_colors
    CA = float(N)
    TF = 0.5
    CF = (N ** 2 - 1.0) / (2.0 * N)
    nf = n_flavors

    b0 = (11.0 / 3.0 * CA - 4.0 / 3.0 * TF * nf) / (16.0 * np.pi ** 2)
    beta_1 = -b0 * coupling ** 3

    b1 = None
    beta_val = beta_1

    if loop_order >= 2:
        b1_num = (34.0 / 3.0 * CA ** 2 - 20.0 / 3.0 * CA * TF * nf - 4.0 * CF * TF * nf)
        b1 = b1_num / (16.0 * np.pi ** 2) ** 2
        beta_val = beta_1 - b1 * coupling ** 5

    asymptotic = b0 > 0

    alpha_s = coupling ** 2 / (4.0 * np.pi)

    result = {
        "beta": float(beta_val),
        "b0": float(b0),
        "asymptotic_freedom": asymptotic,
        "alpha_s": float(alpha_s),
    }
    if b1 is not None:
        result["b1"] = float(b1)
    return result
