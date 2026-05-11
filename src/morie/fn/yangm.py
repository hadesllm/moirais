"""Yang-Mills action functional."""

__all__ = ["yangm"]

import numpy as np


def yangm(
    A: np.ndarray,
    g_coupling: float = 1.0,
    structure_constants: np.ndarray = None,
) -> dict:
    """
    Compute the Yang-Mills field strength tensor and action density.

    Field strength for SU(N) gauge field:

    .. math::

        F^a_{\\mu\\nu} = \\partial_\\mu A^a_\\nu - \\partial_\\nu A^a_\\mu
                        + g f^{abc} A^b_\\mu A^c_\\nu

    Action density:

    .. math::

        \\mathcal{L} = -\\frac{1}{4} F^a_{\\mu\\nu} F^{a\\mu\\nu}

    For SU(2), the structure constants are the Levi-Civita symbol.

    Parameters
    ----------
    A : np.ndarray
        Gauge field (n_generators, 4) where A[a, mu] is the a-th
        color component at spacetime index mu.
    g_coupling : float
        Gauge coupling constant.
    structure_constants : np.ndarray, optional
        f^{abc} array of shape (N, N, N). Default: SU(2) epsilon_{abc}.

    Returns
    -------
    dict
        Keys: field_strength (n_gen, 4, 4), action_density (float),
        structure_constants.
    """
    A = np.asarray(A, dtype=float)
    n_gen = A.shape[0]
    if A.ndim != 2 or A.shape[1] != 4:
        raise ValueError("A must be (n_generators, 4).")

    if structure_constants is None:
        if n_gen != 3:
            raise ValueError("Default structure constants only for SU(2) (3 generators).")
        f = np.zeros((3, 3, 3))
        f[0, 1, 2] = 1.0
        f[1, 2, 0] = 1.0
        f[2, 0, 1] = 1.0
        f[0, 2, 1] = -1.0
        f[2, 1, 0] = -1.0
        f[1, 0, 2] = -1.0
    else:
        f = np.asarray(structure_constants, dtype=float)

    F = np.zeros((n_gen, 4, 4))

    for a in range(n_gen):
        for mu in range(4):
            for nu in range(4):
                for b in range(n_gen):
                    for cc in range(n_gen):
                        F[a, mu, nu] += g_coupling * f[a, b, cc] * A[b, mu] * A[cc, nu]

    action_density = -0.25 * np.sum(F ** 2)

    return {
        "field_strength": F,
        "action_density": float(action_density),
        "structure_constants": f,
    }
