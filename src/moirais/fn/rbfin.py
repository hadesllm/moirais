# moirais.fn — function file (hadesllm/moirais)
"""Radial basis function interpolation. 'I've got a bad feeling about this.' -- Various"""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def rbf_interp(
    x_known: np.ndarray,
    y_known: np.ndarray,
    x_eval: np.ndarray,
    kernel: str = "gaussian",
    epsilon: float = 1.0,
) -> DescriptiveResult:
    """
    Radial basis function interpolation.

    Supported kernels:
    - gaussian: :math:`\\phi(r) = \\exp(-(\\epsilon r)^2)`
    - multiquadric: :math:`\\phi(r) = \\sqrt{1 + (\\epsilon r)^2}`
    - inverse_multiquadric: :math:`\\phi(r) = 1/\\sqrt{1 + (\\epsilon r)^2}`
    - thin_plate: :math:`\\phi(r) = r^2 \\ln(r)`

    :param x_known: Known x-coordinates (1D array).
    :param y_known: Known y-values.
    :param x_eval: Evaluation points.
    :param kernel: RBF kernel name. Default "gaussian".
    :param epsilon: Shape parameter. Default 1.0.
    :return: DescriptiveResult with interpolated values and weights.
    :raises ValueError: If kernel is unknown or arrays are incompatible.

    References
    ----------
    Buhmann, M. D. (2003). *Radial Basis Functions: Theory and
    Implementations*. Cambridge University Press.
    """
    xk = np.asarray(x_known, dtype=np.float64).ravel()
    yk = np.asarray(y_known, dtype=np.float64).ravel()
    xe = np.asarray(x_eval, dtype=np.float64).ravel()

    if len(xk) != len(yk):
        raise ValueError("x_known and y_known must have equal length.")

    kernels = {"gaussian", "multiquadric", "inverse_multiquadric", "thin_plate"}
    if kernel not in kernels:
        raise ValueError(f"Unknown kernel '{kernel}'. Choose from {kernels}.")

    def _phi(r: np.ndarray) -> np.ndarray:
        if kernel == "gaussian":
            return np.exp(-((epsilon * r) ** 2))
        elif kernel == "multiquadric":
            return np.sqrt(1 + (epsilon * r) ** 2)
        elif kernel == "inverse_multiquadric":
            return 1.0 / np.sqrt(1 + (epsilon * r) ** 2)
        else:
            with np.errstate(divide="ignore", invalid="ignore"):
                val = r**2 * np.log(np.maximum(r, 1e-15))
            return np.where(r == 0, 0.0, val)

    n = len(xk)
    dist_mat = np.abs(xk[:, None] - xk[None, :])
    Phi = _phi(dist_mat)

    weights = np.linalg.solve(Phi, yk)

    dist_eval = np.abs(xe[:, None] - xk[None, :])
    Phi_eval = _phi(dist_eval)
    result = Phi_eval @ weights

    return DescriptiveResult(
        name="RBF Interpolation",
        value=float(result[0]) if len(result) == 1 else float(np.mean(result)),
        extra={
            "y_eval": result,
            "weights": weights,
            "kernel": kernel,
            "epsilon": epsilon,
            "n_centers": n,
            "condition_number": float(np.linalg.cond(Phi)),
        },
    )


short = rbf_interp


def cheatsheet() -> str:
    return "rbf_interp({}) -> Radial basis function interpolation. 'I've got a bad feeling"
