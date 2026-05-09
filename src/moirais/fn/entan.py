# moirais.fn — function file (hadesllm/moirais)
"""Von Neumann entanglement entropy -Tr(rho log rho)."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def entanglement_entropy(
    rho: np.ndarray | None = None,
    schmidt_coeffs: np.ndarray | None = None,
) -> DescriptiveResult:
    """Compute von Neumann entanglement entropy.

    .. math::

        S = -\\text{Tr}(\\rho \\log \\rho) = -\\sum_i \\lambda_i \\log \\lambda_i

    :param rho: Density matrix (reduced). If None, use schmidt_coeffs.
    :param schmidt_coeffs: Schmidt coefficients (probabilities). Defaults to maximally entangled.
    :return: DescriptiveResult with entropy value.
    """
    if rho is not None:
        eigenvalues = np.linalg.eigvalsh(rho)
        eigenvalues = eigenvalues[eigenvalues > 1e-15]
    elif schmidt_coeffs is not None:
        eigenvalues = np.asarray(schmidt_coeffs, dtype=float)
        eigenvalues = eigenvalues[eigenvalues > 1e-15]
    else:
        eigenvalues = np.array([0.5, 0.5])

    entropy = -np.sum(eigenvalues * np.log(eigenvalues))
    max_entropy = np.log(len(eigenvalues)) if len(eigenvalues) > 0 else 0.0
    return DescriptiveResult(
        name="entanglement_entropy",
        value=float(entropy),
        extra={
            "entropy": float(entropy),
            "max_entropy": float(max_entropy),
            "n_nonzero": int(len(eigenvalues)),
            "eigenvalues": eigenvalues.tolist(),
            "is_pure": abs(entropy) < 1e-12,
            "is_maximally_entangled": abs(entropy - max_entropy) < 1e-12,
        },
    )


def cheatsheet() -> str:
    return "entanglement_entropy(rho) -> von Neumann entropy -Tr(rho log rho)"


entan = entanglement_entropy
