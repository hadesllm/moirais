# morie.fn — function file (hadesllm/morie)
"""Density matrix operations."""

__all__ = ["densm"]

import numpy as np


def densm(
    state: np.ndarray = None,
    rho: np.ndarray = None,
    operation: str = "construct",
    subsystem_dims: tuple = None,
    trace_out: int = None,
) -> dict:
    r"""
    Density matrix construction and operations.

    Construct from a pure state:

    .. math::

        \\rho = |\\psi\\rangle \\langle\\psi|

    Partial trace for subsystem B of a bipartite system:

    .. math::

        \\rho_A = \\text{Tr}_B(\\rho_{AB})

    Parameters
    ----------
    state : np.ndarray, optional
        State vector (ket). Used when operation='construct'.
    rho : np.ndarray, optional
        Existing density matrix. Used for other operations.
    operation : str
        'construct', 'partial_trace', 'fidelity', 'purity'.
    subsystem_dims : tuple, optional
        (dim_A, dim_B) for bipartite system.
    trace_out : int, optional
        0 to trace out first subsystem, 1 for second.

    Returns
    -------
    dict
        Depends on operation. Always includes 'rho_out'.
    """
    if operation == "construct":
        if state is None:
            raise ValueError("state required for construct.")
        psi = np.asarray(state, dtype=complex).ravel()
        norm = np.linalg.norm(psi)
        if abs(norm) < 1e-15:
            raise ValueError("Zero state vector.")
        psi = psi / norm
        rho_out = np.outer(psi, np.conj(psi))
        return {
            "rho_out": rho_out,
            "purity": float(np.real(np.trace(rho_out @ rho_out))),
            "trace": float(np.real(np.trace(rho_out))),
        }

    elif operation == "partial_trace":
        if rho is None or subsystem_dims is None or trace_out is None:
            raise ValueError("rho, subsystem_dims, trace_out required.")
        rho = np.asarray(rho, dtype=complex)
        dA, dB = subsystem_dims
        rho_reshaped = rho.reshape(dA, dB, dA, dB)
        if trace_out == 1:
            rho_out = np.einsum("ijkj->ik", rho_reshaped)
        elif trace_out == 0:
            rho_out = np.einsum("ijkl->jl", rho_reshaped[:, :, :, :])
            rho_reshaped2 = rho.reshape(dA, dB, dA, dB)
            rho_out = np.einsum("ibjb->bj", rho_reshaped2)
            rho_out = rho_out.reshape(dB, dB)
        else:
            raise ValueError("trace_out must be 0 or 1.")
        return {
            "rho_out": rho_out,
            "trace": float(np.real(np.trace(rho_out))),
        }

    elif operation == "purity":
        if rho is None:
            raise ValueError("rho required.")
        rho = np.asarray(rho, dtype=complex)
        p = float(np.real(np.trace(rho @ rho)))
        return {
            "rho_out": rho,
            "purity": p,
            "is_pure": abs(p - 1.0) < 1e-10,
        }

    else:
        raise ValueError(f"Unknown operation: {operation}")
