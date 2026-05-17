# morie.fn -- function file (hadesllm/morie)
"""Solve the 1D heat equation via explicit finite differences."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult
from ._richresult import RichResult

__all__ = ["heat_diffusion", "diffusion_forward"]


def heat_diffusion(
    T0: np.ndarray,
    alpha: float = 0.01,
    dx: float = 0.1,
    dt: float = 0.01,
    n_steps: int = 100,
) -> DescriptiveResult:
    r"""
    Solve the 1D heat equation via explicit finite differences.

    .. math::

        \\frac{\\partial T}{\\partial t} = \\alpha \\frac{\\partial^2 T}{\\partial x^2}

    Uses forward Euler with Dirichlet boundary conditions (T fixed at
    endpoints).

    :param T0: Initial temperature profile (1D array).
    :param alpha: Thermal diffusivity (m^2/s). Default 0.01.
    :param dx: Spatial step (m). Default 0.1.
    :param dt: Time step (s). Default 0.01.
    :param n_steps: Number of time steps. Default 100.
    :return: DescriptiveResult with final temperature and stability info.
    :raises ValueError: If CFL condition is violated (r > 0.5).

    References
    ----------
    Crank, J. (1975). *The Mathematics of Diffusion*. 2nd ed.
    Oxford University Press.
    """
    T0 = np.asarray(T0, dtype=np.float64)
    if T0.ndim != 1 or len(T0) < 3:
        raise ValueError("T0 must be a 1D array with at least 3 points.")

    r = alpha * dt / (dx**2)
    if r > 0.5:
        raise ValueError(f"CFL condition violated: r = {r:.4f} > 0.5. Reduce dt or increase dx.")

    n_x = len(T0)
    T = T0.copy()
    history = [T.copy()]

    for _ in range(n_steps):
        T_new = T.copy()
        T_new[1:-1] = T[1:-1] + r * (T[2:] - 2 * T[1:-1] + T[:-2])
        T = T_new
        history.append(T.copy())

    return DescriptiveResult(
        name="1D Heat Diffusion",
        value=float(T.mean()),
        extra={
            "T_final": T,
            "T_initial": T0,
            "history": np.array(history),
            "r_stability": float(r),
            "n_steps": n_steps,
            "alpha": alpha,
        },
    )


def diffusion_forward(x0, t: int, betas=None, num_steps: int = 1000,
                      noise=None, seed: int = 0):
    r"""DDPM forward (noising) process (Ho et al. 2020).

    .. math::

        x_t = \\sqrt{\\bar\\alpha_t}\\, x_0
              + \\sqrt{1 - \\bar\\alpha_t}\\, \\varepsilon,
              \\quad \\varepsilon \\sim \\mathcal{N}(0, I)

    where :math:`\\bar\\alpha_t = \\prod_{s=1}^{t}(1 - \\beta_s)` and
    :math:`\\beta_s` is a linear noise schedule from ``1e-4`` to
    ``0.02`` over ``num_steps`` by default.

    Parameters
    ----------
    x0 : array-like
        Clean sample(s).
    t : int
        Diffusion timestep (1..num_steps).
    betas : array-like, optional
        Custom :math:`\\beta` schedule. Overrides ``num_steps``.
    num_steps : int
        Total diffusion steps (default 1000).
    noise : array-like, optional
        Pre-generated Gaussian noise; reproducible via ``seed``.
    seed : int
        RNG seed.

    Returns
    -------
    result : RichResult
        Keys: ``x_t`` / ``estimate``, ``noise``, ``alpha_bar``, ``beta``.

    References
    ----------
    Ho, J., Jain, A., & Abbeel, P. (2020). Denoising diffusion
    probabilistic models. *NeurIPS*.
    """
    x0 = np.asarray(x0, dtype=float)
    if betas is None:
        betas = np.linspace(1e-4, 0.02, int(num_steps))
    betas = np.asarray(betas, dtype=float)
    if not (1 <= t <= len(betas)):
        raise ValueError(f"t must be in [1, {len(betas)}], got {t}.")
    alphas = 1.0 - betas
    alpha_bar = float(np.prod(alphas[:t]))
    if noise is None:
        rng = np.random.default_rng(seed)
        noise = rng.standard_normal(x0.shape)
    else:
        noise = np.asarray(noise, dtype=float)
    x_t = np.sqrt(alpha_bar) * x0 + np.sqrt(1.0 - alpha_bar) * noise
    return RichResult(
        title=f"Diffusion forward (DDPM, t={t})",
        summary_lines=[("t", t), ("alpha_bar_t", alpha_bar),
                       ("beta_t", float(betas[t - 1]))],
        payload={
            "x_t": x_t,
            "estimate": x_t,
            "noise": noise,
            "alpha_bar": alpha_bar,
            "beta": float(betas[t - 1]),
            "method": "DDPM forward diffusion",
        },
    )


# CANONICAL TEST
# diffusion_forward(x0=[1,1], t=1, betas=[0.5], noise=[0,0]).x_t
#   alpha_bar = 0.5, x_t = sqrt(0.5)*[1,1] = [0.7071, 0.7071]


def cheatsheet() -> str:
    return "diffu: heat_diffusion(...) PDE / diffusion_forward(...) DDPM"
