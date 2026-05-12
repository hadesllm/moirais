# morie.fn — function file (hadesllm/morie)
"""Genotype-by-environment interaction (G×E) main-and-interaction model."""
from __future__ import annotations

import numpy as np

from ._richresult import RichResult

__all__ = ["gxe_interaction_model"]


def gxe_interaction_model(x, y, env):
    """Two-way G×E linear model with main effects and interaction.

    Model::

        y_ij = mu + g_i + e_j + (ge)_ij + eps_ij

    where `x` codes genotype (categorical/int label) and `env` codes
    environment.  Each effect is estimated via sum-to-zero OLS; variance
    components for G, E and G×E are then obtained from the ANOVA expected
    mean squares (Montesinos Lopez Ch 11 eq. 11.4).

    Parameters
    ----------
    x : array-like (n,)  — genotype IDs
    y : array-like (n,)  — phenotype
    env : array-like (n,) — environment IDs

    Returns
    -------
    RichResult with payload keys estimate (mu), g, e, ge, var_g, var_e,
    var_ge, var_eps, se, n, method.

    References
    ----------
    Montesinos Lopez et al. (2022), Ch. 11.
    """
    g_id = np.asarray(x).ravel()
    e_id = np.asarray(env).ravel()
    yv = np.asarray(y, dtype=float).ravel()
    n = len(yv)
    if not (len(g_id) == n and len(e_id) == n):
        raise ValueError("x, y, env must be the same length")
    g_levels = np.unique(g_id)
    e_levels = np.unique(e_id)
    G = len(g_levels)
    E = len(e_levels)
    mu = float(np.mean(yv))
    # Main effects via cell means (sum-to-zero)
    g_eff = np.array([float(np.mean(yv[g_id == lev]) - mu) for lev in g_levels])
    e_eff = np.array([float(np.mean(yv[e_id == lev]) - mu) for lev in e_levels])
    # Cell mean matrix [G x E]
    cell_mean = np.full((G, E), np.nan)
    cell_count = np.zeros((G, E), dtype=int)
    for i, gl in enumerate(g_levels):
        for j, el in enumerate(e_levels):
            mask = (g_id == gl) & (e_id == el)
            cell_count[i, j] = int(mask.sum())
            if mask.any():
                cell_mean[i, j] = float(np.mean(yv[mask]))
    ge_eff = cell_mean - mu - g_eff[:, None] - e_eff[None, :]
    # Sums of squares
    n_per_cell = cell_count.mean() if cell_count.sum() else 1
    ss_g = float(np.sum(cell_count.sum(axis=1) * g_eff ** 2))
    ss_e = float(np.sum(cell_count.sum(axis=0) * e_eff ** 2))
    valid = ~np.isnan(ge_eff)
    ss_ge = float(np.sum(cell_count[valid] * ge_eff[valid] ** 2))
    # Residual (within-cell) SS
    resid = np.zeros(n)
    for i, gl in enumerate(g_levels):
        for j, el in enumerate(e_levels):
            mask = (g_id == gl) & (e_id == el)
            if mask.any():
                resid[mask] = yv[mask] - cell_mean[i, j]
    ss_eps = float(np.sum(resid ** 2))
    df_g = max(G - 1, 1)
    df_e = max(E - 1, 1)
    df_ge = max((G - 1) * (E - 1), 1)
    df_eps = max(n - G * E, 1)
    ms_eps = ss_eps / df_eps
    ms_ge = ss_ge / df_ge
    # EMS-based variance components (random-effects model)
    var_eps = ms_eps
    var_ge = max(0.0, (ms_ge - ms_eps) / max(n_per_cell, 1))
    var_g = max(0.0, (ss_g / df_g - ms_ge) / max(E * n_per_cell, 1))
    var_e = max(0.0, (ss_e / df_e - ms_ge) / max(G * n_per_cell, 1))
    se = float(np.sqrt(var_eps))
    return RichResult(
        title="G×E Interaction Model",
        summary_lines=[
            ("n obs", n),
            ("G (genotypes)", G),
            ("E (environments)", E),
            ("mu", mu),
            ("var(G)", var_g),
            ("var(E)", var_e),
            ("var(GE)", var_ge),
            ("var(resid)", var_eps),
        ],
        payload={
            "estimate": mu,
            "g": g_eff,
            "e": e_eff,
            "ge": ge_eff,
            "var_g": var_g,
            "var_e": var_e,
            "var_ge": var_ge,
            "var_eps": var_eps,
            "se": se,
            "n": n,
            "method": "Two-way G×E ANOVA + EMS variance components",
        },
    )


def cheatsheet():
    return "gxemd: G×E interaction model"


# CANONICAL TEST
# x = [1,1,2,2,3,3,1,1,2,2,3,3]; env = [1,1,1,1,1,1,2,2,2,2,2,2]
# y = [1,2,3,4,5,6,2,3,4,5,6,7]
# r = gxe_interaction_model(x, y, env); r.g, r.e, r.ge populated.
