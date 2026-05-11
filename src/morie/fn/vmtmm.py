"""Multitrait-Multimethod (MTMM) matrix analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd
from ._richresult import RichResult


def validity_mtmm(
    data: pd.DataFrame,
    traits: dict[str, list[str]],
    methods: dict[str, list[str]],
) -> dict:
    """Multitrait-Multimethod matrix analysis (Campbell & Fiske, 1959).

    Computes the MTMM correlation matrix and evaluates convergent and
    discriminant validity criteria.

    Parameters
    ----------
    data : DataFrame
        Columns must cover all items referenced in *traits* and *methods*.
    traits : dict
        Mapping of trait name to list of column names measuring that trait.
    methods : dict
        Mapping of method name to list of column names from that method.

    Returns
    -------
    dict
        Keys: ``correlation_matrix`` (DataFrame), ``monotrait_heteromethod``
        (convergent diagonal), ``heterotrait_monomethod``,
        ``heterotrait_heteromethod``, ``convergent_valid`` (bool),
        ``discriminant_valid`` (bool).

    References
    ----------
    Campbell, D. T., & Fiske, D. W. (1959). Convergent and discriminant
    validation by the multitrait-multimethod matrix. *Psychological
    Bulletin*, 56(2), 81--105.
    """
    # Build composite scores for each trait-method combination
    combos: dict[str, np.ndarray] = {}
    for t_name, t_cols in traits.items():
        for m_name, m_cols in methods.items():
            overlap = [c for c in t_cols if c in m_cols]
            if overlap:
                combos[f"{t_name}_{m_name}"] = np.asarray(data[overlap].mean(axis=1), dtype=np.float64)

    if len(combos) < 2:
        return RichResult(payload={"correlation_matrix": pd.DataFrame(), "convergent_valid": False, "discriminant_valid": False})

    names = list(combos.keys())
    X = np.column_stack([combos[n] for n in names])
    R = np.corrcoef(X, rowvar=False)
    corr_df = pd.DataFrame(R, index=names, columns=names)

    # Classify correlations
    mono_hetero = []  # same trait, different method (convergent)
    hetero_mono = []  # different trait, same method
    hetero_hetero = []  # different trait, different method

    trait_names = list(traits.keys())
    method_names = list(methods.keys())

    for i, ni in enumerate(names):
        ti, mi = ni.rsplit("_", 1)
        for j, nj in enumerate(names):
            if j <= i:
                continue
            tj, mj = nj.rsplit("_", 1)
            val = R[i, j]
            if ti == tj and mi != mj:
                mono_hetero.append(val)
            elif ti != tj and mi == mj:
                hetero_mono.append(val)
            elif ti != tj and mi != mj:
                hetero_hetero.append(val)

    avg_conv = float(np.mean(mono_hetero)) if mono_hetero else np.nan
    avg_hm = float(np.mean(hetero_mono)) if hetero_mono else np.nan
    avg_hh = float(np.mean(hetero_hetero)) if hetero_hetero else np.nan

    # Convergent: monotrait-heteromethod should be high
    conv_valid = bool(avg_conv > 0.5) if np.isfinite(avg_conv) else False
    # Discriminant: convergent > heterotrait correlations
    disc_valid = True
    if np.isfinite(avg_conv) and np.isfinite(avg_hm):
        disc_valid = disc_valid and avg_conv > avg_hm
    if np.isfinite(avg_conv) and np.isfinite(avg_hh):
        disc_valid = disc_valid and avg_conv > avg_hh

    return {
        "correlation_matrix": corr_df,
        "monotrait_heteromethod": mono_hetero,
        "heterotrait_monomethod": hetero_mono,
        "heterotrait_heteromethod": hetero_hetero,
        "avg_convergent": avg_conv,
        "avg_heterotrait_monomethod": avg_hm,
        "avg_heterotrait_heteromethod": avg_hh,
        "convergent_valid": conv_valid,
        "discriminant_valid": bool(disc_valid),
    }


def cheatsheet() -> str:
    return "validity_mtmm({}) -> Multitrait-Multimethod (MTMM) matrix analysis."
