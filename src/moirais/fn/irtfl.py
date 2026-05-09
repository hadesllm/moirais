# moirais.fn — function file (hadesllm/moirais)
"""Item fit statistics (infit/outfit) for Rasch-family models."""

from __future__ import annotations

import numpy as np
import pandas as pd


def irtfl(
    data: pd.DataFrame | np.ndarray,
    item_params: dict[str, dict],
    theta: np.ndarray | None = None,
    *,
    model: str = "1PL",
) -> pd.DataFrame:
    """Compute infit and outfit mean-square fit statistics per item.

    Outfit (unweighted) is sensitive to unexpected responses far from item
    difficulty.  Infit (information-weighted) is sensitive to unexpected
    responses near item difficulty.  Acceptable range: 0.5-1.5 (Bond &
    Fox, 2015).

    Parameters
    ----------
    data : DataFrame or ndarray
        Binary item response matrix (n x k), values 0/1.
    item_params : dict
        {item_name: {"a": float, "b": float}}.
    theta : ndarray, optional
        Person ability estimates (length n).  If None, estimated from
        total score via logit transform.
    model : str
        "1PL" or "2PL" (default "1PL").  Controls ICC formula.

    Returns
    -------
    DataFrame
        Columns: item, infit, outfit, infit_z, outfit_z.
        z-standardized values follow Wilson-Hilferty cube-root
        transformation (Wright & Masters, 1982).

    References
    ----------
    Wright, B. D. & Masters, G. N. (1982). Rating Scale Analysis:
    Rasch Measurement. MESA Press.

    Bond, T. G. & Fox, C. M. (2015). Applying the Rasch Model (3rd ed.).
    Routledge.
    """
    X = np.asarray(data, dtype=np.float64)
    n, k = X.shape

    names = list(item_params.keys())
    if len(names) != k:
        raise ValueError(f"item_params has {len(names)} items but data has {k} columns.")

    X = np.where(np.isnan(X), 0.0, X)

    # Person abilities
    if theta is None:
        total = X.sum(axis=1)
        p_person = np.clip(total / k, 0.001, 0.999)
        theta = np.log(p_person / (1.0 - p_person))
    else:
        theta = np.asarray(theta, dtype=np.float64).ravel()
        if len(theta) != n:
            raise ValueError(f"theta length ({len(theta)}) != n ({n}).")

    # Compute expected values and variances
    a_arr = np.array([item_params[nm].get("a", 1.0) for nm in names])
    b_arr = np.array([item_params[nm].get("b", 0.0) for nm in names])

    logits = theta[:, None] * a_arr[None, :] - a_arr[None, :] * b_arr[None, :]
    logits = np.clip(logits, -700, 700)
    E = 1.0 / (1.0 + np.exp(-logits))  # expected P(X=1)
    E = np.clip(E, 1e-10, 1.0 - 1e-10)
    W = E * (1.0 - E)  # variance of Bernoulli = P*Q

    # Standardized residuals
    Z_sq = (X - E) ** 2 / W  # squared standardized residual

    # Outfit: mean of Z^2 across persons for each item
    outfit = Z_sq.mean(axis=0)

    # Infit: information-weighted mean of residuals
    # infit_j = sum_i W_ij * Z_ij^2 / sum_i W_ij
    #         = sum_i (X_ij - E_ij)^2 / sum_i W_ij
    numerator = ((X - E) ** 2).sum(axis=0)
    denominator = W.sum(axis=0)
    infit = numerator / np.where(denominator > 1e-10, denominator, 1e-10)

    # z-standardization (Wilson-Hilferty cube root transformation)
    # For outfit: q_j = sqrt(sum(kurtosis_ij - 1) / N^2)
    # Simplified: use (mnsq^(1/3) - 1) * (3/sd) + (sd/3)
    def _wilson_hilferty_z(mnsq, df_equiv):
        """Wilson-Hilferty z-standardization for mean-square statistics."""
        if df_equiv < 1 or mnsq <= 0:
            return 0.0
        q = max(np.sqrt(2.0 / df_equiv), 1e-10)
        return (mnsq ** (1.0 / 3.0) - 1.0) * (3.0 / q) + (q / 3.0)

    infit_z = np.zeros(k)
    outfit_z = np.zeros(k)
    for j in range(k):
        # Effective df for infit
        Wj = W[:, j]
        sum_W = Wj.sum()
        sum_W2 = (Wj**2).sum()
        df_in = sum_W**2 / max(sum_W2, 1e-10)
        infit_z[j] = _wilson_hilferty_z(infit[j], df_in)

        # Effective df for outfit = N
        outfit_z[j] = _wilson_hilferty_z(outfit[j], n)

    return pd.DataFrame(
        {
            "item": names,
            "infit": infit,
            "outfit": outfit,
            "infit_z": infit_z,
            "outfit_z": outfit_z,
        }
    )


item_fit = irtfl


def cheatsheet() -> str:
    return "irtfl({}) -> Item fit statistics (infit/outfit) for Rasch-family models."
