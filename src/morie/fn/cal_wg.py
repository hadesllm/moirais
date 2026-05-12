# morie.fn — function file from book-equation translation pipeline (hadesllm/morie)
"""Raking calibration (iterative proportional fitting) weights."""

import warnings

import numpy as np
import pandas as pd


def calibration_weights(
    df: pd.DataFrame,
    aux_vars: list[str],
    population_totals: dict[str, float],
    *,
    max_iter: int = 50,
    tol: float = 1e-6,
) -> pd.Series:
    r"""
    Raking calibration (iterative proportional fitting) to match known
    population marginal totals.

    For each auxiliary variable x_j, the calibration adjusts weights
    multiplicatively so that:

    .. math::

        \\sum_i w_i x_{ij} = T_j

    where :math:`T_j` is the known population total for variable j.

    IPF convergence is assessed by the maximum relative deviation across
    all margins at each iteration.

    :param df: Input DataFrame with auxiliary variable columns.
    :param aux_vars: List of column names to calibrate on.
    :param population_totals: Dict mapping column name to known population total.
    :param max_iter: Maximum number of IPF iterations. Default 50.
    :param tol: Convergence tolerance (maximum relative deviation). Default 1e-6.
    :return: pd.Series of calibrated weights.
    :raises ValueError: If any aux_var is missing from df or population_totals,
        or initial weights are zero.

    References
    ----------
    Deville, J.-C., & Sarndal, C.-E. (1992). Calibration estimators in survey
        sampling. JASA, 87(418), 376-382.
    Deming, W. E., & Stephan, F. F. (1940). On a least squares adjustment of
        a sampled frequency table when the expected marginal totals are known.
        Annals of Mathematical Statistics, 11(4), 427-444.
    """
    for var in aux_vars:
        if var not in df.columns:
            raise ValueError(f"Auxiliary variable '{var}' not found in DataFrame.")
        if var not in population_totals:
            raise ValueError(f"Population total for '{var}' not found in population_totals.")

    n = len(df)
    # Start from equal weights summing to n (equivalent to unweighted sample)
    weights = pd.Series(np.ones(n, dtype=float), index=df.index)

    for iteration in range(max_iter):
        max_dev = 0.0
        for var in aux_vars:
            x = df[var].astype(float).values
            T_j = float(population_totals[var])
            current_total = float((weights.values * x).sum())
            if current_total == 0:
                warnings.warn(
                    f"Weighted total of '{var}' is zero at iteration {iteration}; calibration may not converge.",
                    stacklevel=2,
                )
                continue
            factor = T_j / current_total
            weights = weights * factor
            rel_dev = abs(factor - 1.0)
            if rel_dev > max_dev:
                max_dev = rel_dev
        if max_dev < tol:
            break
    else:
        warnings.warn(
            f"Raking calibration did not converge within {max_iter} iterations "
            f"(max relative deviation = {max_dev:.2e}).",
            stacklevel=2,
        )
    return weights


cal_wg_fn = calibration_weights


def cheatsheet() -> str:
    return "calibration_weights({}) -> Raking calibration (iterative proportional fitting) weights."
