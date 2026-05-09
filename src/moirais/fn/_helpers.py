"""
Shared helper utilities for moirais.fn causal estimator functions.

Contains internal functions used across multiple estimator modules,
notably ``_safe_exp`` which is also used by ``moirais.investigation``.
"""

from __future__ import annotations

import numpy as np


def _safe_exp(value):
    """Exponentiate on a clipped log scale to avoid overflow warnings in summaries.

    Accepts scalars, arrays, or Series. Returns float for scalar input,
    ndarray for array/Series input.

    Clips the input to the range [-700, 700] before calling ``np.exp`` to
    avoid ``RuntimeWarning: overflow encountered in exp`` when log-odds are
    very large or very small (e.g. from perfectly separating logistic models).

    :param value: Scalar, array-like, or pandas Series of values to exponentiate.
    :return: Exponentiated value(s).  Float for scalar input, ndarray otherwise.
    """
    arr = np.asarray(value, dtype=float)
    result = np.exp(np.clip(arr, -700, 700))
    return float(result) if result.ndim == 0 else result


safe_exp = _safe_exp


def _validate_df(data, *required_cols: str) -> None:
    """Validate that data is a DataFrame with required columns."""
    import pandas as pd

    if not isinstance(data, pd.DataFrame):
        raise TypeError(f"Expected DataFrame, got {type(data).__name__}")
    missing = [c for c in required_cols if c not in data.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")


def _extract_col(data, col: str) -> np.ndarray:
    """Extract a column from a DataFrame as a float64 array, dropping NaN."""
    import pandas as pd

    if isinstance(data, pd.DataFrame):
        if col not in data.columns:
            raise ValueError(f"Column '{col}' not in DataFrame")
        arr = data[col].to_numpy(dtype=np.float64)
    else:
        arr = np.asarray(data, dtype=np.float64).ravel()
    return arr[np.isfinite(arr)]


def _check_binary(x: np.ndarray) -> None:
    """Raise ValueError if array is not binary (0/1)."""
    unique = np.unique(x[np.isfinite(x)])
    if not np.array_equal(unique, [0, 1]) and not np.array_equal(unique, [0.0, 1.0]):
        raise ValueError(f"Expected binary (0/1), got unique values: {unique}")


def _ci_from_se(estimate: float, se: float, alpha: float = 0.05) -> tuple[float, float]:
    """Compute normal-approximation CI from estimate and SE."""
    from scipy import stats as _st

    z = _st.norm.ppf(1 - alpha / 2)
    return estimate - z * se, estimate + z * se


def _arr(x) -> np.ndarray:
    """Coerce to float64 array, drop NaN."""
    a = np.asarray(x, dtype=np.float64).ravel()
    return a[np.isfinite(a)]


def _bootstrap_ci(func, args, n_boot=2000, confidence=0.95, seed=42):
    """Percentile bootstrap CI and SE for an effect-size function.

    Returns (se, ci_lower, ci_upper).
    """
    rng = np.random.RandomState(seed)
    boot_vals = np.empty(n_boot)
    arrays = [np.asarray(a) for a in args]
    for b in range(n_boot):
        resampled = tuple(a[rng.choice(len(a), len(a), replace=True)] for a in arrays)
        try:
            boot_vals[b] = func(*resampled)
        except Exception:
            boot_vals[b] = np.nan
    boot_vals = boot_vals[np.isfinite(boot_vals)]
    if len(boot_vals) == 0:
        return 0.0, np.nan, np.nan
    alpha = (1 - confidence) / 2
    ci_lo = float(np.percentile(boot_vals, 100 * alpha))
    ci_hi = float(np.percentile(boot_vals, 100 * (1 - alpha)))
    se = float(np.std(boot_vals, ddof=1))
    return se, ci_lo, ci_hi
