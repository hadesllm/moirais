# morie.fn — function file (hadesllm/morie)
"""Expected parameter change from modification indices."""

from __future__ import annotations

import numpy as np
import pandas as pd

from morie.fn._containers import DescriptiveResult


def cfa_expected_change(
    mod_indices: pd.DataFrame | dict,
    se: np.ndarray | list | None = None,
) -> DescriptiveResult:
    """Expected parameter change (EPC) from modification indices.

    Standardised EPC = MI_value / SE. Helps decide whether freeing
    a parameter would meaningfully change the model.

    Parameters
    ----------
    mod_indices : DataFrame or dict
        Must contain 'mi' (modification index) and 'epc' columns.
    se : ndarray or list, optional
        Standard errors for each parameter. If None, uses raw EPC.

    Returns
    -------
    DescriptiveResult
        value=DataFrame with standardised EPCs.

    References
    ----------
    Sorbom, D. (1989). Model modification. Psychometrika, 54(3), 371-384.
    """
    if isinstance(mod_indices, dict):
        df = pd.DataFrame(mod_indices)
    else:
        df = mod_indices.copy()

    if "mi" not in df.columns or "epc" not in df.columns:
        raise ValueError("Input must have 'mi' and 'epc' columns.")

    if se is not None:
        se_arr = np.asarray(se, dtype=np.float64)
        df["se"] = se_arr[: len(df)]
        df["std_epc"] = df["epc"] / np.clip(df["se"], 1e-10, None)
    else:
        df["std_epc"] = df["epc"]

    df = df.sort_values("mi", ascending=False).reset_index(drop=True)

    return DescriptiveResult(
        name="Expected parameter change",
        value=df,
        extra={"n_params": len(df)},
    )


expected_change = cfa_expected_change


def cheatsheet() -> str:
    return "cfa_expected_change({}) -> Expected parameter change from modification indices."
