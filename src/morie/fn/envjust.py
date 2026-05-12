# morie.fn -- function file (hadesllm/morie)
"""Environmental justice exposure-disparity index."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ._containers import DescriptiveResult


def environmental_justice_index(
    data: pd.DataFrame | None = None,
    *,
    exposure: str = "exposure",
    group: str = "group",
    weights: str | None = None,
    reference: str | None = None,
) -> DescriptiveResult:
    """Compute environmental-exposure disparity across demographic groups.

    Produces a ratio-of-means environmental-justice index: the mean
    exposure in each group divided by the mean exposure in the
    reference group (default: overall population). Values > 1 indicate
    disproportionate burden on that group.

    Commonly used for measuring how air pollution, heat, lead, or
    toxic-site proximity loads concentrate by race, income, or
    indigenous-land status (EPA EJSCREEN methodology; Mohai & Bryant
    1992; Mikati et al. 2018).

    Parameters
    ----------
    data : pandas.DataFrame
        One row per person or geography. Must contain `exposure` and
        `group` columns.
    exposure : str
        Column name for the exposure variable (e.g., annual mean PM₂.₅
        in µg/m³, proximity to TRI site in km⁻¹).
    group : str
        Column name for the grouping variable (e.g., race, income
        quintile, Indigenous status).
    weights : str, optional
        Column name for population weights. If None, unweighted means.
    reference : str, optional
        Value of `group` to use as the reference. If None, uses the
        overall (unstratified) population mean as denominator -- this is
        the EJSCREEN convention.

    Returns
    -------
    DescriptiveResult
        value = maximum disparity ratio across groups.
        extra contains per-group mean, count, and disparity ratio.

    Raises
    ------
    ValueError
        If required columns missing, or reference group not found.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...     "pm25": [12, 14, 18, 10, 16, 20],
    ...     "race": ["white","white","white","white","black","black"],
    ... })
    >>> r = environmental_justice_index(df, exposure="pm25", group="race")
    >>> r.extra["disparity"]  # doctest: +SKIP
    {'white': 0.9..., 'black': 1.2...}

    Notes
    -----
    Quote: "The cave. Remember your failure at the cave.",
    on why we cannot ignore where exposure lands.

    References
    ----------
    Mohai, P., & Bryant, B. (1992). Environmental injustice: Weighing
    race and class as factors in the distribution of environmental
    hazards. Univ. Colorado Law Review, 63, 921.

    Mikati, I., Benson, A. F., Luben, T. J., Sacks, J. D., & Richmond-
    Bryant, J. (2018). Disparities in distribution of particulate matter
    emission sources by race and poverty status. American Journal of
    Public Health, 108(4), 480-485.

    US EPA EJSCREEN: Environmental Justice Screening and Mapping Tool
    (Version 2.0), https://www.epa.gov/ejscreen
    """
    if data is None:
        raise ValueError("data= DataFrame required.")
    missing = [c for c in (exposure, group) if c not in data.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    if weights is not None:
        if weights not in data.columns:
            raise ValueError(f"weights column {weights!r} not in data.")
        w = data[weights].to_numpy(dtype=float)

        def _m(x: np.ndarray, idx: np.ndarray) -> float:
            sw = w[idx].sum()
            return float((x[idx] * w[idx]).sum() / sw) if sw > 0 else float("nan")
    else:
        def _m(x: np.ndarray, idx: np.ndarray) -> float:
            return float(x[idx].mean()) if idx.sum() > 0 else float("nan")

    x = data[exposure].to_numpy(dtype=float)
    g = data[group].to_numpy()

    # Reference mean
    if reference is None:
        ref_idx = np.ones_like(g, dtype=bool)
    else:
        ref_idx = g == reference
        if not ref_idx.any():
            raise ValueError(f"reference group {reference!r} not present.")
    ref_mean = _m(x, ref_idx)
    if not np.isfinite(ref_mean) or ref_mean == 0:
        raise ValueError("Reference mean is zero/NaN -- cannot compute ratios.")

    per_group_mean: dict[str, float] = {}
    per_group_n: dict[str, int] = {}
    per_group_ratio: dict[str, float] = {}
    for level in pd.unique(g):
        idx = g == level
        m = _m(x, idx)
        per_group_mean[str(level)] = m
        per_group_n[str(level)] = int(idx.sum())
        per_group_ratio[str(level)] = float(m / ref_mean)

    max_disparity = max(per_group_ratio.values())

    return DescriptiveResult(
        name="environmental_justice_index",
        value=max_disparity,
        extra={
            "mean_by_group": per_group_mean,
            "n_by_group": per_group_n,
            "disparity": per_group_ratio,
            "reference": reference or "overall population",
            "reference_mean": ref_mean,
            "max_disparity": max_disparity,
        },
    )


envjust = environmental_justice_index


def cheatsheet() -> str:
    return "envjust(df, exposure=, group=) -> exposure disparity by demographic."
