# morie.fn -- function file (hadesllm/morie)
"""Person-years at risk."""

import numpy as np

from ._containers import ESRes


def person_years_at_risk(
    entry_dates: np.ndarray,
    exit_dates: np.ndarray,
) -> ESRes:
    r"""Compute total person-years at risk.

    .. math::

        PY = \\sum_i (\\text{exit}_i - \\text{entry}_i) \\;/\\; 365.25

    Parameters
    ----------
    entry_dates : array-like
        Study entry dates (datetime64, str, or numeric days).
    exit_dates : array-like
        Study exit dates.

    Returns
    -------
    ESRes
        estimate = total person-years. extra contains per-person years.

    References
    ----------
    Rothman, K. J., Greenland, S., & Lash, T. L. (2008). Modern
    Epidemiology. 3rd ed. Lippincott Williams & Wilkins.
    """
    entry = np.asarray(entry_dates)
    exit_ = np.asarray(exit_dates)

    if len(entry) != len(exit_):
        raise ValueError("entry_dates and exit_dates must have equal length")
    if len(entry) == 0:
        raise ValueError("Arrays must not be empty")

    if np.issubdtype(entry.dtype, np.datetime64) or entry.dtype == object:
        entry = np.array(entry, dtype="datetime64[D]")
        exit_ = np.array(exit_, dtype="datetime64[D]")
        durations = (exit_ - entry).astype(float)
    else:
        durations = exit_.astype(float) - entry.astype(float)

    py_per_person = durations / 365.25
    total_py = float(np.sum(py_per_person))

    return ESRes(
        measure="Person-years",
        estimate=total_py,
        n=len(entry),
        extra={
            "mean_py": float(np.mean(py_per_person)),
            "median_py": float(np.median(py_per_person)),
        },
    )


pyr = person_years_at_risk


def cheatsheet() -> str:
    return "person_years_at_risk({}) -> Person-years at risk."
