# morie.fn -- function file (hadesllm/morie)
"""IRT scale linking (mean/sigma method)."""

from __future__ import annotations

import numpy as np

from morie.fn._containers import DescriptiveResult


def irt_linking(
    params_form_a: dict,
    params_form_b: dict,
    *,
    method: str = "mean_sigma",
) -> DescriptiveResult:
    """Link two IRT calibrations via mean/sigma or mean/mean method.

    Computes linear transformation constants A and B such that
    theta_b = A * theta_a + B, applied to item parameters.

    Parameters
    ----------
    params_form_a : dict
        {item: {"a": float, "b": float}} from form A (common items).
    params_form_b : dict
        Same structure from form B.
    method : str
        "mean_sigma" or "mean_mean" (default "mean_sigma").

    Returns
    -------
    DescriptiveResult
        extra has A, B, and linked_params.

    References
    ----------
    Marco, G. L. (1977). Item characteristic curve solutions to three
    intractable testing problems. Journal of Educational Measurement.
    """
    common = sorted(set(params_form_a) & set(params_form_b))
    if len(common) < 2:
        raise ValueError("Need at least 2 common items for linking.")

    b_a = np.array([params_form_a[it]["b"] for it in common])
    b_b = np.array([params_form_b[it]["b"] for it in common])
    a_a = np.array([params_form_a[it].get("a", 1.0) for it in common])
    a_b = np.array([params_form_b[it].get("a", 1.0) for it in common])

    if method == "mean_sigma":
        A = np.std(b_b, ddof=1) / max(np.std(b_a, ddof=1), 1e-10)
        B = np.mean(b_b) - A * np.mean(b_a)
    else:
        A = np.mean(a_a) / max(np.mean(a_b), 1e-10)
        B = np.mean(b_b) - A * np.mean(b_a)

    linked = {}
    for item, p in params_form_a.items():
        linked[item] = {
            "a": p.get("a", 1.0) / A,
            "b": A * p["b"] + B,
        }

    return DescriptiveResult(
        name="IRT linking",
        value={"A": float(A), "B": float(B)},
        extra={
            "A": float(A),
            "B": float(B),
            "linked_params": linked,
            "method": method,
            "n_common": len(common),
        },
    )


linking = irt_linking


def cheatsheet() -> str:
    return "irt_linking({}) -> IRT scale linking (mean/sigma method)."
