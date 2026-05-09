"""Content validity ratio (Lawshe CVR)."""

from __future__ import annotations

import numpy as np

from moirais.fn._containers import ESRes


def content_validity_ratio(
    n_essential: int | np.ndarray,
    n_panelists: int,
) -> ESRes:
    """Lawshe's Content Validity Ratio (CVR).

    CVR = (n_e - N/2) / (N/2) where n_e = panelists rating "essential".
    Also computes CVI (Content Validity Index) if n_essential is an array.

    Parameters
    ----------
    n_essential : int or ndarray
        Number of panelists rating item as essential (scalar for one item,
        array for multiple items).
    n_panelists : int
        Total number of panelists.

    Returns
    -------
    ESRes
        measure="CVR".

    References
    ----------
    Lawshe, C. H. (1975). A quantitative approach to content validity.
    Personnel Psychology, 28(4), 563-575.
    """
    ne = np.atleast_1d(np.asarray(n_essential, dtype=np.float64))
    N = float(n_panelists)

    cvr_vals = (ne - N / 2) / (N / 2)
    mean_cvr = float(np.mean(cvr_vals))

    cvi = float(np.mean(ne / N))

    return ESRes(
        measure="CVR",
        estimate=mean_cvr,
        n=n_panelists,
        extra={
            "CVI": cvi,
            "n_items": len(ne),
            "item_cvr": cvr_vals.tolist(),
        },
    )


cvr = content_validity_ratio


def cheatsheet() -> str:
    return "content_validity_ratio({}) -> Content validity ratio (Lawshe CVR)."
