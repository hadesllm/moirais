"""Content validity index (CVI) from expert ratings."""

from __future__ import annotations

import numpy as np
import pandas as pd


def validity_face_content(
    items: list[str] | np.ndarray,
    ratings: np.ndarray | pd.DataFrame,
    *,
    threshold: float = 0.78,
) -> dict:
    """Content Validity Index from expert relevance ratings.

    Each expert rates each item on a 4-point scale (1 = not relevant,
    2 = somewhat, 3 = quite, 4 = highly relevant).  Items rated 3 or 4
    are counted as "relevant".  I-CVI = proportion of experts rating
    relevant; S-CVI/Ave = mean of I-CVIs.

    Parameters
    ----------
    items : list or array
        Item labels.
    ratings : ndarray or DataFrame
        Shape (n_experts, n_items).  Values 1--4.
    threshold : float
        I-CVI threshold for acceptable content validity (default 0.78;
        Lynn, 1986).

    Returns
    -------
    dict
        Keys: ``i_cvi`` (per-item CVI dict), ``s_cvi_ave`` (scale-level),
        ``s_cvi_ua`` (universal agreement), ``n_experts``, ``n_items``,
        ``acceptable_items`` (list).

    References
    ----------
    Lynn, M. R. (1986). Determination and quantification of content
    validity. *Nursing Research*, 35(6), 382--386.
    Polit, D. F., & Beck, C. T. (2006). The content validity index:
    are you sure you know what's being reported? *Research in Nursing
    & Health*, 29(5), 489--497.
    """
    R = np.asarray(ratings, dtype=np.float64)
    if R.ndim == 1:
        R = R.reshape(1, -1)
    n_experts, n_items = R.shape

    item_labels = list(items) if not isinstance(items, list) else items
    if len(item_labels) != n_items:
        raise ValueError(f"items length ({len(item_labels)}) != ratings columns ({n_items})")

    # Relevant = rated 3 or 4
    relevant = (R >= 3).astype(float)
    i_cvi = {item_labels[j]: float(relevant[:, j].mean()) for j in range(n_items)}
    s_cvi_ave = float(np.mean(list(i_cvi.values())))
    # Universal agreement: proportion of items with I-CVI = 1.0
    s_cvi_ua = float(np.mean([v == 1.0 for v in i_cvi.values()]))
    acceptable = [item for item, v in i_cvi.items() if v >= threshold]

    return {
        "i_cvi": i_cvi,
        "s_cvi_ave": s_cvi_ave,
        "s_cvi_ua": s_cvi_ua,
        "n_experts": n_experts,
        "n_items": n_items,
        "threshold": threshold,
        "acceptable_items": acceptable,
    }


def cheatsheet() -> str:
    return "validity_face_content({}) -> Content validity index (CVI) from expert ratings."
