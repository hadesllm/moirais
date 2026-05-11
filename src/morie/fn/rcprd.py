# morie.fn — function file (hadesllm/morie)
"""Predicted choice from ideal points. 'Tri-Beam!' -- Tien, Dragon Ball Z"""

from __future__ import annotations

from ._containers import DescriptiveResult


def predicted_choice(ideal_points, yea_pos, nay_pos):
    """Predict binary choice based on proximity of ideal point to yea/nay positions.

    Parameters
    ----------
    ideal_points : array-like
        Ideal point coordinates (n x p).
    yea_pos : array-like
        Position of the 'yea' option (length p).
    nay_pos : array-like
        Position of the 'nay' option (length p).

    Returns
    -------
    DescriptiveResult
        value = binary predictions (1=yea, 0=nay), length n.
    """
    import numpy as np

    X = np.asarray(ideal_points, dtype=float)
    yea = np.asarray(yea_pos, dtype=float)
    nay = np.asarray(nay_pos, dtype=float)
    d_yea = np.sqrt(np.sum((X - yea) ** 2, axis=1))
    d_nay = np.sqrt(np.sum((X - nay) ** 2, axis=1))
    preds = (d_yea < d_nay).astype(int)
    return DescriptiveResult(
        name="predicted_choice",
        value=preds,
        extra={"n_yea": int(preds.sum()), "n_nay": int(len(preds) - preds.sum())},
    )


rcprd = predicted_choice


def cheatsheet() -> str:
    return "predicted_choice({}) -> Predicted choice from ideal points. 'Tri-Beam!' -- Tien, Dra"
