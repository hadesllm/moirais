# morie.fn — function file (hadesllm/morie)
"""Recidivism by program participation."""

from __future__ import annotations

import numpy as np

from morie.fn._containers import ESRes


def program_recidivism(
    recid_participants: np.ndarray | list[int],
    recid_controls: np.ndarray | list[int],
) -> ESRes:
    """Compare recidivism rates between participants and controls.

    Parameters
    ----------
    recid_participants : array-like
        Binary recidivism indicators for participants (0/1).
    recid_controls : array-like
        Binary recidivism indicators for controls (0/1).

    Returns
    -------
    ESRes
        Risk difference (negative = program reduces recidivism).
    """
    p = np.asarray(recid_participants, dtype=float)
    c = np.asarray(recid_controls, dtype=float)
    if len(p) < 2 or len(c) < 2:
        raise ValueError("Need at least 2 observations per group")
    rate_p = float(np.mean(p))
    rate_c = float(np.mean(c))
    rd = rate_p - rate_c
    se = float(np.sqrt(rate_p * (1 - rate_p) / len(p) + rate_c * (1 - rate_c) / len(c)))
    return ESRes(
        measure="recidivism_risk_difference",
        estimate=rd,
        ci_lower=rd - 1.96 * se,
        ci_upper=rd + 1.96 * se,
        se=se,
        n=len(p) + len(c),
        extra={"rate_participants": rate_p, "rate_controls": rate_c, "n_participants": len(p), "n_controls": len(c)},
    )


prgrc = program_recidivism


def cheatsheet() -> str:
    return "program_recidivism({}) -> Recidivism by program participation."
