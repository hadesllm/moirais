# morie.fn — function file (hadesllm/morie)
"""Agenda-setter model (Romer-Rosenthal; Armstrong Ch 10)."""

import numpy as np

from ._containers import SpatialResult

__all__ = ["agset", "agenda_setter_power"]


def agenda_setter_power(options, setter_ideal, reversion):
    """Romer-Rosenthal (1978) agenda-setter outcome and power.

    The agenda-setter offers a single take-it-or-leave-it proposal from
    `options`; the legislature accepts iff the proposal is closer (in
    1-D Euclidean distance) to the median voter than the reversion
    point. For the simple "setter monopoly" case used in Armstrong Ch
    10, the setter picks the feasible option closest to their own ideal
    point that the legislature will still accept (assumed median voter
    = midpoint between setter ideal and reversion, default behaviour).

    Parameters
    ----------
    options : array-like
        Discrete set of feasible policy proposals.
    setter_ideal : float
        Agenda setter's ideal point.
    reversion : float
        Status-quo / reversion point.

    Returns
    -------
    SpatialResult
        statistic: |chosen - reversion| (power: distance moved from sq)
        extra: chosen, setter_ideal, reversion, win_set_size
    """
    options = np.asarray(options, dtype=float).ravel()
    setter_ideal = float(setter_ideal)
    reversion = float(reversion)
    if options.size == 0:
        return SpatialResult(name="Agenda setter model",
                             statistic=0.0,
                             extra={"chosen": np.nan,
                                    "setter_ideal": setter_ideal,
                                    "reversion": reversion,
                                    "win_set_size": 0})
    # Win-set: options at least as close to reversion as reversion itself
    # is to itself — i.e. options strictly preferred by the median voter
    # over the reversion (single-peaked, voter at midpoint for monopoly
    # setter). For the canonical Romer-Rosenthal monopoly the win-set is
    # the open interval between reversion and 2*median - reversion.
    median = (setter_ideal + reversion) / 2.0
    win_lo, win_hi = sorted([reversion, 2 * median - reversion])
    in_win = (options >= win_lo) & (options <= win_hi)
    if not np.any(in_win):
        chosen = reversion
    else:
        feasible = options[in_win]
        chosen = float(feasible[np.argmin(np.abs(feasible - setter_ideal))])
    power = abs(chosen - reversion)
    return SpatialResult(
        name="Agenda setter model",
        statistic=float(power),
        extra={
            "chosen": chosen,
            "setter_ideal": setter_ideal,
            "reversion": reversion,
            "win_set_size": int(in_win.sum()),
            "win_set_bounds": (float(win_lo), float(win_hi)),
        },
    )


def agset(options, setter_ideal, reversion):
    """Alias of :func:`agenda_setter_power`."""
    return agenda_setter_power(options, setter_ideal, reversion)


def cheatsheet() -> str:
    return ("agset(options, setter_ideal, reversion) -> "
            "Romer-Rosenthal agenda-setter outcome and power.")


# CANONICAL TEST
# >>> r = agset([0.0, 1.0, 2.0], setter_ideal=1.0, reversion=0.0)
# >>> assert r.extra["chosen"] == 1.0
