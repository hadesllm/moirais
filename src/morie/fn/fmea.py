# morie.fn -- function file (hadesllm/morie)
"""FMEA Risk Priority Number scoring. 'I have a bad feeling about this.' -- Everyone"""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def fmea_rpn(
    severity: np.ndarray,
    occurrence: np.ndarray,
    detection: np.ndarray,
) -> DescriptiveResult:
    r"""Compute Failure Mode and Effects Analysis Risk Priority Numbers.

    .. math::

        RPN_i = S_i \times O_i \times D_i

    Each factor is scored 1-10, giving RPN range [1, 1000].

    Parameters
    ----------
    severity : ndarray, shape (n_modes,)
        Severity rating (1=negligible, 10=catastrophic).
    occurrence : ndarray, shape (n_modes,)
        Occurrence likelihood (1=rare, 10=almost certain).
    detection : ndarray, shape (n_modes,)
        Detection difficulty (1=almost certain detect, 10=undetectable).

    Returns
    -------
    DescriptiveResult
        name='FMEA RPN', value=max RPN,
        extra has 'rpn' (ndarray), 'mean_rpn', 'max_rpn',
        'high_risk_mask' (RPN >= threshold), 'severity',
        'occurrence', 'detection'.

    References
    ----------
    Stamatis, D.H. (2003). *Failure Mode and Effect Analysis: FMEA
    from Theory to Execution* (2nd ed.). ASQ Quality Press.

    SAE International (2019). J1739: Potential Failure Mode and
    Effects Analysis in Design and Manufacturing.
    """
    s = np.asarray(severity, dtype=np.float64).ravel()
    o = np.asarray(occurrence, dtype=np.float64).ravel()
    d = np.asarray(detection, dtype=np.float64).ravel()

    if not (len(s) == len(o) == len(d)):
        raise ValueError("severity, occurrence, detection must have same length.")

    for name, arr in [("severity", s), ("occurrence", o), ("detection", d)]:
        if np.any(arr < 1) or np.any(arr > 10):
            raise ValueError(f"{name} values must be in [1, 10].")

    rpn = s * o * d

    threshold = 200.0
    high_risk = rpn >= threshold

    return DescriptiveResult(
        name="FMEA RPN",
        value=float(np.max(rpn)),
        extra={
            "rpn": rpn,
            "mean_rpn": float(np.mean(rpn)),
            "max_rpn": float(np.max(rpn)),
            "high_risk_mask": high_risk,
            "high_risk_count": int(np.sum(high_risk)),
            "threshold": threshold,
            "severity": s,
            "occurrence": o,
            "detection": d,
            "n_modes": len(s),
        },
    )


fmea = fmea_rpn


def cheatsheet() -> str:
    return "fmea_rpn({}) -> FMEA Risk Priority Number scoring. 'I have a bad feeling abo"
