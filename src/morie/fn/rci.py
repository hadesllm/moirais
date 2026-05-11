# morie.fn — function file (hadesllm/morie)
"""Jacobson-Truax Reliable Change Index. 'Hope is like the sun.' -- Poe Dameron"""

from __future__ import annotations

import numpy as np
from scipy import stats as _st

from ._containers import DescriptiveResult


def reliable_change(
    pre: float | np.ndarray,
    post: float | np.ndarray,
    se_meas: float,
    alpha: float = 0.05,
) -> DescriptiveResult:
    """
    Jacobson-Truax Reliable Change Index (RCI).

    .. math::

        RCI = \\frac{X_2 - X_1}{S_{\\text{diff}}}
        \\quad\\text{where}\\quad
        S_{\\text{diff}} = \\sqrt{2 \\, SEM^2}

    A change is reliable if :math:`|RCI| > z_{\\alpha/2}`.

    :param pre: Pre-test score(s).
    :param post: Post-test score(s).
    :param se_meas: Standard error of measurement (SEM).
    :param alpha: Significance level. Default 0.05.
    :return: DescriptiveResult with RCI value(s).
    :raises ValueError: If se_meas <= 0.

    References
    ----------
    Jacobson, N. S., & Truax, P. (1991). Clinical significance: a statistical
    approach to defining meaningful change in psychotherapy research.
    Journal of Consulting and Clinical Psychology, 59(1), 12--19.
    doi:10.1037/0022-006X.59.1.12
    """
    if se_meas <= 0.0:
        raise ValueError(f"se_meas must be > 0, got {se_meas}.")

    pre = np.asarray(pre, dtype=float)
    post = np.asarray(post, dtype=float)
    s_diff = np.sqrt(2.0 * se_meas**2)
    rci_vals = (post - pre) / s_diff

    z_crit = _st.norm.ppf(1.0 - alpha / 2.0)
    reliable = np.abs(rci_vals) > z_crit

    scalar = rci_vals.ndim == 0
    rci_float = float(rci_vals) if scalar else rci_vals

    return DescriptiveResult(
        name="Reliable Change Index",
        value=rci_float,
        extra={
            "rci": rci_float,
            "s_diff": float(np.round(s_diff, 4)),
            "z_critical": float(np.round(z_crit, 4)),
            "reliable": bool(reliable) if scalar else reliable.tolist(),
            "alpha": alpha,
            "se_meas": se_meas,
        },
    )


rci = reliable_change


def cheatsheet() -> str:
    return "reliable_change({}) -> Jacobson-Truax Reliable Change Index. 'Hope is like the sun."
