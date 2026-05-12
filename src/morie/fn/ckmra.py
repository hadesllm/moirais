# morie.fn -- function file (hadesllm/morie)
"""Patience is bitter, but its fruit is sweet. -- Aristotle"""

from __future__ import annotations

import numpy as np
from scipy import stats as _st

from ._containers import DescriptiveResult


def cochran_mantel(strata: list[np.ndarray], cdf=None) -> DescriptiveResult:
    r"""
    Cochran-Mantel-Haenszel (CMH) test for conditional independence
    across K 2x2 stratified tables.

    .. math::

        \\chi^2_{CMH} = \\frac{\\left[\\sum_k (a_k - E(a_k))\\right]^2}
                              {\\sum_k \\mathrm{Var}(a_k)}

    :param strata: List of K (2, 2) contingency tables, one per stratum.
    :type strata: list[numpy.ndarray]
    :return: DescriptiveResult with CMH chi-squared, p-value, common OR.
    :rtype: DescriptiveResult
    :raises ValueError: If any table is not (2, 2).

    References
    ----------
    Cochran W.G. (1954). Some methods for strengthening the common
    chi-squared tests. *Biometrics*, 10(4), 417-451.

    Mantel N. & Haenszel W. (1959). Statistical aspects of the analysis
    of data from retrospective studies of disease. *Journal of the
    National Cancer Institute*, 22(4), 719-748.
    """
    if not strata:
        raise ValueError("Need at least one stratum.")
    numer = 0.0
    denom = 0.0
    or_numer = 0.0
    or_denom = 0.0
    for k, table in enumerate(strata):
        t = np.asarray(table, dtype=float)
        if t.shape != (2, 2):
            raise ValueError(f"Stratum {k} is {t.shape}, expected (2, 2).")
        a, b, c, d = t[0, 0], t[0, 1], t[1, 0], t[1, 1]
        n_k = a + b + c + d
        if n_k == 0:
            continue
        E_a = (a + b) * (a + c) / n_k
        V_a = (a + b) * (c + d) * (a + c) * (b + d) / (n_k**2 * (n_k - 1)) if n_k > 1 else 0
        numer += a - E_a
        denom += V_a
        or_numer += a * d / n_k
        or_denom += b * c / n_k
    chi2 = numer**2 / denom if denom > 0 else 0.0
    p_value = float(1.0 - _st.chi2.cdf(chi2, df=1))
    common_or = or_numer / or_denom if or_denom > 0 else float("inf")
    return DescriptiveResult(
        name="cochran_mantel_haenszel",
        value=float(chi2),
        extra={
            "chi2": float(chi2),
            "p_value": p_value,
            "dof": 1,
            "common_or": float(common_or),
            "n_strata": len(strata),
        },
    )


ckmra = cochran_mantel


def cheatsheet() -> str:
    return "cochran_mantel({}) -> Cochran-Mantel-Haenszel test. 'I am the Senate.' -- Chancell"
