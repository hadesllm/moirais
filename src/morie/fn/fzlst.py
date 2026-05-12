# morie.fn — function file (hadesllm/morie)
"""L-statistic for kernel functionals (Fauzi Ch 5).

L_n = sum_i c_{n,i} X_{(i)} with c_{n,i} = integral_{(i-1)/n}^{i/n} J(u) du
for a user-supplied score function J:(0,1)->R.  Default J=1 yields the
sample mean.

Asymptotic variance via the Stieltjes-integral form:

    Var(L_n) ≈ (1/n) * integral integral J(u) J(v) [min(u,v)-uv]
                          / ( f(F^{-1}(u)) f(F^{-1}(v)) )  du dv,

estimated via a discretised double-Riemann sum on the Silverman-KDE
density-at-quantile.
"""
import numpy as np
from ._richresult import RichResult

__all__ = ["fauzi_l_statistic"]


def fauzi_l_statistic(x, score=None, n_quad=200):
    """L-statistic with score function J.

    Parameters
    ----------
    x : array-like
    score : callable J(u), u in (0,1).  Default = constant 1.
    n_quad : int  quadrature grid size for variance.
    """
    x = np.asarray(x, dtype=float).ravel()
    n = len(x)
    if n < 2:
        return RichResult(payload={"estimate": np.nan, "n": n,
                                    "method": "fzlst — too few obs"})
    if score is None:
        score = lambda u: np.ones_like(u)

    x_sorted = np.sort(x)
    u_grid = np.linspace(0.0, 1.0, n + 1)
    fine = np.linspace(0.0, 1.0, n * 16 + 1)
    J_fine = np.asarray(score(fine), dtype=float)
    cells = np.searchsorted(fine, u_grid)
    weights = np.empty(n)
    for i in range(n):
        a, b = cells[i], cells[i + 1]
        if b <= a:
            b = a + 1
        seg_u = fine[a:b + 1]
        seg_J = J_fine[a:b + 1]
        weights[i] = float(np.trapezoid(seg_J, seg_u))
    L = float(np.sum(weights * x_sorted))

    # Asymptotic variance via double-Riemann
    uu = (np.arange(n_quad) + 0.5) / n_quad
    Q = np.quantile(x, uu)
    sigma = float(np.std(x, ddof=1))
    if sigma <= 0:
        sigma = 1.0
    h = 1.06 * sigma * n ** (-1.0 / 5.0)
    from scipy import stats as _sps
    f_Q = np.array([np.mean(_sps.norm.pdf((q - x) / h) / h) for q in Q])
    J_at_u = np.asarray(score(uu), dtype=float)

    U, V = np.meshgrid(uu, uu, indexing="ij")
    K = (np.minimum(U, V) - U * V) / (np.outer(f_Q, f_Q) + 1e-12)
    JJ = np.outer(J_at_u, J_at_u)
    var = float(np.mean(JJ * K)) / n
    var = max(var, 0.0)
    se = float(np.sqrt(var))

    return RichResult(payload={
        "estimate": L,
        "se": se,
        "n": n,
        "method": "Fauzi L-statistic with user score function J (Ch 5)",
    })


def cheatsheet():
    return "fzlst: L-statistic L_n = sum c_{n,i} X_{(i)} for score J"


# CANONICAL TEST
# >>> import numpy as np
# >>> rng = np.random.default_rng(0)
# >>> x = rng.standard_normal(500)
# >>> r = fauzi_l_statistic(x)
# >>> abs(r["estimate"] - np.mean(x)) < 1e-6
# True
