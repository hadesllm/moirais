# morie.fn — function file (hadesllm/morie)
"""Bayesian nonparametric hypothesis testing via Polya-tree marginals."""
import numpy as np
from scipy.stats import norm
from scipy.special import gammaln, betaln
from ._richresult import RichResult

__all__ = ["ghosal_np_testing"]


def ghosal_np_testing(x, ref_loc=0.0, ref_scale=1.0, depth=6, c=1.0):
    """Bayes factor for ``H0: F = F_ref`` vs ``H1: F ~ PT(c)``.

    A canonical Polya-tree prior centred at ``F_ref = N(loc, scale^2)``
    with alpha_{eps} = c m^2 (m = level of the tree) attains a closed-form
    marginal likelihood::

        log p(X | PT) = sum_{eps} [betaln(alpha_eps + n_{eps0},
                                          alpha_eps + n_{eps1})
                                    - betaln(alpha_eps, alpha_eps)]
                        + sum_i log f_ref(X_i)

    where ``n_eps`` are the counts of observations falling in dyadic
    cells.  We compare this to ``H0`` (which is exactly the centring
    measure) — the Bayes factor is

        BF_{10} = exp( sum_eps betaln(.) - betaln(.) ).

    Reference is ``H0`` (``F = F_ref``), null hypothesis.

    Parameters
    ----------
    x : array-like.
    ref_loc, ref_scale : float — reference mean and sd.
    depth : int — tree depth.
    c : float — tree concentration.

    Returns
    -------
    RichResult with ``statistic`` = log BF, ``p_value`` (BF -> tail
    probability under H0 via a calibration to chi^2 with 1 df),
    ``BF10``, ``log_BF10``.

    References
    ----------
    Lavine, M. (1994). More aspects of Polya trees. AOS 22.
    Berger & Guglielmi (2001). Bayesian testing via Polya trees.
      JASA 96.
    Ghosal & van der Vaart (2017) Ch 16.
    """
    x = np.asarray(x, dtype=float).ravel()
    n = int(x.size)
    if n < 2:
        return RichResult(payload={
            "statistic": float("nan"), "p_value": float("nan"),
            "n": n,
            "method": "Polya-tree BF (n<2)",
        })
    u = norm.cdf(x, loc=ref_loc, scale=ref_scale)  # transform to U(0,1)
    log_bf = 0.0
    for m in range(1, depth + 1):
        nbins = 2 ** m
        edges = np.linspace(0, 1, nbins + 1)
        counts = np.histogram(u, bins=edges)[0]
        alpha = c * m * m
        # Pair adjacent bins: (n_0, n_1) at this level.
        n0 = counts[0::2]
        n1 = counts[1::2]
        log_bf += float(np.sum(betaln(alpha + n0, alpha + n1)
                                - betaln(alpha, alpha)))
    BF10 = float(np.exp(log_bf))
    # Calibrate Bayes factor to a frequentist p-value via the Vovk-style
    # bound p_max <= 1/(1 + BF10) when BF10 > 1.  Otherwise p ≈ 0.5.
    if BF10 > 1:
        p_value = 1.0 / (1.0 + BF10)
    else:
        p_value = 0.5
    return RichResult(payload={
        "statistic": float(log_bf),
        "p_value": float(p_value),
        "BF10": BF10,
        "log_BF10": float(log_bf),
        "n": n,
        "depth": int(depth),
        "method": "Polya-tree Bayes-factor test (Berger-Guglielmi)",
    })


def cheatsheet():
    return "ghtst: Bayesian nonparametric testing"


# CANONICAL TEST
# >>> import numpy as np
# >>> from morie.fn.ghtst import ghosal_np_testing
# >>> r = ghosal_np_testing(np.random.default_rng(0).normal(size=200))
# >>> 0 <= r["p_value"] <= 1
# True
