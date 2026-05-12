# morie.fn — function file (hadesllm/morie)
"""Jensen-Shannon divergence."""

import numpy as np

from ._containers import ESRes


def js_divergence(p: np.ndarray, q: np.ndarray) -> ESRes:
    r"""
    Compute Jensen-Shannon divergence.

    .. math::

        JSD(P \\| Q) = \\frac{1}{2} D_{KL}(P \\| M) + \\frac{1}{2} D_{KL}(Q \\| M)

    where M = (P + Q) / 2. Always symmetric and bounded in [0, ln2].

    :param p: (k,) probability distribution.
    :param q: (k,) probability distribution.
    :return: ESRes with JSD (in nats).

    References
    ----------
    Lin J (1991). Divergence measures based on the Shannon entropy.
    IEEE Transactions on Information Theory, 37(1), 145-151.
    """
    p = np.asarray(p, dtype=np.float64).ravel()
    q = np.asarray(q, dtype=np.float64).ravel()
    if len(p) != len(q):
        raise ValueError("p and q must have same length.")
    p = p / p.sum()
    q = q / q.sum()
    m = (p + q) / 2

    def _kl(a, b):
        mask = a > 0
        return float(np.sum(a[mask] * np.log(a[mask] / b[mask])))

    jsd = 0.5 * _kl(p, m) + 0.5 * _kl(q, m)
    return ESRes(
        measure="js_divergence", estimate=float(jsd), extra={"sqrt_jsd": float(np.sqrt(jsd)), "n_categories": len(p)}
    )


jsdiv = js_divergence


def cheatsheet() -> str:
    return "js_divergence({}) -> Jensen-Shannon divergence."
