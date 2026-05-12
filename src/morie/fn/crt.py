# morie.fn -- function file (hadesllm/morie)
"""Chinese Remainder Theorem solver."""

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "The belonging you seek is not behind you. It is ahead. -- Maz Kanata"


def chinese_remainder(remainders, moduli, **kwargs) -> DescriptiveResult:
    r"""
    Solve a system of simultaneous congruences via the Chinese Remainder Theorem.

    Given :math:`x \\equiv r_i \\pmod{m_i}` for pairwise coprime moduli,
    finds the unique solution :math:`x \\pmod{M}` where
    :math:`M = \\prod m_i`.

    .. math::

        x = \\sum_{i=1}^{k} r_i \\cdot M_i \\cdot y_i \\pmod{M}

    where :math:`M_i = M / m_i` and :math:`y_i = M_i^{-1} \\pmod{m_i}`.

    :param remainders: Array-like of remainders r_i.
    :param moduli: Array-like of pairwise coprime moduli m_i.
    :return: DescriptiveResult with solution x as value.
    :raises ValueError: If moduli are not pairwise coprime or lengths mismatch.

    References
    ----------
    Gauss, C. F. (1801). *Disquisitiones Arithmeticae*, Art. 36.
    """
    remainders = [int(r) for r in np.atleast_1d(remainders)]
    moduli = [int(m) for m in np.atleast_1d(moduli)]

    if len(remainders) != len(moduli):
        raise ValueError("remainders and moduli must have the same length.")
    if any(m <= 0 for m in moduli):
        raise ValueError("All moduli must be positive.")

    from math import gcd

    for i in range(len(moduli)):
        for j in range(i + 1, len(moduli)):
            if gcd(moduli[i], moduli[j]) != 1:
                raise ValueError(f"Moduli {moduli[i]} and {moduli[j]} are not coprime.")

    M = 1
    for m in moduli:
        M *= m

    x = 0
    for r_i, m_i in zip(remainders, moduli):
        M_i = M // m_i
        y_i = pow(M_i, -1, m_i)
        x += r_i * M_i * y_i

    x = x % M

    return DescriptiveResult(
        name="chinese_remainder",
        value=float(x),
        extra={
            "solution": x,
            "product_M": M,
            "remainders": remainders,
            "moduli": moduli,
        },
    )


crt = chinese_remainder


def cheatsheet() -> str:
    return "chinese_remainder({}) -> Chinese Remainder Theorem solver."
