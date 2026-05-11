# morie.fn — function file from book-equation translation pipeline (hadesllm/morie)
"""Mathematics is the music of reason. — James Joseph Sylvester"""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def full_adder(
    a: int | np.ndarray,
    b: int | np.ndarray,
    carry_in: int = 0,
) -> DescriptiveResult:
    """
    Simulate a full adder (or ripple-carry adder for bit arrays).

    For scalars: single-bit full adder.
    For arrays: ripple-carry addition from LSB to MSB.

    .. math::

        S = A \\oplus B \\oplus C_{in}

        C_{out} = (A \\cdot B) + (C_{in} \\cdot (A \\oplus B))

    :param a: Binary value(s) (0 or 1). Scalar or array.
    :param b: Binary value(s) (0 or 1). Same shape as a.
    :param carry_in: Initial carry bit. Default 0.
    :return: DescriptiveResult with sum bits and carry out.
    :raises ValueError: If inputs are not binary.

    References
    ----------
    Ercegovac, M. D., & Lang, T. (2004). *Digital Arithmetic*.
    Morgan Kaufmann.
    """
    a_arr = np.asarray(a, dtype=int).ravel()
    b_arr = np.asarray(b, dtype=int).ravel()

    if len(a_arr) != len(b_arr):
        raise ValueError("a and b must have the same length.")
    if not np.all(np.isin(a_arr, [0, 1])):
        raise ValueError("a must be binary (0 or 1).")
    if not np.all(np.isin(b_arr, [0, 1])):
        raise ValueError("b must be binary (0 or 1).")
    if carry_in not in (0, 1):
        raise ValueError(f"carry_in must be 0 or 1, got {carry_in}.")

    n = len(a_arr)
    s = np.zeros(n, dtype=int)
    c = carry_in

    for i in range(n):
        total = int(a_arr[i]) + int(b_arr[i]) + c
        s[i] = total % 2
        c = total // 2

    decimal_a = sum(int(a_arr[i]) * (2**i) for i in range(n))
    decimal_b = sum(int(b_arr[i]) * (2**i) for i in range(n))
    decimal_s = sum(int(s[i]) * (2**i) for i in range(n)) + c * (2**n)

    return DescriptiveResult(
        name="Full Adder",
        value=int(c),
        extra={
            "sum_bits": s,
            "carry_out": int(c),
            "decimal_a": decimal_a,
            "decimal_b": decimal_b,
            "decimal_sum": decimal_s,
            "n_bits": n,
        },
    )


short = full_adder


def cheatsheet() -> str:
    return "full_adder({}) -> Full adder binary circuit. 'I find your lack of faith distur"
