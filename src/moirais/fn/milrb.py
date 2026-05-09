# moirais.fn — function file (hadesllm/moirais)
"""Miller-Rabin probabilistic primality test."""

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "He who has a why to live can bear almost any how. — Friedrich Nietzsche"


def miller_rabin(n: int, k: int = 10, seed: int = 42, **kwargs) -> DescriptiveResult:
    """
    Miller-Rabin probabilistic primality test.

    Decomposes :math:`n - 1 = 2^r \\cdot d` and tests *k* random witnesses.
    Probability of false positive :math:`\\le 4^{-k}`.

    :param n: Integer to test (must be >= 2).
    :param k: Number of witness rounds. Default 10.
    :param seed: Random seed for reproducibility. Default 42.
    :return: DescriptiveResult with is_probable_prime boolean.
    :raises ValueError: If n < 2 or k < 1.

    References
    ----------
    Miller, G. L. (1976). Riemann's hypothesis and tests for primality.
    *Journal of Computer and System Sciences*, 13(3), 300-317.
    Rabin, M. O. (1980). Probabilistic algorithm for testing primality.
    *Journal of Number Theory*, 12(1), 128-138.
    """
    n = int(n)
    k = int(k)
    if n < 2:
        raise ValueError(f"n must be >= 2, got {n}.")
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}.")

    if n < 4:
        return DescriptiveResult(
            name="miller_rabin",
            value=1.0,
            extra={"is_probable_prime": True, "n": n, "witnesses_tested": 0},
        )
    if n % 2 == 0:
        return DescriptiveResult(
            name="miller_rabin",
            value=0.0,
            extra={"is_probable_prime": False, "n": n, "witnesses_tested": 0},
        )

    r, d = 0, n - 1
    while d % 2 == 0:
        r += 1
        d //= 2

    rng = np.random.default_rng(seed)
    witnesses_tested = 0

    for _ in range(k):
        a = int(rng.integers(2, n - 1))
        witnesses_tested += 1
        x = pow(a, d, n)

        if x == 1 or x == n - 1:
            continue

        composite = True
        for _ in range(r - 1):
            x = pow(x, 2, n)
            if x == n - 1:
                composite = False
                break

        if composite:
            return DescriptiveResult(
                name="miller_rabin",
                value=0.0,
                extra={
                    "is_probable_prime": False,
                    "n": n,
                    "witnesses_tested": witnesses_tested,
                    "composite_witness": a,
                    "false_positive_bound": 4.0 ** (-k),
                },
            )

    return DescriptiveResult(
        name="miller_rabin",
        value=1.0,
        extra={
            "is_probable_prime": True,
            "n": n,
            "witnesses_tested": witnesses_tested,
            "false_positive_bound": 4.0 ** (-k),
            "r": r,
            "d": d,
        },
    )


milrb = miller_rabin


def cheatsheet() -> str:
    return "miller_rabin({}) -> Miller-Rabin probabilistic primality test."
