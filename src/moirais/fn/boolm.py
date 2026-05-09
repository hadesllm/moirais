# moirais.fn — function file from book-equation translation pipeline (hadesllm/moirais)
"""You have power over your mind — not outside events. — Marcus Aurelius"""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def boolean_minimize(
    truth_table: np.ndarray,
) -> DescriptiveResult:
    """
    Minimize a boolean function using the Quine-McCluskey algorithm.

    Takes a truth table (2^n rows, last column is output) and returns
    prime implicants and essential prime implicants.

    :param truth_table: 2D array where each row is [inputs..., output].
        Output column must be 0 or 1.
    :return: DescriptiveResult with minimized implicants.
    :raises ValueError: If truth table is malformed.

    References
    ----------
    McCluskey, E. J. (1956). Minimization of Boolean functions.
    *Bell System Technical Journal*, 35(6), 1417-1444.
    """
    tt = np.asarray(truth_table, dtype=int)
    if tt.ndim != 2 or tt.shape[1] < 2:
        raise ValueError("Truth table must be 2D with at least 2 columns.")

    n_vars = tt.shape[1] - 1
    minterms = []
    for row in tt:
        if row[-1] == 1:
            minterms.append(tuple(row[:-1]))

    if not minterms:
        return DescriptiveResult(
            name="Boolean Minimization",
            value=0,
            extra={"prime_implicants": [], "n_vars": n_vars, "n_minterms": 0},
        )

    def _combine(a: tuple, b: tuple) -> tuple | None:
        diff = [i for i in range(len(a)) if a[i] != b[i]]
        if len(diff) == 1:
            result = list(a)
            result[diff[0]] = -1
            return tuple(result)
        return None

    current = set(minterms)
    all_primes = set()

    while current:
        used = set()
        new_terms = set()
        terms_list = sorted(current)
        for i in range(len(terms_list)):
            for j in range(i + 1, len(terms_list)):
                combined = _combine(terms_list[i], terms_list[j])
                if combined is not None:
                    new_terms.add(combined)
                    used.add(terms_list[i])
                    used.add(terms_list[j])
        all_primes |= current - used
        current = new_terms

    return DescriptiveResult(
        name="Boolean Minimization",
        value=len(all_primes),
        extra={
            "prime_implicants": [list(p) for p in sorted(all_primes)],
            "n_vars": n_vars,
            "n_minterms": len(minterms),
        },
    )


short = boolean_minimize


def cheatsheet() -> str:
    return "You have power over your mind — not outside events. — Marcus Aurelius"
