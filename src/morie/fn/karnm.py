# morie.fn -- function file (hadesllm/morie)
"""Karnaugh map grouping. 'No man ever steps in the same river twice. -- Heraclitus' -- Chirrut Imwe"""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def karnaugh_map(
    truth_table: np.ndarray,
    n_vars: int,
) -> DescriptiveResult:
    """
    Generate Karnaugh map layout and identify minterms.

    Arranges a truth table into K-map format using Gray code ordering
    for up to 4 variables.

    :param truth_table: 1D array of output values (length 2^n_vars).
    :param n_vars: Number of input variables (2, 3, or 4).
    :return: DescriptiveResult with K-map grid and minterm positions.
    :raises ValueError: If n_vars not in {2,3,4} or wrong table size.

    References
    ----------
    Karnaugh, M. (1953). The map method for synthesis of combinational
    logic circuits. *Transactions AIEE*, 72(9), 593-599.
    """
    if n_vars not in (2, 3, 4):
        raise ValueError(f"n_vars must be 2, 3, or 4, got {n_vars}.")

    tt = np.asarray(truth_table, dtype=int).ravel()
    expected = 2**n_vars
    if len(tt) != expected:
        raise ValueError(f"Expected {expected} entries, got {len(tt)}.")

    gray = [0, 1, 3, 2]

    if n_vars == 2:
        rows, cols = [0, 1], [0, 1]
    elif n_vars == 3:
        rows, cols = [0, 1], gray
    else:
        rows, cols = gray, gray

    kmap = np.zeros((len(rows), len(cols)), dtype=int)
    for ri, r in enumerate(rows):
        for ci, c in enumerate(cols):
            if n_vars == 2:
                idx = r * 2 + c
            elif n_vars == 3:
                idx = r * 4 + c
            else:
                idx = r * 4 + c
            kmap[ri, ci] = tt[idx]

    minterms = [i for i in range(len(tt)) if tt[i] == 1]

    return DescriptiveResult(
        name="Karnaugh Map",
        value=len(minterms),
        extra={
            "kmap": kmap,
            "minterms": minterms,
            "n_vars": n_vars,
            "row_gray": rows,
            "col_gray": cols,
        },
    )


short = karnaugh_map


def cheatsheet() -> str:
    return "He who would learn to fly one day must first learn to stand and walk. -- Friedrich Nietzsche"
