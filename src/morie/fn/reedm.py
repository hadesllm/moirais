# morie.fn — function file (hadesllm/morie)
"""Reed-Muller code generator matrix."""

__all__ = ["reedm"]

import numpy as np


def reedm(r: int, m: int) -> dict:
    """
    Construct the generator matrix for Reed-Muller code RM(r, m).

    RM(r, m) has length n = 2^m, dimension k = sum_{i=0}^{r} C(m, i),
    and minimum distance d = 2^{m-r}.

    Parameters
    ----------
    r : int
        Order of the Reed-Muller code, 0 <= r <= m.
    m : int
        Number of variables, m >= 1.

    Returns
    -------
    dict
        'generator' (np.ndarray, k x n binary matrix),
        'n' (code length), 'k' (dimension), 'd_min' (minimum distance),
        'rate' (k / n).

    Raises
    ------
    ValueError
        If r > m or m < 1 or r < 0.

    References
    ----------
    Reed, I. S. (1954). A class of multiple-error-correcting codes and the
    decoding scheme. IEEE Trans. Inform. Theory, 4, 38-49.
    Muller, D. E. (1954). Application of Boolean algebra to switching circuit
    design. IEEE Trans. Computers, 3(3), 6-12.
    """
    if m < 1:
        raise ValueError("m must be >= 1.")
    if r < 0 or r > m:
        raise ValueError(f"r must be in [0, m], got r={r}, m={m}.")

    n = 2 ** m

    variables = np.zeros((m, n), dtype=np.int8)
    for i in range(m):
        for j in range(n):
            variables[i, j] = (j >> (m - 1 - i)) & 1

    rows = [np.ones(n, dtype=np.int8)]

    from itertools import combinations

    for order in range(1, r + 1):
        for combo in combinations(range(m), order):
            row = np.ones(n, dtype=np.int8)
            for var_idx in combo:
                row = row & variables[var_idx]
            rows.append(row)

    G = np.array(rows, dtype=np.int8)
    k = G.shape[0]
    d_min = 2 ** (m - r)

    return {
        "generator": G,
        "n": n,
        "k": k,
        "d_min": d_min,
        "rate": k / n,
    }
