"""Turbo code interleaver design."""

__all__ = ["turbr"]

import numpy as np


def turbr(
    n: int,
    *,
    method: str = "random",
    seed: int = 42,
) -> dict:
    """
    Generate a turbo code interleaver permutation.

    Parameters
    ----------
    n : int
        Block length (number of information bits).
    method : str
        Interleaver type: 'random', 'qpp' (quadratic permutation polynomial),
        or 'golden' (golden-ratio based).
    seed : int
        Random seed for 'random' method.

    Returns
    -------
    dict
        'permutation' (np.ndarray of length n),
        'inverse' (np.ndarray, inverse permutation for decoding),
        'spread' (float, minimum spread metric S = min|pi(i)-pi(j)|/|i-j|),
        'method'.

    Raises
    ------
    ValueError
        If n < 2 or unknown method.

    References
    ----------
    Berrou, C., Glavieux, A. & Thitimajshima, P. (1993). Near Shannon limit
    error-correcting coding and decoding: Turbo-codes. Proc. ICC, 1064-1070.
    Sun, J. & Takeshita, O. (2005). Interleavers for turbo codes using
    permutation polynomials over integer rings. IEEE Trans. Inform. Theory.
    """
    if n < 2:
        raise ValueError("n must be >= 2.")

    if method == "random":
        rng = np.random.default_rng(seed)
        perm = rng.permutation(n)
    elif method == "qpp":
        f1 = 1
        f2 = n // 4 if n >= 4 else 1
        if f2 % 2 == 0 and n % 2 == 0:
            f2 += 1
        perm = np.array([(f1 * i + f2 * i * i) % n for i in range(n)])
        if len(set(perm)) != n:
            rng = np.random.default_rng(seed)
            perm = rng.permutation(n)
    elif method == "golden":
        golden = (1 + np.sqrt(5)) / 2
        perm = np.array([int((i * golden * n) % n) for i in range(n)])
        if len(set(perm)) != n:
            rng = np.random.default_rng(seed)
            perm = rng.permutation(n)
    else:
        raise ValueError(f"Unknown method: {method!r}. Use 'random', 'qpp', or 'golden'.")

    inverse = np.empty(n, dtype=np.int64)
    inverse[perm] = np.arange(n)

    min_spread = float("inf")
    sample_size = min(n, 200)
    rng_s = np.random.default_rng(seed + 1)
    indices = rng_s.choice(n, size=sample_size, replace=False) if n > 200 else np.arange(n)
    for ii in range(len(indices)):
        for jj in range(ii + 1, min(ii + 20, len(indices))):
            i_idx, j_idx = indices[ii], indices[jj]
            denom = abs(int(i_idx) - int(j_idx))
            if denom > 0:
                s = abs(int(perm[i_idx]) - int(perm[j_idx])) / denom
                if s < min_spread:
                    min_spread = s

    if min_spread == float("inf"):
        min_spread = 0.0

    return {
        "permutation": perm,
        "inverse": inverse,
        "spread": min_spread,
        "method": method,
    }
