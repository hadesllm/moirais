# moirais.fn — function file (hadesllm/moirais)
"""Levenshtein edit distance."""

import numpy as np

from ._containers import DescriptiveResult
def edit_distance(str_a: str, str_b: str, **kwargs) -> DescriptiveResult:
    """
    Compute the Levenshtein edit distance between two strings.

    The minimum number of single-character insertions, deletions, and
    substitutions needed to transform ``str_a`` into ``str_b``.

    Uses the Wagner-Fischer dynamic programming algorithm in
    :math:`O(mn)` time and :math:`O(\\min(m, n))` space.

    .. math::

        d(i, j) = \\begin{cases}
            j & \\text{if } i = 0 \\\\
            i & \\text{if } j = 0 \\\\
            d(i-1, j-1) & \\text{if } a_i = b_j \\\\
            1 + \\min(d(i-1,j), d(i,j-1), d(i-1,j-1)) & \\text{otherwise}
        \\end{cases}

    :param str_a: First string.
    :param str_b: Second string.
    :return: DescriptiveResult with edit distance as value.

    References
    ----------
    Wagner, R. A. & Fischer, M. J. (1974). The string-to-string correction
    problem. *Journal of the ACM*, 21(1), 168-173.
    Levenshtein, V. I. (1966). Binary codes capable of correcting deletions,
    insertions, and reversals. *Soviet Physics Doklady*, 10(8), 707-710.
    """
    str_a = str(str_a)
    str_b = str(str_b)
    m, n = len(str_a), len(str_b)

    if m < n:
        str_a, str_b = str_b, str_a
        m, n = n, m

    prev = np.arange(n + 1, dtype=np.int64)
    curr = np.zeros(n + 1, dtype=np.int64)

    for i in range(1, m + 1):
        curr[0] = i
        for j in range(1, n + 1):
            cost = 0 if str_a[i - 1] == str_b[j - 1] else 1
            curr[j] = min(
                prev[j] + 1,
                curr[j - 1] + 1,
                prev[j - 1] + cost,
            )
        prev, curr = curr, prev

    dist = int(prev[n])
    max_len = max(m, n)
    similarity = 1.0 - dist / max_len if max_len > 0 else 1.0

    return DescriptiveResult(
        name="edit_distance",
        value=float(dist),
        extra={
            "edit_distance": dist,
            "normalized": dist / max_len if max_len > 0 else 0.0,
            "similarity": similarity,
            "len_a": len(str_a),
            "len_b": len(str_b),
        },
    )


edtds = edit_distance


def cheatsheet() -> str:
    return "edit_distance({}) -> Levenshtein edit distance."
