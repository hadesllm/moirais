# morie.fn — function file from book-equation translation pipeline (hadesllm/morie)
"""Bell number and partition count."""


from ._containers import DescriptiveResult
def bell_polynomial(n: int, **kwargs) -> DescriptiveResult:
    r"""
    Compute the n-th Bell number :math:`B_n`.

    :math:`B_n` counts the number of partitions of a set of *n* elements.
    Computed via the Bell triangle:

    .. math::

        B_{n+1} = \\sum_{k=0}^{n} \\binom{n}{k} B_k

    First values: :math:`B_0 = 1, B_1 = 1, B_2 = 2, B_3 = 5, B_4 = 15`.

    :param n: Non-negative integer index.
    :return: DescriptiveResult with B_n as value.
    :raises ValueError: If n < 0.

    References
    ----------
    Bell, E. T. (1934). Exponential numbers. *American Mathematical Monthly*,
    41(7), 411-419.
    Rota, G.-C. (1964). The number of partitions of a set. *American
    Mathematical Monthly*, 71(5), 498-504.
    """
    n = int(n)
    if n < 0:
        raise ValueError(f"n must be >= 0, got {n}.")

    if n == 0:
        return DescriptiveResult(
            name="bell_polynomial",
            value=1.0,
            extra={"B_n": 1, "n": 0, "bell_triangle_row": [1]},
        )

    triangle = [[0] * (n + 1) for _ in range(n + 1)]
    triangle[0][0] = 1

    for i in range(1, n + 1):
        triangle[i][0] = triangle[i - 1][i - 1]
        for j in range(1, i + 1):
            triangle[i][j] = triangle[i][j - 1] + triangle[i - 1][j - 1]

    b_n = triangle[n][0]
    first_col = [triangle[i][0] for i in range(n + 1)]

    return DescriptiveResult(
        name="bell_polynomial",
        value=float(b_n),
        extra={
            "B_n": b_n,
            "n": n,
            "bell_sequence": first_col,
        },
    )


bellp = bell_polynomial


def cheatsheet() -> str:
    return "bell_polynomial({}) -> Bell number and partition count."
