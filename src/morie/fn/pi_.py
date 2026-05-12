# morie.fn — function file (hadesllm/morie)
"""Nucleotide diversity (pi) from a sequence alignment."""


from ._containers import GenomicsResult


def nucleotide_diversity(sequences: list[str]) -> GenomicsResult:
    r"""
    Compute nucleotide diversity (pi) from aligned DNA sequences.

    Pi is the average number of pairwise nucleotide differences per site:

    .. math::

        \\pi = \\frac{2}{n(n-1)} \\sum_{i<j} d_{ij}

    where :math:`d_{ij}` is the proportion of differing sites between
    sequences *i* and *j*.

    :param sequences: List of aligned DNA sequences (equal length).
    :return: GenomicsResult with pi as statistic.
    :raises ValueError: If fewer than 2 sequences or unequal lengths.

    References
    ----------
    Nei M & Li WH (1979). Mathematical model for studying genetic
    variation in terms of restriction endonucleases.
    PNAS, 76(10), 5269-5273.
    """
    n = len(sequences)
    if n < 2:
        raise ValueError("Need at least 2 sequences.")
    L = len(sequences[0])
    if any(len(s) != L for s in sequences):
        raise ValueError("All sequences must have equal length.")
    seqs = [s.upper() for s in sequences]
    total_diff = 0.0
    n_pairs = 0
    for i in range(n):
        for j in range(i + 1, n):
            diffs = sum(1 for a, b in zip(seqs[i], seqs[j]) if a != b)
            total_diff += diffs / L
            n_pairs += 1
    pi_val = total_diff / n_pairs
    return GenomicsResult(
        name="nucleotide_diversity", statistic=float(pi_val), n=n, extra={"n_sites": L, "n_pairs": n_pairs}
    )


pi_ = nucleotide_diversity


def cheatsheet() -> str:
    return "nucleotide_diversity({}) -> Nucleotide diversity (pi) from a sequence alignment."
