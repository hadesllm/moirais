# morie.fn -- function file (hadesllm/morie)
"""Codon usage table from a DNA sequence."""


from ._containers import DescriptiveResult


def codon_usage(seq: str) -> DescriptiveResult:
    """
    Compute codon usage table from a DNA sequence.

    Splits sequence into non-overlapping triplets and counts each codon.

    :param seq: DNA sequence (length should be divisible by 3).
    :return: DescriptiveResult with codon counts and frequencies in extra.
    :raises ValueError: If sequence length < 3.

    References
    ----------
    Sharp PM & Li WH (1987). The codon adaptation index.
    Nucleic Acids Research, 15(3), 1281-1295.
    """
    if len(seq) < 3:
        raise ValueError("Sequence must have at least 3 nucleotides.")
    s = seq.upper()
    n_codons = len(s) // 3
    codons = [s[i * 3 : (i + 1) * 3] for i in range(n_codons)]
    counts: dict[str, int] = {}
    for c in codons:
        counts[c] = counts.get(c, 0) + 1
    freqs = {c: v / n_codons for c, v in counts.items()}
    n_unique = len(counts)
    return DescriptiveResult(
        name="codon_usage",
        value=n_unique,
        extra={"counts": counts, "frequencies": freqs, "n_codons": n_codons, "n_unique": n_unique},
    )


codon = codon_usage


def cheatsheet() -> str:
    return "codon_usage({}) -> Codon usage table from a DNA sequence."
