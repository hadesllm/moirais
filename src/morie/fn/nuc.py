# morie.fn — function file (hadesllm/morie)
"""Nucleotide frequency distribution (A/T/G/C counts and proportions)."""


from ._containers import DescriptiveResult


def nucleotide_freq(seq: str) -> DescriptiveResult:
    """
    Compute nucleotide frequency distribution from a DNA sequence.

    Counts occurrences of A, T, G, C and computes proportions.
    Non-ATGC characters are tallied as 'other'.

    :param seq: DNA sequence string (case-insensitive).
    :return: DescriptiveResult with counts and proportions in extra.
    :raises ValueError: If sequence is empty.

    References
    ----------
    Nei M & Kumar S (2000). Molecular Evolution and Phylogenetics.
    Oxford University Press.
    """
    if not seq:
        raise ValueError("Sequence must not be empty.")
    s = seq.upper()
    counts = {b: s.count(b) for b in "ATGC"}
    counts["other"] = len(s) - sum(counts.values())
    total = len(s)
    props = {b: c / total for b, c in counts.items()}
    gc = (counts["G"] + counts["C"]) / total
    return DescriptiveResult(
        name="nucleotide_freq",
        value=gc,
        extra={"counts": counts, "proportions": props, "gc_content": gc, "length": total},
    )


nuc = nucleotide_freq


def cheatsheet() -> str:
    return "nucleotide_freq({}) -> Nucleotide frequency distribution (A/T/G/C counts and propor"
