# morie.fn -- function file (hadesllm/morie)
"""GC content of a DNA sequence."""

from ._containers import ESRes


def gc_content_calc(
    sequence: str,
) -> ESRes:
    """GC content: fraction of guanine (G) and cytosine (C) bases.

    GC% = (G + C) / N

    where N is the total number of valid bases (A, T, G, C).

    Parameters
    ----------
    sequence : str
        DNA sequence string (case-insensitive). Only A, T, G, C
        characters are counted; others are ignored.

    Returns
    -------
    ESRes
        measure="GC_content", estimate=fraction [0, 1].

    Raises
    ------
    ValueError
        If *sequence* is empty or contains no valid bases.

    References
    ----------
    Bernardi, G. (2000). Isochores and the evolutionary genomics of
        vertebrates. Gene, 241(1), 3-17.
    """
    seq = sequence.upper()
    valid = [b for b in seq if b in "ATGC"]
    n = len(valid)
    if n == 0:
        raise ValueError("Sequence must contain at least one valid base (A/T/G/C).")
    gc_count = sum(1 for b in valid if b in "GC")
    return ESRes(
        measure="GC_content",
        estimate=gc_count / n,
        n=n,
    )


gc_content = gc_content_calc


def cheatsheet() -> str:
    return "gc_content_calc({}) -> GC content of a DNA sequence."
