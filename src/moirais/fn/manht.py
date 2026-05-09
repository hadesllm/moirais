# moirais.fn — function file (hadesllm/moirais)
"""Manhattan plot data preparation for GWAS results."""

import numpy as np
import pandas as pd

from ._containers import DescriptiveResult


def manhattan_data(
    p_values: np.ndarray, chromosomes: np.ndarray, positions: np.ndarray, threshold: float = 5e-8
) -> DescriptiveResult:
    """
    Prepare data for a Manhattan plot from GWAS results.

    Computes -log10(p) and cumulative genomic positions for plotting.

    :param p_values: Array of p-values.
    :param chromosomes: Chromosome labels (integer or string).
    :param positions: Base-pair positions within each chromosome.
    :param threshold: Genome-wide significance threshold (default 5e-8).
    :return: DescriptiveResult with plot DataFrame in extra.
    :raises ValueError: If arrays have different lengths.

    References
    ----------
    Pearson TA, Manolio TA (2008). How to interpret a genome-wide
    association study. JAMA, 299(11), 1335-1344.
    """
    pv = np.asarray(p_values, dtype=np.float64)
    chrs = np.asarray(chromosomes)
    pos = np.asarray(positions, dtype=np.float64)
    if not (len(pv) == len(chrs) == len(pos)):
        raise ValueError("All arrays must have the same length.")
    logp = -np.log10(np.clip(pv, 1e-300, 1.0))
    unique_chr = np.unique(chrs)
    offset = 0.0
    cum_pos = np.zeros_like(pos, dtype=np.float64)
    chr_centers = {}
    for c in sorted(unique_chr):
        mask = chrs == c
        cum_pos[mask] = pos[mask] + offset
        chr_centers[c] = float(cum_pos[mask].mean())
        offset = float(cum_pos[mask].max()) + 1e6
    n_sig = int(np.sum(pv < threshold))
    df = pd.DataFrame({"chr": chrs, "pos": pos, "cum_pos": cum_pos, "p": pv, "logp": logp})
    return DescriptiveResult(
        name="manhattan_data",
        value=n_sig,
        extra={"data": df, "chr_centers": chr_centers, "threshold": threshold, "n_significant": n_sig},
    )


manht = manhattan_data


def cheatsheet() -> str:
    return "manhattan_data({}) -> Manhattan plot data preparation for GWAS results."
