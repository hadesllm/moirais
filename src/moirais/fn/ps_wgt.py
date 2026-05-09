# moirais.fn — function file (hadesllm/moirais)
"""Post-stratification weights to match known population distribution."""

import warnings

import numpy as np
import pandas as pd


def poststratification_weights(
    df: pd.DataFrame,
    strata_col: str,
    population_counts: dict[str, int],
) -> pd.Series:
    """
    Compute post-stratification weights so the sample distribution matches
    the known population distribution.

    For each stratum h:

    .. math::

        w_i^{\\text{PS}} = \\frac{N_h / N}{n_h / n}

    where :math:`N_h` is the known population count in stratum h, :math:`N`
    is the total population, :math:`n_h` is the sample count in stratum h,
    and :math:`n` is the total sample size.

    :param df: Input DataFrame.
    :param strata_col: Column name identifying strata.
    :param population_counts: Dict mapping each stratum label (as string) to
        its population count.
    :return: pd.Series of post-stratification weights, indexed like df.
    :raises ValueError: If strata_col not in df, any stratum in df is absent
        from population_counts, or sample stratum count is zero.

    References
    ----------
    Lumley, T. (2010). Complex Surveys: A Guide to Analysis Using R. Wiley. (Chapter 7.)
    Little, R. J. A. (1993). Post-stratification: A modeler's perspective.
        JASA, 88(423), 1001-1012.
    """
    if strata_col not in df.columns:
        raise ValueError(f"Column '{strata_col}' not found in DataFrame.")
    strata_vals = df[strata_col].astype(str)
    sample_counts = strata_vals.value_counts()
    missing_strata = set(strata_vals.unique()) - set(str(k) for k in population_counts)
    if missing_strata:
        raise ValueError(f"Strata {missing_strata} appear in the sample but are missing from population_counts.")

    N_total = sum(population_counts.values())
    n_total = len(df)

    weights = pd.Series(np.ones(len(df), dtype=float), index=df.index)
    for stratum, N_h in population_counts.items():
        stratum_str = str(stratum)
        mask = strata_vals == stratum_str
        n_h = int(mask.sum())
        if n_h == 0:
            warnings.warn(
                f"Stratum '{stratum}' has 0 observations in the sample; "
                "post-stratification weight is undefined for this stratum.",
                stacklevel=2,
            )
            continue
        pop_frac = N_h / N_total
        samp_frac = n_h / n_total
        weights[mask] = pop_frac / samp_frac
    return weights


ps_wgt_fn = poststratification_weights


def cheatsheet() -> str:
    return "poststratification_weights({}) -> Post-stratification weights to match known population distri"
