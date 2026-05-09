# moirais.fn — function file (hadesllm/moirais)
"""Compare posterior parameters."""

from __future__ import annotations

from ._containers import DescriptiveResult


def posterior_compare_params(chain, param_indices) -> DescriptiveResult:
    """Compare posterior distributions of selected parameters.

    .. epigraph:: "Fire and blood." -- Targaryen, Game of Thrones
    """
    import numpy as np

    chain = np.asarray(chain, dtype=float)
    if chain.ndim == 1:
        chain = chain.reshape(-1, 1)
    comparisons = []
    for i, idx in enumerate(param_indices):
        col = chain[:, idx]
        comparisons.append(
            {
                "param_index": int(idx),
                "mean": float(np.mean(col)),
                "sd": float(np.std(col, ddof=1)),
                "ci_lo": float(np.percentile(col, 2.5)),
                "ci_hi": float(np.percentile(col, 97.5)),
            }
        )
    return DescriptiveResult(
        name="posterior_compare_params",
        value=float(len(param_indices)),
        extra={"comparisons": comparisons, "n_params": len(param_indices)},
    )


pstcp = posterior_compare_params


def cheatsheet() -> str:
    return "posterior_compare_params({}) -> Compare posterior parameters."
