# moirais.fn — function file (hadesllm/moirais)
"""Structural causal model — linear SCM with do-calculus."""

import numpy as np

from ._containers import DescriptiveResult


def structural_causal_model(data, edges, intervention=None):
    """
    Fit a linear structural causal model and compute interventional effects.

    :param data: dict mapping variable names to (n,) arrays.
    :param edges: list of (parent, child) tuples defining the DAG.
    :param intervention: dict {var_name: value} for do(X=x) intervention.
    :return: DescriptiveResult with structural coefficients and causal effects.

    References
    ----------
    Pearl J (2009). Causality. 2nd ed. Cambridge University Press.
    """
    if not edges:
        raise ValueError("Need at least one edge in the DAG")

    var_names = sorted(set(v for e in edges for v in e))
    n = len(next(iter(data.values())))
    coeffs = {}

    for child in var_names:
        parents = [p for p, c in edges if c == child]
        if not parents or child not in data:
            continue
        y = np.asarray(data[child], dtype=np.float64)
        X = np.column_stack([np.ones(n)] + [np.asarray(data[p], dtype=np.float64) for p in parents])
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        for i, p in enumerate(parents):
            coeffs[(p, child)] = float(beta[i + 1])

    causal_effects = {}
    if intervention:
        for target in var_names:
            if target in intervention:
                continue
            effect = 0.0
            for src, val in intervention.items():
                if (src, target) in coeffs:
                    effect += coeffs[(src, target)] * val
            causal_effects[target] = effect

    return DescriptiveResult(
        name="structural_causal_model",
        value=len(coeffs),
        extra={
            "structural_coefficients": {f"{p}->{c}": v for (p, c), v in coeffs.items()},
            "causal_effects": causal_effects,
            "variables": var_names,
            "n_edges": len(edges),
            "n": n,
        },
    )


def cheatsheet() -> str:
    return "structural_causal_model({}) -> Structural causal model — linear SCM with do-calculus."
