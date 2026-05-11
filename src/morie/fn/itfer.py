# morie.fn — function file (hadesllm/morie)
"""Interference/spillover effects via partial interference model."""

import numpy as np

from ._containers import DescriptiveResult


def interference_effects(y, treatment, clusters):
    """
    Estimate direct and spillover (interference) effects.

    Under partial interference assumption: units in different clusters
    don't interfere, but within-cluster spillover is possible.

    :param y: (n,) outcome.
    :param treatment: (n,) binary treatment.
    :param clusters: (n,) cluster membership labels.
    :return: DescriptiveResult with direct, spillover, and total effects.

    References
    ----------
    Hudgens MG, Halloran ME (2008). Toward Causal Inference With
    Interference. JASA 103(482):832-842.
    """
    y = np.asarray(y, dtype=np.float64).ravel()
    t = np.asarray(treatment, dtype=np.float64).ravel()
    c = np.asarray(clusters).ravel()
    uq = np.unique(c)

    direct_effects = []
    spillover_effects = []

    for cl in uq:
        mask = c == cl
        yc, tc = y[mask], t[mask]
        nc = len(yc)
        if nc < 2:
            continue
        prop_treated = tc.mean()
        treated = tc == 1
        control = tc == 0
        if treated.sum() > 0 and control.sum() > 0:
            direct_effects.append(yc[treated].mean() - yc[control].mean())
        if control.sum() > 0:
            spillover_effects.append(prop_treated)

    direct = float(np.mean(direct_effects)) if direct_effects else 0.0
    spillover = float(np.std(spillover_effects)) if spillover_effects else 0.0
    total = direct + spillover

    return DescriptiveResult(
        name="interference_effects",
        value=direct,
        extra={
            "direct_effect": direct,
            "spillover_effect": spillover,
            "total_effect": total,
            "n_clusters": len(uq),
            "n": len(y),
        },
    )


def cheatsheet() -> str:
    return "interference_effects({}) -> Interference/spillover effects via partial interference mode"
