# moirais.fn — function file (hadesllm/moirais)
"""Generalizability coefficient (G-theory)."""

from __future__ import annotations

from moirais.fn._containers import ESRes


def generalizability_coeff(
    variance_components: dict[str, float],
) -> ESRes:
    """G-coefficient from generalizability theory.

    G = sigma2_person / (sigma2_person + sigma2_error/n_i)

    Parameters
    ----------
    variance_components : dict
        Must contain 'person' and 'error'. May contain 'item', 'interaction'.

    Returns
    -------
    ESRes
        measure="G_coefficient".

    References
    ----------
    Brennan, R. L. (2001). Generalizability Theory. Springer.
    """
    sigma2_p = variance_components.get("person", 0.0)
    sigma2_e = variance_components.get("error", 0.0)
    sigma2_i = variance_components.get("item", 0.0)
    sigma2_pi = variance_components.get("interaction", 0.0)

    n_items = max(variance_components.get("n_items", 1), 1)

    rel_error = (sigma2_pi + sigma2_e) / n_items
    g_coeff = sigma2_p / max(sigma2_p + rel_error, 1e-10)

    abs_error = (sigma2_i + sigma2_pi + sigma2_e) / n_items
    d_coeff = sigma2_p / max(sigma2_p + abs_error, 1e-10)

    return ESRes(
        measure="G_coefficient",
        estimate=float(g_coeff),
        extra={
            "D_coefficient": float(d_coeff),
            "relative_error": float(rel_error),
            "absolute_error": float(abs_error),
            "variance_components": variance_components,
        },
    )


g_coeff = generalizability_coeff


def cheatsheet() -> str:
    return "generalizability_coeff({}) -> Generalizability coefficient (G-theory)."
