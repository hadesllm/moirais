# morie.fn -- function file (hadesllm/morie)
"""Mamdani fuzzy inference system."""

import numpy as np

from ._containers import DescriptiveResult
def fuzzy_inference(
    rules,
    inputs,
    universe=None,
    n_points: int = 200,
    defuzz_method: str = "centroid",
    **kwargs,
) -> DescriptiveResult:
    """
    Execute a Mamdani-type fuzzy inference system.

    Each rule is a dict with keys:

    - ``"antecedent"``: list of (input_index, mf_type, params)
    - ``"consequent"``: (mf_type, params) for output
    - ``"weight"``: rule weight (default 1.0)

    The inference aggregates via max (union) and defuzzifies.

    :param rules: List of rule dicts.
    :param inputs: List or array of crisp input values.
    :param universe: (low, high) for output universe. Default (0, 1).
    :param n_points: Discretization points for output. Default 200.
    :param defuzz_method: Defuzzification method. Default ``"centroid"``.
    :return: DescriptiveResult with crisp output.
    :raises ValueError: If rules list is empty.

    References
    ----------
    Mamdani, E. H. & Assilian, S. (1975). An experiment in linguistic
    synthesis with a fuzzy logic controller. *International Journal of
    Man-Machine Studies*, 7(1), 1-13.
    Ross, T. J. (2010). *Fuzzy Logic with Engineering Applications* (3rd ed.).
    Wiley.
    """
    if not rules:
        raise ValueError("Rules list must not be empty.")
    inputs = np.atleast_1d(np.asarray(inputs, dtype=np.float64))

    if universe is None:
        universe = (0.0, 1.0)
    x_out = np.linspace(universe[0], universe[1], n_points)
    aggregate = np.zeros(n_points)

    def _eval_mf(val, mf_type, params):
        if mf_type == "triangular":
            a, b, c = params
            if val <= a or val >= c:
                return 0.0
            if val <= b:
                return (val - a) / (b - a) if b > a else 1.0
            return (c - val) / (c - b) if c > b else 1.0
        elif mf_type == "gaussian":
            mean, sigma = params
            return float(np.exp(-0.5 * ((val - mean) / sigma) ** 2))
        elif mf_type == "trapezoidal":
            a, b, c, d = params
            if val <= a or val >= d:
                return 0.0
            if val <= b:
                return (val - a) / (b - a) if b > a else 1.0
            if val <= c:
                return 1.0
            return (d - val) / (d - c) if d > c else 1.0
        return 0.0

    def _mf_array(x_arr, mf_type, params):
        return np.array([_eval_mf(xi, mf_type, params) for xi in x_arr])

    for rule in rules:
        weight = rule.get("weight", 1.0)
        firing = 1.0
        for inp_idx, mf_type, params in rule["antecedent"]:
            val = float(inputs[inp_idx])
            firing = min(firing, _eval_mf(val, mf_type, params))
        firing *= weight
        cons_mf_type, cons_params = rule["consequent"]
        cons_mf = _mf_array(x_out, cons_mf_type, cons_params)
        clipped = np.minimum(cons_mf, firing)
        aggregate = np.maximum(aggregate, clipped)

    total_area = float(np.trapezoid(aggregate, x_out))
    if total_area < 1e-300:
        crisp = float(np.mean(x_out))
    else:
        if defuzz_method == "centroid":
            crisp = float(np.trapezoid(x_out * aggregate, x_out) / total_area)
        elif defuzz_method == "mom":
            max_val = np.max(aggregate)
            crisp = float(np.mean(x_out[np.isclose(aggregate, max_val)]))
        else:
            crisp = float(np.trapezoid(x_out * aggregate, x_out) / total_area)

    return DescriptiveResult(
        name="fuzzy_inference",
        value=crisp,
        extra={
            "crisp_output": crisp,
            "defuzz_method": defuzz_method,
            "aggregate_mf": aggregate,
            "output_universe": x_out,
            "total_area": total_area,
            "n_rules": len(rules),
        },
    )


fzinf = fuzzy_inference


def cheatsheet() -> str:
    return "fuzzy_inference({}) -> Mamdani fuzzy inference system."
