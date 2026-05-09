# moirais.fn — function file (hadesllm/moirais)
"""Kulldorff's spatial scan statistic."""

import numpy as np

from ._containers import DescriptiveResult


def spatial_scan(
    counts: np.ndarray,
    populations: np.ndarray,
    coordinates: np.ndarray,
    max_radius: float | None = None,
    n_simulations: int = 99,
    seed: int = 42,
) -> DescriptiveResult:
    """
    Kulldorff's spatial scan statistic for cluster detection.

    Scans circular windows of varying size centred at each location
    and computes the likelihood ratio for elevated risk.

    :param counts: (n,) observed case counts per location.
    :param populations: (n,) population at risk per location.
    :param coordinates: (n, 2) spatial coordinates.
    :param max_radius: Maximum scanning radius (default: half max distance).
    :param n_simulations: Monte Carlo simulations for p-value.
    :param seed: Random seed.
    :return: DescriptiveResult with best cluster info.

    References
    ----------
    Kulldorff M (1997). A spatial scan statistic.
    Communications in Statistics: Theory and Methods, 26(6), 1481-1496.
    """
    C = np.asarray(counts, dtype=np.float64)
    P = np.asarray(populations, dtype=np.float64)
    coords = np.asarray(coordinates, dtype=np.float64)
    n = len(C)
    C_total = C.sum()
    P_total = P.sum()
    dmat = np.sqrt(((coords[:, None, :] - coords[None, :, :]) ** 2).sum(axis=2))
    if max_radius is None:
        max_radius = float(dmat.max() / 2)

    def scan_llr(c_arr, p_arr):
        best_llr = 0.0
        best_center = 0
        best_radius = 0.0
        for i in range(n):
            order = np.argsort(dmat[i])
            c_in, p_in = 0.0, 0.0
            for j_idx in order:
                if dmat[i, j_idx] > max_radius:
                    break
                c_in += c_arr[j_idx]
                p_in += p_arr[j_idx]
                c_out = C_total - c_in
                p_out = P_total - p_in
                if p_in > 0 and p_out > 0 and c_in > 0 and c_out > 0:
                    rr_in = c_in / p_in
                    rr_out = c_out / p_out
                    if rr_in > rr_out:
                        llr = c_in * np.log(rr_in / (C_total / P_total))
                        llr += c_out * np.log(rr_out / (C_total / P_total))
                        if llr > best_llr:
                            best_llr = llr
                            best_center = i
                            best_radius = dmat[i, j_idx]
        return best_llr, best_center, best_radius

    obs_llr, center, radius = scan_llr(C, P)
    rng = np.random.default_rng(seed)
    count_ge = 0
    for _ in range(n_simulations):
        sim_c = rng.multinomial(int(C_total), P / P_total).astype(float)
        sim_llr, _, _ = scan_llr(sim_c, P)
        if sim_llr >= obs_llr:
            count_ge += 1
    pval = (count_ge + 1) / (n_simulations + 1)
    return DescriptiveResult(
        name="spatial_scan",
        value=float(obs_llr),
        extra={
            "llr": float(obs_llr),
            "center": int(center),
            "radius": float(radius),
            "p_value": float(pval),
            "n_locations": n,
        },
    )


scan = spatial_scan


def cheatsheet() -> str:
    return "spatial_scan({}) -> Kulldorff's spatial scan statistic."
