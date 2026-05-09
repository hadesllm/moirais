"""Space-time scan statistic for cluster detection (Kulldorff 1997)."""

from __future__ import annotations

import numpy as np

from ._containers import SpatialResult


def st_scan_statistic(
    coords: np.ndarray,
    times: np.ndarray,
    cases: np.ndarray,
    population: np.ndarray,
    max_spatial_radius: float | None = None,
    max_temporal_window: float | None = None,
    n_simulations: int = 999,
    seed: int | None = None,
) -> SpatialResult:
    r"""Space-time scan statistic for prospective cluster detection.

    Scans cylindrical windows in (x, y, t) space, computing a
    Poisson-based likelihood ratio for each candidate cluster:

    .. math::

        \Lambda = \frac{L(Z)}{L_0}
        = \left(\frac{c}{E[c]}\right)^{c}
        \left(\frac{C - c}{C - E[c]}\right)^{C - c}
        \mathbf{1}(c > E[c])

    where *c* is the number of cases inside the cylinder, *C* is total
    cases, and :math:`E[c]` is the expected count under constant risk.

    Parameters
    ----------
    coords : np.ndarray
        Spatial coordinates, shape ``(n, 2)``.
    times : np.ndarray
        Time period index, shape ``(n,)``.
    cases : np.ndarray
        Case counts per unit, shape ``(n,)``.
    population : np.ndarray
        Population per unit, shape ``(n,)``.
    max_spatial_radius : float, optional
        Maximum spatial radius. Default: half the max pairwise distance.
    max_temporal_window : float, optional
        Maximum temporal window. Default: half the time range.
    n_simulations : int
        Monte Carlo simulations for p-value. Default 999.
    seed : int, optional
        RNG seed for reproducibility.

    Returns
    -------
    SpatialResult
        ``statistic`` is the maximum log-likelihood ratio.
        ``p_value`` from Monte Carlo.
        ``extra`` contains ``center_idx``, ``radius``, ``time_window``.

    References
    ----------
    Kulldorff M (1997). A spatial scan statistic. *Communications in
    Statistics -- Theory and Methods*, 26(6), 1481-1496.

    Kulldorff M, Heffernan R, Hartman J, Assuncao R, Mostashari F (2005).
    A space-time permutation scan statistic for disease outbreak
    detection. *PLoS Medicine*, 2(3), e59.
    """
    xy = np.asarray(coords, dtype=np.float64)
    t = np.asarray(times, dtype=np.float64).ravel()
    c = np.asarray(cases, dtype=np.float64).ravel()
    pop = np.asarray(population, dtype=np.float64).ravel()
    n = len(c)

    rng = np.random.default_rng(seed)
    C_total = c.sum()
    P_total = pop.sum()

    dists = np.sqrt(((xy[:, None, :] - xy[None, :, :]) ** 2).sum(axis=2))

    if max_spatial_radius is None:
        max_spatial_radius = dists.max() / 2.0
    if max_temporal_window is None:
        max_temporal_window = (t.max() - t.min()) / 2.0

    def _log_lr(c_in, e_in):
        if c_in <= e_in or e_in <= 0 or C_total <= c_in:
            return 0.0
        c_out = C_total - c_in
        e_out = C_total - e_in
        if e_out <= 0:
            return 0.0
        llr = c_in * np.log(c_in / e_in) + c_out * np.log(c_out / e_out)
        return llr

    unique_t = np.unique(t)
    best_llr = 0.0
    best_info = {"center_idx": 0, "radius": 0.0, "time_window": (0.0, 0.0)}

    for i in range(n):
        sorted_d = np.sort(dists[i])
        radii = sorted_d[(sorted_d > 0) & (sorted_d <= max_spatial_radius)]
        radii = np.unique(radii)
        if len(radii) > 20:
            radii = radii[np.linspace(0, len(radii) - 1, 20, dtype=int)]
        for r in radii:
            spatial_mask = dists[i] <= r
            for t_start_idx in range(len(unique_t)):
                for t_end_idx in range(t_start_idx, len(unique_t)):
                    tw = unique_t[t_end_idx] - unique_t[t_start_idx]
                    if tw > max_temporal_window:
                        break
                    temporal_mask = (t >= unique_t[t_start_idx]) & (t <= unique_t[t_end_idx])
                    mask = spatial_mask & temporal_mask
                    c_in = c[mask].sum()
                    pop_in = pop[mask].sum()
                    if pop_in <= 0:
                        continue
                    e_in = C_total * pop_in / P_total
                    llr = _log_lr(c_in, e_in)
                    if llr > best_llr:
                        best_llr = llr
                        best_info = {
                            "center_idx": int(i),
                            "radius": float(r),
                            "time_window": (float(unique_t[t_start_idx]), float(unique_t[t_end_idx])),
                        }

    sim_max = np.empty(n_simulations)
    for sim in range(n_simulations):
        sim_cases = rng.multinomial(int(C_total), pop / P_total).astype(np.float64)
        sim_best = 0.0
        for i in range(0, n, max(1, n // 10)):
            radii_s = np.sort(dists[i])
            radii_s = radii_s[(radii_s > 0) & (radii_s <= max_spatial_radius)]
            if len(radii_s) > 5:
                radii_s = radii_s[np.linspace(0, len(radii_s) - 1, 5, dtype=int)]
            for r in radii_s:
                mask = dists[i] <= r
                c_in = sim_cases[mask].sum()
                pop_in = pop[mask].sum()
                if pop_in <= 0:
                    continue
                e_in = C_total * pop_in / P_total
                llr = _log_lr(c_in, e_in)
                if llr > sim_best:
                    sim_best = llr
        sim_max[sim] = sim_best

    p_value = float((np.sum(sim_max >= best_llr) + 1) / (n_simulations + 1))

    return SpatialResult(
        name="ST_Scan",
        statistic=float(best_llr),
        p_value=p_value,
        extra=best_info,
    )


def cheatsheet() -> str:
    return "st_scan_statistic({}) -> Space-time scan statistic for cluster detection (Kulldorff 1"
