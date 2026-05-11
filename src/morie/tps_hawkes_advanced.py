"""morie.tps_hawkes_advanced — non-stationary Hawkes with non-exponential kernels.

Implements the Kwan-Chen-Dunsmuir (2024, arXiv:2408.09710v1) methodology
for Hawkes process likelihood inference when the baseline intensity is
time-varying *and* the excitation kernel is non-exponential (so the
intensity process is non-Markovian).

Companion to ``morie.tps_stochastic`` — that module contains the
classical exponential-kernel constant-baseline (Markovian / Mohler 2011)
fit; this module adds:

  • Gamma, Weibull, and power-law (Lomax) excitation kernels.
  • Sinusoidal-with-trend time-varying baseline ν(t).
  • Time-rescaling residuals (Brown et al. 2002) for goodness-of-fit
    via Kolmogorov-Smirnov + Q-Q plots.
  • Eight-way model comparison via AIC and an explicit
    Markovian-vs-non-Markovian likelihood-ratio surface.

The complete intensity function is

    λ(t) = ν(t) + ∫_0^{t-} g(t - s) dN_s

with kernel decomposition g(u) = η · g̃(u) where η ∈ (0, 1) is the
branching ratio (mean offspring per event) and g̃ is a probability
density on [0, ∞).  Stationarity requires η < 1.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from scipy import stats as sps
from scipy.optimize import minimize
from scipy.special import gammainc, gammaln  # noqa: F401 — used inline

from .fn._richresult import RichResult

PROJECT = Path(__file__).resolve().parents[5]
FIG_OUT = PROJECT / "data/manifest/outputs/figures/tps_hawkes_advanced"

KernelKind = Literal["exponential", "gamma", "weibull", "lomax"]
BaselineKind = Literal["constant", "sinusoidal"]

KERNELS: tuple[KernelKind, ...] = ("exponential", "gamma", "weibull", "lomax")
BASELINES: tuple[BaselineKind, ...] = ("constant", "sinusoidal")


def _ensure_dirs() -> None:
    FIG_OUT.mkdir(parents=True, exist_ok=True)


# ── Kernel densities and CDFs ───────────────────────────────────────


def _kernel_density(u: np.ndarray, kind: KernelKind,
                    psi: tuple[float, ...]) -> np.ndarray:
    """Normalised excitation density g̃(u) at lags u ≥ 0.

    Parameters
    ----------
    kind:
        One of ``"exponential"``, ``"gamma"``, ``"weibull"``, ``"lomax"``.
    psi:
        Kernel-specific parameter tuple — see ``_n_kernel_params``.
    """
    u = np.asarray(u, dtype=float)
    if kind == "exponential":
        (beta,) = psi
        return beta * np.exp(-beta * u)
    if kind == "gamma":
        alpha, beta = psi
        # density: β^α u^{α-1} e^{-β u} / Γ(α)
        log_d = (alpha * math.log(beta)
                 + (alpha - 1) * np.log(np.maximum(u, 1e-300))
                 - beta * u
                 - math.lgamma(alpha))
        return np.exp(log_d)
    if kind == "weibull":
        alpha, lam = psi
        x = u / lam
        return (alpha / lam) * np.power(np.maximum(x, 1e-300),
                                         alpha - 1) * np.exp(-np.power(x, alpha))
    if kind == "lomax":
        # Pareto-II / Lomax, shape α > 1, scale c > 0.
        # density: (α-1) c^{α-1} (u+c)^{-α}.  Computed in log-space to
        # avoid overflow when c^{α-1} or (u+c)^{α} blow up at large α.
        alpha, c = psi
        log_d = (math.log(alpha - 1)
                 + (alpha - 1) * math.log(c)
                 - alpha * np.log(u + c))
        return np.exp(log_d)
    raise ValueError(f"unknown kernel kind: {kind}")


def _kernel_cdf(u: np.ndarray, kind: KernelKind,
                psi: tuple[float, ...]) -> np.ndarray:
    """Cumulative ∫_0^u g̃(v) dv, used to evaluate ∫_{t_i}^T g(t-t_i) dt."""
    u = np.asarray(u, dtype=float)
    if kind == "exponential":
        (beta,) = psi
        return 1.0 - np.exp(-beta * u)
    if kind == "gamma":
        alpha, beta = psi
        return gammainc(alpha, beta * u)
    if kind == "weibull":
        alpha, lam = psi
        return 1.0 - np.exp(-np.power(u / lam, alpha))
    if kind == "lomax":
        alpha, c = psi
        return 1.0 - np.power(c / (u + c), alpha - 1)
    raise ValueError(f"unknown kernel kind: {kind}")


def _n_kernel_params(kind: KernelKind) -> int:
    return 1 if kind == "exponential" else 2


# ── Baselines and their integrals ───────────────────────────────────


def _baseline(t: np.ndarray, kind: BaselineKind, alpha: tuple[float, ...],
              T: float) -> np.ndarray:
    """Baseline intensity ν(t) on [0, T] in events / day.

    ``"constant"``  →  ν(t) = exp(a₀)  (one parameter, log-link).
    ``"sinusoidal"``→  ν(t) = exp(a₀ + a₁·(t/T) + a₂ sin(2πt/365)
                                       + a₃ cos(2πt/365)).
    """
    t = np.asarray(t, dtype=float)
    if kind == "constant":
        (a0,) = alpha
        return np.full_like(t, math.exp(a0))
    if kind == "sinusoidal":
        a0, a1, a2, a3 = alpha
        return np.exp(a0 + a1 * (t / max(T, 1.0))
                       + a2 * np.sin(2 * math.pi * t / 365.25)
                       + a3 * np.cos(2 * math.pi * t / 365.25))
    raise ValueError(f"unknown baseline kind: {kind}")


def _baseline_integral(T: float, kind: BaselineKind,
                       alpha: tuple[float, ...]) -> float:
    """∫_0^T ν(t) dt — closed form for constant, trapezoidal otherwise."""
    if kind == "constant":
        (a0,) = alpha
        return math.exp(a0) * T
    grid = np.linspace(0.0, T, max(64, int(T) + 1))
    vals = _baseline(grid, kind, alpha, T)
    # NumPy 2.x renamed np.trapz → np.trapezoid; fall back for older.
    trapezoid = getattr(np, "trapezoid", None) or np.trapz
    return float(trapezoid(vals, grid))


def _n_baseline_params(kind: BaselineKind) -> int:
    return 1 if kind == "constant" else 4


# ── Negative log-likelihood ─────────────────────────────────────────


def _split_theta(theta: np.ndarray, kernel_kind: KernelKind,
                 baseline_kind: BaselineKind
                 ) -> tuple[tuple[float, ...], float, tuple[float, ...]]:
    """θ → (α-baseline, η-branching-ratio, ψ-kernel)."""
    nb = _n_baseline_params(baseline_kind)
    nk = _n_kernel_params(kernel_kind)
    if theta.size != nb + 1 + nk:
        raise ValueError(f"expected {nb+1+nk} params, got {theta.size}")
    a = tuple(theta[:nb])
    eta = float(theta[nb])
    psi = tuple(theta[nb + 1:])
    return a, eta, psi


def _neg_loglik_general(theta: np.ndarray, t: np.ndarray, T: float,
                        kernel_kind: KernelKind,
                        baseline_kind: BaselineKind) -> float:
    """Negative log-likelihood for a general non-stationary Hawkes process.

    Uses the form

        ℓ(θ) = Σᵢ log λ(tᵢ) − ∫_0^T λ(s) ds.

    With kernel decomposition g = η · g̃ and CDF F̃,

        ∫_0^T λ(s) ds = ∫_0^T ν(s) ds + η Σᵢ F̃(T − tᵢ).

    The intensity at events is computed by direct O(n²) summation —
    correct for n ≲ 5 000.  For n ≫ 10⁴ a recursive update is needed
    only in the exponential-kernel case (other kernels lack the
    memorylessness required for O(n) recursion).
    """
    nb = _n_baseline_params(baseline_kind)
    nk = _n_kernel_params(kernel_kind)
    a = tuple(theta[:nb])
    eta = float(theta[nb])
    psi = tuple(theta[nb + 1:])

    # box constraints
    if eta <= 1e-6 or eta >= 0.999:
        return 1e12
    if any(p <= 1e-6 for p in psi):
        return 1e12
    if kernel_kind == "lomax" and psi[0] <= 1.001:
        return 1e12  # need α > 1 for finite mean

    # Σᵢ log λ(tᵢ)
    n = t.size
    nu_at_t = _baseline(t, baseline_kind, a, T)
    log_sum = 0.0
    for i in range(n):
        if i == 0:
            lam_i = nu_at_t[0]
        else:
            lags = t[i] - t[:i]
            lam_i = nu_at_t[i] + eta * np.sum(_kernel_density(lags,
                                                                kernel_kind,
                                                                psi))
        if lam_i <= 0:
            return 1e12
        log_sum += math.log(lam_i)

    # ∫_0^T λ(s) ds
    integral = _baseline_integral(T, baseline_kind, a) \
        + eta * float(np.sum(_kernel_cdf(T - t, kernel_kind, psi)))
    return -(log_sum - integral)


# ── Initial-guess heuristics ────────────────────────────────────────


def _x0(kernel_kind: KernelKind, baseline_kind: BaselineKind,
        n: int, T: float, mean_dt: float) -> np.ndarray:
    rate = max(n / T, 1e-3)
    if baseline_kind == "constant":
        a = [math.log(rate * 0.6)]
    else:
        a = [math.log(rate * 0.6), 0.0, 0.0, 0.0]
    eta = [0.4]
    if kernel_kind == "exponential":
        psi = [1.0 / max(mean_dt, 1e-3)]
    elif kernel_kind == "gamma":
        psi = [1.5, 1.0 / max(mean_dt, 1e-3)]
    elif kernel_kind == "weibull":
        psi = [1.5, max(mean_dt, 1e-3) * 1.2]
    else:  # lomax
        psi = [2.5, max(mean_dt, 1e-3) * 5.0]
    return np.array(a + eta + psi, dtype=float)


# ── Public fit + GoF ────────────────────────────────────────────────


def fit_hawkes_general(t: np.ndarray, T: float,
                        kernel_kind: KernelKind = "exponential",
                        baseline_kind: BaselineKind = "constant"
                        ) -> dict:
    """MLE of a non-stationary Hawkes process.

    Returns a dict with ``theta``, ``nll``, ``aic``, ``bic``,
    ``branching_ratio``, ``baseline_params``, ``kernel_params``,
    and the time-rescaling KS statistic.
    """
    n = int(t.size)
    if n < 50:
        raise ValueError(f"too few events ({n}) for non-stationary fit")
    mean_dt = float(np.mean(np.diff(t))) if n > 1 else 1.0
    x0 = _x0(kernel_kind, baseline_kind, n, T, mean_dt)

    res = minimize(_neg_loglik_general, x0,
                    args=(t, T, kernel_kind, baseline_kind),
                    method="Nelder-Mead",
                    options={"xatol": 1e-4, "fatol": 1e-3,
                             "maxiter": 4000, "adaptive": True})
    theta = res.x
    nll = float(res.fun)
    a, eta, psi = _split_theta(theta, kernel_kind, baseline_kind)
    k = theta.size
    aic = 2 * k + 2 * nll
    bic = k * math.log(n) + 2 * nll

    # time-rescaling residuals (Brown et al. 2002)
    u = _time_rescaling_residuals(theta, t, T, kernel_kind, baseline_kind)
    ks = sps.kstest(u, "uniform")
    return {
        "theta": theta.tolist(),
        "baseline_params": list(a),
        "branching_ratio": eta,
        "kernel_params": list(psi),
        "nll": nll,
        "aic": aic,
        "bic": bic,
        "n": n,
        "T_days": float(T),
        "k_params": int(k),
        "ks_stat": float(ks.statistic),
        "ks_pvalue": float(ks.pvalue),
        "rescaled_uniforms": u.tolist()[:1000],
        "kernel_kind": kernel_kind,
        "baseline_kind": baseline_kind,
        "converged": bool(res.success),
    }


def _time_rescaling_residuals(theta: np.ndarray, t: np.ndarray, T: float,
                               kernel_kind: KernelKind,
                               baseline_kind: BaselineKind) -> np.ndarray:
    """Return U_i = 1 - exp(-(Λ(t_i) - Λ(t_{i-1}))) ∈ [0, 1].

    Under correct specification U_i ∼ Uniform(0,1) iid (Brown et al.
    Neural Comput. 14:325-346, 2002).
    """
    nb = _n_baseline_params(baseline_kind)
    a = tuple(theta[:nb])
    eta = float(theta[nb])
    psi = tuple(theta[nb + 1:])

    n = t.size
    # Λ(t_i) = ∫_0^{t_i} ν(s) ds + η Σ_{j<i} F̃(t_i - t_j)
    bl_grid = np.linspace(0.0, float(T), max(256, int(T) + 1))
    bl_vals = _baseline(bl_grid, baseline_kind, a, T)
    cum_baseline = np.concatenate(([0.0], np.cumsum(0.5 *
                                                     (bl_vals[1:] + bl_vals[:-1])
                                                     * np.diff(bl_grid))))

    def Lambda_at(ti: float) -> float:
        bl = float(np.interp(ti, bl_grid, cum_baseline))
        prior = t[t < ti]
        excite = eta * float(np.sum(_kernel_cdf(ti - prior,
                                                  kernel_kind, psi)))
        return bl + excite

    inc = np.empty(n)
    prev = 0.0
    for i, ti in enumerate(t):
        cur = Lambda_at(float(ti))
        inc[i] = max(cur - prev, 1e-12)
        prev = cur
    return 1.0 - np.exp(-inc)


# ── Pretty wrappers (RichResult) ────────────────────────────────────


def _events_to_days(df: pd.DataFrame, max_n: int) -> tuple[np.ndarray, float]:
    """Convert TPS event timestamps to a clean days-since-t0 vector.

    TPS open-data is daily-resolution (OCC_DATE has no hour), which
    creates many ties.  We jitter ties uniformly within the day so the
    kernel-at-zero density does not blow up; this is a standard
    workaround in temporal point-process estimation when only the day
    is observed.  Sub-sample to ``max_n`` for tractable O(n²) MLE.
    """
    from .tps_stochastic import _date_series

    dt = _date_series(df)
    if dt.size > max_n:
        dt = dt.sample(n=max_n, random_state=42).sort_values()
    t0 = dt.min()
    t = (dt - t0).dt.total_seconds().to_numpy() / 86400.0
    rng = np.random.default_rng(42)
    # Add U(0,1)-day jitter to break OCC_DATE ties; the original event
    # ordering by day is preserved because the jitter is at most 1 day.
    t = t + rng.random(t.size)
    t.sort()
    return t, float(t[-1])


def hawkes_advanced_fit(df: pd.DataFrame, *,
                         kernel: KernelKind = "gamma",
                         baseline: BaselineKind = "sinusoidal",
                         ds_name: str = "?",
                         max_n: int = 5000) -> RichResult:
    """Fit a single (kernel, baseline) combination with figures.

    A sibling to ``morie.tps_stochastic.hawkes_temporal_fit`` which is
    locked to the (exponential, constant) Markovian special case.
    """
    from .tps_stochastic import _try_savefig

    if "OCC_DATE" not in df.columns and "REPORT_DATE" not in df.columns:
        return RichResult(title=f"Hawkes-{kernel}/{baseline} — {ds_name}",
                          warnings=["no OCC_DATE or REPORT_DATE column"])
    t, T = _events_to_days(df, max_n)
    if t.size < 100:
        return RichResult(title=f"Hawkes-{kernel}/{baseline} — {ds_name}",
                          warnings=[f"only {t.size} timestamps"])

    result = fit_hawkes_general(t, T, kernel_kind=kernel,
                                  baseline_kind=baseline)

    # QQ figure
    fig_path = None
    try:
        import matplotlib.pyplot as plt
        u = np.array(result["rescaled_uniforms"])
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        sps.probplot(u, dist="uniform", plot=ax[0])
        ax[0].set_title(f"{ds_name} — Q-Q vs Uniform "
                         f"(KS p = {result['ks_pvalue']:.3f})")
        # density vs empirical
        ax[1].hist(u, bins=30, color="#3584e4", alpha=0.7,
                    density=True, label="empirical U_i")
        ax[1].axhline(1.0, color="#e66100", ls="--", label="Uniform(0,1)")
        ax[1].set_xlim(0, 1)
        ax[1].legend()
        ax[1].set_title(f"{kernel}/{baseline} kernel")
        fig.suptitle(f"Time-rescaling residuals — {ds_name}")
        plt.tight_layout()
        fig_path = _try_savefig(
            f"hawkes_qq_{kernel}_{baseline}_{ds_name}.png", fig)
    except Exception:
        pass

    a = result["baseline_params"]
    psi = result["kernel_params"]
    eta = result["branching_ratio"]
    summary = [
        ("Events fitted", result["n"]),
        ("Time window (days)", round(result["T_days"], 1)),
        ("Kernel", kernel),
        ("Baseline", baseline),
        ("η (branching ratio)", round(eta, 4)),
        ("Stationary?", "Yes (η<1)" if eta < 1 else "EXPLOSIVE"),
        ("Kernel params (ψ)", [round(x, 4) for x in psi]),
        ("Baseline params (α)", [round(x, 4) for x in a]),
        ("Negative log-likelihood", round(result["nll"], 1)),
        ("AIC", round(result["aic"], 1)),
        ("BIC", round(result["bic"], 1)),
        ("Time-rescaling KS stat", round(result["ks_stat"], 4)),
        ("Time-rescaling KS p-value", round(result["ks_pvalue"], 4)),
    ]
    interp = (
        f"Branching ratio η = {eta:.3f} → mean {eta:.2f} offspring "
        f"per event. "
        + ("Process is stationary (η < 1)."
           if eta < 1 else "Process is EXPLOSIVE (η ≥ 1).")
        + " Time-rescaling KS p = "
        + ("strong fit; "
           if result["ks_pvalue"] >= 0.05 else "fit is rejected; ")
        + f"residuals {'consistent with' if result['ks_pvalue'] >= 0.05 else 'depart from'} Uniform(0,1)."
    )
    payload = dict(result)
    payload["figure_path"] = fig_path
    return RichResult(
        title=f"Hawkes [{kernel} kernel, {baseline} baseline] — {ds_name}",
        summary_lines=summary,
        interpretation=interp,
        payload=payload,
    )


def compare_hawkes_kernels(df: pd.DataFrame, *,
                             ds_name: str = "?",
                             max_n: int = 4000,
                             baselines: tuple[BaselineKind, ...] = BASELINES,
                             kernels: tuple[KernelKind, ...] = KERNELS
                             ) -> RichResult:
    """Fit every (kernel, baseline) combination and rank by AIC.

    Mirrors Section 5 (numerical examples) of Kwan-Chen-Dunsmuir 2024:
    the Markovian classical Hawkes is the (exponential, constant) row;
    the non-Markovian non-stationary models are everything else.
    """
    if "OCC_DATE" not in df.columns and "REPORT_DATE" not in df.columns:
        return RichResult(title=f"Hawkes comparison — {ds_name}",
                          warnings=["no OCC_DATE or REPORT_DATE column"])
    t, T = _events_to_days(df, max_n)
    if t.size < 100:
        return RichResult(title=f"Hawkes comparison — {ds_name}",
                          warnings=[f"only {t.size} timestamps"])

    rows = []
    for k in kernels:
        for b in baselines:
            try:
                fit = fit_hawkes_general(t, T,
                                          kernel_kind=k, baseline_kind=b)
                rows.append({
                    "kernel": k, "baseline": b,
                    "k_params": fit["k_params"],
                    "nll": round(fit["nll"], 1),
                    "aic": round(fit["aic"], 1),
                    "bic": round(fit["bic"], 1),
                    "branching_ratio": round(fit["branching_ratio"], 3),
                    "ks_pvalue": round(fit["ks_pvalue"], 4),
                    "markovian": (k == "exponential"),
                    "stationary_baseline": (b == "constant"),
                })
            except Exception as exc:  # noqa: BLE001
                rows.append({
                    "kernel": k, "baseline": b, "error": str(exc),
                })

    fitted = [r for r in rows if "error" not in r]
    fitted.sort(key=lambda r: r["aic"])
    best = fitted[0] if fitted else None
    summary = [("Combinations fitted", len(fitted)),
                ("Combinations failed", len(rows) - len(fitted))]
    if best is not None:
        summary += [
            ("Best (lowest AIC)",
              f"{best['kernel']} / {best['baseline']}"),
            ("Δ AIC vs Markovian classical",
              round(
                  next((r["aic"] for r in fitted
                        if r["kernel"] == "exponential"
                        and r["baseline"] == "constant"),
                       float("nan")) - best["aic"], 1)),
        ]
    interp = (
        "Comparison of the eight (kernel × baseline) combinations. "
        "The classical Markovian Hawkes (exponential, constant) is "
        "the special case where the bivariate process (N_t, λ_t) is "
        "Markov. All non-exponential kernels and the time-varying "
        "sinusoidal baseline yield non-Markovian intensity processes; "
        "their large-sample theory is the contribution of "
        "Kwan-Chen-Dunsmuir (2024)."
    )
    return RichResult(
        title=f"Markovian vs non-Markovian Hawkes — {ds_name}",
        summary_lines=summary,
        interpretation=interp,
        payload={"rows": rows, "best": best},
    )


def hawkes_markovian_vs_nonmarkovian(df: pd.DataFrame, *,
                                       ds_name: str = "?",
                                       max_n: int = 4000) -> RichResult:
    """Focused 2-way comparison: classical exp/const vs gamma/sinusoidal.

    The two endpoints of the Kwan-Chen-Dunsmuir framework — quickest to
    run on the dashboard.
    """
    return compare_hawkes_kernels(
        df, ds_name=ds_name, max_n=max_n,
        kernels=("exponential", "gamma"),
        baselines=("constant", "sinusoidal"),
    )
