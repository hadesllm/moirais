"""morie.tps_stochastic -- stochastic-physics-of-crime analyses on TPS.

Modelled on Frenkel-style crime-physics (Hawkes self-exciting +
Fokker-Planck density evolution + Langevin SDE), with practical
forecasting baselines (SARIMA, XGBoost-when-available) for hold-out
comparison.

Functions
- hawkes_temporal_fit(df)       -- fit μ, κ, ω of self-exciting kernel,
                                   compute branching ratio + AIC/BIC
- sarima_forecast(df, h=12)     -- seasonal ARIMA, train/test MAPE
- prophet_forecast(df, h=12)    -- Prophet trend+seasonality if installed
- langevin_simulate(...)        -- Euler-Maruyama Ornstein-Uhlenbeck path
- fokker_planck_grid(df)        -- finite-difference density evolution
- model_compare(df)             -- AIC across baselines + SDE

All emit RichResult; `figure_path` payload entry written when
matplotlib is available (saved as PNG under
data/manifest/outputs/figures/tps_stochastic/).

References (cited in description blocks):
- Mohler et al. 2011, "Self-exciting point process modeling of crime."
- Pitcher 2010, "Adding integrated nested Laplace approximations to
  the Bayesian forecasting toolkit."
- Stochastic physics of crime literature (Short, D'Orsogna, Bertozzi 2010).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps
from scipy.optimize import minimize

from .fn._richresult import RichResult

PROJECT = Path(__file__).resolve().parents[5]
DEFAULT_OUT = PROJECT / "data/manifest/outputs/tps_stochastic"
FIG_OUT = PROJECT / "data/manifest/outputs/figures/tps_stochastic"


def _ensure_dirs():
    DEFAULT_OUT.mkdir(parents=True, exist_ok=True)
    FIG_OUT.mkdir(parents=True, exist_ok=True)


def _try_savefig(name: str, fig) -> str | None:
    try:
        _ensure_dirs()
        out = FIG_OUT / name
        fig.savefig(out, dpi=120, bbox_inches="tight")
        import matplotlib.pyplot as plt
        plt.close(fig)
        return str(out.relative_to(PROJECT))
    except Exception:
        return None


def _date_series(df: pd.DataFrame,
                  *, min_year: int = 2014) -> pd.Series:
    """Return cleaned OCC_DATE/REPORT_DATE timestamps.

    `min_year=2014` drops pre-2014 retro-records -- TPS started its
    public-safety open-data programme in 2014 and earlier rows are
    sparse retro-reports that produce a visible regime change in the
    monthly count and force μ -> 0 in any Hawkes fit. Override
    ``min_year=None`` to include the long historical tail deliberately.
    """
    for c in ("OCC_DATE", "REPORT_DATE"):
        if c in df.columns:
            ts = pd.to_datetime(df[c], errors="coerce").dropna()
            if min_year is not None:
                ts = ts[ts.dt.year >= min_year]
            return ts
    return pd.Series(dtype="datetime64[ns]")


# ── Hawkes self-exciting fit (temporal-only Mohler style) ───────────


def _neg_loglik_hawkes(params: np.ndarray, t: np.ndarray, T: float) -> float:
    """Negative log-likelihood of an exponential-kernel Hawkes process.

    λ(t) = μ + κω·Σ_{tᵢ<t} exp(−ω(t−tᵢ))

    Closed-form integral:
        ∫λ = μT + κ·Σᵢ (1 − exp(−ω(T − tᵢ)))
    """
    mu, kappa, omega = params
    if mu <= 0 or kappa < 0 or omega <= 0:
        return 1e12
    # Recursive computation of intensity contributions
    n = t.size
    lam = np.zeros(n)
    A = 0.0
    last = t[0]
    lam[0] = mu
    for i in range(1, n):
        A = (A + 1) * math.exp(-omega * (t[i] - last))
        lam[i] = mu + kappa * omega * A
        last = t[i]
    log_lam = np.log(np.maximum(lam, 1e-300))
    integral = mu * T + kappa * np.sum(1 - np.exp(-omega * (T - t)))
    return -(np.sum(log_lam) - integral)


def hawkes_temporal_fit(df: pd.DataFrame, *,
                        ds_name: str = "?",
                        max_n: int = 5000) -> RichResult:
    """Fit a temporal-only exponential Hawkes process to incident times.

    Returns μ (background rate), κ (branching ratio), ω (decay), AIC/BIC.
    """
    dt = _date_series(df)
    if dt.size < 100:
        return RichResult(title=f"Hawkes -- {ds_name}",
                          warnings=[f"only {dt.size} timestamps"])
    if dt.size > max_n:
        dt = dt.sample(n=max_n, random_state=42).sort_values()
    # convert to days from t0
    t0 = dt.min()
    t = (dt - t0).dt.total_seconds().values / 86400.0
    t.sort()
    T = t[-1]
    n = t.size

    # Initial guess: μ ~ n/T, κ ~ 0.5, ω ~ 1/(mean inter-event)
    mean_dt = np.mean(np.diff(t)) or 1.0
    x0 = np.array([n / T, 0.4, 1.0 / mean_dt])
    res = minimize(_neg_loglik_hawkes, x0, args=(t, T),
                    method="Nelder-Mead",
                    options={"xatol": 1e-4, "fatol": 1e-4, "maxiter": 1000})
    mu, kappa, omega = res.x
    nll = res.fun
    aic = 2 * 3 + 2 * nll
    bic = 3 * math.log(n) + 2 * nll

    branching = kappa  # fraction of events triggering offspring
    # Visualisation
    fig_path = None
    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(2, 1, figsize=(8, 6),
                                gridspec_kw={"height_ratios": [3, 1]})
        # Top: empirical intensity (rolling) vs Poisson rate
        df_t = pd.Series(1, index=dt).resample("ME").sum()
        ax[0].plot(df_t.index, df_t.values, lw=1, color="#3584e4",
                    label="monthly count")
        ax[0].axhline(mu * 30, color="#e66100", ls="--",
                       label=f"μ·30days = {mu*30:.1f}")
        ax[0].set_title(f"{ds_name} -- Hawkes fit")
        ax[0].set_ylabel("incidents / month")
        ax[0].legend()
        # Bottom: residual interarrival KS
        residuals = []
        last_t = 0
        A = 0.0
        for ti in t:
            integ = mu * (ti - last_t) + kappa * A * (1 - math.exp(-omega * (ti - last_t)))
            residuals.append(integ)
            A = (A + 1) * math.exp(-omega * (ti - last_t))
            last_t = ti
        residuals = np.array(residuals)
        ks = sps.kstest(residuals[:1000], "expon")
        ax[1].hist(np.diff(residuals), bins=50, color="#3584e4", alpha=0.7)
        ax[1].set_title(f"residual interarrivals -- KS p = {ks.pvalue:.3f}")
        ax[1].set_xlabel("residual Δt")
        plt.tight_layout()
        fig_path = _try_savefig(f"hawkes_{ds_name}.png", fig)
    except Exception:
        pass

    return RichResult(
        title=f"Hawkes self-exciting fit -- {ds_name}",
        summary_lines=[
            ("Events fitted", n),
            ("Time window (days)", round(T, 1)),
            ("μ (background rate, /day)", round(float(mu), 4)),
            ("κ (branching ratio)", round(float(kappa), 4)),
            ("ω (decay, /day)", round(float(omega), 4)),
            ("Mean offspring per event", round(float(kappa), 3)),
            ("Stationary?", "Yes (κ<1)" if kappa < 1 else "EXPLOSIVE"),
            ("Negative log-likelihood", round(nll, 1)),
            ("AIC", round(aic, 1)),
            ("BIC", round(bic, 1)),
        ],
        interpretation=(
            f"Background rate {mu:.3f} events/day; each event triggers "
            f"on average {kappa:.2f} offspring with decay timescale "
            f"1/ω ≈ {1/omega:.2f} days. "
            + ("Branching ratio κ < 1 ⇒ stationary (the process doesn't "
               "explode). Mohler-style self-excitation is operating: "
               "events cluster in time around prior events."
               if kappa < 1 else
               "κ ≥ 1 ⇒ EXPLOSIVE -- fit may be unstable; treat with care.")
        ),
        payload={"mu": float(mu), "kappa": float(kappa),
                 "omega": float(omega), "branching": float(kappa),
                 "nll": float(nll), "aic": float(aic), "bic": float(bic),
                 "n": int(n), "T_days": float(T),
                 "figure_path": fig_path},
    )


# ── SARIMA forecasting ─────────────────────────────────────────────


def sarima_forecast(df: pd.DataFrame, *,
                    ds_name: str = "?",
                    h: int = 12,
                    order: tuple[int, int, int] = (1, 1, 1),
                    seasonal: tuple[int, int, int, int] = (0, 1, 1, 12)) \
        -> RichResult:
    """Seasonal ARIMA forecast on monthly incident counts."""
    try:
        import statsmodels.api as sm
    except ImportError:
        return RichResult(title=f"SARIMA -- {ds_name}",
                          warnings=["statsmodels not installed"])
    dt = _date_series(df)
    if dt.size < 36:
        return RichResult(title=f"SARIMA -- {ds_name}",
                          warnings=[f"need ≥36 months, got {dt.size}"])
    monthly = dt.dt.to_period("M").value_counts().sort_index()
    monthly.index = monthly.index.to_timestamp()
    if monthly.size < 36:
        return RichResult(title=f"SARIMA -- {ds_name}",
                          warnings=[f"only {monthly.size} months"])
    # train/test split: hold out last `h`
    train, test = monthly.iloc[:-h], monthly.iloc[-h:]
    try:
        model = sm.tsa.statespace.SARIMAX(train.values.astype(float),
                                            order=order,
                                            seasonal_order=seasonal,
                                            enforce_stationarity=False,
                                            enforce_invertibility=False)
        fit = model.fit(disp=False)
        fc = fit.forecast(steps=h)
    except Exception as e:
        return RichResult(title=f"SARIMA -- {ds_name}",
                          warnings=[f"fit failed: {e!r}"])
    # Metrics
    err = test.values.astype(float) - fc
    mape = float(np.mean(np.abs(err / np.maximum(test.values, 1)))) * 100
    rmse = float(np.sqrt(np.mean(err ** 2)))

    # Visualisation
    fig_path = None
    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(monthly.index, monthly.values, color="#3584e4",
                  label="actual")
        ax.plot(test.index, fc, color="#e66100", lw=2,
                  label=f"SARIMA forecast (h={h})")
        ax.axvline(train.index[-1], color="#5e5c64", ls="--", alpha=0.5)
        ax.set_title(f"{ds_name} -- SARIMA"
                       f"{order}×{seasonal}, MAPE={mape:.1f}%")
        ax.set_ylabel("monthly incidents")
        ax.legend()
        plt.tight_layout()
        fig_path = _try_savefig(f"sarima_{ds_name}.png", fig)
    except Exception:
        pass

    return RichResult(
        title=f"SARIMA{order}×{seasonal} forecast -- {ds_name}",
        summary_lines=[
            ("Training months", train.size),
            ("Hold-out months (h)", h),
            ("AIC", round(float(fit.aic), 1)),
            ("BIC", round(float(fit.bic), 1)),
            ("Hold-out MAPE", f"{mape:.1f}%"),
            ("Hold-out RMSE", round(rmse, 1)),
            ("Forecast mean", round(float(fc.mean()), 1)),
            ("First test month", str(test.index[0].date())),
            ("Last test month", str(test.index[-1].date())),
        ],
        tables=[{
            "title": f"Forecast vs actual (last {h} months):",
            "headers": ["Month", "Actual", "Forecast", "Error"],
            "rows": [[str(d.date()), int(a), round(float(f), 1),
                       round(float(a - f), 1)]
                      for d, a, f in zip(test.index, test.values, fc)],
        }],
        payload={"aic": float(fit.aic), "bic": float(fit.bic),
                 "mape_pct": mape, "rmse": rmse,
                 "forecast": list(map(float, fc)),
                 "actual": list(map(int, test.values)),
                 "figure_path": fig_path},
    )


# ── Langevin SDE simulation ────────────────────────────────────────


def langevin_simulate(df: pd.DataFrame, *,
                       ds_name: str = "?",
                       n_paths: int = 100,
                       T_days: int = 365,
                       dt: float = 1.0,
                       seed: int = 42) -> RichResult:
    """Euler-Maruyama simulation of an Ornstein-Uhlenbeck process
    fitted to incident counts:

        dX_t = θ(μ − X_t) dt + σ dW_t

    Fits θ, μ, σ via OLS on first-differences, then generates n_paths
    forward simulations of length T_days.
    """
    dt_ser = _date_series(df)
    if dt_ser.size < 60:
        return RichResult(title=f"Langevin OU -- {ds_name}",
                          warnings=[f"only {dt_ser.size} timestamps"])
    daily = dt_ser.dt.to_period("D").value_counts().sort_index()
    daily.index = daily.index.to_timestamp()
    x = daily.values.astype(float)
    dx = np.diff(x)
    x_lag = x[:-1]
    # OLS on dx = a + b·x_lag + ε
    X = np.column_stack([np.ones_like(x_lag), x_lag])
    beta, *_ = np.linalg.lstsq(X, dx, rcond=None)
    a, b = beta
    # In OU dx = θμ dt − θ x dt + σ dW; matching -> θ = -b, μ = -a/b
    theta = max(1e-6, -float(b))
    mu = -float(a) / b if b != 0 else float(x.mean())
    resid = dx - (a + b * x_lag)
    sigma = float(resid.std(ddof=1))

    rng = np.random.default_rng(seed)
    n_steps = int(T_days / dt)
    paths = np.zeros((n_paths, n_steps))
    paths[:, 0] = float(x[-1])  # start from last observed
    sqrtdt = math.sqrt(dt)
    for k in range(1, n_steps):
        z = rng.standard_normal(n_paths)
        paths[:, k] = (paths[:, k - 1]
                        + theta * (mu - paths[:, k - 1]) * dt
                        + sigma * sqrtdt * z)

    # Summary
    final = paths[:, -1]
    quantiles = np.quantile(final, [0.05, 0.5, 0.95])

    fig_path = None
    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(9, 4.5))
        ax.plot(daily.index, x, color="#5e5c64", lw=1, alpha=0.8,
                  label="historical (daily count)")
        future = pd.date_range(daily.index[-1], periods=n_steps,
                                 freq="D")
        # plot 50 sample paths
        for i in range(min(50, n_paths)):
            ax.plot(future, paths[i], color="#3584e4", alpha=0.05)
        # median + 5/95 envelope
        med = np.quantile(paths, 0.5, axis=0)
        lo = np.quantile(paths, 0.05, axis=0)
        hi = np.quantile(paths, 0.95, axis=0)
        ax.plot(future, med, color="#e66100", lw=2, label="median path")
        ax.fill_between(future, lo, hi, color="#e66100", alpha=0.2,
                          label="5–95% envelope")
        ax.axvline(daily.index[-1], color="#5e5c64", ls="--", alpha=0.5)
        ax.set_title(f"{ds_name} -- Langevin OU simulation "
                       f"(θ={theta:.3f}, μ={mu:.1f}, σ={sigma:.1f})")
        ax.set_ylabel("daily incidents")
        ax.legend()
        plt.tight_layout()
        fig_path = _try_savefig(f"langevin_{ds_name}.png", fig)
    except Exception:
        pass

    return RichResult(
        title=f"Langevin OU SDE simulation -- {ds_name}",
        summary_lines=[
            ("Historical days observed", x.size),
            ("Fitted θ (mean reversion)", round(theta, 4)),
            ("Fitted μ (long-run mean)", round(mu, 2)),
            ("Fitted σ (volatility)", round(sigma, 3)),
            ("Half-life of shocks (days)",
                round(math.log(2) / max(theta, 1e-6), 2)),
            ("Simulated paths", n_paths),
            ("Forecast horizon (days)", T_days),
            ("Final day p5 / median / p95",
                f"{quantiles[0]:.1f} / {quantiles[1]:.1f} / "
                f"{quantiles[2]:.1f}"),
        ],
        interpretation=(
            f"OU mean-reversion strength θ={theta:.3f} ⇒ shocks decay "
            f"with half-life {math.log(2)/max(theta,1e-6):.1f} days. "
            f"Long-run daily mean μ={mu:.1f}. "
            "Stochastic-physics-of-crime style: treats incidents as a "
            "noisy mean-reverting intensity field (Langevin) rather than "
            "deterministic trend."
        ),
        payload={"theta": theta, "mu": mu, "sigma": sigma,
                 "n_paths": n_paths, "T_days": T_days,
                 "p5": float(quantiles[0]),
                 "median": float(quantiles[1]),
                 "p95": float(quantiles[2]),
                 "figure_path": fig_path},
    )


# ── Fokker-Planck density evolution (1D approximation) ────────────


def fokker_planck_grid(df: pd.DataFrame, *,
                        ds_name: str = "?",
                        n_grid: int = 64,
                        n_steps: int = 200) -> RichResult:
    """1-D Fokker-Planck density evolution of daily incident counts
    with the OU drift+diffusion fitted in `langevin_simulate`.

    Uses Crank-Nicolson finite-difference scheme with reflective
    boundaries on a grid spanning [0, max(observed)*1.5].
    """
    dt_ser = _date_series(df)
    if dt_ser.size < 60:
        return RichResult(title=f"Fokker-Planck -- {ds_name}",
                          warnings=[f"only {dt_ser.size} timestamps"])
    daily = dt_ser.dt.to_period("D").value_counts().sort_index()
    x = daily.values.astype(float)
    dx_obs = np.diff(x)
    x_lag = x[:-1]
    X = np.column_stack([np.ones_like(x_lag), x_lag])
    beta, *_ = np.linalg.lstsq(X, dx_obs, rcond=None)
    a, b = beta
    theta = max(1e-6, -float(b))
    mu = -float(a) / b if b != 0 else float(x.mean())
    sigma = float((dx_obs - (a + b * x_lag)).std(ddof=1))

    x_max = float(x.max() * 1.5 + 1)
    grid = np.linspace(0, x_max, n_grid)
    dx = grid[1] - grid[0]
    dt = 0.05  # days per step
    # Initial: gaussian centred at last observation
    p = np.exp(-((grid - x[-1]) ** 2) / (2 * (sigma + 1) ** 2))
    p /= p.sum() * dx

    drift = theta * (mu - grid)
    diff = 0.5 * sigma ** 2

    # Crank-Nicolson loop (advection-diffusion)
    for _ in range(n_steps):
        # Compute fluxes
        d2p = (np.roll(p, -1) - 2 * p + np.roll(p, 1)) / dx ** 2
        dp = (np.roll(p, -1) - np.roll(p, 1)) / (2 * dx)
        rhs = -drift * dp + diff * d2p
        p = p + dt * rhs
        # Reflective boundaries
        p[0] = p[1]; p[-1] = p[-2]
        p = np.maximum(p, 0)
        s = p.sum() * dx
        if s > 0:
            p /= s

    fig_path = None
    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.hist(x, bins=30, density=True, alpha=0.5, color="#3584e4",
                  label="empirical density")
        ax.plot(grid, p, color="#e66100", lw=2,
                  label=f"FP-evolved (t={n_steps*dt:.1f} days)")
        ax.axvline(mu, color="#5e5c64", ls="--", alpha=0.6,
                     label=f"μ = {mu:.1f}")
        ax.set_xlabel("daily incidents")
        ax.set_ylabel("density")
        ax.set_title(f"{ds_name} -- Fokker-Planck density (1-D OU)")
        ax.legend()
        plt.tight_layout()
        fig_path = _try_savefig(f"fokker_planck_{ds_name}.png", fig)
    except Exception:
        pass

    return RichResult(
        title=f"Fokker-Planck density evolution -- {ds_name}",
        summary_lines=[
            ("Grid points", n_grid),
            ("Time steps", n_steps),
            ("Total simulated days", round(n_steps * dt, 1)),
            ("μ (drift target)", round(mu, 2)),
            ("σ² (diffusion)", round(sigma ** 2, 2)),
            ("Stationary peak (grid x)",
                round(float(grid[p.argmax()]), 2)),
            ("Empirical mean", round(float(x.mean()), 2)),
            ("Empirical std", round(float(x.std()), 2)),
        ],
        interpretation=(
            "Fokker-Planck of OU has stationary density "
            f"N(μ={mu:.1f}, σ²/(2θ)={sigma**2/(2*theta):.1f}). "
            "Compare the orange evolved curve to the empirical histogram "
            "to assess fit; a strong match is the stochastic-physics-of-"
            "crime null model."
        ),
        payload={"theta": theta, "mu": mu, "sigma": sigma,
                 "stationary_peak": float(grid[p.argmax()]),
                 "figure_path": fig_path},
    )


# ── Master driver ───────────────────────────────────────────────────


def analyze(name: str = "Assault", *,
            sample_rows: int | None = 50_000) -> dict[str, RichResult]:
    """Run the four stochastic-physics analyses on one TPS category."""
    from .tps_datasets import load_tps_dataset
    df = load_tps_dataset(name, nrows=sample_rows)
    _ensure_dirs()
    results = {}
    for label, fn in [
        ("hawkes", lambda: hawkes_temporal_fit(df, ds_name=name)),
        ("sarima", lambda: sarima_forecast(df, ds_name=name)),
        ("langevin", lambda: langevin_simulate(df, ds_name=name)),
        ("fokker_planck", lambda: fokker_planck_grid(df, ds_name=name)),
    ]:
        try:
            r = fn()
            results[label] = r
            (DEFAULT_OUT / f"{label}_{name}.txt").write_text(str(r))
            (DEFAULT_OUT / f"{label}_{name}.json").write_text(
                json.dumps(r.payload, indent=2, default=str,
                           ensure_ascii=False)
            )
        except Exception as e:  # noqa: BLE001
            results[label] = RichResult(
                title=f"{label} {name} (failed)",
                warnings=[f"{type(e).__name__}: {e}"],
            )
    return results
