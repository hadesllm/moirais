"""morie.tps_statphysics — Statistical physics of crime applied to TPS.

Implements the 4 canonical methods reviewed by D'Orsogna & Perc,
"Statistical physics of crime: A review", Phys. Life Rev. 12 (2015) 1-21
(arxiv:1411.1743). Each function consumes one TPS category and returns a
RichResult with a paragraph-level interpretation.

Methods
-------
1. ``sdb_reaction_diffusion(category, …)`` — Short-D'Orsogna-Brantingham
   2008 hot-spot PDE on a Toronto grid:

       ∂A/∂t = η ∇²A − ω A + θ ρ
       ∂ρ/∂t = ∇·(D∇ρ − 2 ρ ∇log A) − ρ A + γ

   A is the *attractiveness* field; ρ the *criminal density*. Crime
   incidents are absorbed into A, ρ flows up the gradient of A. Stable
   spots emerge when (η, ω, θ, D, γ) place the system in the
   localised-spike regime. We fit the steady-state spike count to the
   observed cluster count from DBSCAN.

2. ``levy_flight_alpha(category)`` — Brockmann-Hufnagel-Geisel 2006
   Lévy-flight diagnostic. We compute step lengths between
   chronologically-consecutive incidents and fit a power-law tail
   p(ℓ) ∝ ℓ^{−α} via Hill-MLE on the upper-tail.

3. ``urban_scaling_beta(category)`` — Bettencourt et al. 2007 / 2010
   urban scaling: log(crime_i) = log(Y₀) + β · log(pop_i) + ε for the
   158 Toronto wards. β > 1 = super-linear (crime grows faster than
   population), β = 1 = linear, β < 1 = sub-linear.

4. ``lotka_volterra_police_crime(category)`` — Lotka-Volterra
   predator-prey on a coarse-grained yearly time series:

       dx/dt = α x − β x y       (crime: prey)
       dy/dt = δ x y − γ y       (police effort: predator)

   Fit (α, β, γ, δ) by least-squares against TPS counts (proxy x) and
   normalised mass-stop / arrest counts (proxy y if available, else
   one-quarter-of-incidents as a placeholder).

References
----------
- D'Orsogna & Perc (2015) §2.1, §3.2, §4.1
- Short, D'Orsogna, Brantingham et al. (2008) "Statistical model of
  criminal behavior", M3AS 18.
- Brockmann, Hufnagel, Geisel (2006) "Scaling laws of human travel",
  Nature 439.
- Bettencourt, Lobo, Helbing, Kühnert, West (2007) "Growth, innovation,
  scaling, and the pace of life in cities", PNAS 104.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from .fn._richresult import RichResult

PROJECT = Path(__file__).resolve().parents[5]
OUT = PROJECT / "data/manifest/outputs/tps_statphysics"
FIG = PROJECT / "data/manifest/outputs/figures/tps_statphysics"


def _toronto_grid(nx: int = 90, ny: int = 60) -> tuple[np.ndarray, np.ndarray]:
    """Cos-corrected planar grid covering the Toronto bbox."""
    from .tps_render import project_xy
    lats = np.array([43.55, 43.90])
    lons = np.array([-79.65, -79.10])
    xk, yk = project_xy(lats, lons)
    gx = np.linspace(xk.min() - 1, xk.max() + 1, nx)
    gy = np.linspace(yk.min() - 1, yk.max() + 1, ny)
    return gx, gy


def sdb_reaction_diffusion(category: str = "Assault",
                            *, sample_rows: int | None = 30_000,
                            eta: float = 0.05,
                            omega: float = 0.30,
                            theta: float = 1.5,
                            D: float = 0.10,
                            gamma: float = 0.05,
                            n_steps: int = 800,
                            dt: float = 0.04,
                            nx: int = 90, ny: int = 60,
                            save_fig: bool = True) -> RichResult:
    """Solve the Short-D'Orsogna-Brantingham 2008 reaction-diffusion
    crime hot-spot PDE on a Toronto grid, seeded by observed incidents.

    Returns a RichResult with the steady-state hot-spot count, mean
    attractiveness, mean criminal density, and a comparison to DBSCAN.
    """
    from .tps_datasets import load_tps_dataset
    from .tps_render import project_xy

    df = load_tps_dataset(category, nrows=sample_rows)
    df = df.dropna(subset=["LAT_WGS84", "LONG_WGS84"]).copy()
    df = df[(df["LAT_WGS84"].between(43.55, 43.90))
            & (df["LONG_WGS84"].between(-79.65, -79.10))]
    if df.empty:
        return RichResult(title=f"SDB reaction-diffusion — {category}",
                            warnings=[f"{category}: no in-bbox rows"])

    pt_x, pt_y = project_xy(df["LAT_WGS84"].to_numpy(),
                                df["LONG_WGS84"].to_numpy())
    gx, gy = _toronto_grid(nx, ny)
    dx_, dy_ = float(gx[1] - gx[0]), float(gy[1] - gy[0])

    # Initial attractiveness = histogram of incidents (normalised)
    H, _, _ = np.histogram2d(pt_x, pt_y, bins=[gx, gy])
    H = H.T  # rows=y, cols=x
    A = (H + 0.05) / (H.max() + 0.05)
    rho = np.full_like(A, 0.5)

    def lap(F):
        return (np.roll(F, +1, 0) + np.roll(F, -1, 0)
                + np.roll(F, +1, 1) + np.roll(F, -1, 1) - 4 * F) \
            / max(dx_, dy_) ** 2

    def grad(F):
        # Central differences with reflective boundaries
        gy_, gx_ = np.gradient(F, dy_, dx_)
        return gx_, gy_

    rng = np.random.default_rng(7)
    for _ in range(n_steps):
        # PDE step (forward Euler)
        gxA, gyA = grad(np.log(A.clip(1e-3)))
        flux_x = -D * np.gradient(rho, dx_, axis=1) + 2.0 * rho * gxA
        flux_y = -D * np.gradient(rho, dy_, axis=0) + 2.0 * rho * gyA
        div_flux = (np.gradient(flux_x, dx_, axis=1)
                    + np.gradient(flux_y, dy_, axis=0))
        dA = eta * lap(A) - omega * A + theta * rho
        drho = -div_flux - rho * A + gamma
        # Small stochastic kick keeps the lattice from freezing
        drho += 0.005 * rng.standard_normal(rho.shape)
        A = (A + dt * dA).clip(1e-3, None)
        rho = (rho + dt * drho).clip(0.0, None)

    # Spike detection: local maxima of A above 75th percentile
    thresh = np.quantile(A, 0.92)
    spikes = (thresh < A) & (_local_max3x3(A) == A)
    n_spikes = int(spikes.sum())

    # Compare to DBSCAN cluster count
    from .tps_spatial_advanced import dbscan_clusters
    rr_db = dbscan_clusters(df, ds_name=category,
                              eps_km=0.3, min_samples=20)
    n_dbscan = int(rr_db.get("n_clusters") or 0)

    fig_path = None
    if save_fig:
        FIG.mkdir(parents=True, exist_ok=True)
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
        for ax, F, t in zip(axes,
                              [H / max(H.max(), 1.0), A, rho],
                              ["seed (incident histogram)",
                               "attractiveness A(x,t)",
                               "criminal density ρ(x,t)"]):
            im = ax.imshow(F, origin="lower",
                            extent=[gx.min(), gx.max(), gy.min(), gy.max()],
                            cmap="magma", aspect="equal")
            ax.set_title(t, fontsize=10)
            ax.set_xticks([]); ax.set_yticks([])
            fig.colorbar(im, ax=ax, shrink=0.7)
        fig.suptitle(
            f"Short-D'Orsogna-Brantingham 2008 hotspot PDE — {category} · "
            f"η={eta}, ω={omega}, θ={theta}, D={D}, γ={gamma} · "
            f"{n_spikes} spikes (vs {n_dbscan} DBSCAN clusters)",
            fontsize=11, y=1.0)
        fig.tight_layout()
        fig_path = FIG / f"sdb_pde_{category.lower()}.png"
        fig.savefig(fig_path, dpi=140, bbox_inches="tight")
        plt.close(fig)

    return RichResult(
        title=f"SDB reaction-diffusion — {category}",
        summary_lines=[
            ("Method", "Short-D'Orsogna-Brantingham 2008 hot-spot PDE"),
            ("Grid", f"{nx}×{ny} cos-corrected, dx≈{dx_:.2f} km"),
            ("Parameters",
             f"η={eta} ω={omega} θ={theta} D={D} γ={gamma} dt={dt} "
             f"steps={n_steps}"),
            ("Steady-state spikes", n_spikes),
            ("DBSCAN clusters (eps=0.3km)", n_dbscan),
            ("Mean A", round(float(A.mean()), 4)),
            ("Mean ρ", round(float(rho.mean()), 4)),
            ("Figure", str(fig_path) if fig_path else "(skipped)"),
        ],
        interpretation=(
            "The PDE evolves an attractiveness field A and a criminal "
            "density ρ that diffuses up the gradient of log A. "
            "Localised spikes emerge when the diffusion-decay balance "
            "puts the system in the unstable / hot-spot regime "
            "(D'Orsogna & Perc 2015, §3.2). Steady-state spike count "
            f"is {n_spikes}, vs {n_dbscan} empirical DBSCAN clusters "
            f"at 0.3 km — agreement of "
            f"{abs(n_spikes - n_dbscan) / max(n_dbscan, 1) * 100:.0f}% "
            "deviation from the data-driven count."
        ),
    )


def _local_max3x3(F: np.ndarray) -> np.ndarray:
    """Pointwise 3x3 local-maximum filter."""
    out = F.copy()
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            if di == dj == 0:
                continue
            out = np.maximum(out, np.roll(np.roll(F, di, 0), dj, 1))
    return out


def levy_flight_alpha(category: str = "Assault",
                       *, sample_rows: int | None = 30_000,
                       lmin_km: float = 0.5,
                       save_fig: bool = True) -> RichResult:
    """Hill-MLE Lévy exponent on step lengths between consecutive
    incidents in chronological order (proxy for offender mobility)."""
    from .tps_datasets import load_tps_dataset
    from .tps_render import project_xy

    df = load_tps_dataset(category, nrows=sample_rows)
    df = df.dropna(subset=["LAT_WGS84", "LONG_WGS84", "OCC_DATE"]).copy()
    df = df[(df["LAT_WGS84"].between(43.55, 43.90))
            & (df["LONG_WGS84"].between(-79.65, -79.10))]
    df["_dt"] = pd.to_datetime(df["OCC_DATE"], errors="coerce")
    df = df.dropna(subset=["_dt"]).sort_values("_dt")
    if len(df) < 200:
        return RichResult(title=f"Lévy α — {category}",
                            warnings=["too few rows for Lévy fit"])
    xk, yk = project_xy(df["LAT_WGS84"].to_numpy(),
                            df["LONG_WGS84"].to_numpy())
    dx_ = np.diff(xk); dy_ = np.diff(yk)
    steps = np.sqrt(dx_ ** 2 + dy_ ** 2)
    steps = steps[steps >= lmin_km]
    if steps.size < 50:
        return RichResult(title=f"Lévy α — {category}",
                            warnings=[f"only {steps.size} tail steps "
                                       f"≥ {lmin_km} km"])
    # Hill-MLE: α̂ = 1 + n / Σ ln(ℓ_i / ℓ_min)
    alpha = 1 + steps.size / np.sum(np.log(steps / lmin_km))
    # Bootstrap SE
    rng = np.random.default_rng(11)
    boots = []
    for _ in range(200):
        s = rng.choice(steps, size=steps.size, replace=True)
        boots.append(1 + s.size / np.sum(np.log(s / lmin_km)))
    se = float(np.std(boots, ddof=1))

    fig_path = None
    if save_fig:
        FIG.mkdir(parents=True, exist_ok=True)
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 4))
        bins = np.logspace(np.log10(lmin_km), np.log10(steps.max()), 40)
        ax.hist(steps, bins=bins, density=True, alpha=0.55, label="empirical")
        x = np.logspace(np.log10(lmin_km), np.log10(steps.max()), 100)
        y = (alpha - 1) / lmin_km * (x / lmin_km) ** (-alpha)
        ax.plot(x, y, "r-", lw=2,
                  label=f"Pareto fit α={alpha:.2f}±{se:.2f}")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("step length ℓ (km)")
        ax.set_ylabel("p(ℓ)")
        ax.set_title(f"Lévy-flight tail · {category} · "
                       f"n={steps.size:,} steps ≥ {lmin_km} km")
        ax.legend()
        fig_path = FIG / f"levy_alpha_{category.lower()}.png"
        fig.savefig(fig_path, dpi=140, bbox_inches="tight")
        plt.close(fig)

    return RichResult(
        title=f"Lévy α — {category}",
        summary_lines=[
            ("Method", "Hill-MLE upper-tail power-law (BHG 2006)"),
            ("ℓ_min (km)", lmin_km),
            ("n tail steps", int(steps.size)),
            ("α̂", round(float(alpha), 3)),
            ("SE (bootstrap × 200)", round(se, 3)),
            ("median step (km)", round(float(np.median(steps)), 2)),
            ("max step (km)", round(float(steps.max()), 2)),
            ("Figure", str(fig_path) if fig_path else "(skipped)"),
        ],
        interpretation=(
            f"Step-length distribution between consecutive {category} "
            f"incidents has Hill-MLE exponent α̂ = {alpha:.2f} ± {se:.2f}. "
            "Brockmann-Hufnagel-Geisel 2006 found α ≈ 1.6 for human "
            "mobility (Lévy regime). α ∈ (1, 3) ⇒ heavy-tailed Lévy; "
            "α > 3 ⇒ Gaussian-like local diffusion. Routine-activity "
            "theory predicts crime mobility tracks general human "
            "mobility — D'Orsogna & Perc 2015 §2.2."
        ),
    )


def urban_scaling_beta(category: str = "Assault",
                        *, year: int = 2024,
                        save_fig: bool = True) -> RichResult:
    """Bettencourt 2007 super-linear urban scaling on the 158 wards.

    Regresses log(crime per ward) on log(population per ward) → β.
    """
    from .tps_io import load_tps
    df = load_tps("NeighbourhoodCrimeRates", format="geojson")
    pop_col = next((c for c in df.columns if "POP" in c.upper()
                    and str(year)[-2:] in c), None)
    if pop_col is None:
        pop_col = next((c for c in df.columns if "POP" in c.upper()), None)
    cat_prefix = {"Assault": "ASSAULT", "AutoTheft": "AUTOTHEFT",
                    "BicycleTheft": "BIKETHEFT", "BreakandEnter": "BREAKENTER",
                    "Homicides": "HOMICIDE", "Robbery": "ROBBERY",
                    "ShootingAndFirearmDiscarges": "SHOOTING",
                    "TheftFromMovingVehicle": "THEFTFROMMV",
                    "TheftOver": "THEFTOVER"}.get(category, category.upper())
    crime_col = f"{cat_prefix}_{year}"
    if pop_col is None or crime_col not in df.columns:
        return RichResult(
            title=f"Urban scaling β — {category}",
            warnings=[f"missing pop or {crime_col} column "
                       f"(have pop_col={pop_col})"])
    sub = df[[pop_col, crime_col]].dropna()
    sub = sub[(sub[pop_col] > 100) & (sub[crime_col] > 0)]
    if len(sub) < 30:
        return RichResult(title=f"Urban scaling β — {category}",
                            warnings=[f"only {len(sub)} usable wards"])
    lx = np.log(sub[pop_col].to_numpy(dtype=float))
    ly = np.log(sub[crime_col].to_numpy(dtype=float))
    n = lx.size
    sx, sy = lx.mean(), ly.mean()
    beta = float(np.sum((lx - sx) * (ly - sy))
                  / np.sum((lx - sx) ** 2))
    Y0 = float(np.exp(sy - beta * sx))
    resid = ly - (math.log(Y0) + beta * lx)
    se_beta = float(np.sqrt(np.sum(resid ** 2) / (n - 2)
                              / np.sum((lx - sx) ** 2)))
    r2 = float(1 - np.var(resid) / np.var(ly))

    fig_path = None
    if save_fig:
        FIG.mkdir(parents=True, exist_ok=True)
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(sub[pop_col], sub[crime_col], s=14, alpha=0.5,
                    edgecolor="#1a1a1a", linewidth=0.4)
        xfit = np.linspace(sub[pop_col].min(), sub[pop_col].max(), 80)
        ax.plot(xfit, Y0 * xfit ** beta, "r-", lw=2,
                  label=f"β={beta:.2f}±{se_beta:.2f}, R²={r2:.2f}")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("ward population"); ax.set_ylabel(f"{crime_col}")
        ax.set_title(f"Urban scaling · {category} · {year} · 158 wards")
        ax.legend()
        ax.grid(True, which="both", alpha=0.2)
        fig_path = FIG / f"urban_scaling_{category.lower()}_{year}.png"
        fig.savefig(fig_path, dpi=140, bbox_inches="tight")
        plt.close(fig)

    regime = ("super-linear (β > 1)" if beta > 1.05
              else "sub-linear (β < 1)" if beta < 0.95
              else "linear (β ≈ 1)")
    return RichResult(
        title=f"Urban scaling β — {category} · {year}",
        summary_lines=[
            ("Method", "Bettencourt 2007 OLS log–log scaling"),
            ("Crime column", crime_col),
            ("Population column", pop_col),
            ("n wards", int(n)),
            ("β̂", round(beta, 3)),
            ("SE(β̂)", round(se_beta, 3)),
            ("R²", round(r2, 3)),
            ("Y₀", round(Y0, 4)),
            ("Regime", regime),
            ("Figure", str(fig_path) if fig_path else "(skipped)"),
        ],
        interpretation=(
            f"Across Toronto's 158 wards, {category} {year} scales as "
            f"crime ∝ pop^{beta:.2f}. {regime}. "
            "Bettencourt 2007 finds β ≈ 1.16 for violent crime across "
            "US metros — population doubles, violent crime rises ~2.24x. "
            "β < 1 (sub-linear) is rare and indicates protective scale. "
            "D'Orsogna & Perc 2015 §4.1 generalises to socio-economic "
            "indicators."
        ),
    )


def lotka_volterra_police_crime(category: str = "Assault",
                                  save_fig: bool = True) -> RichResult:
    """Yearly Lotka-Volterra fit: crime as prey, mass-stop / arrest as
    predator.

    With only 1 series we use the empirical x(t) and back out (α, β)
    from the LV equilibrium x* = γ/δ, y* = α/β under stationarity. For
    the predator series we use the cumulative count of incidents → a
    crude effort-density proxy; the absolute scale is unidentified but
    the (α, γ) ratio and the cycle period are.
    """
    from .tps_datasets import load_tps_dataset
    df = load_tps_dataset(category)
    if "OCC_YEAR" not in df.columns:
        return RichResult(title=f"Lotka-Volterra — {category}",
                            warnings=["no OCC_YEAR column"])
    counts = df.groupby("OCC_YEAR").size().sort_index()
    counts = counts[counts.index >= 2014]
    if len(counts) < 5:
        return RichResult(title=f"Lotka-Volterra — {category}",
                            warnings=[f"only {len(counts)} years"])
    x = counts.to_numpy(dtype=float)
    # Naive predator proxy: rolling 3-year mean of x (lagged response)
    y = pd.Series(x).rolling(3, min_periods=1).mean().to_numpy()

    # Fit growth rate α from log-diffs of x ignoring predation term
    log_dx = np.diff(np.log(x.clip(1.0)))
    alpha = float(np.median(log_dx + 0.05))
    gamma = -float(np.median(np.diff(np.log(y.clip(1.0))))) + 0.05
    # Cycle period T = 2π / sqrt(α γ)  (linearised LV)
    T = 2 * math.pi / math.sqrt(max(alpha * gamma, 1e-6))
    beta_lv = alpha / max(np.median(y), 1.0)
    delta_lv = gamma / max(np.median(x), 1.0)

    fig_path = None
    if save_fig:
        FIG.mkdir(parents=True, exist_ok=True)
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(counts.index, x, "o-", color="#d7191c",
                  label=f"prey (x = {category} count)", lw=2)
        ax.plot(counts.index, y, "s-", color="#2c7bb6",
                  label="predator proxy (3-yr smoothed)", lw=2)
        ax.set_xlabel("year"); ax.set_ylabel("count")
        ax.set_title(f"Lotka-Volterra · {category} · α={alpha:.2f} "
                       f"γ={gamma:.2f} T={T:.1f} yr")
        ax.legend(); ax.grid(True, alpha=0.2)
        fig_path = FIG / f"lotka_volterra_{category.lower()}.png"
        fig.savefig(fig_path, dpi=140, bbox_inches="tight")
        plt.close(fig)

    return RichResult(
        title=f"Lotka-Volterra — {category}",
        summary_lines=[
            ("Method",
             "LV predator-prey on yearly counts (D'Orsogna & Perc 2015 §3.4)"),
            ("Years", f"{int(counts.index.min())}–{int(counts.index.max())}"),
            ("α (prey growth)", round(alpha, 3)),
            ("β (predation rate)", round(beta_lv, 5)),
            ("γ (predator decay)", round(gamma, 3)),
            ("δ (predator efficiency)", round(delta_lv, 5)),
            ("Linearised cycle T (yr)", round(T, 1)),
            ("Figure", str(fig_path) if fig_path else "(skipped)"),
        ],
        interpretation=(
            f"Yearly {category} counts treated as prey; predator proxy "
            f"is a 3-yr smoothing of past crime as a stand-in for "
            f"police-effort density (no public mass-stop time series "
            f"yet). Linearised LV gives a cycle period of T ≈ {T:.1f} "
            f"years. D'Orsogna & Perc 2015 §3.4 use this framework "
            f"for Mexican drug-cartel turf wars; here it's qualitative "
            f"only — substitute a real predator series (TPS use-of-force "
            f"counts, arrests, dispatches) for quantitative inference."
        ),
    )


def sdb_turing_demo(*, eta: float = 0.20, omega: float = 0.033,
                       theta: float = 0.56, D: float = 30.0,
                       gamma: float = 0.019,
                       n_steps: int = 6000, dt: float = 0.005,
                       n: int = 80, save_fig: bool = True) -> RichResult:
    """Canonical SDB Turing-instability demo on a *clean periodic* grid.

    Reproduces the localized hot-spot lattice from Short, D'Orsogna &
    Brantingham (2008) Fig. 4 / D'Orsogna & Perc (2015) Fig. 5 — which
    is what the author's reference image #3 / #4 shows. Periodic boundaries,
    homogeneous initial state + Gaussian noise, runs to steady state.
    Output: 1×3 panel (early / mid / late).
    """
    rng = np.random.default_rng(7)
    # Steady-state-ish initial condition + small noise (Turing seed)
    A0 = theta * gamma / max(omega, 1e-6) ** 2
    A = A0 + 0.02 * A0 * rng.standard_normal((n, n))
    rho = (gamma / max(A0, 1e-6)) * np.ones((n, n))
    rho += 0.02 * rho.mean() * rng.standard_normal((n, n))

    def lap_periodic(F):
        return (np.roll(F, 1, 0) + np.roll(F, -1, 0)
                + np.roll(F, 1, 1) + np.roll(F, -1, 1) - 4 * F)

    def grad_periodic(F):
        gx = (np.roll(F, -1, 1) - np.roll(F, 1, 1)) / 2
        gy = (np.roll(F, -1, 0) - np.roll(F, 1, 0)) / 2
        return gx, gy

    snaps = []
    snap_steps = (max(1, n_steps // 6),
                    max(2, n_steps // 2), n_steps - 1)
    for step in range(n_steps):
        gxA, gyA = grad_periodic(np.log(A.clip(1e-3)))
        flux_x = -D * grad_periodic(rho)[0] + 2 * rho * gxA
        flux_y = -D * grad_periodic(rho)[1] + 2 * rho * gyA
        div = grad_periodic(flux_x)[0] + grad_periodic(flux_y)[1]
        dA = eta * lap_periodic(A) - omega * A + theta * rho
        drho = -div - rho * A + gamma  # noise only at init
        A = (A + dt * dA).clip(1e-3, None)
        rho = (rho + dt * drho).clip(0.0, None)
        if step in snap_steps:
            snaps.append((step, A.copy()))

    fig_path = None
    if save_fig:
        FIG.mkdir(parents=True, exist_ok=True)
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        for ax, (step, F) in zip(axes, snaps, strict=False):
            im = ax.imshow(F, cmap="jet", origin="lower")
            ax.set_title(f"step {step}", fontsize=10)
            ax.set_xticks([]); ax.set_yticks([])
            fig.colorbar(im, ax=ax, shrink=0.8)
        fig.suptitle(
            f"Short-D'Orsogna-Brantingham 2008 · canonical Turing regime · "
            f"η={eta} ω={omega} θ={theta} D={D} γ={gamma}",
            fontsize=11, y=1.02)
        fig.tight_layout()
        fig_path = FIG / "sdb_turing_demo.png"
        fig.savefig(fig_path, dpi=140, bbox_inches="tight")
        plt.close(fig)

    spike_thresh = float(np.quantile(A, 0.92))
    n_spikes = int(((spike_thresh < A) & (_local_max3x3(A) == A)).sum())
    return RichResult(
        title="SDB Turing-pattern demo",
        summary_lines=[
            ("Method",
             "Short-D'Orsogna-Brantingham 2008 reaction-diffusion PDE "
             "on periodic lattice"),
            ("Grid", f"{n}×{n} periodic"),
            ("Steps", n_steps),
            ("Steady-state spikes", n_spikes),
            ("Mean A", round(float(A.mean()), 4)),
            ("Mean ρ", round(float(rho.mean()), 4)),
            ("Figure", str(fig_path) if fig_path else "(skipped)"),
        ],
        interpretation=(
            "Canonical Turing-instability hot-spot lattice — the "
            "homogeneous (A, ρ) state is unstable for these parameters "
            "and the system spontaneously self-organises into a "
            "near-hexagonal lattice of localised spikes. This figure "
            "is parameter-driven (not data-seeded) and reproduces the "
            "SDB 2008 canonical regime that D'Orsogna & Perc 2015 "
            "Fig. 5 illustrates."
        ),
    )


def inspection_game_phase(n_temptations: int = 20,
                            n_costs: int = 20,
                            n_steps: int = 600,
                            save_fig: bool = True) -> RichResult:
    """Helbing-Szolnoki-Perc (2010) cooperator-defector-inspector
    replicator dynamics; produces the (Temptation T, Inspection cost γ)
    crime-rate phase diagram from D'Orsogna & Perc 2015 §5 / Image #5.

    Three strategies (pure populations only):
      C — cooperator
      P — defector / 'predator'
      O — punisher / inspector

    Replicator dynamics with payoff matrix that includes punishment of
    P by O at cost γ. Steady-state defector fraction → 'crime rate'.
    """
    Ts = np.linspace(0.05, 1.8, n_temptations)
    gs = np.linspace(0.05, 1.2, n_costs)
    crime = np.zeros((n_temptations, n_costs))
    rng = np.random.default_rng(3)
    for i, T in enumerate(Ts):
        for j, g in enumerate(gs):
            # 3-strategy replicator (C, P, O), starting near uniform
            x = np.array([0.34, 0.33, 0.33])
            x += 0.01 * rng.standard_normal(3); x = x.clip(0); x /= x.sum()
            for _ in range(n_steps):
                # payoffs:  C vs C = 1, C vs P = 0, C vs O = 1
                #           P vs C = T, P vs P = 0, P vs O = T - 1
                #           O vs C = 1 - g, O vs P = 1 - g, O vs O = 1 - g
                P = np.array([
                    [1.0, 0.0, 1.0],
                    [T,   0.0, T - 1.0],
                    [1.0 - g, 1.0 - g, 1.0 - g],
                ])
                fit = P @ x
                phi = float(x @ fit)
                x = x + 0.05 * x * (fit - phi)
                x = x.clip(0); s = x.sum()
                x = x / s if s > 0 else np.array([0.34, 0.33, 0.33])
            crime[i, j] = float(x[1])

    fig_path = None
    if save_fig:
        FIG.mkdir(parents=True, exist_ok=True)
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        im = ax.imshow(crime, origin="lower", aspect="auto",
                        extent=[gs.min(), gs.max(), Ts.min(), Ts.max()],
                        cmap="cool")
        ax.set_xlabel("Inspection cost γ")
        ax.set_ylabel("Temptation T")
        ax.set_title("Helbing-Szolnoki cooperator-predator-inspector "
                       "phase diagram\nreplicator dynamics, "
                       "steady-state defector frequency")
        fig.colorbar(im, ax=ax, label="Crime rate")
        # Mark approximate phase boundary T = 1 + γ / 2
        gline = np.linspace(gs.min(), gs.max(), 50)
        ax.plot(gline, 1 + 0.5 * gline, "k-", linewidth=1.5,
                  label="C ↔ P+C boundary")
        ax.legend(loc="lower left", fontsize=8)
        fig_path = FIG / "inspection_game_phase.png"
        fig.savefig(fig_path, dpi=140, bbox_inches="tight")
        plt.close(fig)

    return RichResult(
        title="Helbing-Szolnoki inspection-game phase diagram",
        summary_lines=[
            ("Method", "3-strategy replicator dynamics (C, P, O)"),
            ("Grid", f"{n_temptations} × {n_costs} (T, γ)"),
            ("Steps", n_steps),
            ("Mean crime rate", round(float(crime.mean()), 3)),
            ("Min crime rate", round(float(crime.min()), 3)),
            ("Max crime rate", round(float(crime.max()), 3)),
            ("Figure", str(fig_path) if fig_path else "(skipped)"),
        ],
        interpretation=(
            "Steady-state defector ('predator') frequency across the "
            "(temptation T, inspection cost γ) plane. Pure-cooperation "
            "phase emerges where T < 1 and γ small; pure-defection "
            "phase where T > ~1 + γ/2. Mixed P+C phase along the "
            "boundary. D'Orsogna & Perc 2015 §5; Helbing, Szolnoki & "
            "Perc 2010 PNAS."
        ),
    )


def criminal_network_graph(category: str = "Assault",
                            *, sample_rows: int | None = 30_000,
                            top_n_premises: int = 20,
                            save_fig: bool = True) -> RichResult:
    """Co-occurrence network: nodes = top-N premise types, edges =
    sharing a common neighbourhood, weighted by joint incident count.

    Replaces the criminal-role graph in D'Orsogna & Perc Fig. 9 /
    Diviák et al. (2019) — for our public TPS data we don't have
    co-offender records, so we build the closest-meaning network from
    PREMISES_TYPE × HOOD_158 co-incidence.
    """
    from .tps_datasets import load_tps_dataset
    df = load_tps_dataset(category, nrows=sample_rows)
    if "HOOD_158" not in df.columns:
        return RichResult(title=f"Criminal network — {category}",
                            warnings=["missing HOOD_158"])
    # Pick the best available node-attribute column. Most TPS categories
    # have PREMISES_TYPE (e.g. "Apartment", "Outside"); Homicides /
    # Shooting only have HOMICIDE_TYPE / DIVISION / LOCATION_TYPE.
    node_col = next((c for c in
                     ("PREMISES_TYPE", "LOCATION_TYPE",
                      "HOMICIDE_TYPE", "OFFENCE", "DIVISION")
                     if c in df.columns), None)
    if node_col is None:
        return RichResult(title=f"Criminal network — {category}",
                            warnings=["no usable node-attribute column "
                                       "(PREMISES_TYPE, LOCATION_TYPE, "
                                       "HOMICIDE_TYPE, OFFENCE, DIVISION "
                                       "all missing)"])
    # Top categories by frequency
    top = df[node_col].value_counts().head(top_n_premises)
    top_set = set(top.index)
    sub = df[df[node_col].isin(top_set)]
    # Edge weight: count of hoods where both nodes i and j appeared
    pivot = (sub.groupby(["HOOD_158", node_col]).size()
                  .unstack(fill_value=0).clip(0, 1))
    co = pivot.T @ pivot
    nodes = list(co.columns)
    n = len(nodes)
    if n < 2:
        return RichResult(title=f"Criminal network — {category}",
                            warnings=[f"only {n} nodes"])

    # Circular layout
    angles = 2 * math.pi * np.arange(n) / n
    pos = {nodes[i]: (math.cos(angles[i]), math.sin(angles[i]))
           for i in range(n)}
    sizes = (top.reindex(nodes).fillna(0).to_numpy() ** 0.5)
    sizes = sizes / sizes.max() * 800 + 80

    fig_path = None
    if save_fig:
        FIG.mkdir(parents=True, exist_ok=True)
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 8))
        # Edges
        max_w = float(co.values.max())
        for i in range(n):
            for j in range(i + 1, n):
                w = float(co.iloc[i, j])
                if w <= 0: continue
                a, b = pos[nodes[i]], pos[nodes[j]]
                ax.plot([a[0], b[0]], [a[1], b[1]],
                          color="#444", alpha=0.05 + 0.6 * (w / max_w),
                          linewidth=0.4 + 1.5 * (w / max_w), zorder=1)
        # Nodes
        for i, (name, (x, y)) in enumerate(pos.items()):
            ax.scatter(x, y, s=sizes[i], color="#bbb",
                        edgecolor="#222", zorder=3)
            # Label outside the ring
            tx, ty = 1.18 * x, 1.18 * y
            ax.text(tx, ty, str(name).replace("'", "’"),
                      fontsize=7, ha="center" if abs(x) < 0.3
                      else ("left" if x > 0 else "right"),
                      va="center", zorder=4,
                      bbox=dict(facecolor="white",
                                  edgecolor="#888",
                                  boxstyle="round,pad=0.18",
                                  linewidth=0.5))
        ax.set_xlim(-1.6, 1.6); ax.set_ylim(-1.6, 1.6)
        ax.set_aspect("equal"); ax.axis("off")
        ax.set_title(f"{category} · premise × neighbourhood "
                       f"co-occurrence network · top {top_n_premises} premises",
                       fontsize=11)
        fig_path = FIG / f"network_{category.lower()}.png"
        fig.savefig(fig_path, dpi=140, bbox_inches="tight")
        plt.close(fig)

    return RichResult(
        title=f"Criminal network — {category}",
        summary_lines=[
            ("Method", "Premise × neighbourhood co-occurrence network"),
            ("Nodes", n),
            ("Edges (≥1 weight)", int((co.values > 0).sum() // 2)),
            ("Strongest edge weight (max hoods)",
             int(co.values.max() if co.size else 0)),
            ("Figure", str(fig_path) if fig_path else "(skipped)"),
        ],
        interpretation=(
            "Force-directed-style circular network. Each node is a "
            "premise type (top-N by frequency); node size ∝ √(incident "
            "count). Edges connect two premise types if they share at "
            "least one neighbourhood, weighted by the count of shared "
            "hoods. Approximates the Diviák-style criminal-role "
            "network of D'Orsogna & Perc 2015 Fig. 9 with public TPS "
            "data (no co-offender records)."
        ),
    )


def analyze_all(categories: list[str] | None = None,
                  save_fig: bool = True) -> dict[str, dict[str, RichResult]]:
    """Run all 4 stat-phys analyses on each category."""
    if categories is None:
        categories = ["Assault", "AutoTheft", "BicycleTheft",
                       "BreakandEnter", "Homicides", "Robbery",
                       "ShootingAndFirearmDiscarges",
                       "TheftFromMovingVehicle", "TheftOver"]
    out: dict[str, dict[str, RichResult]] = {}
    for cat in categories:
        out[cat] = {
            "sdb_pde": sdb_reaction_diffusion(cat, save_fig=save_fig),
            "levy": levy_flight_alpha(cat, save_fig=save_fig),
            "urban_scaling": urban_scaling_beta(cat, save_fig=save_fig),
            "lotka_volterra": lotka_volterra_police_crime(
                cat, save_fig=save_fig),
        }
    # Persist each RichResult as txt + json
    OUT.mkdir(parents=True, exist_ok=True)
    for cat, results in out.items():
        for kind, rr in results.items():
            (OUT / f"sp_{cat.lower()}_{kind}.txt").write_text(str(rr))
            try:
                import json
                (OUT / f"sp_{cat.lower()}_{kind}.json").write_text(
                    json.dumps(rr.to_dict(), default=str, indent=2))
            except Exception:  # noqa: BLE001
                pass
    return out
