"""morie.tps_temporal — temporal analyses for TPS crime data.

- year_over_year_trend: linear regression on yearly counts
- seasonal_pattern: monthly / DOW / hour-of-day cyclic stats
- arima_forecast: simple ARIMA(1,1,1) on monthly counts
- changepoint_detection: PELT-style changepoint on yearly counts

All emit RichResult.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .fn._richresult import RichResult


def year_over_year_trend(df: pd.DataFrame, *,
                          year_col: str = "OCC_YEAR",
                          ds_name: str = "?") -> RichResult:
    """Linear regression of incident counts vs year."""
    if year_col not in df.columns:
        return RichResult(title=f"YoY trend — {ds_name}",
                          warnings=[f"{year_col} missing"])
    counts = df.groupby(year_col).size().sort_index()
    valid = counts[counts.index.notna() & (counts.index >= 1990) &
                   (counts.index <= 2030)] if hasattr(counts.index, "notna") \
            else counts
    valid = valid.dropna()
    valid = valid[(valid.index >= 1990) & (valid.index <= 2030)]
    if valid.size < 3:
        return RichResult(title=f"YoY trend — {ds_name}",
                          warnings=[f"only {valid.size} usable years"])
    years = valid.index.values.astype(float)
    y = valid.values.astype(float)
    x = years - years.mean()
    slope = float((x * (y - y.mean())).sum() / ((x ** 2).sum() + 1e-300))
    intercept = float(y.mean() - slope * years.mean())
    y_hat = slope * years + intercept
    ss_res = float(((y - y_hat) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    direction = ("INCREASING" if slope > 0
                 else "DECREASING" if slope < 0
                 else "FLAT")

    return RichResult(
        title=f"Year-over-year trend — {ds_name}",
        summary_lines=[
            ("Years", f"{int(years.min())}–{int(years.max())}"),
            ("n years", int(valid.size)),
            ("Slope (incidents/year)", round(slope, 2)),
            ("Intercept", round(intercept, 2)),
            ("R²", round(float(r2), 4)),
            ("Direction", direction),
            ("Mean count/year", round(float(y.mean()), 1)),
            ("Min year, count", f"{int(years[y.argmin()])}, {int(y.min())}"),
            ("Max year, count", f"{int(years[y.argmax()])}, {int(y.max())}"),
        ],
        tables=[{
            "title": "Year-by-year counts:",
            "headers": [year_col, "Count", "Fitted"],
            "rows": [[int(yr), int(c), round(float(f), 1)]
                     for yr, c, f in zip(years, y, y_hat)],
        }],
        interpretation=(
            f"Linear fit: count ≈ {slope:.1f} · year + {intercept:.0f}, "
            f"R² = {r2:.3f}. {direction.title()} trend over the "
            f"{int(years.min())}–{int(years.max())} window."
        ),
        payload={"slope": slope, "intercept": intercept, "r2": float(r2),
                 "direction": direction},
    )


def seasonal_pattern(df: pd.DataFrame, *,
                      ds_name: str = "?") -> RichResult:
    """Month / day-of-week / hour-of-day cyclic patterns + chi-square
    test of uniformity.
    """
    from scipy.stats import chisquare
    sections = []

    def _cycle(col: str, expected_levels: int, label: str) -> dict | None:
        if col not in df.columns:
            return None
        s = df[col].dropna()
        if s.size == 0:
            return None
        counts = s.value_counts().sort_index()
        # Chi-square against uniform
        try:
            obs = counts.values
            exp = np.full_like(obs, fill_value=obs.sum() / obs.size,
                                dtype=float)
            chi2, p = chisquare(obs, exp)
        except Exception:
            chi2, p = float("nan"), float("nan")
        return {
            "label": label,
            "rows": [[str(idx), int(c)] for idx, c in counts.items()],
            "chi2": float(chi2), "p": float(p),
        }

    month = _cycle("OCC_MONTH", 12, "Month of occurrence (1=Jan)")
    dow = _cycle("OCC_DOW", 7, "Day of week of occurrence")
    hour = _cycle("OCC_HOUR", 24, "Hour of day of occurrence")

    summary = [("Dataset", ds_name), ("Incidents", int(df.shape[0]))]
    tables = []
    for c in [month, dow, hour]:
        if c:
            summary.append((f"{c['label']} chi² uniformity p",
                            round(c["p"], 6)))
            tables.append({
                "title": f"{c['label']}:",
                "headers": ["Bucket", "Count"],
                "rows": c["rows"],
            })
    return RichResult(
        title=f"Seasonal / cyclic patterns — {ds_name}",
        summary_lines=summary,
        tables=tables,
        interpretation=(
            "p < 0.05 in any cycle ⇒ incident times are NOT uniformly "
            "distributed over that cycle (e.g. weekday vs weekend, "
            "evening vs morning)."
        ),
    )


def changepoint_detection(df: pd.DataFrame, *,
                           year_col: str = "OCC_YEAR",
                           ds_name: str = "?") -> RichResult:
    """Simple Pettitt-style change-point on yearly counts."""
    if year_col not in df.columns:
        return RichResult(title=f"Change-point — {ds_name}",
                          warnings=[f"{year_col} missing"])
    counts = df.groupby(year_col).size().sort_index()
    counts = counts[(counts.index >= 1990) & (counts.index <= 2030)]
    if counts.size < 6:
        return RichResult(title=f"Change-point — {ds_name}",
                          warnings=[f"need ≥6 years, got {counts.size}"])
    x = counts.values.astype(float)
    n = x.size
    # Pettitt's test: U_t = sum_i sum_j sign(x_i - x_j)
    U = np.zeros(n)
    for t in range(n):
        s = 0
        for i in range(t + 1):
            for j in range(t + 1, n):
                s += np.sign(x[i] - x[j])
        U[t] = s
    K = int(np.argmax(np.abs(U)))
    K_stat = float(np.abs(U[K]))
    # Approx p
    p = 2 * np.exp(-6 * K_stat ** 2 / (n ** 3 + n ** 2))
    p = float(min(1.0, p))
    bp_year = int(counts.index[K])
    pre = float(x[: K + 1].mean())
    post = float(x[K + 1 :].mean()) if n > K + 1 else float("nan")
    return RichResult(
        title=f"Change-point (Pettitt) — {ds_name}",
        summary_lines=[
            ("Years", f"{int(counts.index.min())}–"
                     f"{int(counts.index.max())}"),
            ("n years", int(n)),
            ("Estimated change-point year", bp_year),
            ("Pettitt K statistic", round(K_stat, 2)),
            ("Approx p-value", round(p, 6)),
            ("Mean count BEFORE change-point", round(pre, 1)),
            ("Mean count AFTER change-point", round(post, 1)),
            ("Δ mean", round(post - pre, 1) if not np.isnan(post)
                       else "n/a"),
        ],
        interpretation=(
            f"Estimated structural break in {bp_year}: pre-mean {pre:.1f}, "
            f"post-mean {post:.1f}. p={p:.4f} — {'significant' if p < 0.05 else 'not significant'} at α=0.05."
        ),
    )


def arima_forecast(df: pd.DataFrame, *,
                    h: int = 12,
                    ds_name: str = "?") -> RichResult:
    """ARIMA(1,1,1) on monthly counts; forecast `h` periods ahead."""
    try:
        import statsmodels.api as sm
    except ImportError:
        return RichResult(title=f"ARIMA — {ds_name}",
                          warnings=["statsmodels not installed"])
    # Build monthly time series
    if "OCC_DATE" in df.columns:
        dt = pd.to_datetime(df["OCC_DATE"], errors="coerce").dropna()
    elif "REPORT_DATE" in df.columns:
        dt = pd.to_datetime(df["REPORT_DATE"], errors="coerce").dropna()
    else:
        return RichResult(title=f"ARIMA — {ds_name}",
                          warnings=["no OCC_DATE / REPORT_DATE"])
    monthly = dt.dt.to_period("M").value_counts().sort_index()
    monthly.index = monthly.index.to_timestamp()
    if monthly.size < 24:
        return RichResult(title=f"ARIMA — {ds_name}",
                          warnings=[f"need ≥24 months, got {monthly.size}"])
    try:
        model = sm.tsa.ARIMA(monthly.values.astype(float), order=(1, 1, 1))
        fit = model.fit()
        fc = fit.forecast(steps=h)
    except Exception as e:
        return RichResult(title=f"ARIMA — {ds_name}",
                          warnings=[f"fit failed: {e!r}"])
    return RichResult(
        title=f"ARIMA(1,1,1) {h}-month forecast — {ds_name}",
        summary_lines=[
            ("Training months", int(monthly.size)),
            ("Last training month",
                monthly.index[-1].strftime("%Y-%m")),
            ("AIC", round(float(fit.aic), 1)),
            ("BIC", round(float(fit.bic), 1)),
            ("Forecast horizon (h)", h),
            ("Forecast mean", round(float(np.mean(fc)), 1)),
        ],
        tables=[{
            "title": f"Forecast next {h} months:",
            "headers": ["Step", "Forecast"],
            "rows": [[i + 1, round(float(v), 1)] for i, v in enumerate(fc)],
        }],
        payload={"forecast": list(map(float, fc)),
                 "aic": float(fit.aic), "bic": float(fit.bic)},
    )
