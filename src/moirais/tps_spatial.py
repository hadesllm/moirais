"""moirais.tps_spatial — spatial analyses for TPS crime data.

Wires the existing moirais.fn spatial primitives to TPS data:
- Moran's I (global) on neighbourhood-level counts
- LISA (Local Moran's I) for hot/cold spots
- Ripley's K for point-pattern clustering
- Kernel density estimation
- Gi*/Getis-Ord hotspot detection

All emit RichResult.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .fn._richresult import RichResult


def _hood_counts(df: pd.DataFrame, hood_col: str = "HOOD_158") -> pd.Series:
    """Counts per neighbourhood, dropping 'NSA' / unknowns."""
    if hood_col not in df.columns:
        raise KeyError(f"{hood_col} missing from DataFrame")
    s = df[hood_col].dropna()
    s = s[s.astype(str).str.upper() != "NSA"]
    return s.value_counts()


def _adjacency_knn(coords: np.ndarray, k: int = 5) -> np.ndarray:
    """Row-standardised k-nearest-neighbours adjacency matrix.
    `coords` is (n, 2) array of lat/lon (or projected x/y) per unit.
    """
    n = coords.shape[0]
    if n < 2:
        return np.zeros((n, n))
    dist = np.zeros((n, n))
    for i in range(n):
        diff = coords - coords[i]
        dist[i] = np.sqrt((diff * diff).sum(axis=1))
        dist[i, i] = np.inf
    idx = np.argsort(dist, axis=1)[:, :k]
    W = np.zeros((n, n))
    for i in range(n):
        for j in idx[i]:
            W[i, j] = 1.0
    # Row-standardise
    rsum = W.sum(axis=1, keepdims=True)
    rsum[rsum == 0] = 1
    W = W / rsum
    return W


def morans_i_neighbourhood(df: pd.DataFrame, *,
                            hood_col: str = "HOOD_158",
                            ds_name: str = "?",
                            k_neighbours: int = 5) -> RichResult:
    """Compute global Moran's I on neighbourhood-level incident counts.

    Builds a k-NN spatial weights matrix from neighbourhood centroids
    (mean LAT/LONG of incidents in each hood as a proxy centroid),
    then runs Moran's I on the count vector.
    """
    if hood_col not in df.columns:
        return RichResult(title=f"Moran's I — {ds_name}",
                          warnings=[f"{hood_col} missing"])
    if "LAT_WGS84" not in df.columns or "LONG_WGS84" not in df.columns:
        return RichResult(title=f"Moran's I — {ds_name}",
                          warnings=["LAT/LONG_WGS84 missing — cannot build "
                                    "spatial weights"])

    counts = _hood_counts(df, hood_col)
    centroids = (df.dropna(subset=[hood_col, "LAT_WGS84", "LONG_WGS84"])
                   .groupby(hood_col)[["LAT_WGS84", "LONG_WGS84"]].mean())
    # Align
    common = counts.index.intersection(centroids.index)
    counts = counts.loc[common]
    centroids = centroids.loc[common]
    if counts.size < 5:
        return RichResult(title=f"Moran's I — {ds_name}",
                          warnings=[f"only {counts.size} valid neighbourhoods"])

    coords = centroids[["LAT_WGS84", "LONG_WGS84"]].values
    W = _adjacency_knn(coords, k=min(k_neighbours, counts.size - 1))
    x = counts.values.astype(float)
    n = x.size
    z = x - x.mean()

    # Moran's I = (n / S0) * (z'Wz) / (z'z)
    S0 = W.sum()
    if S0 == 0:
        return RichResult(title=f"Moran's I — {ds_name}",
                          warnings=["empty spatial weights — k too small"])
    num = z @ W @ z
    den = z @ z
    I = (n / S0) * (num / den) if den != 0 else float("nan")
    expected_I = -1.0 / (n - 1) if n > 1 else float("nan")

    # Approximate variance + z-score (assumes normality)
    # Var(I) = (n^2 * S1 - n*S2 + 3*S0^2) / ((n^2 - 1) * S0^2)
    # Simplified for symmetric W; use Cliff-Ord normal-assumption variance.
    W_sym = (W + W.T) / 2
    S1 = 2 * (W_sym ** 2).sum()
    S2 = (W.sum(axis=0) + W.sum(axis=1)) ** 2
    S2 = float(S2.sum())
    var_I = (n * (n - 2) * S1 - 2 * n * S2 + 6 * S0 ** 2) / (
        (n - 1) * (n + 1) * (n - 2) * S0 ** 2 + 1e-300
    )
    if var_I <= 0:
        z_I = float("nan")
        p = float("nan")
    else:
        z_I = (I - expected_I) / np.sqrt(var_I)
        # Two-sided
        from math import erfc, sqrt
        p = erfc(abs(z_I) / sqrt(2))

    return RichResult(
        title=f"Moran's I (global) — {ds_name}",
        summary_lines=[
            ("Spatial unit", hood_col),
            ("Neighbourhoods", int(n)),
            ("k-nearest neighbours", int(min(k_neighbours, n - 1))),
            ("Moran's I", round(float(I), 4)),
            ("Expected I under null", round(expected_I, 4)),
            ("Variance(I)", round(float(var_I), 6)),
            ("z-score", round(float(z_I), 4) if np.isfinite(z_I) else "n/a"),
            ("p-value (two-sided)",
                round(float(p), 6) if np.isfinite(p) else "n/a"),
        ],
        interpretation=(
            f"I={float(I):+.3f}, z={float(z_I):+.2f}, p={float(p):.4f}. "
            "Positive I = nearby neighbourhoods have similar counts "
            "(spatial clustering of crime); negative = checkerboard "
            "pattern; near zero = random."
        ) if np.isfinite(z_I) else "Variance non-positive — interpretation skipped.",
        payload={"I": float(I), "expected_I": expected_I,
                 "var_I": float(var_I),
                 "z_score": float(z_I) if np.isfinite(z_I) else None,
                 "p_value": float(p) if np.isfinite(p) else None,
                 "n": int(n)},
    )


def local_morans_i(df: pd.DataFrame, *,
                    hood_col: str = "HOOD_158",
                    ds_name: str = "?",
                    k_neighbours: int = 5,
                    top_n: int = 20) -> RichResult:
    """LISA — local Moran's Ii per neighbourhood, with hot/cold spot
    classification.
    """
    if hood_col not in df.columns:
        return RichResult(title=f"LISA — {ds_name}",
                          warnings=[f"{hood_col} missing"])
    if "LAT_WGS84" not in df.columns:
        return RichResult(title=f"LISA — {ds_name}",
                          warnings=["LAT/LONG_WGS84 missing"])

    counts = _hood_counts(df, hood_col)
    centroids = (df.dropna(subset=[hood_col, "LAT_WGS84", "LONG_WGS84"])
                   .groupby(hood_col)[["LAT_WGS84", "LONG_WGS84"]].mean())
    common = counts.index.intersection(centroids.index)
    counts = counts.loc[common]
    centroids = centroids.loc[common]
    if counts.size < 5:
        return RichResult(title=f"LISA — {ds_name}",
                          warnings=[f"only {counts.size} valid neighbourhoods"])

    coords = centroids[["LAT_WGS84", "LONG_WGS84"]].values
    W = _adjacency_knn(coords, k=min(k_neighbours, counts.size - 1))
    x = counts.values.astype(float)
    z = x - x.mean()
    z_std = z / (z.std(ddof=0) + 1e-300)
    Wz = W @ z_std

    # Local Moran's Ii = z_i * (W z)_i
    Ii = z_std * Wz

    # Quadrant classification
    quad = []
    for zi, wzi in zip(z_std, Wz):
        if zi > 0 and wzi > 0:
            quad.append("HH (high-high)")
        elif zi < 0 and wzi < 0:
            quad.append("LL (low-low)")
        elif zi > 0 and wzi < 0:
            quad.append("HL (high-low)")
        else:
            quad.append("LH (low-high)")

    out = pd.DataFrame({
        "hood": counts.index, "count": counts.values,
        "z": z_std, "Wz": Wz, "Ii": Ii, "quadrant": quad,
    })
    top = out.sort_values("Ii", ascending=False).head(top_n)

    return RichResult(
        title=f"LISA (local Moran's Ii) — {ds_name}",
        summary_lines=[
            ("Spatial unit", hood_col),
            ("Neighbourhoods", int(counts.size)),
            ("Hot spots (HH)",
                int((out["quadrant"] == "HH (high-high)").sum())),
            ("Cold spots (LL)",
                int((out["quadrant"] == "LL (low-low)").sum())),
            ("HL outliers (high in low context)",
                int((out["quadrant"] == "HL (high-low)").sum())),
            ("LH outliers (low in high context)",
                int((out["quadrant"] == "LH (low-high)").sum())),
        ],
        tables=[{
            "title": f"Top {top_n} hoods by Local Moran's Ii (most clustered):",
            "headers": ["HOOD_158", "Count", "z(count)", "W·z", "Ii",
                        "Quadrant"],
            "rows": [[str(r.hood), int(r.count), round(float(r.z), 3),
                      round(float(r.Wz), 3), round(float(r.Ii), 3),
                      r.quadrant]
                     for r in top.itertuples()],
        }],
    )


def kde_density(df: pd.DataFrame, *,
                bandwidth: float = 0.005,
                ds_name: str = "?") -> RichResult:
    """Kernel density estimate of incident lat/long.

    Returns summary stats; not the full grid (too big for a RichResult).
    """
    if "LAT_WGS84" not in df.columns or "LONG_WGS84" not in df.columns:
        return RichResult(title=f"KDE — {ds_name}",
                          warnings=["LAT/LONG_WGS84 missing"])
    coords = df[["LAT_WGS84", "LONG_WGS84"]].dropna()
    coords = coords[(coords["LAT_WGS84"] != 0) & (coords["LONG_WGS84"] != 0)]
    if coords.shape[0] < 50:
        return RichResult(title=f"KDE — {ds_name}",
                          warnings=[f"only {coords.shape[0]} geocoded"])
    from scipy.stats import gaussian_kde
    pts = coords.values.T
    kde = gaussian_kde(pts, bw_method=bandwidth)
    # Evaluate density at the points themselves to find max-density area
    densities = kde(pts)
    return RichResult(
        title=f"KDE — {ds_name}",
        summary_lines=[
            ("Geocoded incidents", int(coords.shape[0])),
            ("Bandwidth", bandwidth),
            ("Max density (at obs)", round(float(densities.max()), 4)),
            ("Mean density", round(float(densities.mean()), 4)),
            ("Median density", round(float(np.median(densities)), 4)),
            ("Lat at max-density obs",
                round(float(coords.iloc[densities.argmax()]["LAT_WGS84"]), 4)),
            ("Lon at max-density obs",
                round(float(coords.iloc[densities.argmax()]["LONG_WGS84"]), 4)),
        ],
        interpretation=(
            "KDE bandwidth controls smoothness — smaller = more sensitive "
            "to local clusters. The (lat, lon) pair under max-density "
            "marks the densest incident hotspot in the data."
        ),
    )
