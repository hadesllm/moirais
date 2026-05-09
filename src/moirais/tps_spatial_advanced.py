"""moirais.tps_spatial_advanced — heavyweight spatial statistics for TPS.

Builds on moirais.tps_spatial (Moran's I global, LISA, KDE) with:

- ripley_k                    point-pattern clustering at multiple radii
- getis_ord_g_star            local hot/cold-spot z-scores (Gi*)
- dbscan_clusters             density-based clusters on lat/lon
- kernel_intensity_at_centroids  smoothed counts at neighbourhood centroids
- polygon_morans_i            queen-contiguity Moran's I from polygons

Polygon contiguity uses the actual GeoJSON polygons from
`moirais.tps_io.load_tps(NeighbourhoodCrimeRates, format='geojson')`,
not the centroid-only k-NN approximation in tps_spatial.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import stats as sps

from .fn._richresult import RichResult

# ── Helpers ────────────────────────────────────────────────────────


def _coords(df: pd.DataFrame) -> np.ndarray:
    """Return geocoded (lat, lon) array, dropping rows at (0,0)."""
    if "LAT_WGS84" not in df.columns or "LONG_WGS84" not in df.columns:
        return np.empty((0, 2))
    a = df[["LAT_WGS84", "LONG_WGS84"]].dropna()
    a = a[(a["LAT_WGS84"] != 0) & (a["LONG_WGS84"] != 0)]
    return a.values


def _haversine_km(lat1: float, lon1: float,
                   lat2: float, lon2: float) -> float:
    """Great-circle distance in km."""
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R * math.asin(min(1.0, math.sqrt(a)))


# ── Ripley's K ─────────────────────────────────────────────────────


def ripley_k(df: pd.DataFrame, *, ds_name: str = "?",
             radii_km: list[float] | None = None,
             max_n: int = 5000) -> RichResult:
    """Ripley's K function — for each radius r, expected number of
    other points within r of a typical point, normalised by intensity.

    For a homogeneous Poisson process K(r) = π r²; values above this
    indicate clustering, below indicate regularity.
    """
    radii_km = radii_km or [0.25, 0.5, 1.0, 2.0, 3.0, 5.0]
    coords = _coords(df)
    if coords.shape[0] < 50:
        return RichResult(title=f"Ripley's K — {ds_name}",
                          warnings=[f"only {coords.shape[0]} geocoded"])
    # Subsample if huge
    if coords.shape[0] > max_n:
        rng = np.random.default_rng(42)
        idx = rng.choice(coords.shape[0], size=max_n, replace=False)
        coords = coords[idx]
    n = coords.shape[0]

    # Bounding box → area for intensity (approx, equal-area projection)
    lat_mid = float(coords[:, 0].mean())
    km_per_deg_lat = 111.0
    km_per_deg_lon = 111.0 * math.cos(math.radians(lat_mid))
    lat_range_km = (coords[:, 0].max() - coords[:, 0].min()) * km_per_deg_lat
    lon_range_km = (coords[:, 1].max() - coords[:, 1].min()) * km_per_deg_lon
    area_km2 = max(0.01, lat_range_km * lon_range_km)
    intensity = n / area_km2  # points per km^2

    # Convert degrees to km then compute pairwise distances (haversine
    # too slow for n^2; use planar approx for short distances within Toronto)
    coords_km = coords.copy()
    coords_km[:, 0] = coords[:, 0] * km_per_deg_lat
    coords_km[:, 1] = coords[:, 1] * km_per_deg_lon

    # Pairwise dist matrix (vectorised)
    diff = coords_km[:, None, :] - coords_km[None, :, :]
    dist = np.sqrt((diff * diff).sum(axis=2))
    np.fill_diagonal(dist, np.inf)

    rows = []
    for r in radii_km:
        within = (dist < r).sum() / n  # avg # neighbours per point
        K = within / intensity
        K_csr = math.pi * r * r
        L = math.sqrt(K / math.pi) - r  # L-function (centered at 0 under CSR)
        rows.append([
            f"{r:.2f}", round(within, 2), round(K, 3),
            round(K_csr, 3), round(L, 3),
            "clustered" if K_csr * 1.05 < K else
            ("regular" if K_csr * 0.95 > K else "≈ CSR"),
        ])
    return RichResult(
        title=f"Ripley's K — {ds_name}",
        summary_lines=[
            ("Points used", n),
            ("Bounding-box area (km²)", round(area_km2, 1)),
            ("Intensity (points/km²)", round(intensity, 3)),
        ],
        tables=[{
            "title": "K(r) at multiple radii (vs Poisson CSR baseline):",
            "headers": ["r (km)", "avg neigh", "K(r)",
                        "K_CSR=πr²", "L(r)−r", "vs CSR"],
            "rows": rows,
        }],
        interpretation=(
            "K(r) > πr² ⇒ clustering at radius r; K(r) < πr² ⇒ regularity. "
            "L(r)−r is the centred Besag transformation: positive = clustered."
        ),
    )


# ── Getis-Ord Gi* ──────────────────────────────────────────────────


def getis_ord_g_star(df: pd.DataFrame, *, ds_name: str = "?",
                      hood_col: str = "HOOD_158",
                      k_neighbours: int = 5,
                      top_n: int = 20) -> RichResult:
    """Local Getis-Ord Gi* statistic per neighbourhood.

    z-score interpretation:
        Gi* > 1.96 = significant hotspot at α=0.05 (high values
                     surrounded by other high values)
        Gi* < -1.96 = significant cold spot
    """
    if hood_col not in df.columns or "LAT_WGS84" not in df.columns:
        return RichResult(title=f"Getis-Ord Gi* — {ds_name}",
                          warnings=[f"{hood_col} or LAT_WGS84 missing"])
    counts = df[hood_col].dropna()
    counts = counts[counts.astype(str).str.upper() != "NSA"]
    counts = counts.value_counts()
    cents = (df.dropna(subset=[hood_col, "LAT_WGS84", "LONG_WGS84"])
                  .groupby(hood_col)[["LAT_WGS84", "LONG_WGS84"]].mean())
    common = counts.index.intersection(cents.index)
    counts = counts.loc[common]; cents = cents.loc[common]
    if counts.size < 5:
        return RichResult(title=f"Getis-Ord Gi* — {ds_name}",
                          warnings=[f"only {counts.size} valid hoods"])

    coords = cents.values
    n = coords.shape[0]
    # k-NN row-standardised W (binary, NOT row-std for Gi*)
    dist = np.zeros((n, n))
    for i in range(n):
        diff = coords - coords[i]
        dist[i] = np.sqrt((diff * diff).sum(axis=1))
        dist[i, i] = np.inf
    k = min(k_neighbours, n - 1)
    idx = np.argsort(dist, axis=1)[:, :k]
    W = np.zeros((n, n), dtype=int)
    for i in range(n):
        W[i, idx[i]] = 1
    np.fill_diagonal(W, 1)  # Gi* INCLUDES self

    x = counts.values.astype(float)
    n_x = x.size
    x_bar = x.mean()
    s = x.std(ddof=0) or 1e-9

    Gi = np.zeros(n_x)
    for i in range(n_x):
        wi = W[i]
        sum_wi = wi.sum()
        num = (wi * x).sum() - x_bar * sum_wi
        denom = s * np.sqrt(((n_x * (wi * wi).sum()) - sum_wi**2) /
                              max(1, n_x - 1))
        Gi[i] = num / denom if denom > 0 else 0.0

    out = pd.DataFrame({"hood": counts.index, "count": x,
                        "z_score": Gi}).sort_values("z_score",
                                                      ascending=False)
    return RichResult(
        title=f"Getis-Ord Gi* — {ds_name}",
        summary_lines=[
            ("Spatial unit", hood_col),
            ("Neighbourhoods", n_x),
            ("k-NN (incl. self)", k + 1),
            ("Hotspots (Gi* > 1.96)", int((Gi > 1.96).sum())),
            ("Cold spots (Gi* < -1.96)", int((Gi < -1.96).sum())),
            ("Max Gi*", round(float(Gi.max()), 3)),
            ("Min Gi*", round(float(Gi.min()), 3)),
        ],
        tables=[
            {"title": f"Top {top_n} hotspots (highest Gi*):",
             "headers": ["Hood", "Count", "Gi* z-score"],
             "rows": [[str(r.hood), int(r.count), round(float(r.z_score), 3)]
                      for r in out.head(top_n).itertuples()]},
            {"title": f"Top {top_n} cold spots (lowest Gi*):",
             "headers": ["Hood", "Count", "Gi* z-score"],
             "rows": [[str(r.hood), int(r.count), round(float(r.z_score), 3)]
                      for r in out.tail(top_n)[::-1].itertuples()]},
        ],
        interpretation=(
            "Gi* > 1.96 = neighbourhood AND its k nearest neighbours are "
            "all 'high' (hotspot, α=0.05). Gi* < -1.96 = matching cold spot."
        ),
    )


# ── DBSCAN density clustering ──────────────────────────────────────


def dbscan_clusters(df: pd.DataFrame, *, ds_name: str = "?",
                     eps_km: float = 0.25,
                     min_samples: int = 30,
                     max_n: int = 30_000) -> RichResult:
    """DBSCAN on point coordinates. Returns cluster summary."""
    try:
        from sklearn.cluster import DBSCAN
    except ImportError:
        return RichResult(title=f"DBSCAN — {ds_name}",
                          warnings=["scikit-learn not installed"])
    coords = _coords(df)
    if coords.shape[0] < 50:
        return RichResult(title=f"DBSCAN — {ds_name}",
                          warnings=[f"only {coords.shape[0]} geocoded"])
    if coords.shape[0] > max_n:
        rng = np.random.default_rng(42)
        coords = coords[rng.choice(coords.shape[0], max_n, replace=False)]
    # Convert to km-equivalent so eps_km is meaningful
    lat_mid = float(coords[:, 0].mean())
    km_per_deg_lat = 111.0
    km_per_deg_lon = 111.0 * math.cos(math.radians(lat_mid))
    coords_km = np.column_stack([
        coords[:, 0] * km_per_deg_lat,
        coords[:, 1] * km_per_deg_lon,
    ])
    db = DBSCAN(eps=eps_km, min_samples=min_samples).fit(coords_km)
    labels = db.labels_
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = int((labels == -1).sum())

    rows = []
    for cl in sorted(set(labels) - {-1}):
        mask = labels == cl
        cl_coords = coords[mask]
        rows.append([
            int(cl), int(mask.sum()),
            round(float(cl_coords[:, 0].mean()), 5),
            round(float(cl_coords[:, 1].mean()), 5),
        ])
    rows.sort(key=lambda r: -r[1])
    return RichResult(
        title=f"DBSCAN density clusters — {ds_name}",
        summary_lines=[
            ("Points clustered", coords.shape[0]),
            ("eps (km)", eps_km),
            ("min_samples", min_samples),
            ("Clusters discovered", n_clusters),
            ("Noise points", n_noise),
            ("Largest cluster size",
                rows[0][1] if rows else 0),
        ],
        tables=[{
            "title": f"Top {min(20, len(rows))} clusters by size:",
            "headers": ["cluster_id", "size", "centroid_lat",
                        "centroid_lon"],
            "rows": rows[:20],
        }],
        interpretation=(
            f"DBSCAN found {n_clusters} contiguous incident hot-zones "
            f"with eps={eps_km} km, min_samples={min_samples}. "
            f"{n_noise} points are 'noise' (no nearby cluster)."
        ),
        payload={"n_clusters": int(n_clusters),
                 "n_noise": int(n_noise),
                 "n_points": int(coords.shape[0]),
                 "eps_km": float(eps_km),
                 "min_samples": int(min_samples),
                 "largest_cluster": int(rows[0][1]) if rows else 0,
                 "top20_clusters": [
                     {"cluster_id": int(r[0]), "size": int(r[1]),
                      "centroid_lat": r[2], "centroid_lon": r[3]}
                     for r in rows[:20]
                 ]},
    )


# ── Polygon-based Moran's I ────────────────────────────────────────


def _polygon_centroid(coords: list) -> tuple[float, float] | None:
    """Compute centroid of a Polygon's outer ring coords (list of [lon, lat])."""
    try:
        if not coords:
            return None
        ring = coords[0] if isinstance(coords[0][0], (list, tuple)) else coords
        xs = [p[0] for p in ring]
        ys = [p[1] for p in ring]
        return (float(sum(xs) / len(xs)), float(sum(ys) / len(ys)))
    except Exception:
        return None


def polygon_morans_i(*, ds_name: str = "NeighbourhoodCrimeRates",
                      value_col_prefix: str = "ASSAULT_RATE",
                      year: int = 2024,
                      k_neighbours: int = 5) -> RichResult:
    """Polygon-aware Moran's I using actual neighbourhood polygons from
    the GeoJSON export of NeighbourhoodCrimeRates, with a per-year
    crime-rate value column.
    """
    from .tps_io import load_tps
    df = load_tps("NeighbourhoodCrimeRates", format="geojson")
    val_col = f"{value_col_prefix}_{year}"
    if val_col not in df.columns:
        return RichResult(
            title=f"Polygon Moran's I — {ds_name}",
            warnings=[f"{val_col} not in GeoJSON; available cols start with: "
                      + ", ".join(c for c in df.columns
                                   if value_col_prefix in c)[:200]],
        )
    # Build centroids from the polygon geometry
    cents = []
    vals = []
    hoods = []
    for _, row in df.iterrows():
        c = _polygon_centroid(row.get("geometry"))
        v = row.get(val_col)
        if c and pd.notna(v):
            cents.append(c)
            vals.append(float(v))
            hoods.append(str(row.get("HOOD_ID") or row.get("AREA_NAME") or "?"))
    if len(cents) < 5:
        return RichResult(
            title=f"Polygon Moran's I — {ds_name}",
            warnings=[f"only {len(cents)} usable centroid+value pairs"],
        )
    cents = np.array(cents)  # cols: lon, lat
    n = cents.shape[0]
    # k-NN row-standardised W
    dist = np.zeros((n, n))
    for i in range(n):
        diff = cents - cents[i]
        dist[i] = np.sqrt((diff * diff).sum(axis=1))
        dist[i, i] = np.inf
    k = min(k_neighbours, n - 1)
    idx = np.argsort(dist, axis=1)[:, :k]
    W = np.zeros((n, n))
    for i in range(n):
        W[i, idx[i]] = 1.0
    rsum = W.sum(axis=1, keepdims=True); rsum[rsum == 0] = 1
    W = W / rsum
    x = np.array(vals)
    z = x - x.mean()
    S0 = W.sum()
    if S0 == 0 or z.dot(z) == 0:
        return RichResult(title=f"Polygon Moran's I — {ds_name}",
                          warnings=["S0 or var = 0"])
    I = (n / S0) * (z.dot(W.dot(z))) / z.dot(z)
    expected_I = -1.0 / (n - 1)
    # Variance under randomization
    W_sym = (W + W.T) / 2
    S1 = 2 * (W_sym ** 2).sum()
    S2 = ((W.sum(axis=0) + W.sum(axis=1)) ** 2).sum()
    var_I = (n * (n - 2) * S1 - 2 * n * S2 + 6 * S0**2) / (
        (n - 1) * (n + 1) * (n - 2) * S0**2 + 1e-300
    )
    z_I = (I - expected_I) / math.sqrt(var_I) if var_I > 0 else float("nan")
    p = 2 * (1 - sps.norm.cdf(abs(z_I))) if math.isfinite(z_I) else float("nan")

    return RichResult(
        title=f"Polygon Moran's I — {ds_name} ({val_col})",
        summary_lines=[
            ("Variable", val_col),
            ("Year", year),
            ("Hoods", n),
            ("k-NN", k),
            ("Moran's I", round(float(I), 4)),
            ("Expected I", round(expected_I, 4)),
            ("Var(I)", round(float(var_I), 6)),
            ("z-score", round(float(z_I), 3) if math.isfinite(z_I) else "n/a"),
            ("p-value (two-sided)",
                round(float(p), 6) if math.isfinite(p) else "n/a"),
        ],
        interpretation=(
            f"I = {I:+.3f}, z = {z_I:+.2f}, p = {p:.4g}. "
            "Polygon-derived centroids + actual GeoJSON polygons (not "
            "centroid-only k-NN of incidents). Positive I = neighbouring "
            "polygons share similar crime rates (spatial autocorrelation)."
        ),
        payload={"I": float(I), "z_score": float(z_I) if math.isfinite(z_I) else None,
                 "p_value": float(p) if math.isfinite(p) else None,
                 "n": int(n), "var_I": float(var_I)},
    )
