"""moirais.tps_crime — cross-category crime analyses for TPS datasets.

Builds on top of moirais.tps_datasets + tps_io to compare incidents
ACROSS the 13 TPS categories. Useful for:
- Comparing year-over-year trends side-by-side
- Identifying neighbourhoods that are 'hot' across many crime types
- Computing bivariate spatial correlation (Moran's BV) between two
  crime categories on the same neighbourhoods
- Building a composite crime-risk index per neighbourhood

All emit RichResult.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .fn._richresult import RichResult
from .tps_datasets import TPS_REGISTRY, load_tps_dataset


def _hood_counts(df: pd.DataFrame, col: str = "HOOD_158") -> pd.Series:
    """Counts per neighbourhood, dropping 'NSA' / unknowns."""
    if col not in df.columns:
        return pd.Series(dtype=int)
    s = df[col].dropna()
    s = s[s.astype(str).str.upper() != "NSA"]
    return s.value_counts()


def yoy_panel(categories: list[str] | None = None,
              *, sample_rows: int | None = 50_000) -> RichResult:
    """Side-by-side year-over-year panel across TPS categories."""
    cats = categories or sorted(TPS_REGISTRY.keys())
    series = {}
    for cat in cats:
        try:
            df = load_tps_dataset(cat, nrows=sample_rows)
            yc = "OCC_YEAR" if "OCC_YEAR" in df.columns else (
                "REPORT_YEAR" if "REPORT_YEAR" in df.columns else None
            )
            if yc:
                s = df.groupby(yc).size()
                s = s[(s.index >= 1990) & (s.index <= 2030)]
                series[cat] = s
        except Exception:
            pass
    if not series:
        return RichResult(
            title="TPS YoY panel — cross-category",
            warnings=["no datasets loaded with year column"],
        )
    panel = pd.concat(series, axis=1).fillna(0).astype(int).sort_index()
    return RichResult(
        title="TPS — year-over-year panel (cross-category)",
        summary_lines=[
            ("Categories", len(series)),
            ("Years", f"{int(panel.index.min())}–{int(panel.index.max())}"),
            ("Total category-years", int(panel.shape[0] * panel.shape[1])),
        ],
        tables=[{
            "title": "Incidents per OCC_YEAR (rows) × Category (cols):",
            "headers": ["OCC_YEAR"] + list(panel.columns),
            "rows": [[int(y)] + list(map(int, row))
                     for y, row in panel.iterrows()],
        }],
    )


def composite_index(categories: list[str] | None = None,
                    *, sample_rows: int | None = 50_000,
                    weights: dict[str, float] | None = None,
                    top_n: int = 25) -> RichResult:
    """Build a composite per-neighbourhood crime-risk index.

    For each category, count incidents per HOOD_158, z-standardise
    across neighbourhoods, then sum (or weight-and-sum) to get a single
    composite. Higher index = neighbourhood with elevated incidence
    across many crime types.

    `weights` lets you tune (e.g. weights={'Homicides': 5.0,
    'BicycleTheft': 0.5}). Default = unit weights for all loaded cats.
    """
    cats = categories or sorted(TPS_REGISTRY.keys())
    z_panels = []
    used = []
    for cat in cats:
        try:
            df = load_tps_dataset(cat, nrows=sample_rows)
            counts = _hood_counts(df)
            if counts.size < 5:
                continue
            mu = counts.mean()
            sd = counts.std(ddof=0) or 1.0
            z = (counts - mu) / sd
            z.name = cat
            z_panels.append(z)
            used.append(cat)
        except Exception:
            pass
    if not z_panels:
        return RichResult(
            title="TPS composite index",
            warnings=["no usable categories"],
        )
    panel = pd.concat(z_panels, axis=1).fillna(0)
    w = weights or {c: 1.0 for c in panel.columns}
    weight_vec = np.array([w.get(c, 0.0) for c in panel.columns])
    composite = panel.values @ weight_vec
    out = pd.DataFrame({
        "hood": panel.index, "composite": composite,
    }).sort_values("composite", ascending=False)
    return RichResult(
        title="TPS — composite crime-risk index per neighbourhood",
        summary_lines=[
            ("Categories used", len(used)),
            ("Neighbourhoods scored", int(panel.shape[0])),
            ("Mean weight", round(float(weight_vec.mean()), 3)),
            ("Mean composite", round(float(composite.mean()), 3)),
            ("Max composite",
                round(float(composite.max()), 3)),
        ],
        tables=[{
            "title": f"Top {top_n} neighbourhoods (highest composite):",
            "headers": ["HOOD_158", "Composite z-sum"],
            "rows": [[str(r.hood), round(float(r.composite), 3)]
                     for r in out.head(top_n).itertuples()],
        }, {
            "title": f"Bottom {top_n} neighbourhoods (lowest composite):",
            "headers": ["HOOD_158", "Composite z-sum"],
            "rows": [[str(r.hood), round(float(r.composite), 3)]
                     for r in out.tail(top_n)[::-1].itertuples()],
        }],
        interpretation=(
            "Composite is the unweighted (or weighted) sum of "
            "z-standardised counts across all loaded TPS categories. "
            "Positive = neighbourhood with elevated incidence across "
            "the included crime types; near-zero = average; negative = "
            "below-average exposure across categories."
        ),
        payload={"used": used,
                 "ranking": out.head(50).to_dict("records")},
    )


def bivariate_morans_i(cat_a: str, cat_b: str,
                       *, sample_rows: int | None = 50_000,
                       k_neighbours: int = 5) -> RichResult:
    """Bivariate Moran's I — does category A's count in a hood
    correlate with category B's count in NEIGHBOURING hoods?
    """
    df_a = load_tps_dataset(cat_a, nrows=sample_rows)
    df_b = load_tps_dataset(cat_b, nrows=sample_rows)
    a = _hood_counts(df_a)
    b = _hood_counts(df_b)
    common = a.index.intersection(b.index)
    if common.size < 5:
        return RichResult(
            title=f"Bivariate Moran's I — {cat_a} vs {cat_b}",
            warnings=[f"only {common.size} common hoods"],
        )
    a = a.loc[common].astype(float)
    b = b.loc[common].astype(float)
    # Centroid per hood from category A's lat/lon (close enough)
    cents = (df_a.dropna(subset=["HOOD_158", "LAT_WGS84", "LONG_WGS84"])
                  .groupby("HOOD_158")[["LAT_WGS84", "LONG_WGS84"]].mean())
    cents = cents.loc[cents.index.intersection(common)]
    coords = cents[["LAT_WGS84", "LONG_WGS84"]].values
    n = coords.shape[0]
    if n < 5:
        return RichResult(
            title=f"Bivariate Moran's I — {cat_a} vs {cat_b}",
            warnings=[f"only {n} hoods with valid centroids"],
        )
    # k-NN row-standardised W
    dist = np.zeros((n, n))
    for i in range(n):
        diff = coords - coords[i]
        dist[i] = np.sqrt((diff * diff).sum(axis=1))
        dist[i, i] = np.inf
    idx = np.argsort(dist, axis=1)[:, : min(k_neighbours, n - 1)]
    W = np.zeros((n, n))
    for i in range(n):
        for j in idx[i]:
            W[i, j] = 1.0
    rsum = W.sum(axis=1, keepdims=True)
    rsum[rsum == 0] = 1
    W = W / rsum
    # Standardise both vars to common index
    a = a.reindex(cents.index).values
    b = b.reindex(cents.index).values
    z_a = (a - a.mean()) / (a.std(ddof=0) + 1e-300)
    z_b = (b - b.mean()) / (b.std(ddof=0) + 1e-300)
    Wz_b = W @ z_b
    # Bivariate Moran's I_AB = sum(z_A_i * (W z_B)_i) / n
    I_ab = float(np.mean(z_a * Wz_b))
    # Pearson r as comparison (no spatial lag)
    r = float(np.corrcoef(a, b)[0, 1])
    return RichResult(
        title=f"Bivariate Moran's I — {cat_a} ↔ {cat_b}",
        summary_lines=[
            ("Hoods compared", int(n)),
            ("Bivariate Moran's I_AB", round(I_ab, 4)),
            ("Pearson r (non-spatial)", round(r, 4)),
            ("k-nearest neighbours", min(k_neighbours, n - 1)),
            ("z-A range",
                f"{z_a.min():.2f} … {z_a.max():.2f}"),
            ("z-B range",
                f"{z_b.min():.2f} … {z_b.max():.2f}"),
        ],
        interpretation=(
            f"I_AB={I_ab:+.3f}, Pearson r={r:+.3f}. Positive I_AB means "
            f"{cat_a} counts in a hood track {cat_b} counts in NEIGHBOURING "
            "hoods (spatial spillover). Compare to Pearson r to see if the "
            "association is purely co-located vs spatially-lagged."
        ),
    )


def category_correlation_matrix(*, sample_rows: int | None = 50_000) -> RichResult:
    """Pearson correlation of per-hood incident counts across all 13
    TPS categories.
    """
    cats = sorted(TPS_REGISTRY.keys())
    series = {}
    for c in cats:
        try:
            df = load_tps_dataset(c, nrows=sample_rows)
            series[c] = _hood_counts(df)
        except Exception:
            pass
    if not series:
        return RichResult(title="TPS correlation matrix",
                          warnings=["no data"])
    panel = pd.concat(series, axis=1).fillna(0).astype(float)
    corr = panel.corr().round(3)
    return RichResult(
        title="TPS — per-hood correlation across categories",
        summary_lines=[
            ("Categories", len(series)),
            ("Hoods (union)", int(panel.shape[0])),
        ],
        tables=[{
            "title": "Pearson r between per-hood counts (categories ↔ categories):",
            "headers": [""] + list(corr.columns),
            "rows": [[idx] + [float(v) for v in row]
                     for idx, row in corr.iterrows()],
        }],
    )
