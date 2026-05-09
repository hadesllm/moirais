"""moirais.tps_render — clean Toronto map rendering.

Two design rules per the author, 2026-05-07:

1. NO floating neighbourhood text labels on the map (no
   "Downtown Yonge-East", "Yonge-Bay Corridor", "Mimico Queensway", …).
   Hot-spot identification is delivered via the RichResult tables,
   not via on-canvas text.

2. Map is rotated ~17° CLOCKWISE in projected space so Lake Ontario's
   shoreline sits level horizontally — matching the Sigar Li 2022
   "Hotspot Policing for the City of Toronto" poster aesthetic and
   the Hohl 2024 ALMI homicide-cluster map.

Rotation works in cos-latitude-corrected metres (not raw degrees), so
shapes stay metric-true regardless of where in Toronto's bbox they sit.

Public API
----------
``project_xy(lat, lon)``     — degrees → rotated planar metres (km)
``render_choropleth(rate_col, year, ...)``   — polygon choropleth
``render_dbscan(category, ...)``             — point-pattern + clusters
``render_yearly_grid(prefix, years, ...)``   — small-multiples
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------
# Toronto rotation constants
# Centre = downtown intersection (~Yonge/Dundas).
# 17.5° matches the Sigar Li / Hohl-ALMI orientation: the long axis
# of Toronto (Etobicoke→Scarborough) becomes horizontal.
# ----------------------------------------------------------------------

_LAT_C: float = 43.7000
_LON_C: float = -79.4000
_ROT_DEG_CW: float = 17.5
_KM_PER_DEG_LAT: float = 110.574
_KM_PER_DEG_LON: float = 111.320 * math.cos(math.radians(_LAT_C))

PROJECT = Path(__file__).resolve().parents[5]
FIG_DIR = PROJECT / "data/manifest/outputs/figures/diagnostics"


# Per-category palette (matches Hohl 2024 ALMI quad aesthetic + Sigar Li
# 2022 Toronto poster). Two cmaps per category — sequential for rate
# panels, divergent for cluster/LISA panels.
# Toronto's six former-municipality districts (pre-1998 amalgamation),
# used for district-colour layouts. Roughly bbox-defined; close to the
# OFFICIAL community-council boundaries within ~1 km. Order matters
# for legend layout.
TORONTO_DISTRICTS: list[tuple[str, str, dict]] = [
    ("Etobicoke",  "#cfe7c8", dict(lon_max=-79.475)),
    ("North York", "#fff2b3", dict(lat_min=43.745, lon_min=-79.475,
                                     lon_max=-79.275)),
    ("Scarborough","#cfe2f3", dict(lon_min=-79.275)),
    ("York",       "#e8b6cf", dict(lat_max=43.745, lon_min=-79.475,
                                     lon_max=-79.430)),
    ("Old Toronto","#f7c69b", dict(lat_max=43.745, lon_min=-79.430,
                                     lon_max=-79.310)),
    ("East York",  "#d2c2a0", dict(lat_max=43.745, lat_min=43.685,
                                     lon_min=-79.355, lon_max=-79.300)),
]


def _district_for_centroid(lat: float, lon: float) -> str:
    """Return the former-borough district name for a (lat, lon) centroid."""
    for name, _, box in TORONTO_DISTRICTS:
        ok = True
        for k, v in box.items():
            if k == "lon_max" and lon > v: ok = False
            if k == "lon_min" and lon < v: ok = False
            if k == "lat_max" and lat > v: ok = False
            if k == "lat_min" and lat < v: ok = False
        if ok:
            return name
    return "Old Toronto"


CATEGORY_CMAP: dict[str, tuple[str, str]] = {
    "Assault":                       ("YlOrRd",  "RdBu_r"),
    "AutoTheft":                     ("YlGnBu",  "RdBu_r"),
    "BicycleTheft":                  ("BuGn",    "PuOr_r"),
    "BreakandEnter":                 ("Greens",  "RdBu_r"),
    "Homicides":                     ("YlOrBr",  "RdBu_r"),
    "Robbery":                       ("Purples", "PiYG_r"),
    "ShootingAndFirearmDiscarges":   ("Reds",    "RdBu_r"),
    "TheftFromMovingVehicle":        ("PuBu",    "RdBu_r"),
    "TheftOver":                     ("OrRd",    "RdBu_r"),
    "HateCrimes":                    ("RdPu",    "RdBu_r"),
    "IntimatePartnerAndFamilyViolence": ("RdPu", "RdBu_r"),
    "CommunitySafetyIndicators":     ("magma",   "RdBu_r"),
    "NeighbourhoodCrimeRates":       ("YlOrRd",  "RdBu_r"),
}


def pretty_label(s: str) -> str:
    """`ASSAULT_RATE_2024` → `Assault rate · 2024`. Strips underscores
    and casing so titles & legend labels look like prose."""
    parts = s.split("_")
    out: list[str] = []
    for p in parts:
        if p.isdigit() and len(p) == 4:
            out.append(p)             # year stays as e.g. 2024
        else:
            out.append(p.lower())
    if not out:
        return s
    out[0] = out[0].capitalize()
    return " · ".join([" ".join(out[:-1]).strip(), out[-1]]) \
        if (out[-1].isdigit() and len(out) > 1) else " ".join(out)


def project_xy(lat: np.ndarray | float,
               lon: np.ndarray | float,
               *, rot_deg_cw: float = _ROT_DEG_CW,
               lat_c: float = _LAT_C,
               lon_c: float = _LON_C) -> tuple[np.ndarray, np.ndarray]:
    """Project (lat, lon) → (x_km, y_km) with a clockwise rotation.

    Equirectangular projection centred at (lat_c, lon_c), then rotated
    `rot_deg_cw` degrees clockwise. Returns kilometres east-of-centre
    (after rotation) and kilometres north-of-centre (after rotation).

    Clockwise convention: positive `rot_deg_cw` rotates the map so a
    line that previously sloped up-right slopes less (or down-right).
    """
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    x = (lon - lon_c) * _KM_PER_DEG_LON
    y = (lat - lat_c) * _KM_PER_DEG_LAT
    # Clockwise rotation matrix (x', y') = (x cosα + y sinα, -x sinα + y cosα)
    a = math.radians(rot_deg_cw)
    cs, sn = math.cos(a), math.sin(a)
    return x * cs + y * sn, -x * sn + y * cs


def _setup_axes(ax, *, title: str, basemap_alpha: float = 0.0):
    """Common axis setup — equal aspect, no degree ticks, light grid."""
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("east of centre (km)")
    ax.set_ylabel("north of centre (km)")
    ax.set_title(title)
    ax.grid(True, alpha=0.15, linestyle=":")


def _polygon_pieces(df) -> list[tuple[np.ndarray, dict]]:
    """Return [(xy_km, row_attrs)] per ring after Toronto-bbox filter."""
    out: list[tuple[np.ndarray, dict]] = []
    for _, row in df.iterrows():
        geom = row.get("geometry")
        if geom is None:
            continue
        rings = geom if isinstance(geom[0][0], list) else [geom]
        for ring in rings:
            arr = np.asarray(ring, dtype=float)
            if arr.ndim != 2 or arr.shape[1] < 2:
                continue
            lats, lons = arr[:, 1], arr[:, 0]
            if (lats.min() < 43.55 or lats.max() > 43.90
                    or lons.min() < -79.65 or lons.max() > -79.10):
                continue
            xk, yk = project_xy(lats, lons)
            out.append((np.column_stack([xk, yk]),
                          {k: row.get(k) for k in row.index}))
    return out


def render_choropleth(*, rate_col: str = "ASSAULT_RATE_2024",
                       title: str | None = None,
                       cmap: str | None = None,
                       outfile: str | Path | None = None,
                       fig_h: float = 7.0,
                       fig_w: float = 12.0,
                       show_ids: bool = True,
                       border_color: str = "#1a1a1a",
                       border_lw: float = 0.7) -> Path:
    """Polygon choropleth using NeighbourhoodCrimeRates GeoJSON polys.

    Style matches the Hohl 2024 ALMI / Sigar Li 2022 Toronto poster:
    - thick black polygon borders (`border_color`/`border_lw`)
    - sequential cmap chosen per category (default Reds family)
    - small numeric polygon-ID labels (`show_ids=True`) — IDs only,
      never neighbourhood names per the author 2026-05-07
    - title and colour-bar legend with proper spaces (no underscores)
    """
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection

    from .tps_io import load_tps

    df = load_tps("NeighbourhoodCrimeRates", format="geojson")
    if rate_col not in df.columns:
        raise KeyError(f"{rate_col!r} not in NeighbourhoodCrimeRates "
                        f"({len(df.columns)} cols).")
    pieces = _polygon_pieces(df)
    polys = [p for p, _ in pieces]
    vals: list[float] = []
    for _, attrs in pieces:
        try:
            vals.append(float(attrs.get(rate_col)))
        except (TypeError, ValueError):
            vals.append(np.nan)

    if cmap is None:
        cmap = "YlOrRd"

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    pc = PolyCollection(polys, array=np.asarray(vals),
                          edgecolor=border_color, linewidth=border_lw,
                          cmap=cmap)
    ax.add_collection(pc)
    ax.autoscale_view()
    cb = fig.colorbar(pc, ax=ax, shrink=0.8)
    cb.set_label(pretty_label(rate_col))

    if show_ids:
        import matplotlib.patheffects as pe
        for poly, attrs in pieces:
            hood_id = attrs.get("HOOD_ID") or attrs.get("AREA_ID")
            if hood_id is None or pd.isna(hood_id):
                continue
            cx, cy = poly.mean(axis=0)
            # bbox of this poly in km — used to scale font so labels stay inside
            w = float(poly[:, 0].max() - poly[:, 0].min())
            h = float(poly[:, 1].max() - poly[:, 1].min())
            small = min(w, h)
            # 3-tier scale: tiny downtown wards (< 0.8 km) use 4 pt;
            # mid-size 5 pt; large 6 pt. White halo keeps it readable
            # over any base colour without using a background box.
            if small < 0.8:
                fs = 4.0
            elif small < 1.5:
                fs = 5.0
            else:
                fs = 6.0
            ax.text(cx, cy, str(int(hood_id)),
                     ha="center", va="center",
                     fontsize=fs, color="#0a0a0a", fontweight="bold",
                     clip_on=True,
                     path_effects=[pe.withStroke(linewidth=1.6,
                                                  foreground="white")])

    _setup_axes(ax, title=title
                  or f"Toronto · {pretty_label(rate_col)} · 158 wards")
    out = Path(outfile) if outfile else FIG_DIR / \
        f"choropleth_{rate_col.lower()}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out


def render_quad(category: str = "Homicides",
                 *, year: int = 2024,
                 outfile: str | Path | None = None,
                 fig_h: float = 9.0,
                 fig_w: float = 13.0,
                 sample_rows: int | None = 30_000,
                 border_color: str = "#1a1a1a",
                 border_lw: float = 0.6) -> Path:
    """Hohl 2024 ALMI-style 4-panel quad for any TPS category.

    Panels (a-d):
      a) Kernel density (incidents per sq km) — sequential
      b) Per-100k rate from NeighbourhoodCrimeRates — sequential
      c) Spatial clusters / outliers (LISA HH/LL/HL/LH) — diverging
      d) Significant Gi* hot/cold spots — diverging
    """
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    from matplotlib.colors import Normalize
    from scipy.stats import gaussian_kde

    from .tps_datasets import load_tps_dataset
    from .tps_io import load_tps
    from .tps_spatial import local_morans_i
    from .tps_spatial_advanced import getis_ord_g_star

    seq_cmap, div_cmap = CATEGORY_CMAP.get(category, ("YlOrRd", "RdBu_r"))

    # ---- polygons (one geometry per ward) -----------------------------
    poly_df = load_tps("NeighbourhoodCrimeRates", format="geojson")
    pieces = _polygon_pieces(poly_df)
    polys = [p for p, _ in pieces]
    poly_attrs = [a for _, a in pieces]

    # ---- incident points --------------------------------------------
    df = load_tps_dataset(category, nrows=sample_rows)
    df = df.dropna(subset=["LAT_WGS84", "LONG_WGS84"]).copy()
    df = df[(df["LAT_WGS84"].between(43.55, 43.90))
            & (df["LONG_WGS84"].between(-79.65, -79.10))]
    pt_x, pt_y = project_xy(df["LAT_WGS84"].to_numpy(),
                                df["LONG_WGS84"].to_numpy())

    fig, axes = plt.subplots(2, 2, figsize=(fig_w, fig_h))
    panel_letters = ["a", "b", "c", "d"]

    # ---- (a) kernel density ------------------------------------------
    axA = axes[0, 0]
    if len(pt_x) >= 50:
        kde = gaussian_kde(np.vstack([pt_x, pt_y]),
                              bw_method="silverman")
        gx = np.linspace(pt_x.min() - 1, pt_x.max() + 1, 220)
        gy = np.linspace(pt_y.min() - 1, pt_y.max() + 1, 180)
        gxx, gyy = np.meshgrid(gx, gy)
        z = kde(np.vstack([gxx.ravel(), gyy.ravel()])).reshape(gxx.shape)
        cs = axA.contourf(gxx, gyy, z, levels=14, cmap=seq_cmap, alpha=0.92)
        fig.colorbar(cs, ax=axA, shrink=0.7,
                       label=f"{category} density (per sq km)")
    pc_outline = PolyCollection(polys, facecolors="none",
                                  edgecolors=border_color,
                                  linewidth=border_lw)
    axA.add_collection(pc_outline)
    axA.autoscale_view()

    # ---- (b) per-100k rate -------------------------------------------
    axB = axes[0, 1]
    cat_prefix_map = {
        "Assault": "ASSAULT",
        "AutoTheft": "AUTOTHEFT",
        "BicycleTheft": "BIKETHEFT",
        "BreakandEnter": "BREAKENTER",
        "Homicides": "HOMICIDE",
        "Robbery": "ROBBERY",
        "ShootingAndFirearmDiscarges": "SHOOTING",
        "TheftFromMovingVehicle": "THEFTFROMMV",
        "TheftOver": "THEFTOVER",
    }
    px = cat_prefix_map.get(category, category.upper())
    rate_col_candidates = [f"{px}_RATE_{year}", f"{category.upper()}_RATE_{year}"]
    rate_col = next((c for c in rate_col_candidates
                     if c in poly_df.columns), None)
    rate_vals: list[float] = []
    if rate_col is not None:
        for _, attrs in pieces:
            try:
                rate_vals.append(float(attrs.get(rate_col)))
            except (TypeError, ValueError):
                rate_vals.append(np.nan)
        pcB = PolyCollection(polys, array=np.asarray(rate_vals),
                               edgecolor=border_color, linewidth=border_lw,
                               cmap=seq_cmap)
        axB.add_collection(pcB)
        fig.colorbar(pcB, ax=axB, shrink=0.7,
                       label=pretty_label(rate_col))
    else:
        axB.text(0.5, 0.5, f"no per-100k {category} {year} column",
                  ha="center", va="center", transform=axB.transAxes)
        pcB = PolyCollection(polys, facecolors="white",
                                edgecolor=border_color, linewidth=border_lw)
        axB.add_collection(pcB)
    axB.autoscale_view()

    # ---- (c) LISA HH/LL/HL/LH ----------------------------------------
    axC = axes[1, 0]
    rr_lisa = local_morans_i(df, ds_name=category, k_neighbours=5,
                                 top_n=200)
    quad_color = {"HH": "#d7191c", "HL": "#fdae61",
                    "LH": "#abd9e9", "LL": "#2c7bb6"}
    hood_q: dict[int, str] = {}
    for tbl in (rr_lisa.tables or []):
        rows = tbl["rows"] if isinstance(tbl, dict) \
            else getattr(tbl, "rows", [])
        for r in rows:
            try:
                # quadrant cells look like "HH (high-high)" — keep "HH"
                hood_q[int(r[0])] = str(r[-1]).strip()[:2].upper()
            except (TypeError, ValueError, IndexError):
                pass
    poly_face = []
    for _, attrs in pieces:
        hid = attrs.get("HOOD_ID") or attrs.get("AREA_ID")
        try:
            hid_i = int(hid)
        except (TypeError, ValueError):
            hid_i = -1
        q = hood_q.get(hid_i)
        poly_face.append(quad_color.get(q, "#f5f5f5"))
    pcC = PolyCollection(polys, facecolors=poly_face,
                           edgecolor=border_color, linewidth=border_lw)
    axC.add_collection(pcC)
    axC.autoscale_view()
    # custom legend for HH/HL/LH/LL
    from matplotlib.patches import Patch
    axC.legend(handles=[Patch(facecolor=v, edgecolor="black",
                                label=k) for k, v in quad_color.items()],
                loc="lower right", fontsize=8, framealpha=0.95, ncols=2,
                title="LISA quadrant")

    # ---- (d) Gi* significant hot/cold spots --------------------------
    axD = axes[1, 1]
    rr_gi = getis_ord_g_star(df, ds_name=category, k_neighbours=5)
    z_by_hood: dict[int, float] = {}
    for tbl in (rr_gi.tables or []):
        rows = tbl["rows"] if isinstance(tbl, dict) \
            else getattr(tbl, "rows", [])
        headers = tbl["headers"] if isinstance(tbl, dict) \
            else getattr(tbl, "headers", [])
        # Column index for z-score: prefer "z(count)" / "z_score" / "z"
        z_idx = next((i for i, h in enumerate(headers)
                       if isinstance(h, str)
                       and h.lower() in ("z_score", "z(count)", "z")), -1)
        for r in rows:
            try:
                z_by_hood[int(r[0])] = float(r[z_idx])
            except (TypeError, ValueError, IndexError):
                pass
    z_vals = []
    for _, attrs in pieces:
        hid = attrs.get("HOOD_ID") or attrs.get("AREA_ID")
        try:
            hid_i = int(hid)
        except (TypeError, ValueError):
            hid_i = -1
        z_vals.append(z_by_hood.get(hid_i, 0.0))
    z_arr = np.asarray(z_vals)
    vmax = float(max(2.5, np.nanmax(np.abs(z_arr)) if z_arr.size else 2.5))
    pcD = PolyCollection(polys, array=z_arr,
                           edgecolor=border_color, linewidth=border_lw,
                           cmap=div_cmap,
                           norm=Normalize(vmin=-vmax, vmax=vmax))
    axD.add_collection(pcD)
    fig.colorbar(pcD, ax=axD, shrink=0.7, label="Gi* z-score")
    axD.autoscale_view()

    # common formatting
    titles = [
        f"{category} density (incidents per sq km)",
        f"{category} rate {year} (per 100k)",
        f"{category} spatial clusters (LISA, k=5)",
        f"{category} significant hot/cold spots (Gi*, k=5)",
    ]
    for letter, ax, title in zip(panel_letters, axes.ravel(), titles,
                                    strict=False):
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.text(0.02, 0.97, letter, transform=ax.transAxes,
                fontsize=14, fontweight="bold", va="top")
        ax.set_title(title, fontsize=11)
    fig.suptitle(
        f"Toronto · {category} · 4-panel spatial analysis "
        f"· 158 wards", fontsize=12, y=0.995)
    out = Path(outfile) if outfile else FIG_DIR / \
        f"quad_{category.lower()}_{year}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out


def render_dbscan(category: str = "Assault",
                   *, eps_km: float = 0.3,
                   min_samples: int = 20,
                   sample_rows: int | None = 30_000,
                   outfile: str | Path | None = None,
                   show_top: int = 12,
                   fig_h: float = 7.5,
                   fig_w: float = 12.0) -> Path:
    """Point-pattern map: incidents + DBSCAN cluster IDs.

    No floating hood labels; clusters are shown by colour with a
    compact legend. Aspect/rotation matches the rest of the suite.
    """
    import matplotlib.pyplot as plt

    from .tps_datasets import load_tps_dataset

    df = load_tps_dataset(category, nrows=sample_rows)
    df = df.dropna(subset=["LAT_WGS84", "LONG_WGS84"]).copy()
    # Toronto bbox sanity filter — drops any rows accidentally at
    # (0,0), in another country, or with junk values.
    df = df[(df["LAT_WGS84"].between(43.55, 43.90))
            & (df["LONG_WGS84"].between(-79.65, -79.10))]
    if df.empty:
        raise ValueError(f"{category}: no in-bbox LAT/LONG rows")

    from sklearn.cluster import DBSCAN
    xk, yk = project_xy(df["LAT_WGS84"].to_numpy(),
                          df["LONG_WGS84"].to_numpy())
    labels = DBSCAN(eps=eps_km, min_samples=min_samples).fit_predict(
        np.column_stack([xk, yk]))

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    noise_mask = labels == -1
    ax.scatter(xk[noise_mask], yk[noise_mask], s=1.5,
                c="#888", alpha=0.25, label=f"noise ({int(noise_mask.sum())})")
    # Top-N clusters by size
    cluster_ids, sizes = np.unique(labels[~noise_mask], return_counts=True)
    order = np.argsort(-sizes)[:show_top]
    palette = plt.cm.tab20(np.linspace(0, 1, len(order)))
    for col, idx in zip(palette, order, strict=False):
        cid = int(cluster_ids[idx])
        m = labels == cid
        ax.scatter(xk[m], yk[m], s=4, color=col,
                    label=f"cluster {cid} (n={int(m.sum())})")
    ax.legend(loc="lower right", fontsize=7, frameon=True,
                facecolor="white", framealpha=0.85, ncols=2)
    n_clusters = int((cluster_ids).size)
    n_noise = int(noise_mask.sum())
    _setup_axes(
        ax,
        title=(f"Toronto {category} — DBSCAN (eps={eps_km}km, "
                f"min={min_samples}) — {n_clusters} clusters, "
                f"{n_noise:,} noise"))
    out = Path(outfile) if outfile else FIG_DIR / \
        f"map_dbscan_{category.lower()}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out


def _draw_compass(ax, *, x_frac: float = 0.93, y_frac: float = 0.83,
                    size_frac: float = 0.10) -> None:
    """North arrow rotated to compensate for the 17.5° CW projection.

    Drawn in axes-fraction coords. The arrow tip points to *true* north,
    which on a 17.5° CW-rotated canvas is up-and-to-the-right (rotated
    17.5° CW from straight-up).
    """
    import matplotlib.patches as mpatches
    cx, cy = x_frac, y_frac
    # On a CW-rotated map, north on the page is rotated 17.5° CCW from
    # straight up — i.e. arrow points slightly LEFT of vertical. But
    # the human-readable convention in the Hohl reference shows north
    # leaning slightly RIGHT (matching the 17.5° CW projection). We
    # follow that convention: arrow direction (dx, dy) using CW angle.
    a = math.radians(_ROT_DEG_CW)
    dx = math.sin(a) * size_frac * 0.8
    dy = math.cos(a) * size_frac * 0.8
    arr = mpatches.FancyArrow(cx - dx / 2, cy - dy / 2, dx, dy,
                                width=size_frac * 0.04,
                                head_width=size_frac * 0.18,
                                head_length=size_frac * 0.22,
                                length_includes_head=True,
                                color="#0d3a8a",
                                transform=ax.transAxes, zorder=8)
    ax.add_patch(arr)
    ax.text(cx + dx, cy + dy, "N",
              ha="center", va="bottom", fontsize=10,
              fontweight="bold", color="#0d3a8a",
              transform=ax.transAxes, zorder=9)


def _draw_scalebar(ax, *, x_frac: float = 0.05, y_frac: float = 0.06,
                     length_km: float = 10.0) -> None:
    """Simple scale bar in km; assumes ax is in projected-km space."""
    x0, y0 = ax.transAxes.transform((x_frac, y_frac))
    inv = ax.transData.inverted()
    dx0, dy0 = inv.transform((x0, y0))
    ax.plot([dx0, dx0 + length_km], [dy0, dy0],
              color="#1a1a1a", linewidth=2.0, zorder=8)
    ax.text(dx0 + length_km / 2, dy0 + 0.4,
              f"{int(length_km)} km", ha="center", va="bottom",
              fontsize=8, color="#1a1a1a",
              path_effects=[__import__("matplotlib.patheffects",
                  fromlist=["pe"]).withStroke(linewidth=2,
                      foreground="white")])


def render_district_proportional(metric: str = "ASSAULT_2024",
                                    *, title: str | None = None,
                                    outfile: str | Path | None = None,
                                    fig_h: float = 7.0,
                                    fig_w: float = 12.5,
                                    border_color: str = "#1a1a1a",
                                    border_lw: float = 0.55) -> Path:
    """District-coloured map with proportional-symbol black circles.

    Inspired by the author's reference (Toronto news-coverage proportional-
    symbol map). Toronto's 158 wards are coloured by former-borough
    district (Etobicoke / North York / Scarborough / York / Old Toronto
    / East York), with a black circle drawn at each ward centroid sized
    proportional to ``metric`` (any column from NeighbourhoodCrimeRates).

    Includes a "Map Divisions" legend with district swatches and a
    "Number of <metric>" graduated-symbol legend with 5 size tiers.
    """
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    from matplotlib.patches import Circle, Rectangle

    from .tps_io import load_tps

    df = load_tps("NeighbourhoodCrimeRates", format="geojson")
    if metric not in df.columns:
        raise KeyError(f"{metric!r} not in NeighbourhoodCrimeRates")
    pieces = _polygon_pieces(df)
    polys = [p for p, _ in pieces]
    poly_attrs = [a for _, a in pieces]

    # Per-polygon district lookup via raw lat/lon centroid (before rotation)
    district_face = []
    centroids_raw = []
    for _, attrs in pieces:
        # raw centroid: pull the geometry again from the row's geometry
        geom = attrs.get("geometry")
        if geom is None:
            district_face.append("#eaeaea")
            centroids_raw.append((43.7, -79.4))
            continue
        rings = geom if isinstance(geom[0][0], list) else [geom]
        # Use the largest ring (outer)
        ring = max(rings, key=lambda r: len(r))
        arr = np.asarray(ring, dtype=float)
        clat = float(arr[:, 1].mean())
        clon = float(arr[:, 0].mean())
        d = _district_for_centroid(clat, clon)
        face = next((c for n, c, _ in TORONTO_DISTRICTS if n == d), "#eee")
        district_face.append(face)
        centroids_raw.append((clat, clon))

    fig = plt.figure(figsize=(fig_w, fig_h))
    ax = fig.add_axes([0.04, 0.06, 0.66, 0.90])

    pc = PolyCollection(polys, facecolors=district_face,
                          edgecolor=border_color, linewidth=border_lw)
    ax.add_collection(pc)

    # Proportional-symbol circles at projected centroids
    vals = np.asarray([float(a.get(metric)) if a.get(metric) is not None
                        else 0.0 for a in poly_attrs])
    finite = vals[np.isfinite(vals) & (vals > 0)]
    if finite.size == 0:
        return Path("")
    vmin, vmax = float(finite.min()), float(finite.max())
    # Square-root scaling so area is proportional to value
    def _radius(v: float) -> float:
        if v <= 0: return 0.0
        return 0.25 + 1.4 * math.sqrt((v - vmin) / max(vmax - vmin, 1e-6))
    for poly, v in zip(polys, vals, strict=False):
        if not (np.isfinite(v) and v > 0):
            continue
        cx, cy = poly.mean(axis=0)
        r = _radius(float(v))
        ax.add_patch(Circle((cx, cy), r,
                              facecolor="#1a1a1a", edgecolor="none",
                              alpha=0.92, zorder=4))

    ax.set_aspect("equal", adjustable="box")
    ax.autoscale_view()
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title or f"Toronto · {pretty_label(metric)} · district map "
                            f"· district map",
                   fontsize=12)

    _draw_compass(ax)
    _draw_scalebar(ax, length_km=10)

    # ------------------------------------------------------------------
    # Right-side legend axes
    leg = fig.add_axes([0.71, 0.06, 0.27, 0.90])
    leg.set_xlim(0, 1); leg.set_ylim(0, 1); leg.axis("off")

    leg.text(0.50, 0.97, "Map Divisions", ha="center", va="top",
              fontsize=11, fontweight="bold")
    for i, (name, color, _) in enumerate(TORONTO_DISTRICTS):
        y = 0.92 - i * 0.045
        leg.add_patch(Rectangle((0.12, y - 0.018), 0.10, 0.034,
                                  facecolor=color, edgecolor="#1a1a1a",
                                  linewidth=0.6, transform=leg.transAxes))
        leg.text(0.27, y, name, va="center", fontsize=9.5)

    # Graduated-symbol legend
    leg.text(0.50, 0.55, f"Number of {pretty_label(metric)}",
              ha="center", va="top",
              fontsize=11, fontweight="bold")
    tiers = np.linspace(vmin, vmax, 5)
    for i, t in enumerate(tiers):
        y = 0.50 - i * 0.082
        r_pts = _radius(float(t))
        leg.add_patch(Circle((0.18, y), r_pts * 0.012,
                                facecolor="#1a1a1a", edgecolor="none",
                                transform=leg.transAxes))
        leg.text(0.34, y, f"{t:.0f}", va="center", fontsize=9)

    out = Path(outfile) if outfile else FIG_DIR / \
        f"district_prop_{metric.lower()}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out


def render_satscan_panel(category: str = "Homicides",
                          *, sample_rows: int | None = 30_000,
                          n_clusters: int = 2,
                          n_mc: int = 199,
                          year_window_yrs: int = 4,
                          outfile: str | Path | None = None,
                          fig_h: float = 7.5,
                          fig_w: float = 11.5,
                          border_color: str = "#1a1a1a",
                          border_lw: float = 0.55) -> Path:
    """Hohl 2024 ALMI "panel d" — Kulldorff-style space-time scan
    statistic with significant clusters as pink filled circles, red
    dots inside for sig. locations, yellow polygons for sig. wards,
    and an info-box (n_cases / expected / RR / p-value) on the right.

    Implementation is a simplified Kulldorff Poisson scan over a grid
    of candidate (x, y, r, t-window) cells: for each cell compute a
    log-likelihood ratio assuming Poisson(λ_in) inside vs Poisson(λ_out)
    outside; null distribution by Monte-Carlo (`n_mc` permutations of
    incident timestamps within the bbox).
    """
    import matplotlib.patheffects as pe
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    from matplotlib.patches import Circle

    from .tps_datasets import load_tps_dataset
    from .tps_io import load_tps

    poly_df = load_tps("NeighbourhoodCrimeRates", format="geojson")
    pieces = _polygon_pieces(poly_df)
    polys = [p for p, _ in pieces]
    poly_attrs = [a for _, a in pieces]

    df = load_tps_dataset(category, nrows=sample_rows)
    df = df.dropna(subset=["LAT_WGS84", "LONG_WGS84", "OCC_DATE"]).copy()
    df = df[(df["LAT_WGS84"].between(43.55, 43.90))
             & (df["LONG_WGS84"].between(-79.65, -79.10))]
    df["_dt"] = pd.to_datetime(df["OCC_DATE"], errors="coerce")
    df = df.dropna(subset=["_dt"])
    if df.empty or len(df) < 50:
        return Path("")

    pt_x, pt_y = project_xy(df["LAT_WGS84"].to_numpy(),
                                df["LONG_WGS84"].to_numpy())
    years = df["_dt"].dt.year.to_numpy()
    n_total = len(df)
    yr_min, yr_max = int(years.min()), int(years.max())

    # Candidate centres = polygon centroids; r ∈ {1, 2, 3, 5, 8} km
    cents = np.array([p.mean(axis=0) for p in polys])
    radii_km = [1.0, 2.0, 3.0, 5.0, 8.0]

    def _llr(n_in, n_out, mu_in, mu_out):
        """Kulldorff Poisson LLR (returns 0 if n_in / mu_in ≤ n_out / mu_out)"""
        if n_in == 0 or mu_in <= 0 or mu_out <= 0:
            return 0.0
        if n_in / mu_in <= n_out / mu_out:
            return 0.0
        a = n_in * math.log(n_in / mu_in)
        b = n_out * math.log(n_out / mu_out) if n_out > 0 else 0.0
        return a + b - (n_in + n_out) * math.log(
            (n_in + n_out) / (mu_in + mu_out))

    # Time windows of length `year_window_yrs` sliding through (yr_min, yr_max)
    twins = []
    for ts in range(yr_min, yr_max - year_window_yrs + 2):
        twins.append((ts, ts + year_window_yrs - 1))
    if not twins:
        twins = [(yr_min, yr_max)]

    # Pre-build per-cluster candidate lookup. For Kulldorff baseline mu_in
    # we use space-time fraction: mu_in = n_total · |inside_area · twindow|
    bbox_area = ((pt_x.max() - pt_x.min())
                 * (pt_y.max() - pt_y.min())) or 1.0
    span_yrs = (yr_max - yr_min + 1)

    candidates = []
    for ci, c in enumerate(cents):
        for r in radii_km:
            inside_space = ((pt_x - c[0]) ** 2 + (pt_y - c[1]) ** 2) <= r ** 2
            for (t0, t1) in twins:
                inside_time = (years >= t0) & (years <= t1)
                inside = inside_space & inside_time
                n_in = int(inside.sum())
                if n_in < 5:
                    continue
                mu_in = n_total * (math.pi * r ** 2 / bbox_area) \
                    * ((t1 - t0 + 1) / span_yrs)
                mu_in = max(mu_in, 1e-3)
                mu_out = n_total - mu_in
                n_out = n_total - n_in
                llr = _llr(n_in, n_out, mu_in, mu_out)
                if llr > 0:
                    candidates.append((llr, ci, r, t0, t1, n_in, mu_in))
    if not candidates:
        return Path("")
    candidates.sort(key=lambda r: -r[0])

    # Greedy non-overlapping top-K
    chosen = []
    for cand in candidates:
        llr, ci, r, t0, t1, n_in, mu_in = cand
        c = cents[ci]
        if any(((c[0] - cents[c2[1]][0]) ** 2
                 + (c[1] - cents[c2[1]][1]) ** 2)
                < max(r, c2[2]) ** 2
                 for c2 in chosen):
            continue
        chosen.append(cand)
        if len(chosen) >= n_clusters:
            break

    # Monte-Carlo p-values for each chosen cluster
    rng = np.random.default_rng(13)
    null_max_llr = []
    for _ in range(n_mc):
        perm_years = rng.permutation(years)
        best = 0.0
        for cand in candidates[:200]:  # check just the top-200 cells
            _, ci, r, t0, t1, n_in_obs, mu_in = cand
            c = cents[ci]
            inside_space = ((pt_x - c[0]) ** 2 + (pt_y - c[1]) ** 2) <= r ** 2
            inside_time = (perm_years >= t0) & (perm_years <= t1)
            n_in = int((inside_space & inside_time).sum())
            n_out = n_total - n_in
            mu_out = n_total - mu_in
            llr = _llr(n_in, n_out, mu_in, mu_out)
            if llr > best:
                best = llr
        null_max_llr.append(best)

    fig = plt.figure(figsize=(fig_w, fig_h))
    ax = fig.add_axes([0.04, 0.04, 0.62, 0.92])
    # info axis is a transparent canvas in figure-fraction so that
    # ConnectionPatch coords align with the boxes we draw on it
    info = fig.add_axes([0.68, 0.04, 0.30, 0.92])
    info.set_xlim(0, 1)
    info.set_ylim(0, 1)
    info.axis("off")

    pc_base = PolyCollection(polys, facecolors="white",
                                 edgecolor=border_color, linewidth=border_lw)
    ax.add_collection(pc_base)

    # Yellow-highlight wards inside any chosen cluster
    yellow_idx = set()
    for cand in chosen:
        _, ci, r, _, _, _, _ = cand
        c = cents[ci]
        for pi_, p in enumerate(polys):
            cx, cy = p.mean(axis=0)
            if (cx - c[0]) ** 2 + (cy - c[1]) ** 2 <= r ** 2:
                yellow_idx.add(pi_)
    pc_yellow = PolyCollection([polys[i] for i in yellow_idx],
                                  facecolors="#fff7a8",
                                  edgecolor=border_color,
                                  linewidth=border_lw + 0.2)
    ax.add_collection(pc_yellow)

    # Tiny ward IDs (Hohl-style) — same adaptive sizing as choropleths
    for poly, attrs in zip(polys, poly_attrs, strict=False):
        hood_id = attrs.get("HOOD_ID") or attrs.get("AREA_ID")
        if hood_id is None or pd.isna(hood_id):
            continue
        cx, cy = poly.mean(axis=0)
        w = float(poly[:, 0].max() - poly[:, 0].min())
        h = float(poly[:, 1].max() - poly[:, 1].min())
        small = min(w, h)
        fs = 4.0 if small < 0.8 else 5.0 if small < 1.5 else 6.0
        ax.text(cx, cy, str(int(hood_id)), ha="center", va="center",
                  fontsize=fs, color="#0a0a0a", fontweight="bold",
                  clip_on=True, zorder=3,
                  path_effects=[pe.withStroke(linewidth=1.6,
                                               foreground="white")])

    from matplotlib.patches import ConnectionPatch, FancyBboxPatch

    sig_loc_count = 0
    info.text(0.5, 0.985,
                f"{category} · Space-time scan (Kulldorff)",
                ha="center", va="top",
                fontsize=10.5, fontweight="bold", color="#0d3a8a")
    info.text(0.5, 0.955,
                f"{year_window_yrs}-yr window · {yr_min}–{yr_max} · "
                f"{n_mc} MC permutations",
                ha="center", va="top",
                fontsize=8, color="#444")

    # Reserve top y=0.92 down to y=0.10 for K cluster cards (0.82 height)
    n_chosen = max(len(chosen), 1)
    card_top = 0.92
    card_h = 0.78 / n_chosen
    card_gap = 0.03

    for k, cand in enumerate(chosen, start=1):
        llr, ci, r, t0, t1, n_in, mu_in = cand
        c = cents[ci]
        # Significant points = incidents inside this cluster's circle+window
        inside_space = ((pt_x - c[0]) ** 2 + (pt_y - c[1]) ** 2) <= r ** 2
        inside_time = (years >= t0) & (years <= t1)
        sel = inside_space & inside_time
        sig_loc_count += int(sel.sum())
        # Pink halo — translucent so wards/dots show through, like Hohl
        ax.add_patch(Circle(c, r, facecolor="#f4b6c2",
                              edgecolor="#9a1d3a",
                              linewidth=1.0, alpha=0.32, zorder=2))
        # Red dots = significant locations
        ax.scatter(pt_x[sel], pt_y[sel], s=14, c="#d7191c",
                    edgecolor="#7a0000", linewidths=0.3, zorder=4)
        # Cluster index — BIG blue numeral on top of everything
        ax.text(c[0], c[1], f"{k}", ha="center", va="center",
                  fontsize=26, fontweight="bold", color="#0d3a8a",
                  zorder=6,
                  path_effects=[pe.withStroke(linewidth=3,
                                               foreground="white")])
        # MC p-value
        p_val = (1 + sum(1 for v in null_max_llr if v >= llr)) \
            / (1 + n_mc)
        rr = (n_in / mu_in) if mu_in > 0 else float("inf")
        # Per-cluster boxed card on the right side
        y_top = card_top - (k - 1) * (card_h + card_gap)
        y_bot = y_top - card_h
        y_mid = (y_top + y_bot) / 2
        card = FancyBboxPatch((0.04, y_bot), 0.92, card_h,
                                boxstyle="round,pad=0.012",
                                linewidth=1.2, edgecolor="#1a1a1a",
                                facecolor="white",
                                transform=info.transAxes)
        info.add_patch(card)
        body = (
            f"Cluster {k}\n"
            f"Time frame    : {t0}–{t1}\n"
            f"Centre (km)   : ({c[0]:.1f}, {c[1]:.1f})\n"
            f"Radius (km)   : {r:.1f}\n"
            f"Number cases  : {n_in}\n"
            f"Expected cases: {mu_in:.2f}\n"
            f"Observed/exp  : {rr:.2f}\n"
            f"Relative risk : {rr:.2f}\n"
            f"P-value       : "
            + ("< 0.001" if p_val < 0.001 else f"{p_val:.3f}")
        )
        info.text(0.10, y_top - 0.012, body, va="top", ha="left",
                    family="monospace", fontsize=9.0)
        # Leader line — from circle right-edge (in ax data coords) to
        # card centre-left (in info axes coords) — gives the
        # "pulled-out" 3D-callout look like Hohl 2024
        con = ConnectionPatch(
            xyA=(c[0] + r * 0.95, c[1] + r * 0.5),
            coordsA=ax.transData,
            xyB=(0.04, y_mid), coordsB=info.transAxes,
            arrowstyle="-", color="#7a7a7a",
            linewidth=0.9, linestyle=":", zorder=10)
        fig.add_artist(con)

    # Legend (bottom-right of map): sig locations / sig clusters /
    # sig neighbourhoods / wards
    legend_y = 0.04
    handles = [
        plt.Line2D([], [], marker="o", color="w",
                    markerfacecolor="#d7191c",
                    markeredgecolor="#7a0000",
                    markersize=7, label=f"Sig. locations (n={sig_loc_count})"),
        plt.Line2D([], [], marker="o", color="w",
                    markerfacecolor="#ffd6e7",
                    markeredgecolor="#c2185b", markersize=12,
                    label=f"Sig. clusters (n={len(chosen)})"),
        plt.Line2D([], [], marker="s", color="w",
                    markerfacecolor="#fff7a8",
                    markeredgecolor="#1a1a1a", markersize=10,
                    label=f"Sig. neighbourhoods (n={len(yellow_idx)})"),
        plt.Line2D([], [], marker="s", color="w",
                    markerfacecolor="white",
                    markeredgecolor="#1a1a1a", markersize=10,
                    label="Wards by ID (158)"),
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=8,
                framealpha=0.92, ncols=1)
    ax.set_aspect("equal", adjustable="box")
    ax.autoscale_view()
    ax.set_xticks([]); ax.set_yticks([])
    _draw_compass(ax)
    _draw_scalebar(ax, length_km=10)
    ax.set_title(f"{category} · Space-time significant clusters",
                    fontsize=11)
    out = Path(outfile) if outfile else FIG_DIR / \
        f"satscan_d_{category.lower()}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out


def render_yearly_grid(*, prefix: str = "ASSAULT_RATE",
                         years: Iterable[int] = range(2014, 2025),
                         cmap: str = "Reds",
                         outfile: str | Path | None = None,
                         ncols: int = 4) -> Path:
    """Small-multiples panel of per-year choropleths, no hood labels."""
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection

    from .tps_io import load_tps

    df = load_tps("NeighbourhoodCrimeRates", format="geojson")
    years = [y for y in years if f"{prefix}_{y}" in df.columns]
    if not years:
        raise KeyError(f"no {prefix}_<year> columns found")

    polys: list[np.ndarray] = []
    raw_lats: list[float] = []
    for _, row in df.iterrows():
        geom = row.get("geometry")
        if geom is None:
            continue
        rings = geom if isinstance(geom[0][0], list) else [geom]
        for ring in rings:
            arr = np.asarray(ring, dtype=float)
            if arr.ndim != 2 or arr.shape[1] < 2:
                continue
            xk, yk = project_xy(arr[:, 1], arr[:, 0])
            polys.append(np.column_stack([xk, yk]))
            raw_lats.append(float(arr[:, 1].mean()))

    n = len(years)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(3.5 * ncols, 2.4 * nrows),
                              squeeze=False)
    vmin = min(df[f"{prefix}_{y}"].min() for y in years)
    vmax = max(df[f"{prefix}_{y}"].quantile(0.97) for y in years)
    pc = None
    for k, year in enumerate(years):
        r, c = divmod(k, ncols)
        ax = axes[r][c]
        vals = []
        for _, row in df.iterrows():
            geom = row.get("geometry")
            if geom is None:
                continue
            rings = geom if isinstance(geom[0][0], list) else [geom]
            for _ in rings:
                try:
                    vals.append(float(row[f"{prefix}_{year}"]))
                except (TypeError, ValueError, KeyError):
                    vals.append(np.nan)
        from matplotlib.colors import Normalize
        pc = PolyCollection(polys, array=np.asarray(vals),
                              edgecolor="white", linewidth=0.1,
                              cmap=cmap, norm=Normalize(vmin=vmin, vmax=vmax))
        ax.add_collection(pc)
        ax.autoscale_view()
        ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(str(year), fontsize=10)
    for k in range(n, nrows * ncols):
        r, c = divmod(k, ncols)
        axes[r][c].axis("off")
    if pc is not None:
        fig.colorbar(pc, ax=axes.ravel().tolist(),
                       shrink=0.6, label=f"{prefix}_<year>")
    fig.suptitle(f"Toronto {prefix} · yearly small-multiples", y=1.0)
    out = Path(outfile) if outfile else FIG_DIR / \
        f"choropleth_{prefix.lower()}_yearly.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out
