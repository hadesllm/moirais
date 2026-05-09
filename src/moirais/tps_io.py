"""moirais.tps_io — multi-format readers for TPS open-data exports.

Each TPS category at data/datasets/TPS/<Category>/ has 9 sibling format
exports of the same incident records:

    CSV               — pandas.read_csv (canonical fast path)
    Excel             — pandas.read_excel via openpyxl
    GeoJSON           — pure JSON, geometry as Python list-of-coords
    FeatureCollection — ESRI JSON variant of GeoJSON
    KML / KMZ         — zipfile + ElementTree, geometry as coord list
    GeoPackage        — sqlite3, geometry stored as WKB blobs
    SQLiteGeodatabase — sqlite3, ESRI variant of GeoPackage
    Shapefile         — needs pyshp; graceful degrade if not installed
    FileGeoDatabase   — needs fiona / gdal; graceful degrade

This module returns a `pandas.DataFrame` for every format, with the
geometry (when available) attached as a `geometry` column whose values
are simple Python lists/tuples — no shapely / geopandas dependency.

For NeighbourhoodCrimeRates the geometry is `Polygon`; for incident
datasets the geometry is `Point`. Either way, the geometry column
contains the (longitude, latitude) tuples in WGS84.

Public API:
    load_tps(name, format='csv', nrows=None) -> pd.DataFrame
    list_tps_formats(name)                   -> dict[str, Path]
    available_formats()                      -> list of supported format names
"""

from __future__ import annotations

import json
import sqlite3
import struct
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path
from typing import Any

import pandas as pd

from .tps_datasets import TPS_DATA_DIR, TPS_REGISTRY

SUPPORTED_FORMATS = (
    "csv", "excel",
    "geojson", "featurecollection",
    "kml",
    "geopackage", "sqlitegeodatabase",
)
NEEDS_LIB_FORMATS = ("shapefile", "filegeodatabase")
ALL_FORMATS = SUPPORTED_FORMATS + NEEDS_LIB_FORMATS


def _category_dir(name: str, fmt_subdir: str) -> Path:
    canonical = next(
        (k for k in TPS_REGISTRY if k.lower() == name.lower()), None
    )
    if canonical is None:
        raise KeyError(
            f"unknown TPS dataset {name!r}; "
            f"valid: {sorted(TPS_REGISTRY.keys())}"
        )
    return TPS_DATA_DIR / canonical / fmt_subdir


def _pick_one(d: Path, *exts: str) -> Path:
    for ext in exts:
        candidates = list(d.glob(f"*.{ext}"))
        if candidates:
            return candidates[0]
    raise FileNotFoundError(f"no matching file in {d} (exts: {exts})")


# ── CSV / Excel ────────────────────────────────────────────────────


def _read_csv(name: str, nrows: int | None) -> pd.DataFrame:
    p = _pick_one(_category_dir(name, "CSV"), "csv")
    return pd.read_csv(p, nrows=nrows)


def _read_excel(name: str, nrows: int | None) -> pd.DataFrame:
    p = _pick_one(_category_dir(name, "Excel"), "xlsx", "xls")
    return pd.read_excel(p, nrows=nrows, engine="openpyxl")


# ── GeoJSON / FeatureCollection ────────────────────────────────────


def _read_geojson_text(text: str, nrows: int | None) -> pd.DataFrame:
    """Parse a GeoJSON / ESRI FeatureCollection text into a DataFrame.
    Geometry is kept as a python coord list under column `geometry`.

    Handles both:
    - GeoJSON `{"features": [...]}`
    - ESRI Layer-JSON `{"layers": [{"featureSet": {"features": [...]}}]}`
    """
    data = json.loads(text)
    feats = data.get("features", [])
    if not feats and "layers" in data and data["layers"]:
        # ESRI Layer-JSON: peel down to layer[0].featureSet.features
        layer0 = data["layers"][0]
        feats = (layer0.get("featureSet") or {}).get("features", []) \
                or layer0.get("features", [])
    if nrows:
        feats = feats[:nrows]
    rows = []
    for f in feats:
        props = f.get("properties") or f.get("attributes") or {}
        geom = f.get("geometry") or {}
        gtype = geom.get("type")
        coords = geom.get("coordinates")
        # ESRI variant: rings/paths instead of coordinates
        if coords is None:
            coords = geom.get("rings") or geom.get("paths") or geom.get("points")
            if "rings" in geom:
                gtype = gtype or "Polygon"
            elif "paths" in geom:
                gtype = gtype or "LineString"
        row = dict(props)
        row["geometry_type"] = gtype
        row["geometry"] = coords
        rows.append(row)
    return pd.DataFrame(rows)


def _read_geojson(name: str, nrows: int | None) -> pd.DataFrame:
    p = _pick_one(_category_dir(name, "GeoJSON"), "geojson", "json")
    return _read_geojson_text(p.read_text(encoding="utf-8"), nrows)


def _read_featurecollection(name: str, nrows: int | None) -> pd.DataFrame:
    # ESRI FeatureCollection exports are .txt with JSON inside
    p = _pick_one(_category_dir(name, "FeatureCollection"),
                  "txt", "json", "geojson")
    return _read_geojson_text(p.read_text(encoding="utf-8"), nrows)


# ── KML / KMZ ──────────────────────────────────────────────────────


_KML_NS = {"kml": "http://www.opengis.net/kml/2.2"}


def _kml_to_dataframe(xml_text: str, nrows: int | None) -> pd.DataFrame:
    root = ET.fromstring(xml_text)
    placemarks = root.findall(".//kml:Placemark", _KML_NS)
    if nrows:
        placemarks = placemarks[:nrows]
    rows = []
    for pm in placemarks:
        row: dict[str, Any] = {}
        # ExtendedData → SimpleData
        for sd in pm.findall(".//kml:SimpleData", _KML_NS):
            row[sd.attrib.get("name", "data")] = sd.text
        # Try geometry
        pt = pm.find(".//kml:Point/kml:coordinates", _KML_NS)
        if pt is not None and pt.text:
            try:
                lon, lat, *_ = [float(x) for x in pt.text.strip().split(",")]
                row["geometry_type"] = "Point"
                row["geometry"] = (lon, lat)
            except Exception:
                pass
        else:
            poly = pm.find(".//kml:Polygon//kml:coordinates", _KML_NS)
            if poly is not None and poly.text:
                try:
                    coords = [
                        tuple(float(x) for x in c.split(","))[:2]
                        for c in poly.text.strip().split()
                    ]
                    row["geometry_type"] = "Polygon"
                    row["geometry"] = coords
                except Exception:
                    pass
        rows.append(row)
    return pd.DataFrame(rows)


def _read_kml(name: str, nrows: int | None) -> pd.DataFrame:
    d = _category_dir(name, "KML")
    candidates = list(d.glob("*.kmz")) + list(d.glob("*.kml"))
    if not candidates:
        raise FileNotFoundError(f"no KML/KMZ in {d}")
    p = candidates[0]
    if p.suffix.lower() == ".kmz":
        with zipfile.ZipFile(p) as zf:
            kml_names = [n for n in zf.namelist() if n.endswith(".kml")]
            if not kml_names:
                raise FileNotFoundError(f"no .kml inside {p}")
            with zf.open(kml_names[0]) as fp:
                return _kml_to_dataframe(
                    fp.read().decode("utf-8", errors="replace"), nrows
                )
    return _kml_to_dataframe(p.read_text(encoding="utf-8"), nrows)


# ── GeoPackage / SQLiteGeodatabase ─────────────────────────────────


def _wkb_point(blob: bytes) -> tuple[float, float] | None:
    """Decode a WKB Point to (lon, lat). Returns None on failure."""
    try:
        if not blob or len(blob) < 21:
            return None
        # Skip GPKG flag header if present (starts with b'GP')
        if blob[:2] == b"GP":
            # GPKG envelope/header — skip past to WKB body. Header has
            # variable size; locate first 0x01 (LE) or 0x00 (BE) WKB byte
            # after the envelope. Simplification: WKB body usually starts
            # at offset 8 + 4*envelope_size. Use a tolerant scan: skip
            # the first 8 bytes (magic+flags+srs_id), then check for
            # endian byte (0x00 or 0x01).
            for off in (8, 16, 24, 32, 40, 48):
                if off >= len(blob):
                    break
                if blob[off] in (0x00, 0x01):
                    blob = blob[off:]
                    break
        if len(blob) < 21:
            return None
        endian = blob[0]
        prefix = "<" if endian else ">"
        gtype = struct.unpack(prefix + "I", blob[1:5])[0]
        # 1 = Point
        if gtype == 1:
            x, y = struct.unpack(prefix + "dd", blob[5:21])
            return (float(x), float(y))
    except Exception:
        return None
    return None


def _read_sqlite_geo(name: str, fmt_subdir: str, ext: str,
                    nrows: int | None) -> pd.DataFrame:
    p = _pick_one(_category_dir(name, fmt_subdir), ext)
    con = sqlite3.connect(p)
    try:
        cur = con.cursor()
        # Find a feature table: first user table whose first column is
        # likely "OBJECTID" or whose count > 100.
        cur.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name NOT LIKE 'sqlite_%' "
            "AND name NOT LIKE 'gpkg_%' "
            "AND name NOT LIKE 'rtree_%' "
            "AND name NOT LIKE 'GDB_%' "
            "AND name NOT LIKE 'st_%';"
        )
        tables = [t[0] for t in cur.fetchall()]
        if not tables:
            return pd.DataFrame()
        # Pick the largest table
        best, best_n = None, -1
        for t in tables:
            try:
                n = cur.execute(
                    f'SELECT COUNT(*) FROM "{t}";'
                ).fetchone()[0]
            except sqlite3.DatabaseError:
                continue
            if n > best_n:
                best_n, best = n, t
        if best is None:
            return pd.DataFrame()
        limit = f" LIMIT {nrows}" if nrows else ""
        df = pd.read_sql(f'SELECT * FROM "{best}"{limit};', con)
        # Decode geometry blob if a geometry column exists
        for col in df.columns:
            if df[col].dtype == object and df[col].iloc[:1].apply(
                lambda v: isinstance(v, (bytes, bytearray))
            ).any():
                df["geometry"] = df[col].apply(
                    lambda b: _wkb_point(b) if isinstance(
                        b, (bytes, bytearray)
                    ) else None
                )
                break
        return df
    finally:
        con.close()


def _read_geopackage(name: str, nrows: int | None) -> pd.DataFrame:
    return _read_sqlite_geo(name, "GeoPackage", "gpkg", nrows)


def _read_sqlite_geodatabase(name: str, nrows: int | None) -> pd.DataFrame:
    return _read_sqlite_geo(name, "SQLiteGeodatabase", "geodatabase", nrows)


# ── Shapefile (optional dep) ───────────────────────────────────────


def _read_shapefile(name: str, nrows: int | None) -> pd.DataFrame:
    try:
        import shapefile  # pyshp
    except ImportError:
        raise ImportError(
            "Shapefile reading needs `pyshp` (pip install pyshp). "
            "Falling back to CSV is recommended for now."
        ) from None
    p = _pick_one(_category_dir(name, "Shapefile"), "zip", "shp")
    # If zip, extract in-memory
    if p.suffix.lower() == ".zip":
        with zipfile.ZipFile(p) as zf:
            shp = next(n for n in zf.namelist() if n.endswith(".shp"))
            with zf.open(shp) as shp_fp, \
                 zf.open(shp.replace(".shp", ".dbf")) as dbf_fp, \
                 zf.open(shp.replace(".shp", ".shx")) as shx_fp:
                reader = shapefile.Reader(
                    shp=shp_fp, dbf=dbf_fp, shx=shx_fp
                )
                fields = [f[0] for f in reader.fields[1:]]
                rows = []
                for i, sr in enumerate(reader.iterShapeRecords()):
                    if nrows and i >= nrows:
                        break
                    row = dict(zip(fields, sr.record))
                    row["geometry"] = list(sr.shape.points)
                    rows.append(row)
                return pd.DataFrame(rows)
    reader = shapefile.Reader(str(p))
    fields = [f[0] for f in reader.fields[1:]]
    rows = []
    for i, sr in enumerate(reader.iterShapeRecords()):
        if nrows and i >= nrows:
            break
        row = dict(zip(fields, sr.record))
        row["geometry"] = list(sr.shape.points)
        rows.append(row)
    return pd.DataFrame(rows)


# ── Master dispatcher ──────────────────────────────────────────────


_DISPATCH = {
    "csv": _read_csv,
    "excel": _read_excel,
    "geojson": _read_geojson,
    "featurecollection": _read_featurecollection,
    "kml": _read_kml,
    "geopackage": _read_geopackage,
    "sqlitegeodatabase": _read_sqlite_geodatabase,
    "shapefile": _read_shapefile,
}


def load_tps(name: str, format: str = "csv",
             nrows: int | None = None) -> pd.DataFrame:
    """Load TPS dataset `name` in the given `format`.

    Parameters
    ----------
    name   : "Assault", "Homicides", … (case-insensitive)
    format : one of csv | excel | geojson | featurecollection |
             kml | geopackage | sqlitegeodatabase | shapefile.
             "filegeodatabase" requires fiona/gdal — not supported on
             this build.
    nrows  : sample size cap; None = full dataset.
    """
    fmt = format.lower()
    if fmt == "filegeodatabase":
        raise NotImplementedError(
            "FileGeoDatabase requires fiona/gdal. "
            "Install via `pip install fiona` and re-run."
        )
    if fmt not in _DISPATCH:
        raise ValueError(
            f"unknown format {format!r}; valid: {SUPPORTED_FORMATS}"
        )
    return _DISPATCH[fmt](name, nrows)


def list_tps_formats(name: str) -> dict[str, Path]:
    """Map format name → path of the file that would be loaded.
    For formats not present on disk, the path is omitted from the dict.
    """
    out: dict[str, Path] = {}
    canonical = next(
        (k for k in TPS_REGISTRY if k.lower() == name.lower()), None
    )
    if canonical is None:
        return out
    base = TPS_DATA_DIR / canonical
    fmt_dirs = {
        "csv": ("CSV", ["csv"]),
        "excel": ("Excel", ["xlsx", "xls"]),
        "geojson": ("GeoJSON", ["geojson", "json"]),
        "featurecollection": ("FeatureCollection",
                              ["txt", "json", "geojson"]),
        "kml": ("KML", ["kmz", "kml"]),
        "geopackage": ("GeoPackage", ["gpkg"]),
        "sqlitegeodatabase": ("SQLiteGeodatabase", ["geodatabase"]),
        "shapefile": ("Shapefile", ["zip", "shp"]),
        "filegeodatabase": ("FileGeoDatabase", ["zip"]),
    }
    for fmt, (subdir, exts) in fmt_dirs.items():
        d = base / subdir
        if not d.exists():
            continue
        for ext in exts:
            cands = list(d.glob(f"*.{ext}"))
            if cands:
                out[fmt] = cands[0]
                break
    return out


def available_formats() -> list[str]:
    """List of formats this build can actually load."""
    out = list(SUPPORTED_FORMATS)
    try:
        import shapefile  # noqa: F401
        out.append("shapefile")
    except ImportError:
        pass
    return sorted(out)
