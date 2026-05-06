"""
Cook County geography for pipeline HTML reports.

**Political townships** use Cook County GIS ``cookVwrDynmc`` MapServer layer
**43 — Political Township** (38 polygons, WGS84). This matches CCAO
``meta_township_name`` after normalizing *North/South/West Chicago* to GIS
``NORTH`` / ``SOUTH`` / ``WEST``. Aligns with
`CookViewer <https://maps.cookcountyil.gov/cookviewer/>`__ and published assessor maps.

**PUMA (2020)** — U.S. Census TIGER/Line for secondary finer choropleths.

References: IAAO *Standard on Ratio Studies*; ±5% mean % error band for map coloring.
"""

from __future__ import annotations

import json
import re
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parent.parent
_GEO_CACHE = _REPO / "data" / "geo"

# IAAO-inspired **level** tolerance for **mean % error** maps (symmetric).
IAAO_MEAN_PCT_ERROR_BAND: float = 5.0

_CENSUS_COSUB_QUERY = (
    "https://tigerweb.geo.census.gov/arcgis/rest/services/"
    "TIGERweb/tigerWMS_ACS2021/MapServer/18/query"
)
_CENSUS_PUMA_QUERY = (
    "https://tigerweb.geo.census.gov/arcgis/rest/services/"
    "TIGERweb/tigerWMS_Current/MapServer/0/query"
)

# Cook County GIS — Political Township (38 assessor townships; WGS84 query).
_COOK_POLITICAL_TOWNSHIP_QUERY = (
    "https://gis.cookcountyil.gov/traditional/rest/services/"
    "cookVwrDynmc/MapServer/43/query"
)

_TRIAD_LINE_COLORS: Dict[str, str] = {
    "City": "#7c3aed",
    "North": "#0d9488",
    "South": "#ea580c",
}


def _download_json_url(url: str) -> Dict[str, Any]:
    with urllib.request.urlopen(url, timeout=120) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _ensure_cache_dir() -> Path:
    _GEO_CACHE.mkdir(parents=True, exist_ok=True)
    return _GEO_CACHE


def load_or_fetch_cook_cousub_geojson(*, force: bool = False) -> Dict[str, Any]:
    """Cook County, IL county subdivisions (Census `BASENAME` ~ assessor townships for suburbs)."""
    path = _ensure_cache_dir() / "cook_il_cousub_acs2021.geojson"
    if path.is_file() and not force:
        return json.loads(path.read_text(encoding="utf-8"))
    q = (
        f"{_CENSUS_COSUB_QUERY}?"
        "where=STATE%3D%2717%27+AND+COUNTY%3D%27031%27"
        "&outFields=BASENAME,GEOID,NAMELSAD,LSADC"
        "&returnGeometry=true"
        "&resultRecordCount=100"
        "&f=geojson"
    )
    data = _download_json_url(q)
    path.write_text(json.dumps(data, separators=(",", ":")), encoding="utf-8")
    return data


def load_or_fetch_cook_puma_geojson(*, force: bool = False) -> Dict[str, Any]:
    """Cook-nesting 2020 PUMA polygons (`GEOID` begins with `17031`)."""
    path = _ensure_cache_dir() / "cook_il_puma2020.geojson"
    if path.is_file() and not force:
        return json.loads(path.read_text(encoding="utf-8"))
    q = (
        f"{_CENSUS_PUMA_QUERY}?"
        "where=GEOID+LIKE+%2717031%25%27"
        "&outFields=GEOID,BASENAME,NAME,PUMA,STATE"
        "&returnGeometry=true"
        "&resultRecordCount=200"
        "&f=geojson"
    )
    data = _download_json_url(q)
    path.write_text(json.dumps(data, separators=(",", ":")), encoding="utf-8")
    return data


# CCAO ``meta_township_name`` (training column) → Cook GIS Political Township ``NAME``.
_CCAO_META_TO_GIS_NAME: Dict[str, str] = {
    "North Chicago": "NORTH",
    "South Chicago": "SOUTH",
    "West Chicago": "WEST",
}

# Assessor triad by GIS township (matches common Cook County North / Chicago / South grouping).
COOK_GIS_TOWNSHIP_TRIAD: Dict[str, str] = {
    **{k: "North" for k in (
        "BARRINGTON", "PALATINE", "WHEELING", "NORTHFIELD", "NEW TRIER", "EVANSTON",
        "HANOVER", "SCHAUMBURG", "ELK GROVE", "MAINE", "NILES",
    )},
    **{k: "City" for k in (
        "ROGERS PARK", "JEFFERSON", "LAKE VIEW", "NORTH", "SOUTH", "WEST", "LAKE", "HYDE PARK",
    )},
    **{k: "South" for k in (
        "LEYDEN", "NORWOOD PARK", "RIVER FOREST", "OAK PARK", "PROVISO", "RIVERSIDE", "BERWYN",
        "CICERO", "LYONS", "STICKNEY", "LEMONT", "PALOS", "WORTH", "CALUMET", "ORLAND", "BREMEN",
        "THORNTON", "RICH", "BLOOM",
    )},
}


def ccao_meta_township_to_gis_name(meta_township: str) -> str:
    """Map CCAO ``meta_township_name`` to Cook GIS layer ``NAME`` (uppercase token)."""
    m = str(meta_township).strip()
    if m in _CCAO_META_TO_GIS_NAME:
        return _CCAO_META_TO_GIS_NAME[m]
    return " ".join(part.upper() for part in m.split())


def gis_name_to_display_label(gis_name: str) -> str:
    """Human-readable label for hover (reverses Chicago aliases)."""
    inv = {v: k for k, v in _CCAO_META_TO_GIS_NAME.items()}
    if gis_name in inv:
        return inv[gis_name]
    # "LAKE VIEW" -> "Lake View"
    return str(gis_name).title()


def cook_political_township_triad_by_gis_name() -> Dict[str, str]:
    return dict(COOK_GIS_TOWNSHIP_TRIAD)


def load_or_fetch_cook_political_township_geojson(*, force: bool = False) -> Dict[str, Any]:
    """
    All **38** Cook County political townships — Cook County GIS MapServer layer 43, WGS84.

    Each feature gets ``properties.twn`` = GIS ``NAME`` (e.g. ``NORTH``, ``ORLAND``) for
    Plotly choropleth joins. Boundaries follow official county GIS (shoreline / land).
    """
    path = _ensure_cache_dir() / "cook_il_political_township_ccgis.geojson"
    if path.is_file() and not force:
        return json.loads(path.read_text(encoding="utf-8"))
    q = (
        f"{_COOK_POLITICAL_TOWNSHIP_QUERY}?"
        "where=1%3D1"
        "&outFields=NAME,OBJECTID"
        "&returnGeometry=true"
        "&outSR=4326"
        "&resultRecordCount=100"
        "&f=geojson"
    )
    raw = _download_json_url(q)
    feats_out: List[Dict[str, Any]] = []
    for feat in raw.get("features") or []:
        props = dict(feat.get("properties") or {})
        nm = str(props.get("NAME", "")).strip()
        if not nm:
            continue
        props["twn"] = nm
        props["label"] = gis_name_to_display_label(nm)
        feat = {"type": "Feature", "properties": props, "geometry": feat.get("geometry")}
        feats_out.append(feat)
    data = {"type": "FeatureCollection", "features": feats_out}
    if len(feats_out) != 38:
        raise RuntimeError(
            f"Expected 38 Cook political townships; got {len(feats_out)}. "
            "Cook County GIS service may have changed."
        )
    path.write_text(json.dumps(data, separators=(",", ":")), encoding="utf-8")
    return data


def _normalize_place(s: str) -> str:
    t = str(s).strip().lower()
    t = t.split(",")[0].strip()
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"\s+(township|village|city|cdp)\s*$", "", t)
    t = t.removesuffix(" township")
    t = t.removesuffix(" city")
    return t.strip()


def _cousub_basename_index(geojson: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for feat in geojson.get("features") or []:
        props = feat.get("properties") or {}
        bn = props.get("BASENAME") or props.get("NAME")
        if not bn or bn == "County subdivisions not defined":
            continue
        key = _normalize_place(str(bn))
        out[key] = feat
    return out


def _cousub_namelsad_index(geojson: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Secondary lookup when BASENAME alone does not match assessor spelling."""
    out: Dict[str, Dict[str, Any]] = {}
    for feat in geojson.get("features") or []:
        props = feat.get("properties") or {}
        nslad = props.get("NAMELSAD")
        if not nslad or nslad == "County subdivisions not defined":
            continue
        key = _normalize_place(str(nslad))
        if key:
            out[key] = feat
    return out


def _geometry_from_pins(lon: np.ndarray, lat: np.ndarray) -> Optional[Dict[str, Any]]:
    """Fallback polygon from sale pins (WGS84): convex hull, with buffers for sparse pins."""
    from shapely.geometry import LineString, MultiPoint, Point, mapping

    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)
    m = np.isfinite(lon) & np.isfinite(lat)
    lon, lat = lon[m], lat[m]
    if lon.size == 0:
        return None
    if lon.size == 1:
        g = Point(float(lon[0]), float(lat[0])).buffer(0.0085)
    elif lon.size == 2:
        g = LineString(
            [(float(lon[0]), float(lat[0])), (float(lon[1]), float(lat[1]))]
        ).buffer(0.0038)
    else:
        g = MultiPoint(list(zip(lon.tolist(), lat.tolist()))).convex_hull
        if g.geom_type == "LineString":
            g = g.buffer(0.0022)
        elif g.geom_type == "Point":
            g = g.buffer(0.0085)
    return mapping(g)


def _township_pin_lon_lat_by_name(fallback_df: pd.DataFrame) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    need = {"meta_township_name", "loc_longitude", "loc_latitude"}
    if not need.issubset(fallback_df.columns):
        return {}
    base = fallback_df.dropna(subset=["meta_township_name", "loc_longitude", "loc_latitude"]).copy()
    if base.empty:
        return {}
    if "row_id" in base.columns:
        base = base.drop_duplicates(subset=["row_id"], keep="first")
    out: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for twn, sub in base.groupby(base["meta_township_name"].astype(str)):
        if not twn or twn in ("Unknown", "nan"):
            continue
        lo = pd.to_numeric(sub["loc_longitude"], errors="coerce").to_numpy(dtype=float)
        la = pd.to_numeric(sub["loc_latitude"], errors="coerce").to_numpy(dtype=float)
        out[str(twn)] = (lo, la)
    return out


def _puma_geoid_index(geojson: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for feat in geojson.get("features") or []:
        props = feat.get("properties") or {}
        gid = str(props.get("GEOID", "")).strip()
        if gid:
            out[gid] = feat
    return out


def township_to_modal_puma(training_or_df: pd.DataFrame) -> Dict[str, str]:
    """Assessor township -> modal ``loc_census_puma_geoid`` (string)."""
    need = {"meta_township_name", "loc_census_puma_geoid"}
    if not need.issubset(training_or_df.columns):
        return {}
    df = training_or_df.dropna(subset=["meta_township_name"]).copy()
    df["meta_township_name"] = df["meta_township_name"].astype(str)
    df = df.loc[df["loc_census_puma_geoid"].notna()].copy()
    if df.empty:
        return {}
    df["loc_census_puma_geoid"] = (
        df["loc_census_puma_geoid"].astype(str).str.replace(r"\.0$", "", regex=True)
    )
    df = df.loc[~df["loc_census_puma_geoid"].isin(["", "nan", "None"])].copy()
    rows: Dict[str, str] = {}
    for twn, sub in df.groupby("meta_township_name"):
        if not twn or twn in ("Unknown", "nan"):
            continue
        vc = sub["loc_census_puma_geoid"].value_counts()
        if vc.empty:
            continue
        rows[str(twn)] = str(vc.index[0])
    return rows


def township_to_modal_triad(training_or_df: pd.DataFrame) -> Dict[str, str]:
    """Assessor township -> modal ``meta_triad_name``."""
    need = {"meta_township_name", "meta_triad_name"}
    if not need.issubset(training_or_df.columns):
        return {}
    df = training_or_df.dropna(subset=["meta_township_name", "meta_triad_name"]).copy()
    rows: Dict[str, str] = {}
    for twn, sub in df.groupby(df["meta_township_name"].astype(str)):
        if not twn or twn in ("Unknown", "nan"):
            continue
        m = sub["meta_triad_name"].astype(str).mode()
        if len(m):
            rows[str(twn)] = str(m.iloc[0])
    return rows


def build_assessor_township_official_geojson(
    *,
    assessor_township_names: Sequence[str],
    township_modal_puma: Dict[str, str],
    cousub: Dict[str, Any],
    puma_gj: Dict[str, Any],
    coordinate_fallback_df: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """
    One Feature per assessor label in data: **cousub** (BASENAME or NAMELSAD), else
    modal **PUMA**, else convex hull of unique sale pins (last resort).
    """
    by_bn = _cousub_basename_index(cousub)
    by_nl = _cousub_namelsad_index(cousub)
    by_puma = _puma_geoid_index(puma_gj)
    pins_by_twn = _township_pin_lon_lat_by_name(coordinate_fallback_df) if coordinate_fallback_df is not None else {}

    feats: List[Dict[str, Any]] = []
    for raw in assessor_township_names:
        twn = str(raw).strip()
        if not twn or twn in ("Unknown", "nan"):
            continue
        key = _normalize_place(twn)
        chosen: Optional[Dict[str, Any]] = None
        src = ""
        if key in by_bn:
            chosen = by_bn[key]
            src = "cousub"
        elif key in by_nl:
            chosen = by_nl[key]
            src = "cousub_namelsad"
        else:
            pgeo = township_modal_puma.get(twn)
            if pgeo and pgeo in by_puma:
                chosen = by_puma[pgeo]
                src = "puma"
        if chosen is not None:
            geom = chosen.get("geometry")
            if geom:
                feats.append(
                    {
                        "type": "Feature",
                        "properties": {
                            "twn": twn,
                            "src": src,
                            "geoid": chosen.get("properties", {}).get("GEOID", ""),
                        },
                        "geometry": geom,
                    }
                )
                continue

        lo_la = pins_by_twn.get(twn)
        if lo_la is not None:
            geom_dict = _geometry_from_pins(lo_la[0], lo_la[1])
            if geom_dict:
                feats.append(
                    {
                        "type": "Feature",
                        "properties": {"twn": twn, "src": "pin_hull", "geoid": ""},
                        "geometry": geom_dict,
                    }
                )
    return {"type": "FeatureCollection", "features": feats}


def build_puma_official_geojson(puma_gj: Dict[str, Any]) -> Dict[str, Any]:
    """Relabel PUMA features with ``properties.puma_id = GEOID`` for choropleth joins."""
    feats: List[Dict[str, Any]] = []
    for f in puma_gj.get("features") or []:
        props = dict(f.get("properties") or {})
        gid = str(props.get("GEOID", "")).strip()
        if not gid:
            continue
        props["puma_id"] = gid
        feats.append({"type": "Feature", "properties": props, "geometry": f.get("geometry")})
    return {"type": "FeatureCollection", "features": feats}


def triad_outline_traces(
    *,
    township_geojson: Dict[str, Any],
    township_to_triad: Dict[str, str],
) -> List[Dict[str, Any]]:
    """
    Plotly ``scattermapbox`` line traces (boundaries only) — one trace per triad.
    """
    from shapely.geometry import shape
    from shapely.ops import unary_union

    by_triad: Dict[str, List[Any]] = {"City": [], "North": [], "South": []}
    for feat in township_geojson.get("features") or []:
        twn = str((feat.get("properties") or {}).get("twn", "")).strip()
        tri = township_to_triad.get(twn)
        if tri not in by_triad:
            continue
        g = feat.get("geometry")
        if not g:
            continue
        by_triad[tri].append(shape(g))

    def _append_boundary_coords(geom: Any, lats: List[Any], lons: List[Any]) -> None:
        """Extract line coordinates from a Shapely boundary sub-geometry."""
        if geom is None or geom.is_empty:
            return
        gt = geom.geom_type
        if gt == "LineString":
            xs, ys = geom.xy  # type: ignore[union-attr]
            lons.extend(xs.tolist())
            lats.extend(ys.tolist())
        elif gt == "MultiLineString":
            for seg in geom.geoms:  # type: ignore[union-attr]
                xs, ys = seg.xy
                lons.extend(xs.tolist())
                lats.extend(ys.tolist())
                lons.append(None)
                lats.append(None)
        elif gt == "GeometryCollection":
            for g in geom.geoms:  # type: ignore[union-attr]
                _append_boundary_coords(g, lats, lons)

    traces: List[Dict[str, Any]] = []
    # Draw North → South → City so **City** renders on top (clearer versus suburban rings).
    triad_order = ("North", "South", "City")
    for tri in triad_order:
        geoms = by_triad.get(tri) or []
        if not geoms:
            continue
        u = unary_union(geoms)
        b = u.boundary
        lats: List[Any] = []
        lons: List[Any] = []
        _append_boundary_coords(b, lats, lons)
        if not lats or not lons:
            continue
        line_width = 4.9 if tri == "City" else 3.1
        traces.append(
            {
                "type": "scattermapbox",
                "lat": lats,
                "lon": lons,
                "mode": "lines",
                "name": f"{tri} triad",
                "line": {"width": line_width, "color": _TRIAD_LINE_COLORS.get(tri, "#555")},
                "showlegend": True,
                "hoverinfo": "skip",
            }
        )
    return traces


def mean_pct_error_tri_colorscale(*, zmax: float, band: float = IAAO_MEAN_PCT_ERROR_BAND) -> List[List]:
    """
    Piecewise Plotly colorscale for z in [-zmax, zmax]:
    blue (under) → green (|z|≤band) → red (over).
    """
    Z = float(max(zmax, band + 1e-6))
    B = float(min(band, Z - 1e-6))
    p_bm = (Z - B) / (2 * Z)  # position of -B on [-Z,Z]
    p_bp = (Z + B) / (2 * Z)  # position of +B
    return [
        [0.0, "rgb(30,64,175)"],
        [max(0.0, p_bm - 0.08), "rgb(96,165,250)"],
        [p_bm, "rgb(74,222,128)"],
        [0.5, "rgb(22,163,74)"],
        [p_bp, "rgb(74,222,128)"],
        [min(1.0, p_bp + 0.08), "rgb(252,165,165)"],
        [1.0, "rgb(185,28,28)"],
    ]


def map_summary_stats(
    tdf: pd.DataFrame,
    *,
    acceptable_band: float,
    label_column: str = "meta_township_name",
    region_word: str = "townships",
) -> str:
    """Compact HTML for one model's aggregated geography table."""
    if tdf is None or tdf.empty:
        return "<p class='note small'>No rows to summarize.</p>"
    if label_column not in tdf.columns:
        return "<p class='note small'>Summary unavailable (missing label column).</p>"
    m = tdf["mean_pct_error"].astype(float)
    idx_max = int(m.idxmax())
    idx_min = int(m.idxmin())
    worst = tdf.loc[idx_max]
    best = tdf.loc[idx_min]
    in_band = float(np.mean(np.abs(m.to_numpy()) <= acceptable_band) * 100.0)
    mae = float(np.mean(np.abs(m.to_numpy())))
    return (
        "<ul class='small' style='margin:8px 0 12px 18px;'>"
        f"<li><strong>Worst mean error:</strong> {worst[label_column]} "
        f"({worst['mean_pct_error']:+.2f}%, n={int(worst['n_obs'])})</li>"
        f"<li><strong>Best mean error:</strong> {best[label_column]} "
        f"({best['mean_pct_error']:+.2f}%, n={int(best['n_obs'])})</li>"
        f"<li><strong>Within ±{acceptable_band:.0f}% (IAAO-style level band):</strong> "
        f"{in_band:.1f}% of {region_word} (by count)</li>"
        f"<li><strong>Mean absolute bias (|mean error|) across {region_word}:</strong> {mae:.2f}%</li>"
        "</ul>"
    )
