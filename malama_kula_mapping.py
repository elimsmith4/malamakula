#!/usr/bin/env python3
"""
malama_kula_mapping.py
----------------------
Generate the Malama Kula interactive property map from the v3 CSV and the
Maui County parcel boundaries GeoJSON. Runs end-to-end with no code editing
required for the common case.

Inputs (default filenames; all overridable via CLI flags):
    malama_kula_properties_v3.csv
    Maui_Parcels.geojson

Outputs (written to --output-dir, default: current directory):
    malama_kula_interactive_map.html
    malama_kula_geocoded.csv
    malama_kula_parcels_by_id.geojson
    malama_kula_parcels_by_geocode.geojson
    malama_kula_all_parcels.geojson

Usage:
    python malama_kula_mapping.py
    python malama_kula_mapping.py --csv data/malama_kula_properties_v3.csv \\
                                  --parcels data/Maui_Parcels.geojson \\
                                  --output-dir build/
    python malama_kula_mapping.py --csv-url https://docs.google.com/.../pub?output=csv

Dependencies:
    pip install pandas geopandas folium geopy shapely
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Optional

import pandas as pd
import geopandas as gpd
import folium
from folium import plugins
from geopy.geocoders import ArcGIS
from geopy.exc import GeocoderTimedOut, GeocoderUnavailable
from shapely.geometry import Point

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DEFAULT_CSV = "malama_kula_properties_v3.csv"
DEFAULT_PARCELS = "Maui_Parcels.geojson"

# Service columns in the v3 CSV, preserving original spelling (including the
# "Infrastucture" typo that matches the header).
SERVICE_COLUMNS = [
    "Green Waste Pickup",
    "Kupuna Pickup",
    "Property Cleanup",
    "Debris Removal",
    "Wood Chipping",
    "Wood Chip Delivery",
    "Event",
    "Fencing Support",
    "Infrastucture / Drainage",
    "Recycling Pickup",
]

# Kula, Maui fallback center (upcountry reference point)
KULA_CENTER = (20.7918, -156.3277)

GEOCODE_CACHE = "geocode_cache.csv"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

log = logging.getLogger("malama_kula")


def setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
    )


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_properties(csv_source: str) -> pd.DataFrame:
    """
    Load the Malama Kula properties CSV. Accepts a local path or a URL
    (e.g. a published Google Sheet).
    """
    log.info("Loading properties from: %s", csv_source)
    df = pd.read_csv(csv_source)
    df.columns = df.columns.str.strip()

    if "Parcel ID" in df.columns:
        df["Parcel ID"] = df["Parcel ID"].apply(
            lambda x: str(int(x)) if pd.notna(x) else pd.NA
        )

    # Keep only service columns that actually exist (resilient if schema changes)
    present_services = [c for c in SERVICE_COLUMNS if c in df.columns]
    missing = set(SERVICE_COLUMNS) - set(present_services)
    if missing:
        log.warning("Service columns missing from CSV (ignored): %s", sorted(missing))

    log.info(
        "  %d properties | %d with address | %d with Parcel ID | %d service columns",
        len(df),
        df["Address"].notna().sum() if "Address" in df.columns else 0,
        df["Parcel ID"].notna().sum() if "Parcel ID" in df.columns else 0,
        len(present_services),
    )
    return df


def load_parcels(parcels_path: Path) -> gpd.GeoDataFrame:
    if not parcels_path.exists():
        raise FileNotFoundError(
            f"Parcel file not found: {parcels_path}\n"
            "Download from: "
            "https://geoportal.hawaii.gov/datasets/HiStateGIS::parcels-maui-county/about"
        )

    size_mb = parcels_path.stat().st_size / (1024 ** 2)
    log.info("Loading parcels from %s (%.1f MB) — this takes a minute…",
             parcels_path, size_mb)
    gdf = gpd.read_file(parcels_path)
    log.info("  %d parcels, CRS=%s", len(gdf), gdf.crs)

    if gdf.crs is None or str(gdf.crs) != "EPSG:4326":
        log.info("  Reprojecting to EPSG:4326")
        gdf = gdf.to_crs("EPSG:4326")

    return gdf


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def services_for_row(row: pd.Series, service_columns: list[str]) -> str:
    present = [
        c for c in service_columns
        if c in row.index
        and pd.notna(row[c])
        and str(row[c]).strip().lower() == "x"
    ]
    return ", ".join(present) if present else "No services recorded"


def normalize_tmk(tmk_value, island_code: str = "2") -> Optional[str]:
    """
    Normalize Hawaii TMK values to the full 9-digit form used by the
    state GIS parcel layer.

    Full TMK format: Island (1) + Zone (1) + Section (1) + Plat (3) + Parcel (3)
    Maui's island code is 2, so a complete Maui TMK looks like "223013026".

    Malama Kula's CSV stores 8-digit TMKs with the island code stripped
    (e.g. "23013026"). We prepend the island code to restore the full form.
    Already-9-digit values and longer variants are returned as-is.
    """
    if pd.isna(tmk_value):
        return None
    s = str(tmk_value).replace("-", "").replace(" ", "").strip()
    if "e" in s.lower():
        s = f"{float(s):.0f}"
    s = s.split(".")[0]
    if len(s) == 8:
        s = island_code + s
    return s


# ---------------------------------------------------------------------------
# Matching pipeline
# ---------------------------------------------------------------------------

def match_by_parcel_id(
    properties_df: pd.DataFrame,
    parcels_gdf: gpd.GeoDataFrame,
    tmk_field: str,
    service_columns: list[str],
) -> tuple[Optional[gpd.GeoDataFrame], list[int]]:
    """Direct TMK → parcel match. Returns (matched_gdf, matched_indices)."""
    properties_df = properties_df.copy()
    properties_df["TMK_normalized"] = properties_df["Parcel ID"].apply(normalize_tmk)
    parcels_gdf = parcels_gdf.copy()
    parcels_gdf["TMK_normalized"] = parcels_gdf[tmk_field].apply(normalize_tmk)

    matched, matched_indices = [], []
    to_match = properties_df[properties_df["Parcel ID"].notna()]
    log.info("Matching %d Parcel IDs directly…", len(to_match))

    for idx, row in to_match.iterrows():
        hits = parcels_gdf[parcels_gdf["TMK_normalized"] == row["TMK_normalized"]]
        if len(hits) == 0:
            log.warning("  No parcel match for TMK %s", row["Parcel ID"])
            continue
        parcel = hits.iloc[0].copy()
        parcel["malama_parcel_id"] = row["Parcel ID"]
        parcel["malama_acres"] = row["Acres"]
        parcel["malama_address"] = row["Address"] if pd.notna(row["Address"]) else "N/A"
        parcel["match_method"] = "Parcel ID"
        parcel["malama_services"] = services_for_row(row, service_columns)
        matched.append(parcel)
        matched_indices.append(idx)

    if not matched:
        return None, []

    gdf = gpd.GeoDataFrame(matched, crs=parcels_gdf.crs)
    log.info("  Matched %d parcels by TMK (exact)", len(gdf))
    return gdf, matched_indices


def normalize_address_for_cache(address: str) -> str:
    """Normalize address text so small formatting differences don't create duplicate cache entries."""
    return str(address).strip().lower()


def geocode_addresses(
    addresses_df: pd.DataFrame,
    checkpoint_every: int = 20,
    cache_path: str = GEOCODE_CACHE,
) -> pd.DataFrame:
    """
    Geocode rows with non-null Address values, using geocode_cache.csv
    so previously geocoded addresses are reused on future runs.
    """
    geolocator = ArcGIS(timeout=10)

    cache_file = Path(cache_path)

    if cache_file.exists():
        cache_df = pd.read_csv(cache_file)
        if "address_key" not in cache_df.columns:
            cache_df["address_key"] = cache_df["Address"].apply(normalize_address_for_cache)
    else:
        cache_df = pd.DataFrame(columns=[
            "address_key",
            "Address",
            "latitude",
            "longitude",
            "formatted_address",
            "geocode_status",
        ])

    cache_lookup = {
        row["address_key"]: row.to_dict()
        for _, row in cache_df.iterrows()
    }

    def geocode_one(address: str, max_retries: int = 3) -> dict:
        full = f"{address}, Maui, Hawaii, USA"
        for attempt in range(max_retries):
            try:
                loc = geolocator.geocode(full, timeout=10)
                if loc:
                    return {
                        "latitude": loc.latitude,
                        "longitude": loc.longitude,
                        "formatted_address": loc.address,
                        "geocode_status": "success",
                    }
                return {
                    "latitude": None,
                    "longitude": None,
                    "formatted_address": None,
                    "geocode_status": "not_found",
                }
            except (GeocoderTimedOut, GeocoderUnavailable):
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                return {
                    "latitude": None,
                    "longitude": None,
                    "formatted_address": None,
                    "geocode_status": "timeout",
                }
            except Exception as exc:
                return {
                    "latitude": None,
                    "longitude": None,
                    "formatted_address": None,
                    "geocode_status": f"error: {exc}",
                }

        return {
            "latitude": None,
            "longitude": None,
            "formatted_address": None,
            "geocode_status": "failed",
        }

    out = addresses_df.copy()
    results = []
    new_cache_rows = []
    total = len(out)

    log.info("Geocoding %d addresses via ArcGIS with cache…", total)

    for i, (_, row) in enumerate(out.iterrows(), start=1):
        address = row["Address"]
        address_key = normalize_address_for_cache(address)

        if address_key in cache_lookup:
            cached = cache_lookup[address_key]
            res = {
                "latitude": cached.get("latitude"),
                "longitude": cached.get("longitude"),
                "formatted_address": cached.get("formatted_address"),
                "geocode_status": cached.get("geocode_status"),
            }
            log.debug("  [%3d/%d] CACHE %-40s", i, total, str(address)[:40])
        else:
            res = geocode_one(address)
            new_cache_rows.append({
                "address_key": address_key,
                "Address": address,
                **res,
            })
            log.debug("  [%3d/%d] NEW   %-40s %s", i, total, str(address)[:40], res["geocode_status"])
            time.sleep(0.1)

        results.append(res)

        if i % checkpoint_every == 0:
            cached_count = total - len(new_cache_rows)
            log.info("  progress: %d/%d | %d new geocodes so far", i, total, len(new_cache_rows))

    res_df = pd.DataFrame(results, index=out.index)
    out = pd.concat([out, res_df], axis=1)

    if new_cache_rows:
        new_cache_df = pd.DataFrame(new_cache_rows)
        cache_df = pd.concat([cache_df, new_cache_df], ignore_index=True)
        cache_df = cache_df.drop_duplicates(subset=["address_key"], keep="last")
        cache_df.to_csv(cache_file, index=False)
        log.info("  Added %d new address(es) to %s", len(new_cache_rows), cache_path)
    else:
        log.info("  No new addresses to add to geocode cache")

    successes = (out["geocode_status"] == "success").sum()
    log.info(
        "  Geocoded/reused %d/%d successfully (%.1f%%)",
        successes,
        total,
        100 * successes / total if total else 0,
    )
    return out


def match_by_geocoding(
    geocoded_df: pd.DataFrame,
    parcels_gdf: gpd.GeoDataFrame,
    service_columns: list[str],
) -> tuple[Optional[gpd.GeoDataFrame], Optional[gpd.GeoDataFrame]]:
    """
    Build a point GeoDataFrame from geocoded addresses and spatially join to
    parcels. Returns (geocoded_points_gdf, matched_parcels_gdf).
    """
    valid = geocoded_df[geocoded_df["latitude"].notna()].copy()
    if valid.empty:
        log.warning("No successfully geocoded addresses to spatially join")
        return None, None

    geometry = [Point(lon, lat) for lon, lat in zip(valid["longitude"], valid["latitude"])]
    points_gdf = gpd.GeoDataFrame(valid, geometry=geometry, crs="EPSG:4326")

    log.info("Spatially joining %d points to parcels…", len(points_gdf))
    joined = gpd.sjoin(points_gdf, parcels_gdf, how="left", predicate="within")
    matched_count = joined["index_right"].notna().sum()
    log.info("  %d/%d points fell inside a parcel (%.1f%%)",
             matched_count, len(joined),
             100 * matched_count / len(joined) if len(joined) else 0)

    if matched_count == 0:
        return points_gdf, None

    matched_rows = joined[joined["index_right"].notna()].copy()
    parcel_idxs = matched_rows["index_right"].astype(int).unique()
    matched_parcels = parcels_gdf.iloc[parcel_idxs].copy()

    # Annotate each matched parcel with Malama data (first matching property wins)
    for pidx in parcel_idxs:
        prop = matched_rows[matched_rows["index_right"] == pidx].iloc[0]
        mask = matched_parcels.index == pidx
        matched_parcels.loc[mask, "malama_address"] = prop["Address"]
        matched_parcels.loc[mask, "malama_acres"] = prop["Acres"]
        matched_parcels.loc[mask, "malama_services"] = services_for_row(prop, service_columns)
    matched_parcels["match_method"] = "Geocoding"

    return points_gdf, matched_parcels


# ---------------------------------------------------------------------------
# Map building
# ---------------------------------------------------------------------------

def build_map(
    parcels_by_id: Optional[gpd.GeoDataFrame],
    parcels_by_geocode: Optional[gpd.GeoDataFrame],
    geocoded_points: Optional[gpd.GeoDataFrame],
    service_columns: list[str],
    total_properties: int,
) -> folium.Map:
    # Choose a center: the centroid of everything we mapped, else Kula default
    geometries = []
    if parcels_by_id is not None:
        geometries.extend(parcels_by_id.geometry.tolist())
    if parcels_by_geocode is not None:
        geometries.extend(parcels_by_geocode.geometry.tolist())
    if geometries:
        bbox = gpd.GeoDataFrame(geometry=geometries, crs="EPSG:4326").total_bounds
        center = ((bbox[1] + bbox[3]) / 2, (bbox[0] + bbox[2]) / 2)
    else:
        center = KULA_CENTER

    m = folium.Map(location=list(center), zoom_start=13,
                   tiles="OpenStreetMap", control_scale=True)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/"
              "World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri",
        name="Satellite",
        overlay=False,
        control=True,
    ).add_to(m)

    if parcels_by_id is not None and len(parcels_by_id) > 0:
        folium.GeoJson(
            parcels_by_id,
            name="Parcels — ID Matched (Exact)",
            style_function=lambda _: {
                "fillColor": "#2ecc71", "color": "#27ae60",
                "weight": 2, "fillOpacity": 0.6, "opacity": 0.9,
            },
            highlight_function=lambda _: {
                "fillColor": "#58d68d", "color": "#27ae60",
                "weight": 3, "fillOpacity": 0.8,
            },
            tooltip=folium.GeoJsonTooltip(
                fields=["malama_address", "malama_acres", "malama_services"],
                aliases=["Address:", "Acres:", "Services:"],
                sticky=True,
            ),
        ).add_to(m)

    if parcels_by_geocode is not None and len(parcels_by_geocode) > 0:
        folium.GeoJson(
            parcels_by_geocode,
            name="Parcels — Geocoded (Approximate)",
            style_function=lambda _: {
                "fillColor": "#fc8d59", "color": "#d73027",
                "weight": 2, "fillOpacity": 0.5, "opacity": 0.8,
            },
            highlight_function=lambda _: {
                "fillColor": "#fdae61", "color": "#d73027",
                "weight": 3, "fillOpacity": 0.7,
            },
            tooltip=folium.GeoJsonTooltip(
                fields=["malama_address", "malama_acres", "malama_services"],
                aliases=["Address:", "Acres:", "Services:"],
                sticky=True,
            ),
        ).add_to(m)

    if geocoded_points is not None and len(geocoded_points) > 0:
        cluster = plugins.MarkerCluster(name="Address Points").add_to(m)
        for _, row in geocoded_points.iterrows():
            popup_html = f"""
            <div style="font-family: Arial; font-size: 12px; width: 250px;">
              <h4 style="margin: 0 0 10px 0; color: #2c5aa0;">Malama Kula Service Location</h4>
              <p style="margin: 5px 0;"><strong>Address:</strong><br>{row['Address']}</p>
              <p style="margin: 5px 0;"><strong>Acres:</strong> {row['Acres'] if pd.notna(row['Acres']) else 'N/A'}</p>
              <p style="margin: 5px 0;"><strong>Services:</strong><br>{services_for_row(row, service_columns)}</p>
              <p style="margin: 5px 0;"><strong>Coordinates:</strong><br>
                 Lat: {row['latitude']:.6f}<br>
                 Lon: {row['longitude']:.6f}</p>
            </div>
            """
            folium.CircleMarker(
                location=[row["latitude"], row["longitude"]],
                radius=6,
                popup=folium.Popup(popup_html, max_width=300),
                color="darkred", fillColor="red", fillOpacity=0.8, weight=2,
            ).add_to(cluster)

    # Title / legend
    id_count = len(parcels_by_id) if parcels_by_id is not None else 0
    geo_count = len(parcels_by_geocode) if parcels_by_geocode is not None else 0
    title_html = f"""
    <div style="position: fixed; top: 10px; left: 50px; width: 480px; height: 110px;
                background-color: white; border:2px solid #333; z-index:9999;
                font-size:14px; padding: 15px; opacity: 0.95; border-radius: 5px;
                box-shadow: 0 2px 6px rgba(0,0,0,0.3);">
      <h3 style="margin: 0 0 5px 0; color: #2c5aa0;">Malama Kula</h3>
      <p style="margin: 5px 0; font-size: 12px; color: #666;">
        2023 Maui Wildfire Response, Hazard Mitigation, and Community Services
      </p>
      <p style="margin: 5px 0; font-size: 11px;">
        <strong>Total Parcels Mapped:</strong> {id_count + geo_count} &nbsp;·&nbsp;
        <strong>Total Properties:</strong> {total_properties}
      </p>
    </div>
    """
    m.get_root().html.add_child(folium.Element(title_html))

    legend_html = """
    <div style="position: fixed; bottom: 30px; right: 10px; width: 230px;
                background-color: white; border:2px solid #333; z-index:9999;
                font-size: 12px; padding: 12px; opacity: 0.95; border-radius: 5px;
                box-shadow: 0 2px 6px rgba(0,0,0,0.3);">
      <p style="margin: 0 0 8px 0;"><strong>Legend</strong></p>
      <p style="margin: 4px 0;">
        <span style="display:inline-block; width:14px; height:14px;
                     background:#2ecc71; border:1px solid #27ae60;
                     vertical-align:middle;"></span>
        &nbsp; Parcel ID match (exact)
      </p>
      <p style="margin: 4px 0;">
        <span style="display:inline-block; width:14px; height:14px;
                     background:#fc8d59; border:1px solid #d73027;
                     vertical-align:middle;"></span>
        &nbsp; Geocoded match (~95%)
      </p>
      <p style="margin: 4px 0;">
        <span style="display:inline-block; width:10px; height:10px;
                     background:red; border:2px solid darkred;
                     border-radius:50%; vertical-align:middle;"></span>
        &nbsp; Geocoded address point
      </p>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    folium.LayerControl(collapsed=False).add_to(m)
    return m


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def save_outputs(
    output_dir: Path,
    map_obj: folium.Map,
    geocoded_points: Optional[gpd.GeoDataFrame],
    parcels_by_id: Optional[gpd.GeoDataFrame],
    parcels_by_geocode: Optional[gpd.GeoDataFrame],
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    map_path = output_dir / "malama_kula_interactive_map.html"
    map_obj.save(str(map_path))
    written.append(map_path)

    if geocoded_points is not None:
        p = output_dir / "malama_kula_geocoded.csv"
        geocoded_points.drop(columns="geometry").to_csv(p, index=False)
        written.append(p)

    if parcels_by_id is not None:
        p = output_dir / "malama_kula_parcels_by_id.geojson"
        parcels_by_id.to_file(p, driver="GeoJSON")
        written.append(p)

    if parcels_by_geocode is not None:
        p = output_dir / "malama_kula_parcels_by_geocode.geojson"
        parcels_by_geocode.to_file(p, driver="GeoJSON")
        written.append(p)

    combined = []
    if parcels_by_id is not None:
        combined.append(parcels_by_id)
    if parcels_by_geocode is not None:
        combined.append(parcels_by_geocode)
    if combined:
        p = output_dir / "malama_kula_all_parcels.geojson"
        pd.concat(combined, ignore_index=True).to_file(p, driver="GeoJSON")
        written.append(p)

    return written


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate the Malama Kula interactive property map.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    src = p.add_mutually_exclusive_group()
    src.add_argument("--csv", default=DEFAULT_CSV,
                     help=f"Path to properties CSV (default: {DEFAULT_CSV})")
    src.add_argument("--csv-url",
                     help="URL to properties CSV, e.g. a published Google Sheet")
    p.add_argument("--parcels", default=DEFAULT_PARCELS,
                   help=f"Path to Maui parcels GeoJSON (default: {DEFAULT_PARCELS})")
    p.add_argument("--output-dir", default=".",
                   help="Directory to write outputs (default: current dir)")
    p.add_argument("--skip-geocode", action="store_true",
                   help="Skip geocoding step (useful for quick iteration on map styling)")
    p.add_argument("-v", "--verbose", action="store_true",
                   help="Verbose per-address logging")
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    setup_logging(args.verbose)

    try:
        csv_source = args.csv_url if args.csv_url else args.csv
        properties_df = load_properties(csv_source)
        parcels_gdf = load_parcels(Path(args.parcels))

        tmk_field = next(
            (c for c in parcels_gdf.columns if c.lower() == "tmk"),
            None,
        )
        if tmk_field is None:
            log.error("No 'tmk' field found in parcel data. Columns: %s",
                      list(parcels_gdf.columns))
            return 1

        service_columns = [c for c in SERVICE_COLUMNS if c in properties_df.columns]

        # 1. Parcel-ID match (exact)
        parcels_by_id, matched_idx = match_by_parcel_id(
            properties_df, parcels_gdf, tmk_field, service_columns
        )

        # 2. Geocode the remaining rows that have an address
        to_geocode = properties_df.loc[
            ~properties_df.index.isin(matched_idx)
            & properties_df["Address"].notna()
        ].copy()

        geocoded_points = None
        parcels_by_geocode = None

        if args.skip_geocode:
            log.warning("Skipping geocoding step (--skip-geocode)")
        elif to_geocode.empty:
            log.info("No remaining addresses to geocode")
        else:
            geocoded_df = geocode_addresses(to_geocode)
            geocoded_points, parcels_by_geocode = match_by_geocoding(
                geocoded_df, parcels_gdf, service_columns
            )

        # 3. Build the map
        log.info("Building interactive map…")
        m = build_map(
            parcels_by_id, parcels_by_geocode, geocoded_points,
            service_columns, total_properties=len(properties_df),
        )

        # 4. Save outputs
        written = save_outputs(
            Path(args.output_dir), m, geocoded_points, parcels_by_id, parcels_by_geocode
        )

        log.info("Done. Wrote %d file(s):", len(written))
        for p in written:
            log.info("  %s", p)
        return 0

    except FileNotFoundError as exc:
        log.error("%s", exc)
        return 2
    except Exception as exc:  # noqa: BLE001
        log.exception("Unexpected error: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
