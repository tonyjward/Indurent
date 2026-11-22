"""Location checks: how far each point is from the UK coast and from its labelled region, and its nearest city.

Implements EDA 2.2 and section 5. Boundaries are the UK land outline and regions in
`data/geo/uk_boundaries.geojson` (Natural Earth 1:10m, public domain; built by `scripts/build_boundaries.py`),
with regions named as the dataset names them. Distances are measured in metres on the British National Grid.
"""

from functools import cache
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from indurent_matching.config import COAST_TOLERANCE_KM, MAJOR_CITIES, REGION_TOLERANCE_KM

BOUNDARIES = Path(__file__).resolve().parents[2] / "data" / "geo" / "uk_boundaries.geojson"
BRITISH_NATIONAL_GRID = 27700
EARTH_RADIUS_MILES = 3958.8


@cache
def _boundaries() -> tuple[object, gpd.GeoSeries]:
    """The UK land outline and the regions (indexed by name), in British National Grid. Read once, then cached."""
    shapes = gpd.read_file(BOUNDARIES).to_crs(BRITISH_NATIONAL_GRID)
    land = shapes.loc[shapes["kind"] == "land", "geometry"].iloc[0]
    regions = shapes.loc[shapes["kind"] == "region"].set_index("name")["geometry"]
    return land, regions


def assess_locations(df: pd.DataFrame) -> pd.DataFrame:
    """Check each row's coordinates against the coastline and its labelled region.

    Args:
        df: rows with `latitude`, `longitude` and `region` (the dataset's own label).

    Returns:
        A frame on the same index with:

        * `km_offshore`: distance from the UK coast (0 on land); NaN without coordinates.
        * `km_to_region`: distance from the labelled region (0 inside it); NaN without coordinates or label.
        * `location_confidence`: `high` within `REGION_TOLERANCE_KM` of the labelled region; `medium` on
          land (or within `COAST_TOLERANCE_KM` of the coast) but outside it; `unknown` otherwise.
        * `derived_region`: the region the coordinates fall in (nearest region), for high and medium only.
    """
    land, regions = _boundaries()
    out = pd.DataFrame(index=df.index)
    out["km_offshore"] = np.nan
    out["km_to_region"] = np.nan
    out["location_confidence"] = "unknown"
    out["derived_region"] = None

    located = df[df["latitude"].notna() & df["longitude"].notna()]
    points = gpd.GeoSeries(
        gpd.points_from_xy(located["longitude"], located["latitude"]), index=located.index, crs=4326
    ).to_crs(BRITISH_NATIONAL_GRID)

    # Distance (km) from every located point to every region: one column per region.
    km_to_each = pd.DataFrame({name: points.distance(shape) / 1000 for name, shape in regions.items()})
    labelled = located["region"].where(located["region"].isin(regions.index))
    out.loc[located.index, "km_offshore"] = points.distance(land) / 1000
    out.loc[located.index, "km_to_region"] = [
        km_to_each.at[idx, label] if pd.notna(label) else np.nan for idx, label in labelled.items()
    ]

    on_land = out["km_offshore"] < COAST_TOLERANCE_KM
    agrees = out["km_to_region"] <= REGION_TOLERANCE_KM  # regions are land, so this implies on land too
    out.loc[on_land, "location_confidence"] = "medium"
    out.loc[agrees, "location_confidence"] = "high"

    usable = out.index[out["location_confidence"] != "unknown"]
    out.loc[usable, "derived_region"] = km_to_each.loc[usable].idxmin(axis=1)
    return out


def haversine_miles(lat1, lon1, lat2, lon2):
    """Great-circle distance in miles between points given in degrees (works on arrays)."""
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * EARTH_RADIUS_MILES * np.arcsin(np.sqrt(a))


def nearest_city(df: pd.DataFrame, usable: pd.Series) -> pd.DataFrame:
    """Nearest of `MAJOR_CITIES` and the distance to it in miles, for rows where `usable` is True.

    Rows whose location isn't trusted get NaN, so they fall into the "unknown" group of any distance filter
    instead of being counted as near or far (EDA section 5).
    """
    rows = df[usable]
    miles = pd.DataFrame(
        {
            city: haversine_miles(rows["latitude"], rows["longitude"], lat, lon)
            for city, (lat, lon) in MAJOR_CITIES.items()
        },
        index=rows.index,
    )
    out = pd.DataFrame({"nearest_city": miles.idxmin(axis=1), "miles_to_city": miles.min(axis=1)})
    return out.reindex(df.index)
