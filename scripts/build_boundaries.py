"""Build the UK boundary file used by `indurent_matching.geo`.

Extracts the UK land outline and the regions (named as the dataset names them) from Natural Earth 1:10m
(public domain, https://www.naturalearthdata.com/) and writes them to `data/geo/uk_boundaries.geojson`.
That file is committed (the rest of data/ is gitignored), so no download is needed at run time. Rerun only to
refresh the boundaries:

    uv run python scripts/build_boundaries.py

The outline is not simplified: the UK-only file is under 1 MB at full Natural Earth detail.
"""

import urllib.request
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "external"  # gitignored download cache
OUT = ROOT / "data" / "geo" / "uk_boundaries.geojson"
NE_URL = "https://naciscdn.org/naturalearth/10m/cultural/{}.zip"

# The dataset's region labels are coarser than England's official regions, so official regions are merged to
# match them (EDA 2.2). Scotland, Wales and Northern Ireland keep their own names.
ENGLAND_TO_LABEL = {
    "East": "East of England",
    "East Midlands": "Midlands",
    "West Midlands": "Midlands",
    "Greater London": "London",
    "Yorkshire and the Humber": "Yorks. & Humber",
}


def natural_earth(name: str) -> Path:
    """Path to a Natural Earth zip in the cache, downloading it first if needed."""
    path = CACHE / f"{name}.zip"
    if not path.exists():
        CACHE.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(NE_URL.format(name), path)
    return path


def main() -> None:
    countries = gpd.read_file(natural_earth("ne_10m_admin_0_countries"))
    land = countries.loc[countries["ADMIN"] == "United Kingdom", ["geometry"]].dissolve()
    land["kind"], land["name"] = "land", "United Kingdom"

    areas = gpd.read_file(natural_earth("ne_10m_admin_1_states_provinces"), where="adm0_a3 = 'GBR'")
    areas["name"] = np.where(
        areas["geonunit"] == "England", areas["region"].replace(ENGLAND_TO_LABEL), areas["geonunit"]
    )
    regions = areas.dissolve("name")[["geometry"]].reset_index()
    regions["kind"] = "region"

    out = gpd.GeoDataFrame(pd.concat([land, regions], ignore_index=True)[["kind", "name", "geometry"]], crs=4326)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_file(OUT, driver="GeoJSON")
    print(f"wrote {OUT.relative_to(ROOT)} ({OUT.stat().st_size / 1e6:.1f} MB): {', '.join(out['name'])}")


if __name__ == "__main__":
    main()
