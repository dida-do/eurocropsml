"""Adding NUTS region information to dataset."""

import logging
import sys
from pathlib import Path
from typing import cast

import geopandas as gpd
import pandas as pd
import pyogrio

from eurocropsml.acquisition.config import CollectorConfig
from eurocropsml.acquisition.utils import _nuts_region_downloader

logger = logging.getLogger(__name__)


def add_nuts_regions(
    config: CollectorConfig,
    output_dir: Path,
    shape_dir: Path,
    nuts_dir: Path,
    final_output_dir: Path,
) -> None:
    """Get NUTS regions for parcels.

    Args:
        config: Country-specific configuration for acquiring EuroCrops reflectance data.
        output_dir: Directory path where intermediate results will be stored.
        shape_dir: File path of EuroCrops shapefile.
        nuts_dir: Directory where NUTS-region shapefiles are stored.
        final_output_dir: Directory where the final DataFrame will be stored.

    Raises:
        ValueError: If NUTS-files are not available.

    """
    url: str = (
        "https://ec.europa.eu/eurostat/de/web/gisco/geodata/"
        "statistical-units/territorial-units-statistics"
    )

    label_dir = final_output_dir.joinpath("labels")
    geom_dir = final_output_dir.joinpath("geometries")

    if not (
        label_dir.joinpath(f"{config.country}_labels.parquet").exists()
        and final_output_dir.joinpath(f"{config.country}.parquet").exists()
        and geom_dir.joinpath(f"{config.country}.geojson").exists()
    ):
        shapefile: gpd.GeoDataFrame = pyogrio.read_dataframe(shape_dir)

        # fix invalid geometries
        shapefile["geometry"] = shapefile["geometry"].buffer(0)

        crs: str = str(shapefile.crs).split(":")[-1]

        # get nuts regions
        nuts_region_filename = f"NUTS_RG_01M_{config.year}_{crs}.geojson"
        nuts_regions_file = nuts_dir.joinpath(nuts_region_filename)
        if not nuts_dir.exists() or len(list((nuts_dir.iterdir()))) <= 3:
            nuts_dir.mkdir(exist_ok=True, parents=True)
            _nuts_region_downloader(url, nuts_dir, crs, config.year)

        if len(list((nuts_dir.iterdir()))) == 0:
            logger.info(
                "There are no NUTS region files available."
                f" Please download them manually from {url}."
            )
            sys.exit()
        try:
            nuts: gpd.GeoDataFrame = pyogrio.read_dataframe(nuts_regions_file)

        except Exception:
            available_crs: list[str] = []
            for file in nuts_dir.iterdir():
                available_crs.append(file.stem.split("_")[-1])
            crs = available_crs[0]
            shapefile = shapefile.to_crs(crs)
            nuts_region_filename = f"NUTS_RG_01M_{config.year}_{crs}.geojson"
            nuts_regions_file = nuts_dir.joinpath(nuts_region_filename)

            nuts = pyogrio.read_dataframe(nuts_regions_file)

        nuts_df = nuts[nuts["CNTR_CODE"] == config.country_code]

        parcel_id_name: str = cast(str, config.parcel_id_name)

        cols_shapefile = [parcel_id_name, "geometry", "EC_hcat_n", "EC_hcat_c", "nuts1"]

        for nuts_level in [1, 2, 3]:
            nuts_filtered = nuts_df[nuts_df["LEVL_CODE"] == nuts_level]
            # we use intersect instead of within since some parcels are at the border of two regions
            shapefile = gpd.sjoin(shapefile, nuts_filtered, how="left", predicate="intersects")
            no_intersections = shapefile[shapefile.index_right.isna()]
            no_intersections = no_intersections.to_crs("EPSG:3857")
            nuts_proj = nuts_filtered.to_crs("EPSG:3857")
            for idx, row in no_intersections.iterrows():
                point = row["geometry"]
                closest_geom_idx = nuts_proj.geometry.distance(point).idxmin()
                shapefile.at[idx, "index_right"] = closest_geom_idx
                shapefile.at[idx, "NUTS_ID"] = nuts_proj.at[closest_geom_idx, "NUTS_ID"]
            multiple_intersections = shapefile[
                shapefile.duplicated(subset=parcel_id_name, keep=False)
            ]

            # Keep only the row with the largest intersection for each duplicated parcel_id
            if not multiple_intersections.empty:
                biggest_intersection = multiple_intersections.sort_values(
                    "index_right"
                ).drop_duplicates(subset=parcel_id_name, keep="last")
                shapefile = shapefile[
                    ~shapefile[parcel_id_name].isin(multiple_intersections[parcel_id_name])
                ]
                shapefile = pd.concat([shapefile, biggest_intersection], ignore_index=True)
            shapefile = shapefile.rename(columns={"NUTS_ID": f"nuts{nuts_level}"})
            shapefile = shapefile[cols_shapefile]
            cols_shapefile = cols_shapefile + [f"nuts{nuts_level+1}"]

        # add nuts region to final reflectance dataframe
        full_df: pd.DataFrame = pd.read_parquet(output_dir.joinpath("clipper", "clipped.parquet"))

        shapefile[parcel_id_name] = shapefile[parcel_id_name].astype(int)
        joined_final = pd.merge(full_df, shapefile, on=parcel_id_name, how="left")

        # reorder columns
        cols = joined_final.columns.tolist()
        # Sort columns by date
        cols_dates = pd.to_datetime(cols[1:-6])
        cols_dates = sorted(cols_dates)
        dates_strings = [dt.strftime("%Y-%m-%d") for dt in cols_dates]

        cols = [parcel_id_name] + cols[-6:] + dates_strings
        joined_final = joined_final[cols]

        joined_final = joined_final.rename(columns={f"{parcel_id_name}": "parcel_id"})

        classes_df = joined_final[["parcel_id", "EC_hcat_c", "EC_hcat_n"]]
        geometry_df = joined_final[["parcel_id", "geometry"]]
        geometry_df = gpd.GeoDataFrame(geometry_df, geometry=geometry_df["geometry"])
        joined_final = joined_final.drop(columns="geometry", axis=1)

        # Replace list of nan with None and convert floats to integers
        joined_final[dates_strings] = joined_final[dates_strings].apply(
            lambda col: col.apply(
                lambda x: (None if x is not None and all(pd.isna(val) for val in x) else x)
            )
        )
        joined_final[dates_strings] = joined_final[dates_strings].apply(
            lambda col: col.apply(lambda x: [int(val) for val in x] if x is not None else x)
        )

        label_dir.mkdir(exist_ok=True, parents=True)
        geom_dir.mkdir(exist_ok=True, parents=True)

        classes_df.to_parquet(
            label_dir.joinpath(f"{config.ec_filename}_{config.year}_labels.parquet"), index=False
        )

        joined_final.to_parquet(
            final_output_dir.joinpath(f"{config.ec_filename}_{config.year}.parquet"), index=False
        )

        geometry_df.to_file(
            geom_dir.joinpath(f"{config.ec_filename}_{config.year}.geojson"),
            driver="GeoJSON",
        )

    else:
        logger.info(
            "Raw_data files already exists. Please delete them if you want to recreate them."
        )
