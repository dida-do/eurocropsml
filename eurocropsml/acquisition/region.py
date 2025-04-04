"""Adding NUTS region information to dataset."""

import logging
from pathlib import Path
from typing import cast

import geopandas as gpd
import pandas as pd
import pyogrio
from pyogrio.errors import DataSourceError
from tqdm import tqdm

from eurocropsml.acquisition.config import CollectorConfig
from eurocropsml.acquisition.utils import _get_closest_year, _nuts_region_downloader

logger = logging.getLogger(__name__)


def add_nuts_regions(
    config: CollectorConfig,
    raw_data_dir: Path,
    output_dir: Path,
    shape_dir: Path,
    nuts_dir: Path,
    final_output_dir: Path,
    rebuild: bool = False,
) -> None:
    """Get NUTS regions for parcels.

    Args:
        config: Country-specific configuration for acquiring EuroCrops reflectance data.
        raw_data_dir: Directory where raw data that is independent of the satellite
            is stored. This concerns the labels and geometries.
        output_dir: Directory path where intermediate results will be stored.
        shape_dir: File path of EuroCrops shapefile.
        nuts_dir: Directory where NUTS-region shapefiles are stored.
        final_output_dir: Directory where the final DataFrame will be stored.
            This depends on the satellite.
        rebuild: Whether to re-build the final clipped parquet file for each month.
            This will overwrite the existing ones.

    Raises:
        FileNotFoundError: If NUTS-files are not available.
        FileNotFoundError: If the merged clipped file from clipping module does not exist.

    """
    url: str = (
        "https://ec.europa.eu/eurostat/de/web/gisco/geodata/"
        "statistical-units/territorial-units-statistics"
    )

    shapefile: gpd.GeoDataFrame = pyogrio.read_dataframe(shape_dir)

    # fix invalid geometries
    shapefile["geometry"] = shapefile["geometry"].buffer(0)

    crs: str = str(shapefile.crs).split(":")[-1]

    # get nuts regions
    nuts_region_filename = f"NUTS_RG_01M_{config.year}_{crs}.geojson"
    nuts_regions_file = nuts_dir.joinpath(nuts_region_filename)

    if not nuts_dir.exists() or len(list((nuts_dir.iterdir()))) < 3:
        nuts_dir.mkdir(exist_ok=True, parents=True)
        _nuts_region_downloader(
            url, nuts_dir, crs, config.year, [file.name for file in nuts_dir.iterdir()]
        )

    if not any(nuts_dir.iterdir()):
        raise FileNotFoundError(
            "There are no NUTS region files available."
            f" Please download them manually from {url}."
        )
    try:
        nuts: gpd.GeoDataFrame = pyogrio.read_dataframe(nuts_regions_file)

    except DataSourceError:

        available_crs: list[str] = []
        available_years: list[int] = []
        for file in nuts_dir.iterdir():
            available_crs.append(file.stem.split("_")[-1])
            available_years.append(int(file.stem.split("_")[-2]))
        nuts_year = _get_closest_year(available_years, config.year)
        new_crs = available_crs[0]
        logger.info(f"NUTS-file with CRS {crs} could not be found. Using CRS {new_crs}.")
        shapefile = shapefile.to_crs(new_crs)
        nuts_region_filename = f"NUTS_RG_01M_{nuts_year}_{new_crs}.geojson"
        nuts_regions_file = nuts_dir.joinpath(nuts_region_filename)

        nuts = pyogrio.read_dataframe(nuts_regions_file)

    nuts_df = nuts[nuts["CNTR_CODE"] == config.country_code]

    parcel_id_name: str = cast(str, config.parcel_id_name)

    cols_shapefile = [parcel_id_name, "geometry", "EC_hcat_n", "EC_hcat_c", "nuts1"]

    for nuts_level in [1, 2, 3]:
        nuts_filtered = nuts_df[nuts_df["LEVL_CODE"] == nuts_level]
        # we use intersect instead of within
        # since some parcels are at the border of two regions
        shapefile = gpd.sjoin(shapefile, nuts_filtered, how="left", predicate="intersects")
        no_intersections = shapefile[shapefile.index_right.isna()]
        no_intersections = no_intersections.to_crs("EPSG:3857")
        nuts_proj = nuts_filtered.to_crs("EPSG:3857")
        for idx, row in no_intersections.iterrows():
            point = row["geometry"]
            closest_geom_idx = nuts_proj.geometry.distance(point).idxmin()
            shapefile.at[idx, "index_right"] = closest_geom_idx
            shapefile.at[idx, "NUTS_ID"] = nuts_proj.at[closest_geom_idx, "NUTS_ID"]
        multiple_intersections = shapefile[shapefile.duplicated(subset=parcel_id_name, keep=False)]

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
        shapefile[parcel_id_name] = shapefile[parcel_id_name].astype(int)
        cols_shapefile = cols_shapefile + [f"nuts{nuts_level+1}"]

    label_dir = raw_data_dir.joinpath("labels", str(config.year))
    geom_dir = raw_data_dir.joinpath("geometries", str(config.year))

    label_dir.mkdir(exist_ok=True, parents=True)
    geom_dir.mkdir(exist_ok=True, parents=True)

    classes_df = shapefile[[f"{parcel_id_name}", "EC_hcat_c", "EC_hcat_n"]]
    geometry_df = shapefile[[f"{parcel_id_name}", "geometry"]]

    classes_df = classes_df.rename(columns={f"{parcel_id_name}": "parcel_id"})
    geometry_df = geometry_df.rename(columns={f"{parcel_id_name}": "parcel_id"})

    classes_df.to_parquet(
        label_dir.joinpath(f"{config.ec_filename}_labels.parquet"),
        index=False,
    )

    geometry_df.to_file(
        geom_dir.joinpath(f"{config.ec_filename}.geojson"),
        driver="GeoJSON",
    )

    for month in tqdm(
        range(config.months[0], config.months[1] + 1), desc="Adding NUTS regions to data..."
    ):
        month = f"{month:02d}"
        month_dir: Path = final_output_dir.joinpath(f"{month}")
        if rebuild or not month_dir.joinpath(f"{config.ec_filename}.parquet").exists():
            month_dir.mkdir(exist_ok=True, parents=True)
            # add nuts region to final reflectance dataframe
            try:
                full_df: pd.DataFrame = pd.read_parquet(
                    output_dir.joinpath("clipper", f"{month}", "clipped.parquet")
                )
            except FileNotFoundError as e:
                raise FileNotFoundError(
                    f"{output_dir.joinpath('clipper', f'{month}', 'clipped.parquet')} does not "
                    "exist. Run the clipping process again or change the acquisition month in "
                    "acquisiton.config.CollectorConfig.months"
                ) from e

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

            joined_final = joined_final.drop(columns="geometry", axis=1)

            # Replace list of nan with None
            joined_final[dates_strings] = joined_final[dates_strings].apply(
                lambda col: col.apply(
                    lambda x: (
                        None
                        if (
                            x is None
                            or (isinstance(x, float) and pd.isna(x))
                            or (isinstance(x, list) and all(pd.isna(val) for val in x))
                        )
                        else x
                    )
                )
            )

            joined_final.to_parquet(
                month_dir.joinpath(f"{config.ec_filename}.parquet"),
                index=False,
            )

        else:
            logger.info(
                f"{month_dir.joinpath(f'{config.ec_filename}.parquet')} already exists. "
                "Please delete it if you want to recreate it or set rebuild to False."
            )
