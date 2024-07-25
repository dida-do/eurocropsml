"""Collecting SAFE-files via EO-Lab finder API."""

#####################################################################
# Initial idea to obtain the SAFE-files via an API call: David Gackstetter
# Copyright: Copyright 2022, Technical University of Munich
# Email: david.gackstetter@tum.de
#####################################################################
# Script majorly revised for EuroCrops by Joana Reuss
# Copyright: Copyright 20223, Technical University of Munich
# Email: joana.reuss@tum.de
#####################################################################

import calendar
import json
import logging
import multiprocessing
import multiprocessing as mp_orig
import time
from functools import partial
from pathlib import Path
from typing import Any, cast
from xml.etree import ElementTree

import geopandas as gpd
import pandas as pd
import pyogrio
import requests
from pyproj import CRS
from shapely.geometry.polygon import Polygon
from tqdm import tqdm

from eurocropsml.acquisition.config import CollectorConfig
from eurocropsml.acquisition.utils import _get_dict_value_by_name, _load_pkg

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def _eolab_finder(
    year: int,
    months: list[int],
    geom_polygon: str,
    max_cloud_cover: int,
    processing_level: str,
    products_per_request: int,
) -> dict[str, str | dict | list]:
    """Getting list of Sentinel .SAFE files that overlap the given geometry."""
    _, num_days = calendar.monthrange(year, months[1])
    months_list: list[str] = ["0{0}".format(m) if m < 10 else "{0}".format(m) for m in months]

    request_url = """https://datahub.creodias.eu/odata/v1/Products?$filter=(Attributes/OData.CSC.D\
oubleAttribute/any(i0:i0/Name eq 'cloudCover' and i0/Value le {0}) and (ContentDate/Start ge {1}-\
{2}-01T00:00:00.000Z and ContentDate/Start le {1}-{3}-{4}T23:59:59.999Z) and (Online eq true) and \
(OData.CSC.Intersects(Footprint=geography'SRID=4326;{6}')) and (((((Collection/Name eq 'SENTINEL-2\
') and not contains(Name, '_N0500') and (((Attributes/Odata.CSC.StringAttribute/any(i0:i0/Name eq \
'productType' and i0/Value eq 'S2MSI{5}')))))))))&$expand=Attributes&$expand=Assets&$orderby=Conte\
ntDate/Start asc&$top={7}""".format(
        max_cloud_cover,
        year,
        months_list[0],
        months_list[1],
        num_days,
        processing_level[1:],
        geom_polygon,
        products_per_request,
    )

    response_text = requests.get(request_url).text

    return cast(dict, json.loads(response_text))


def acquire_s2_tiles(
    config: CollectorConfig,
    output_dir: Path,
    shape_dir: Path,
    shape_dir_clean: Path,
    eodata_dir: str | None,
    workers: int,
    batch_size: int = 10000,
) -> None:
    """
    Function to acquire Sentinel 2 tiles.

    Args:
        config: Country-specific configuration for acquiring EuroCrops reflectance data.
        output_dir: Directory path where intermediate results will be stored.
        shape_dir: File path of EuroCrops shapefile.
        shape_dir_clean: Directory where the cleaned shapefile will be stored.
        eodata_dir: Directory where Sentinel-2 data is stored.
            If None, `eodata` is used since this will be returned by the API call.
        workers: Maximum number of workers used for multiprocessing.
        batch_size: Batch size used for multiprocessed merging of SAFE-files and parcels.

    """

    output_dir.mkdir(exist_ok=True, parents=True)

    download_kwargs: dict[str, Any] = {
        "year": config.year,
        "country": config.country,
        "country_code": config.country_code,
        "shape_dir": shape_dir,
        "shape_dir_clean": shape_dir_clean,
        "eodata_dir": eodata_dir,
        "parcel_id_name": config.parcel_id_name,
        "months": config.months,
        "geom_polygon": config.polygon,
        "max_cloud_cover": config.max_cloud_cover,
        "processing_level": config.processing_level,
        "max_requested_products": config.max_requested_products,
        "output_dir": output_dir,
        "workers": workers,
        "batch_size": batch_size,
    }

    _downloader(**download_kwargs)


def _downloader(
    year: int,
    country: str,
    shape_dir: Path,
    shape_dir_clean: Path,
    eodata_dir: str | None,
    parcel_id_name: str,
    months: list[int],
    geom_polygon: str,
    max_cloud_cover: int,
    processing_level: str,
    max_requested_products: int,
    output_dir: Path,
    workers: int,
    batch_size: int,
) -> None:
    request_path = output_dir.joinpath("requests")
    request_path.mkdir(exist_ok=True, parents=True)
    eofinder_request: Path = request_path.joinpath(f"{country}_{year}.json")

    if not eofinder_request.exists():
        # if not eofinder_request_new.exists():
        logger.info("Requesting list of .SAFE-files from Creodias...")
        # Request all sentinel tiles from eo-lab API based
        # on given geometry (e.g. national boerders) and transform to dataframe
        run_loop: bool = True
        while run_loop:
            try:
                requests: dict = _eolab_finder(
                    year,
                    months,
                    geom_polygon,
                    max_cloud_cover,
                    processing_level,
                    max_requested_products,
                )
                run_loop = False
                logger.info("API-request was successful!")
            except ConnectionError:
                time.sleep(2000)

        # Extra loop in case that available products exceed number of maximum requests
        # These requestes are executed on a monthly basis.
        if len(requests["value"]) == max_requested_products:
            logger.info("Too many requested product. Executing monthly requests.")
            all_months: list = list(range(months[0], months[1] + 1))
            for idx, month in enumerate(all_months):
                run_loop = True
                while run_loop:
                    try:
                        requests_monthly: dict[str, Any] = _eolab_finder(
                            year,
                            [month, month],
                            geom_polygon,
                            max_cloud_cover,
                            processing_level,
                            max_requested_products,
                        )
                        run_loop = False
                        logger.info(f"API-request for {calendar.month_name[month]} was successful!")

                        if idx == 0:
                            accum_requests = requests_monthly["value"]
                        else:
                            accum_requests = accum_requests + requests_monthly["value"]

                    except ConnectionError:
                        time.sleep(2000)
                run_loop = True

            requests = {}
            requests["value"] = accum_requests

        with open(eofinder_request, "w") as outfile:
            json.dump(requests, outfile)

    else:
        with open(eofinder_request) as outfile:
            requests = json.load(outfile)

    products = requests["value"]

    request_files = output_dir.joinpath("requests", "request_safe_files.pkg")
    max_workers = min(mp_orig.cpu_count(), max(1, min(len(products), workers)))

    if not request_files.exists():
        # creating GeoDataFrame from .SAFE-files
        results: list[list] = []
        with mp_orig.Pool(processes=max_workers) as p:
            func = partial(_get_tiles, eodata_dir)
            process_iter = p.imap(func, products, chunksize=1000)
            ti = tqdm(total=len(products), desc="Processing requested SAFE files.")

            for result in process_iter:
                if result is not None:
                    results.append(result)
                ti.update(n=1)
            ti.close()

        request_df: pd.DataFrame = pd.DataFrame(
            results,
            columns=(
                "geometry",
                "productIdentifier",
                "completionDate",
                "cloudCover",
                "crs",
            ),
        )

        request_df = request_df.sort_values(by=["completionDate"])
        request_df = request_df.reset_index().drop(columns=["index"])

        request_df.to_pickle(request_files)

        logger.info(f"Finished acquiring SAFE-files for {country} for {year}.")

    else:
        request_df = pd.read_pickle(request_files)

    unique_crs: list[int] = request_df["crs"].unique().tolist()

    request_df_list = [
        gpd.GeoDataFrame(
            request_df[request_df["crs"] == crs],
            crs=crs,
            geometry=request_df[request_df["crs"] == crs]["geometry"],
        )
        for crs in unique_crs
    ]

    if not (
        output_dir.joinpath("full_safe_file_list.pkg").exists()
        and output_dir.joinpath("full_parcel_list.pkg").exists()
    ):
        if not shape_dir_clean.exists():
            # Cleaning up country's shapefile
            # Load in SHP-File
            shapefile: gpd.GeoDataFrame = pyogrio.read_dataframe(shape_dir)
            if "EC_NUTS3" in shapefile.columns.tolist():
                shapefile.drop(["EC_NUTS3"], axis=1)
            # sort shapefile s.t. NULL classes are at the end
            shapefile.sort_values(by="EC_hcat_c", na_position="last", inplace=True)
            # drop duplicate geometries
            # this will drop NULL classes if they are duplicates
            shapefile.drop_duplicates(subset=["geometry"], keep="first", inplace=True)

            # create parcel ID if none exists
            if parcel_id_name not in shapefile.columns:
                shapefile[parcel_id_name] = [
                    f"{i+1:0{len(str(len(shapefile)))}d}" for i in range(len(shapefile))
                ]

            shapefile.to_file(shape_dir_clean)
        else:
            shapefile = pyogrio.read_dataframe(shape_dir_clean)

        results = []
        parcel_path = output_dir.joinpath("parcels")
        parcel_path.mkdir(exist_ok=True, parents=True)

        subset_cols = [parcel_id_name, "geometry"]
        shapefile = shapefile[subset_cols]

        for request_df in request_df_list:
            # transforming shapefile into CRS of SAFE files
            tile_crs: str = request_df.crs.srs
            if tile_crs != shapefile.crs.srs:
                parcel_df = shapefile.to_crs(tile_crs)
            else:
                parcel_df = shapefile.copy()
            with multiprocessing.Pool(processes=max_workers) as p:
                args_list = [
                    (i, batch_size, parcel_df, request_df, parcel_path)
                    for i in range(0, len(parcel_df), batch_size)
                ]

                result_list = list(parcel_path.iterdir())

                if len(result_list) < len(args_list):
                    process_iter = p.imap(_process_batch, args_list)
                    ti = tqdm(total=len(args_list), desc="Matching SAFE-files and parcels.")
                    for _ in process_iter:
                        ti.update(n=1)
                    ti.close()

        del parcel_df
        del shapefile
        del request_df_list

        with multiprocessing.Pool(processes=max_workers) as p:
            result_list = list(parcel_path.iterdir())
            process_iter = p.imap(_load_pkg, result_list)
            ti = tqdm(total=len(args_list), desc="Loading DataFrames.")
            for result in process_iter:
                results.append(result)  # type: ignore[arg-type]
                ti.update(n=1)
            ti.close()

        # Combine the results if needed
        combined_result = pd.concat(results, ignore_index=True)

        combined_result["completionDate"] = combined_result["completionDate"].str[0:10]

        # if a parcel has multiple sentinel tiles for the same date, we take the one with
        # the lowest cloud cover value
        subset_cols = [parcel_id_name, "completionDate"]

        duplicates = combined_result[combined_result.duplicated(subset=subset_cols, keep=False)]
        combined_result = combined_result[
            ~combined_result.duplicated(subset=subset_cols, keep=False)
        ]

        min_index = duplicates.groupby(subset_cols)["cloudCover"].idxmin()
        duplicates_filtered = duplicates.loc[min_index]
        combined_result = pd.concat([duplicates_filtered, combined_result])

        # saving DataFrame that matches .SAFE-files with parcels
        combined_result.to_pickle(output_dir.joinpath("full_parcel_list.pkg"))

        unique_safe_files = combined_result["productIdentifier"].unique()

        # DataFrame of unique .SAFE-files
        safefiles_df = pd.DataFrame({"productIdentifier": unique_safe_files})
        safefiles_df.to_pickle(output_dir.joinpath("full_safe_file_list.pkg"))

        logger.info(f"Finished merging .SAFE-files and parcels for {country} for {year}.")


def _process_batch(args: tuple[int, int, gpd.GeoDataFrame, gpd.GeoDataFrame, Path]) -> None:
    """Checking for intersections between raster tiles and parcel polygons."""
    i, batch_size, parcel_df, request_df, parcel_path = args
    if not parcel_path.joinpath(f"parcel_list_{i}.pkg").exists():
        batch_parcel_df = parcel_df[i : i + batch_size]
        result = gpd.sjoin(batch_parcel_df, request_df, how="left", predicate="intersects")
        result = result[result["index_right"].notna()]
        result = result.drop(["index_right", "crs", "geometry"], axis=1)
        result.to_pickle(parcel_path.joinpath(f"parcel_list_{i}.pkg"))


def _get_tiles(
    eodata_dir: str | None,
    tile: dict,
) -> list | None:
    """Getting information from raster .SAFE-files."""
    safe_file: str = tile["S3Path"]  # product Identifier
    cloudcover: float | None = cast(
        float, _get_dict_value_by_name(tile["Attributes"], "cloudCover")
    )
    endingdate: str | None = cast(
        str, _get_dict_value_by_name(tile["Attributes"], "endingDateTime")
    )

    if endingdate is None:
        logger.warning("Wasn't able to obtain observation date. This .SAFE-file is being skipped.")
        return None

    if cloudcover is None:
        cloudcover = 0.0

    if eodata_dir is not None:
        safe_file = safe_file.replace("eodata", eodata_dir)
    try:
        granule_path = Path(safe_file).joinpath("GRANULE")
        folder: list = list(granule_path.iterdir())
        tree = ElementTree.parse(folder[0].joinpath("MTD_TL.xml"))
        root = tree.getroot()
        spatial_ref_element: ElementTree.Element = cast(
            ElementTree.Element, root.find(".//HORIZONTAL_CS_NAME")
        )
        gcs: str = cast(str, spatial_ref_element.text)
        gcs = gcs.split(" ")[0]
        crs: int = cast(int, CRS.from_string(gcs).to_epsg())
        request: list | None = [
            Polygon(tile["GeoFootprint"]["coordinates"][0]),
            safe_file,
            endingdate,
            cloudcover,
            int(crs),
        ]
    except ValueError:
        logger.warning(
            f"The geometry of {safe_file} could not be transformed into a"
            " shapely Polygon correctly. This .SAFE-file is being skipped."
        )
        request = None

    return request
