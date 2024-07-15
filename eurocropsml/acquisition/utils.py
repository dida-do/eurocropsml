"""Utilities for the dataset acquisition."""

import logging
import sys
import time
from pathlib import Path
from typing import Any

import bs4
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import requests
from rasterio.io import DatasetReader
from rasterio.mask import mask
from rasterio.plot import reshape_as_image
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as ec
from selenium.webdriver.support.ui import WebDriverWait
from tqdm import tqdm

logger = logging.getLogger(__name__)

pd.options.mode.chained_assignment = None


def mask_polygon_raster(
    tilepaths: list[Path],
    polygon_df: pd.DataFrame,
    parcel_id_name: str,
    product_date: str,
) -> pd.DataFrame:
    """Clipping parcels from raster files (per band) and calculating median pixel value per band.

    Args:
        tilepaths: Paths to the raster's band tiles.
        polygon_df: GeoDataFrame of all parcels to be clipped.
        parcel_id_name: The country's parcel ID name (varies from country to country).
        product_date: Date on which the raster tile was obtained.

    Returns:
        Dataframe with clipped values.
    """

    parcels_dict: dict[int, list[int | None]] = {
        parcel_id: [] for parcel_id in polygon_df[parcel_id_name].unique()
    }

    # removing any self-intersections or inconsistencies in geometries
    polygon_df["geometry"] = polygon_df["geometry"].buffer(0)
    polygon_df = polygon_df.reset_index(drop=True)

    for b, band_path in enumerate(tilepaths):
        with rasterio.open(band_path, "r") as raster_tile:
            if b == 0 and polygon_df.crs.srs != raster_tile.crs.data["init"]:
                # transforming shapefile into CRS of raster tile
                polygon_df = polygon_df.to_crs(raster_tile.crs.data["init"])
            # clippping geometry out of raster tile and saving in dictionary
            polygon_df.apply(
                lambda row: _process_row(row, raster_tile, parcels_dict, parcel_id_name),
                axis=1,
            )

    parcels_df = pd.DataFrame(list(parcels_dict.items()), columns=[parcel_id_name, product_date])

    return parcels_df


def _process_row(
    row: pd.Series,
    raster_tile: DatasetReader,
    parcels_dict: dict[int, list[int | None]],
    parcel_id_name: str,
) -> None:
    """Masking geometry from raster tiles and calculating median pixel value."""
    parcel_id: int = row[parcel_id_name]
    geom = row["geometry"]
    try:
        masked_img, _ = mask(raster_tile, [geom], crop=True, nodata=0)
        masked_img = reshape_as_image(masked_img)

        # Calculate the median of each patch where the clipped values are not zero
        # third dimension has index 0 since we only have one band
        not_zero = masked_img[:, :, 0] != 0
        if not not_zero.any():
            patch_median: int = 0
        else:
            patch_median = np.median(masked_img[:, :, 0][not_zero]).astype(np.int16)
            patch_median = max(0, patch_median)
        parcels_dict[parcel_id].append(patch_median)
    except ValueError:
        # in case geometry is not inside raster tile
        parcels_dict[parcel_id].append(None)


def _merge_clipper(
    full_df: gpd.GeoDataFrame,
    clipped_output_dir: Path,
    output_dir: Path,
    parcel_id_name: str,
) -> None:
    logger.info("Starting merging of DataFrames...")
    df_list: list = [file for file in clipped_output_dir.iterdir() if "Final_" in file.name]

    # setting parcel_id column to index
    full_df.set_index(parcel_id_name, inplace=True)
    for file in tqdm(df_list):
        full_df = full_df.fillna(pd.read_pickle(file))

    # reset index column
    full_df = full_df.reset_index()

    full_df.to_parquet(output_dir.joinpath("clipped.parquet"))
    logger.info("Saved final clipped file.")


def _get_options_from_field(url: str, field_id: str) -> bs4.element.ResultSet:
    response = requests.get(url)
    response.raise_for_status()
    soup = bs4.BeautifulSoup(response.content, "html.parser")
    select_element = soup.find("select", {"id": field_id})
    options: bs4.element.ResultSet = select_element.find_all("option")

    return options


def _get_closest_year(url: str, year: int) -> str:
    """Get available years and select the one closest to the acquisition year."""
    selected_year: int

    try:
        options: bs4.element.ResultSet = _get_options_from_field(url, "year")

        year_options: list[int] = sorted(
            [int(option.get("value")) for option in options if len(option.get("value")) > 0]
        )

        for number in year_options:
            if number == year:
                return str(year)
            elif number < year:
                selected_year = number

        return str(selected_year)

    except ValueError:
        logger.warning(
            "Failed to acquire available years for NUTS region files."
            f" Please download NUTS files manually from {url}."
        )

        sys.exit()


def _get_proj_options(url: str) -> list[str]:
    """Get all available projections."""
    try:
        options: bs4.element.ResultSet = _get_options_from_field(url, "proj")

        proj_options: list[str] = sorted([option.get("value") for option in options])

        return proj_options

    except ValueError:
        logger.warning(
            "Failed to acquire projection options for NUTS region files."
            f" Please download NUTS files manually from {url}."
        )

        sys.exit()


def _nuts_region_downloader(url: str, download_dir: Path, crs: str, year: int) -> None:

    chrome_options: Options = Options()
    chrome_options.add_argument("--headless")  # Run Chrome in headless mode

    timeout: int = 120  # Timeout after 120 seconds
    polling_interval: int = 5  # Check every 5 second
    file_count = 0

    chrome_prefs = {
        "profile.default_content_settings.popups": 0,
        "download.default_directory": str(download_dir),
        "directory_upgrade": True,
        "safebrowsing.enabled": True,
    }

    chrome_options.add_experimental_option("prefs", chrome_prefs)

    selected_year: str = _get_closest_year(url, year)
    projections: list[str] = _get_proj_options(url)

    driver: webdriver.Chrome = webdriver.Chrome(options=chrome_options)
    driver.get(url)

    field_format: webdriver.remote.webelement.WebElement = driver.find_element(By.ID, "format")
    driver.execute_script("arguments[0].value = 'geojson';", field_format)

    field_geom: webdriver.remote.webelement.WebElement = driver.find_element(By.ID, "geomType")
    driver.execute_script("arguments[0].value = 'RG';", field_geom)

    field_scale: webdriver.remote.webelement.WebElement = driver.find_element(By.ID, "scale")
    driver.execute_script("arguments[0].value = '01M';", field_scale)

    field4_year: webdriver.remote.webelement.WebElement = driver.find_element(By.ID, "year")
    driver.execute_script(f"arguments[0].value = {selected_year};", field4_year)

    try:
        if crs in projections:
            projections = [crs]  # only download the required projection, otherwise download all
        for proj in projections:
            filename = (
                f"NUTS_RG_01M_{year}_{proj}.geojson"  # Replace with the actual expected file name
            )
            filepath = download_dir / filename
            if not filepath.exists():

                field_proj: webdriver.remote.webelement.WebElement = driver.find_element(
                    By.ID, "proj"
                )
                driver.execute_script(f"arguments[0].value = {proj};", field_proj)

                # Find and click the download button
                download_button: webdriver.remote.webelement.WebElement = WebDriverWait(
                    driver, 100
                ).until(
                    ec.element_to_be_clickable((By.XPATH, '//button[contains(text(), "Download")]'))
                )

                driver.execute_script("arguments[0].scrollIntoView(true);", download_button)
                download_button.click()

                # Wait until download has started
                WebDriverWait(driver, 10).until(
                    ec.visibility_of_element_located((By.ID, "loadingJsonFile"))
                )
                logger.info(f"Downloading NUTS file EPSG:{proj} for {year}...")

                # Wait for the file to appear in the download directory
                start_time: float = time.time()

                while not filepath.exists():
                    if time.time() - start_time > timeout:
                        logger.info(
                            "Couldn't finish downloading."
                            f" Skipping NUTS file EPSG:{proj} for {year}."
                        )
                    time.sleep(polling_interval)
                logger.info(f"Finished downloading NUTS file EPSG:{proj} for {year}.")
                file_count += 1

            else:
                file_count += 1

    finally:
        driver.quit()
        if file_count == len(projections):
            logger.info("All files have been downloaded.")
        else:
            logger.info(
                f"Only {file_count} files could be downloaded. Please check which ones are"
                " missing and download the remaining ones manually from {url}."
            )


def _get_dict_value_by_name(
    attributes_list: list[dict[str, Any]], attribute: str
) -> str | float | None:
    val: str | float
    for item in attributes_list:
        if item["Name"] == attribute:
            val = item["Value"]
            return val
    logger.info(f"Wasn't able to find a {attribute} value. Returning None")
    return None


def _load_pkg(file: Path) -> pd.DataFrame:
    return pd.read_pickle(file)
