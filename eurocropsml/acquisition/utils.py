"""Utilities for the dataset acquisition."""

import logging
import sys
import time
from pathlib import Path
from typing import Any

import bs4
import pandas as pd
import requests
import typer
from selenium import webdriver
from selenium.common.exceptions import WebDriverException
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as ec
from selenium.webdriver.support.ui import WebDriverWait
from webdriver_manager.chrome import ChromeDriverManager

logger = logging.getLogger(__name__)

pd.options.mode.chained_assignment = None


def _get_options_from_field(url: str, field_id: str) -> bs4.element.ResultSet:
    response = requests.get(url)
    response.raise_for_status()
    soup = bs4.BeautifulSoup(response.content, "html.parser")
    select_element = soup.find("select", {"id": field_id})
    if not isinstance(select_element, bs4.element.Tag):
        raise ValueError(
            f"Could not access field options for field id {field_id} when downloading NUTS region "
            "files. Please download them manually."
        )

    options: bs4.element.ResultSet = select_element.find_all("option")
    return options


def _get_closest_year(year_options: list[int], year: int) -> str:
    selected_year: int
    for number in year_options:
        if number == year:
            return str(year)
        elif number < year:
            selected_year = number

    return str(selected_year)


def _select_year_from_url(url: str, year: int) -> str:
    """Get available years and select the one closest to the acquisition year."""

    try:
        options: bs4.element.ResultSet = _get_options_from_field(url, "year")

        year_options: list[int] = sorted(
            [int(option.get("value")) for option in options if len(option.get("value")) > 0]
        )

        return _get_closest_year(year_options, year)

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


def _nuts_region_downloader(
    url: str, download_dir: Path, crs: str, year: int, files: list[str]
) -> None:

    timeout: int = 120  # Timeout after 120 seconds
    polling_interval: int = 5  # Check every 5 second
    file_count = 0

    # Set Chrome options for headless mode
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")

    chrome_prefs = {
        "profile.default_content_settings.popups": 0,
        "download.prompt_for_download": False,
        "download.default_directory": str(download_dir),
        "directory_upgrade": True,
        "safebrowsing.enabled": True,
    }

    options.add_experimental_option("prefs", chrome_prefs)

    # Setup ChromeDriver service using webdriver_manager
    service = Service(ChromeDriverManager().install())

    selected_year: str = _select_year_from_url(url, year)
    projections: list[str] = _get_proj_options(url)

    try:
        driver = webdriver.Chrome(service=service, options=options)
        driver.get(url)
    except WebDriverException:
        _manual_download(url, [])

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
                files.append(filename)

            else:
                file_count += 1

    finally:
        driver.quit()
        if file_count == len(projections):
            logger.info("All files have been downloaded.")
        else:
            _manual_download(url, files)


def _manual_download(url: str, files: list[str]) -> None:
    if len(files) == 0:
        logger.warning(
            "No NUTS files could be downloaded. Please download them manually from "
            f"{url}. The folder structure should look like this:\n"
            "path/to/data/directory\n"
            "└── meta_data/\n"
            "    ├── NUTS/\n"
            "    │   ├── NUTS_RG_01M_2021_3035.geojson\n"
            "    │   ├── NUTS_RG_01M_2021_3857.geojson\n"
            "    │   └── NUTS_RG_01M_2021_4326.geojson\n"
            "    └── ...\n"
        )
        sys.exit()
    else:
        manual_download = typer.confirm(
            f"Only {', '.join(files)} could be downloaded. Do you want to download the missing "
            f"ones manually from {url}? This will exit the script. If manually downloaded, "
            "the folder sturcture should look like this:\n"
            "path/to/data/directory\n"
            "└── meta_data/\n"
            "    ├── NUTS/\n"
            "    │   ├── NUTS_RG_01M_2021_3035.geojson\n"
            "    │   ├── NUTS_RG_01M_2021_3857.geojson\n"
            "    │   └── NUTS_RG_01M_2021_4326.geojson\n"
            "    └── ...\n"
        )
        if manual_download:
            sys.exit()
        else:
            return


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
