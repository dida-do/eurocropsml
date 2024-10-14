"""Utilities for applying radiometric calibration to S-1 data."""

from pathlib import Path
from typing import Any, cast
from xml.etree import ElementTree

import numpy as np
import xarray as xr
import xmlschema


def _get_xml_file(filepath: Path, band: str, identifier: str = "calibration") -> Path:
    files: list[Path] = list(filepath.iterdir())
    return [file for file in files if f"{band.lower()}" in str(file) and identifier in str(file)][0]


def _parse_tag_as_list(
    xml_path: Path,
    query: str,
    schema_dir: Path,
    validation: str = "skip",
) -> list[dict[str, Any]]:
    """Function to parse xml tags into list.

    Adjusted from xarray-sentinel (https://github.com/bopen/xarray-sentinel).
    """
    schema = xmlschema.XMLSchema(schema_dir)
    xml_tree = ElementTree.parse(xml_path)
    tag: Any = schema.decode(xml_tree, query, validation=validation)
    if tag is None:
        tag = []
    elif isinstance(tag, dict):
        tag = [tag]
    tag_list: list[dict[str, Any]] = tag
    assert isinstance(tag_list, list), f"{type(tag_list)} is not list"
    return tag_list


def _open_noise_dataset(safe_dir: Path, band: str) -> xr.Dataset:
    """Function to read noise from LUT.

    This reads the NADS (Noise Annotation Data Set) file.

    Adjusted from xarray-sentinel (https://github.com/bopen/xarray-sentinel).
    """
    xml_dir: Path = safe_dir / "annotation" / "calibration"
    schema_dir: Path = safe_dir / "support" / "s1-level-1-noise.xsd"
    xml_calibration_file = _get_xml_file(xml_dir, band, identifier="noise")
    noise_vectors = _parse_tag_as_list(xml_calibration_file, ".//noiseRangeVector", schema_dir)

    pixel_list = []
    line_list = []
    noise_range_lut_list = []
    for vector in noise_vectors:
        line_list.append(vector["line"])
        pixel = np.fromstring(vector["pixel"]["$"], dtype=int, sep=" ")
        pixel_list.append(pixel)
        noise_range_lut = np.fromstring(vector["noiseRangeLut"]["$"], dtype=np.float32, sep=" ")
        noise_range_lut_list.append(noise_range_lut)

    pixel = np.array(pixel_list)
    if (pixel - pixel[0]).any():
        raise ValueError("Unable to organize noise vectors in a regular line-pixel grid")
    data_vars = {
        "noiseRangeLut": (("line", "pixel"), noise_range_lut_list),
    }
    coords = {"line": line_list, "pixel": pixel_list[0]}

    return xr.Dataset(data_vars=data_vars, coords=coords)


def _open_calibration_dataset(safe_dir: Path, band: str) -> xr.Dataset:
    """Function to read calibration from LUT.

    This reads the CADS (Calibration Annotation Data Set) file.

    Adjusted from xarray-sentinel (https://github.com/bopen/xarray-sentinel).
    """
    xml_dir: Path = safe_dir / "annotation" / "calibration"
    schema_dir: Path = safe_dir / "support" / "s1-level-1-calibration.xsd"
    xml_calibration_file = _get_xml_file(xml_dir, band)
    calibration_vectors = _parse_tag_as_list(
        xml_calibration_file, ".//calibrationVector", schema_dir
    )

    pixel_list = []
    line_list = []
    sigmanought_list = []
    for vector in calibration_vectors:
        line_list.append(vector["line"])
        pixel = np.fromstring(vector["pixel"]["$"], dtype=int, sep=" ")
        pixel_list.append(pixel)
        sigma_nought = np.fromstring(vector["sigmaNought"]["$"], dtype=np.float32, sep=" ")
        sigmanought_list.append(sigma_nought)

    pixel = np.array(pixel_list)
    if (pixel - pixel[0]).any():
        raise ValueError("Unable to organise calibration vectors in a regular line-pixel grid")
    data_vars = {
        "sigmaNought": (("line", "pixel"), sigmanought_list),
    }
    coords = {"line": line_list, "pixel": pixel_list[0]}

    return xr.Dataset(data_vars=data_vars, coords=coords)


def _get_lut_value(
    digital_number: xr.DataArray, available_lut: xr.DataArray, **kwargs: Any
) -> xr.DataArray:
    lut_mean = available_lut.mean()
    if np.allclose(lut_mean, available_lut, **kwargs):
        lut: xr.DataArray = lut_mean.astype(np.float32)
    else:
        lut = available_lut.interp(
            line=digital_number.line,
            pixel=digital_number.pixel,
        ).astype(np.float32)
        if digital_number.chunks is not None:
            lut = lut.chunk(digital_number.chunksizes)

    return lut


def _calibrate(
    digital_number: xr.DataArray,
    backscatter_calibration_lut: xr.DataArray,
    thermal_noise_lut: xr.DataArray | None = None,
    **kwargs: Any,
) -> xr.DataArray:
    """Return the calibrated sigma nought (backscatter coefficient) using the calibration LUT.

    Apply thermal noise removal if wanted.
    Adjusted from xarray-sentinel (https://github.com/bopen/xarray-sentinel).

    digital_number: Digital numbers from the original raster tile to be calibrated.
    backscatter_calibration_lut: Calibration LUT for backscatter (sigma nought).
    thermal_noise_lut: Thermal noise LUT to remove sensor noise.
    """
    radar_intensity = digital_number**2
    backscatter_calibration = _get_lut_value(digital_number, backscatter_calibration_lut, **kwargs)
    if thermal_noise_lut is not None:
        thermal_noise = _get_lut_value(digital_number, thermal_noise_lut, **kwargs)
        radar_intensity = radar_intensity - thermal_noise
    sigma_nought: xr.DataArray = radar_intensity / backscatter_calibration**2
    return abs(sigma_nought)


def _calibrate_digital_number_in_db(
    digital_number: xr.DataArray,
    backscatter_calibration_lut: xr.DataArray,
    thermal_noise_lut: xr.DataArray | None = None,
    min_db: float | None = -50.0,
) -> np.ndarray:
    """Return calibrated sigma nought (backscatter coefficient) in dB using the calibration LUT.

    Adjusted from xarray-sentinel (https://github.com/bopen/xarray-sentinel).

    digital_number: Digital numbers from the original raster tile to be calibrated.
    backscatter_calibration_lut: calibration LUT (sigmaNought).
    thermal_noise_lut: Thermal noise LUT.
    min_db: minimal value in db, to avoid infinity values.
    """
    sigma_nought = _calibrate(digital_number, backscatter_calibration_lut, thermal_noise_lut)
    # convert to decibels (dB)
    sigma_nought_db = 10.0 * np.log10(np.maximum(sigma_nought, 1e-10))  # prevent division by 0

    if min_db is not None:
        sigma_nought_db = cast(xr.DataArray, np.maximum(sigma_nought_db, min_db))

    return cast(np.ndarray, sigma_nought_db.values)
