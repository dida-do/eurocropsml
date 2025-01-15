"""Utilities for pre-processing Sentinel-1 data.

esa_snappy code taken from https://github.com/wajuqi/Sentinel-1-preprocessing-using-Snappy.
Slightly adjusted by Ekaterina Gikalo, Joana Reuss (TUM), 2025.

"""

import logging
from typing import cast

import jpy
import numpy as np
from esa_snappy import GPF, HashMap, Product

logger = logging.getLogger(__name__)


def apply_orbit_file(prdct: Product) -> Product:
    """Apply orbit file correction to the source data.

    Args:
        prdct: The input product to which the orbit file correction will be applied.

    Returns:
        Product: The product with the orbit file correction applied.
    """
    logger.info("Apply orbit file...")
    parameters = HashMap()
    parameters.put("Apply-Orbit-File", True)
    output = GPF.createProduct("Apply-Orbit-File", parameters, prdct)
    return output


def thermal_noise_removal(prdct: Product) -> Product:
    """Perform thermal noise removal on the source data.

    Args:
        prdct: The input product on which thermal noise removal will be performed.

    Returns:
        The product with thermal noise removed.
    """
    logger.info("Remove thermal noise...")
    parameters = HashMap()
    parameters.put("removeThermalNoise", True)
    output = GPF.createProduct("ThermalNoiseRemoval", parameters, prdct)
    return output


def calibration(prdct: Product, pols: str) -> Product:
    """Perform radiometric calibration on the source data.

    Args:
        prdct: The input product to calibrate.
        pols: The selected polarizations.

    Returns:
        Product converted to backscatter values.
    """
    logger.info("Apply radiometric calibration...")
    parameters = HashMap()
    parameters.put("outputSigmaBand", True)  # output sigma0
    parameters.put("sourceBands", f"Intensity_{pols.split(',')[0]},Intensity_{pols.split(',')[1]}")
    parameters.put("selectedPolarisations", pols)
    parameters.put("outputImageScaleInDb", False)
    output = GPF.createProduct("Calibration", parameters, prdct)
    return output


def convert_to_db(
    sigma_nought: np.ndarray, min_db: float | None = -50.0, max_db: float | None = 0.0
) -> np.ndarray:
    """Convert Sigma Nought linear values into decibels.

    Args:
        sigma_nought: Sigma nought linear values.
        min_db: Minimum value to clip db values to.
        max_db: Maximum value to clip db values to.

    Returns:
        Sigma Nought values in decibel.
    """
    logger.info("Convert to decibel...")
    sigma_nought_db = 10.0 * np.log10(np.maximum(sigma_nought, 1e-10))  # prevent division by 0
    if min_db is not None:
        sigma_nought_db = np.maximum(sigma_nought_db, min_db)
    if max_db is not None:
        sigma_nought_db = np.minimum(sigma_nought_db, max_db)

    return cast(np.ndarray, sigma_nought_db)


def speckle_filtering(prdct: Product) -> Product:
    """Perform speckle filtering on the source data.

    Args:
        prdct: The input product on which speckle filtering will be performed.

    Returns:
        Product: The product with speckle filtering applied.
    """
    logger.info("Speckle filtering...")
    java_integer = jpy.get_type("java.lang.Integer")
    parameters = HashMap()
    parameters.put("filter", "Lee")
    parameters.put("filterSizeX", java_integer(5))
    parameters.put("filterSizeY", java_integer(5))
    output = GPF.createProduct("Speckle-Filter", parameters, prdct)
    return output


def terrain_correction(
    prdct: Product,
    dem_name: str = "SRTM 1Sec HGT",
    pixel_spacing: float | None = None,
    proj: str | None = None,
) -> Product:
    """
    Perform terrain correction (orthorectification) on the source data.

    Args:
        prdct: The input product to process.
        dem_name: Name of the DEM to use for terrain correction.
        pixel_spacing: Pixel spacing used for (optional) spatial downsamplng.
        proj: The projection system (e.g., UTM or WGS84).

    Returns:
        Product: The product with terrain correction applied.
    """
    logger.info("Apply terrain correction...")
    parameters = HashMap()
    parameters.put("demName", dem_name)
    parameters.put("imgResamplingMethod", "BILINEAR_INTERPOLATION")
    # if need to convert to UTM/WGS84, default is WGS84
    if proj is not None:
        parameters.put("mapProjection", proj)
    parameters.put("saveProjectedLocalIncidenceAngle", True)
    parameters.put("saveSelectedSourceBand", True)
    if pixel_spacing is not None:
        parameters.put("pixelSpacingInMeter", pixel_spacing)
    output = GPF.createProduct("Terrain-Correction", parameters, prdct)
    return output


def subset(prdct: Product, wkt: str) -> Product:
    """Perform a subset operation on the source data based on a geographic region.

    Args:
        prdct: The input product to subset.
        wkt: The Well-Known Text (WKT) string defining the geographic region.

    Returns:
        Product: The subset product.
    """
    parameters = HashMap()
    parameters.put("geoRegion", wkt)
    output = GPF.createProduct("Subset", parameters, prdct)
    return output
