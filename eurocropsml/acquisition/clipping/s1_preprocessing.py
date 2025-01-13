"""Code taken from https://github.com/wajuqi/Sentinel-1-preprocessing-using-Snappy."""

from typing import Optional

import jpy
from esa_snappy import GPF, HashMap, Product


def do_apply_orbit_file(source: Product) -> Product:
    """Apply orbit file correction to the source data.

    Args:
        source (Product): The input product to which the orbit file correction will be applied.

    Returns:
        Product: The product with the orbit file correction applied.
    """
    print("\tApply orbit file...")
    parameters = HashMap()
    parameters.put("Apply-Orbit-File", True)
    output = GPF.createProduct("Apply-Orbit-File", parameters, source)
    return output


def do_thermal_noise_removal(source: Product) -> Product:
    """Perform thermal noise removal on the source data.

    Args:
        source (Product): The input product on which thermal noise removal will be performed.

    Returns:
        Product: The product with thermal noise removed.
    """
    print("\tThermal noise removal...")
    parameters = HashMap()
    parameters.put("removeThermalNoise", True)
    output = GPF.createProduct("ThermalNoiseRemoval", parameters, source)
    return output


def do_calibration(source: Product, polarization: str, pols: str) -> Product:
    """Perform radiometric calibration on the source data.

    Args:
        source (Product): The input product to calibrate.
        polarization (str): The type of polarization (e.g., "DH", "DV", "SH").
        pols (str): The selected polarizations.

    Returns:
        Product: The calibrated product.
    """
    print("\tCalibration...")
    parameters = HashMap()
    parameters.put("outputSigmaBand", True)
    if polarization == "DH":
        parameters.put("sourceBands", "Intensity_HH,Intensity_HV")
    elif polarization == "DV":
        parameters.put("sourceBands", "Intensity_VH,Intensity_VV")
    elif polarization == "SH" or polarization == "HH":
        parameters.put("sourceBands", "Intensity_HH")
    elif polarization == "SV":
        parameters.put("sourceBands", "Intensity_VV")
    else:
        print("different polarization!")
    parameters.put("selectedPolarisations", pols)
    parameters.put("outputImageScaleInDb", False)
    output = GPF.createProduct("Calibration", parameters, source)
    return output


def do_speckle_filtering(source: Product) -> Product:
    """Perform speckle filtering on the source data.

    Args:
        source (Product): The input product on which speckle filtering will be performed.

    Returns:
        Product: The product with speckle filtering applied.
    """
    print("\tSpeckle filtering...")
    java_integer = jpy.get_type("java.lang.Integer")
    parameters = HashMap()
    parameters.put("filter", "Lee")
    parameters.put("filterSizeX", java_integer(5))
    parameters.put("filterSizeY", java_integer(5))
    output = GPF.createProduct("Speckle-Filter", parameters, source)
    return output


def do_terrain_correction(source: Product, downsample: int, proj: Optional[str] = None) -> Product:
    """
    Perform terrain correction on the source data.

    Args:
        source (Product): The input product to process.
        downsample (int): Whether to downsample (1 for yes, 0 for no).
        proj (Optional[str]): The projection system (e.g., UTM or WGS84).

    Returns:
        Product: The product with terrain correction applied.
    """
    print("\tTerrain correction...")
    parameters = HashMap()
    parameters.put("demName", "GETASSE30")
    parameters.put("imgResamplingMethod", "BILINEAR_INTERPOLATION")
    # comment this line if no need to convert to UTM/WGS84, default is WGS84
    # parameters.put('mapProjection', proj)
    parameters.put("saveProjectedLocalIncidenceAngle", True)
    parameters.put("saveSelectedSourceBand", True)
    while downsample == 1:  # downsample: 1 -- need downsample to 40m, 0 -- no need to downsample
        parameters.put("pixelSpacingInMeter", 40.0)
        break
    output = GPF.createProduct("Terrain-Correction", parameters, source)
    return output


def do_subset(source: Product, wkt: str) -> Product:
    """Perform a subset operation on the source data based on a geographic region.

    Args:
        source (Product): The input product to subset.
        wkt (str): The Well-Known Text (WKT) string defining the geographic region.

    Returns:
        Product: The subset product.
    """
    parameters = HashMap()
    parameters.put("geoRegion", wkt)
    output = GPF.createProduct("Subset", parameters, source)
    return output
