"""
verma_net_radiation.model
-------------------------

This module provides the main implementation for calculating instantaneous net radiation and its components
based on the methodology described in:

    Verma, M., Fisher, J. B., Mallick, K., Ryu, Y., Kobayashi, H., Guillaume, A., Moore, G., Ramakrishnan, L., 
    Hendrix, V. C., Wolf, S., Sikka, M., Kiely, G., Wohlfahrt, G., Gielen, B., Roupsard, O., Toscano, P., 
    Arain, A., & Cescatti, A. (2016). Global surface net-radiation at 5 km from MODIS Terra. 
    Remote Sensing, 8, 739. https://api.semanticscholar.org/CorpusID:1517647

The core function, `verma_net_radiation`, computes the following radiation components:
    - Outgoing shortwave radiation (SWout)
    - Incoming longwave radiation (LWin)
    - Outgoing longwave radiation (LWout)
    - Instantaneous net radiation (Rn)

Inputs can be provided as Raster objects, NumPy arrays, or scalars, and the function supports optional
cloud masking and resampling. The module relies on supporting functions for atmospheric emissivity and
longwave radiation calculations.

If certain parameters (SWin, Ta_C, RH) are not provided, and both `geometry` and `time_UTC` are given,
the function will automatically retrieve these variables from the NASA GEOS-5 FP reanalysis dataset
using the `GEOS5FP` interface. This allows for seamless integration of meteorological data when only
surface properties and spatial/temporal context are available.

Dependencies:
    - numpy
    - rasters
    - GEOS5FP
    - .constants
    - .brutsaert_atmospheric_emissivity
    - .incoming_longwave_radiation
    - .outgoing_longwave_radiation
"""

from typing import Union, Dict
import numpy as np
import warnings
from datetime import datetime
from rasters import Raster, RasterGeometry
import logging
from check_distribution import check_distribution
from GEOS5FP import GEOS5FP

from .constants import *
from .brutsaert_atmospheric_emissivity import brutsaert_atmospheric_emissivity
from .incoming_longwave_radiation import incoming_longwave_radiation
from .outgoing_longwave_radiation import outgoing_longwave_radiation
from .daylight_Rn_integration_verma import daylight_Rn_integration_verma

logger = logging.getLogger(__name__)

def verma_net_radiation(
        ST_C: Union[Raster, np.ndarray, float],
        emissivity: Union[Raster, np.ndarray, float],
        albedo: Union[Raster, np.ndarray, float],
        SWin_Wm2: Union[Raster, np.ndarray, float] = None, 
        Ta_C: Union[Raster, np.ndarray, float] = None,
        RH: Union[Raster, np.ndarray, float] = None,
        geometry: RasterGeometry = None,
        time_UTC: datetime = None,
        GEOS5FP_connection: GEOS5FP = None,
        resampling: str = RESAMPLING_METHOD,
        cloud_mask: Union[Raster, np.ndarray, float, None] = None,
    upscale_to_daylight: bool = UPSCALE_TO_DAYLIGHT,
        ) -> Dict[str, Union[Raster, np.ndarray, float]]:
    """
    Calculate instantaneous net radiation and its components.

    This function implements the net radiation and component fluxes as described in:
    Verma, M., Fisher, J. B., Mallick, K., Ryu, Y., Kobayashi, H., Guillaume, A., Moore, G., Ramakrishnan, L., Hendrix, V. C., Wolf, S., Sikka, M., Kiely, G., Wohlfahrt, G., Gielen, B., Roupsard, O., Toscano, P., Arain, A., & Cescatti, A. (2016). Global surface net-radiation at 5 km from MODIS Terra. Remote Sensing, 8, 739. https://api.semanticscholar.org/CorpusID:1517647

    If any of the parameters SWin_Wm2 (incoming shortwave radiation), Ta_C (air temperature), or RH (relative humidity) are not provided, and both `geometry` and `time_UTC` are given, the function will automatically retrieve the missing variables from the NASA GEOS-5 FP reanalysis dataset using the `GEOS5FP` interface. This enables automatic integration of meteorological data when only surface properties and spatial/temporal context are available.

    Parameters:
        ST_C (Raster, np.ndarray, float): Surface temperature in Celsius.
        emissivity (Raster, np.ndarray, float): Surface emissivity (unitless, constrained between 0 and 1).
        albedo (Raster, np.ndarray, float): Surface albedo (unitless, constrained between 0 and 1).
        SWin_Wm2 (Raster, np.ndarray, float, optional): Incoming shortwave radiation (W/m²). If not provided, will be retrieved from GEOS-5 FP if geometry and time_UTC are given.
        Ta_C (Raster, np.ndarray, float, optional): Air temperature in Celsius. If not provided, will be retrieved from GEOS-5 FP if geometry and time_UTC are given.
        RH (Raster, np.ndarray, float, optional): Relative humidity (fractional, e.g., 0.5 for 50%). If not provided, will be retrieved from GEOS-5 FP if geometry and time_UTC are given.
        geometry (RasterGeometry, optional): Spatial geometry for GEOS-5 FP retrievals.
        time_UTC (datetime, optional): UTC time for GEOS-5 FP retrievals.
        GEOS5FP_connection (GEOS5FP, optional): Existing GEOS5FP connection to use for data retrievals.
        resampling (str, optional): Resampling method for GEOS-5 FP data retrievals.
        cloud_mask (Raster, np.ndarray, float, optional): Boolean mask indicating cloudy areas (True for cloudy).

    Returns:
        Dict[str, Raster, np.ndarray, float]: A dictionary containing:
            - "SWout_Wm2": Outgoing shortwave radiation (W/m²).
            - "SWnet_Wm2": Net shortwave radiation (W/m²).
            - "LWin_Wm2": Incoming longwave radiation (W/m²).
            - "LWout_Wm2": Outgoing longwave radiation (W/m²).
            - "LWnet_Wm2": Net longwave radiation (W/m²).
            - "Rn_Wm2": Instantaneous net radiation (W/m²).
    """
    from pytictoc import TicToc
    t = TicToc()
    t.tic()
    results = {}

    logger.info("starting Verma net radiation processing")

    if geometry is None and isinstance(ST_C, Raster):
        geometry = ST_C.geometry

    raster_processing = isinstance(geometry, RasterGeometry) or isinstance(ST_C, Raster)
    spatial_temporal_processing = geometry is not None and time_UTC is not None

    # Create GEOS5FP connection if not provided
    if GEOS5FP_connection is None:
        GEOS5FP_connection = GEOS5FP()

    # Retrieve incoming shortwave if not provided
    if SWin_Wm2 is None and spatial_temporal_processing:
        SWin_Wm2 = GEOS5FP_connection.SWin(
            time_UTC=time_UTC,
            geometry=geometry,
            resampling=resampling
        )
    
    if SWin_Wm2 is None:
        raise ValueError("incoming shortwave radiation (SWin) not given")

    results["SWin_Wm2"] = SWin_Wm2

    # Retrieve air temperature if not provided, using GEOS5FP and geometry/time
    if Ta_C is None and spatial_temporal_processing:
        Ta_C = GEOS5FP_connection.Ta_C(
            time_UTC=time_UTC,
            geometry=geometry,
            resampling=resampling
        )

    if Ta_C is None:
        raise ValueError("air temperature (Ta_C) not given")
    
    # Retrieve relative humidity if not provided, using GEOS5FP and geometry/time
    if RH is None and spatial_temporal_processing:
        RH = GEOS5FP_connection.RH(
            time_UTC=time_UTC,
            geometry=geometry,
            resampling=resampling
        )

    if RH is None:
        raise ValueError("relative humidity (RH) not given")

    # Convert surface temperature from Celsius to Kelvin
    ST_K = ST_C + 273.15

    # Convert air temperature from Celsius to Kelvin
    Ta_K = Ta_C + 273.15

    # Calculate water vapor pressure in Pascals using air temperature and relative humidity
    Ea_Pa = (RH * 0.6113 * (10 ** (7.5 * (Ta_K - 273.15) / (Ta_K - 35.85)))) * 1000
    
    # Constrain albedo between 0 and 1
    albedo = np.clip(albedo, 0, 1)

    # Calculate outgoing shortwave from incoming shortwave and albedo
    SWout_Wm2 = np.clip(SWin_Wm2 * albedo, 0, None)
    check_distribution(SWout_Wm2, "SWout_Wm2")
    results["SWout_Wm2"] = SWout_Wm2

    # Calculate instantaneous net radiation from components
    SWnet_Wm2 = np.clip(SWin_Wm2 - SWout_Wm2, 0, None)
    check_distribution(SWnet_Wm2, "SWnet_Wm2")
    results["SWnet_Wm2"] = SWnet_Wm2

    # Calculate atmospheric emissivity using Brutsaert (1975) model
    atmospheric_emissivity = brutsaert_atmospheric_emissivity(Ea_Pa, Ta_K)

    # Calculate incoming longwave radiation (clear/cloudy)
    LWin_Wm2 = incoming_longwave_radiation(atmospheric_emissivity, Ta_K, cloud_mask)
    check_distribution(LWin_Wm2, "LWin_Wm2")
    results["LWin_Wm2"] = LWin_Wm2

    # Constrain emissivity between 0 and 1
    emissivity = np.clip(emissivity, 0, 1)

    # Calculate outgoing longwave from land surface temperature and emissivity
    LWout_Wm2 = outgoing_longwave_radiation(emissivity, ST_K)
    check_distribution(LWout_Wm2, "LWout_Wm2")
    results["LWout_Wm2"] = LWout_Wm2

    # Calculate net longwave radiation
    LWnet_Wm2 = LWin_Wm2 - LWout_Wm2

    # Constrain negative values of instantaneous net radiation
    Rn_Wm2 = np.clip(SWnet_Wm2 + LWnet_Wm2, 0, None)
    check_distribution(Rn_Wm2, "Rn_Wm2")
    results["Rn_Wm2"] = Rn_Wm2
 
    if upscale_to_daylight and time_UTC is not None:
        Rn_daylight_Wm2 = daylight_Rn_integration_verma(
            Rn_Wm2=Rn_Wm2,
            geometry=geometry,
            time_UTC=time_UTC
        )

        check_distribution(Rn_daylight_Wm2, "Rn_daylight_Wm2")
        results["Rn_daylight_Wm2"] = Rn_daylight_Wm2

    if raster_processing:
        for key, value in results.items():
            if not isinstance(results[key], Raster):
                results[key] = Raster(value, geometry=geometry)

    elapsed = t.tocvalue()
    logger.info(f"Verma net radiation processing complete in {elapsed:.2f} seconds")

    return results
