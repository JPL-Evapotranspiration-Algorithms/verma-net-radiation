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

from GEOS5FP import GEOS5FP

from .constants import *
from .brutsaert_atmospheric_emissivity import brutsaert_atmospheric_emissivity
from .incoming_longwave_radiation import incoming_longwave_radiation
from .outgoing_longwave_radiation import outgoing_longwave_radiation

def verma_net_radiation(
        ST_C: Union[Raster, np.ndarray, float],
        emissivity: Union[Raster, np.ndarray, float],
        albedo: Union[Raster, np.ndarray, float],
        SWin: Union[Raster, np.ndarray, float] = None, 
        Ta_C: Union[Raster, np.ndarray, float] = None,
        RH: Union[Raster, np.ndarray, float] = None,
        geometry: RasterGeometry = None,
        time_UTC: datetime = None,
        GEOS5FP_connection: GEOS5FP = None,
        resampling: str = RESAMPLING_METHOD,
        cloud_mask: Union[Raster, np.ndarray, float, None] = None
        ) -> Dict[str, Union[Raster, np.ndarray, float]]:
    """
    Calculate instantaneous net radiation and its components.

    This function implements the net radiation and component fluxes as described in:
    Verma, M., Fisher, J. B., Mallick, K., Ryu, Y., Kobayashi, H., Guillaume, A., Moore, G., Ramakrishnan, L., Hendrix, V. C., Wolf, S., Sikka, M., Kiely, G., Wohlfahrt, G., Gielen, B., Roupsard, O., Toscano, P., Arain, A., & Cescatti, A. (2016). Global surface net-radiation at 5 km from MODIS Terra. Remote Sensing, 8, 739. https://api.semanticscholar.org/CorpusID:1517647

    If any of the parameters SWin (incoming shortwave radiation), Ta_C (air temperature), or RH (relative humidity) are not provided, and both `geometry` and `time_UTC` are given, the function will automatically retrieve the missing variables from the NASA GEOS-5 FP reanalysis dataset using the `GEOS5FP` interface. This enables automatic integration of meteorological data when only surface properties and spatial/temporal context are available.


    Parameters:
        ST_C (np.ndarray): Surface temperature in Celsius.
        emissivity (np.ndarray): Surface emissivity (unitless, constrained between 0 and 1).
        albedo (np.ndarray): Surface albedo (unitless, constrained between 0 and 1).
        SWin (np.ndarray, optional): Incoming shortwave radiation (W/m²). If not provided, will be retrieved from GEOS-5 FP if geometry and time_UTC are given.
        Ta_C (np.ndarray, optional): Air temperature in Celsius. If not provided, will be retrieved from GEOS-5 FP if geometry and time_UTC are given.
        RH (np.ndarray, optional): Relative humidity (fractional, e.g., 0.5 for 50%). If not provided, will be retrieved from GEOS-5 FP if geometry and time_UTC are given.
        geometry (RasterGeometry, optional): Spatial geometry for GEOS-5 FP retrievals.
        time_UTC (datetime, optional): UTC time for GEOS-5 FP retrievals.
        GEOS5FP_connection (GEOS5FP, optional): Existing GEOS5FP connection to use for data retrievals.
        resampling (str, optional): Resampling method for GEOS-5 FP data retrievals.
        cloud_mask (np.ndarray, optional): Boolean mask indicating cloudy areas (True for cloudy).

    Returns:
        Dict: A dictionary containing:
            - "SWout": Outgoing shortwave radiation (W/m²).
            - "LWin": Incoming longwave radiation (W/m²).
            - "LWout": Outgoing longwave radiation (W/m²).
            - "Rn": Instantaneous net radiation (W/m²).
    """
    results = {}

    if geometry is None and isinstance(ST_C, Raster):
        geometry = ST_C.geometry

    raster_processing = isinstance(geometry, RasterGeometry) or isinstance(ST_C, Raster)
    spatial_temporal_processing = geometry is not None and time_UTC is not None
    # print("spatial_temporal_processing:", spatial_temporal_processing)

    # Create GEOS5FP connection if not provided
    if GEOS5FP_connection is None:
        GEOS5FP_connection = GEOS5FP()

    # Retrieve incoming shortwave if not provided
    if SWin is None and spatial_temporal_processing:
        # print("Retrieving incoming shortwave radiation (SWin) from GEOS-5 FP...")
        SWin = GEOS5FP_connection.SWin(
            time_UTC=time_UTC,
            geometry=geometry,
            resampling=resampling
        )
    
    if SWin is None:
        raise ValueError("incoming shortwave radiation (SWin) not given")

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
    SWout = np.clip(SWin * albedo, 0, None)
    results["SWout"] = SWout

    # Calculate instantaneous net radiation from components
    SWnet = np.clip(SWin - SWout, 0, None)

    # Calculate atmospheric emissivity using Brutsaert (1975) model
    atmospheric_emissivity = brutsaert_atmospheric_emissivity(Ea_Pa, Ta_K)

    # Calculate incoming longwave radiation (clear/cloudy)
    LWin = incoming_longwave_radiation(atmospheric_emissivity, Ta_K, cloud_mask)
    
    results["LWin"] = LWin

    # Constrain emissivity between 0 and 1
    emissivity = np.clip(emissivity, 0, 1)

    # Calculate outgoing longwave from land surface temperature and emissivity
    LWout = outgoing_longwave_radiation(emissivity, ST_K)
    results["LWout"] = LWout

    # Calculate net longwave radiation
    LWnet = LWin - LWout

    # Constrain negative values of instantaneous net radiation
    Rn = np.clip(SWnet + LWnet, 0, None)
    results["Rn"] = Rn

    if raster_processing:
        for key, value in results.items():
            if not isinstance(results[key], Raster):
                results[key] = Raster(value, geometry=geometry)

    return results
