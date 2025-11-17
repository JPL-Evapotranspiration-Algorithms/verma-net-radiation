"""
Daylight Net Radiation Integration (Verma et al., 2016)
====================================================

This module provides a function to integrate instantaneous net radiation to daylight values using solar geometry parameters.
It is based on the methodology described in Verma et al. (2016) for global surface net radiation estimation from MODIS Terra data.

Key Features:
-------------
- Integrates instantaneous net radiation (Rn) to daylight values using hour of day, latitude, and solar angles.
- Accepts Raster, numpy array, or float inputs for geospatial and scientific workflows.
- Handles calculation of daylight hours and sunrise time if not provided.

Reference:
----------
Verma, M., Fisher, J. B., Mallick, K., Ryu, Y., Kobayashi, H., Guillaume, A., Moore, G., Ramakrishnan, L., Hendrix, V. C., Wolf, S., Sikka, M., Kiely, G., Wohlfahrt, G., Gielen, B., Roupsard, O., Toscano, P., Arain, A., & Cescatti, A. (2016). Global surface net-radiation at 5 km from MODIS Terra. Remote Sensing, 8, 739. https://api.semanticscholar.org/CorpusID:1517647

Example Usage:
--------------
>>> from daylight_Rn_integration_verma import daylight_Rn_integration_verma
>>> Rn_daylight = daylight_Rn_integration_verma(Rn=400, hour_of_day=12, doy=180, lat=35)
"""

from datetime import datetime
from dateutil import parser
from typing import Union
import warnings
from geopandas import GeoSeries
import numpy as np

from rasters import Raster
from rasters import SpatialGeometry

from solar_apparent_time import calculate_solar_day_of_year, calculate_solar_hour_of_day
from sun_angles import daylight_from_SHA, sunrise_from_SHA, SHA_deg_from_DOY_lat

import logging

logger = logging.getLogger(__name__)

def daylight_Rn_integration_verma(
        Rn_Wm2: Union[Raster, np.ndarray, float],
        time_UTC: Union[datetime, str, list, np.ndarray] = None,
        geometry: Union[SpatialGeometry, GeoSeries] = None,
        hour_of_day: Union[Raster, np.ndarray, float] = None,
        day_of_year: Union[Raster, np.ndarray, int] = None,
        lat: Union[Raster, np.ndarray, float] = None,
        lon: Union[Raster, np.ndarray, float] = None,
        sunrise_hour: Union[Raster, np.ndarray, float] = None,
        daylight_hours: Union[Raster, np.ndarray, float] = None
        ) -> Union[Raster, np.ndarray, float]:
    """
    Integrate instantaneous net radiation (Rn) to daylight average values using solar geometry parameters.

    This function estimates the daylight average net radiation (W/m²) from instantaneous measurements, accounting for solar position and daylight duration. It supports Raster, numpy array, or float inputs for geospatial and scientific workflows. If sunrise time or daylight hours are not provided, they are calculated from day of year and latitude.

    Parameters:
        Rn_Wm2 (Union[Raster, np.ndarray, float]): Instantaneous net radiation (W/m²).
        hour_of_day (Union[Raster, np.ndarray, float]): Hour of the day (0-24) when Rn is measured.
        day_of_year (Union[Raster, np.ndarray, float], optional): Day of the year (1-365).
        lat (Union[Raster, np.ndarray, float], optional): Latitude in degrees.
        sunrise_hour (Union[Raster, np.ndarray, float], optional): Hour of sunrise (local time).
        daylight_hours (Union[Raster, np.ndarray, float], optional): Total daylight hours.

    Returns:
        Union[Raster, np.ndarray, float]: Daylight average net radiation (W/m²).

    Notes:
        - To obtain total daylight energy (J/m²), multiply the result by (daylight_hours * 3600).
        - If sunrise_hour or daylight_hours are not provided, they are computed from day_of_year and latitude using solar geometry.

    Reference:
        Verma, M., Fisher, J. B., Mallick, K., Ryu, Y., Kobayashi, H., Guillaume, A., Moore, G., Ramakrishnan, L., Hendrix, V. C., Wolf, S., Sikka, M., Kiely, G., Wohlfahrt, G., Gielen, B., Roupsard, O., Toscano, P., Arain, A., & Cescatti, A. (2016). Global surface net-radiation at 5 km from MODIS Terra. Remote Sensing, 8, 739. https://api.semanticscholar.org/CorpusID:1517647
    """
    if Rn_Wm2 is None:
        raise ValueError("Instantaneous net radiation (Rn) must be provided.")

    # Handle string or list of strings for time_UTC
    if isinstance(time_UTC, str):
        time_UTC = parser.parse(time_UTC)
    elif isinstance(time_UTC, list):
        time_UTC = [parser.parse(t) if isinstance(t, str) else t for t in time_UTC]
    elif isinstance(time_UTC, np.ndarray) and time_UTC.dtype.type is np.str_:
        time_UTC = np.array([parser.parse(t) for t in time_UTC])

    # If latitude is not provided, try to extract from geometry
    if lat is None and isinstance(geometry, SpatialGeometry):
        lat = geometry.lat
    elif lat is None and isinstance(geometry, GeoSeries):
        lat = geometry.y

    if lon is None and isinstance(geometry, SpatialGeometry):
        lon = geometry.lon
    elif lon is None and isinstance(geometry, GeoSeries):
        lon = geometry.x

    # Handle day_of_year input: convert lists to np.ndarray
    if day_of_year is not None:
        if isinstance(day_of_year, list):
            day_of_year = np.array(day_of_year)

    # Handle lat input: convert lists to np.ndarray
    if lat is not None:
        if isinstance(lat, list):
            lat = np.array(lat)

    # If day_of_year is not provided, try to infer from time_UTC
    if day_of_year is None and time_UTC is not None:
        logger.info("calculating solar day of year")
        day_of_year = calculate_solar_day_of_year(
            time_UTC=time_UTC,
            geometry=geometry,
            lat=lat,
            lon=lon
        )    

    if hour_of_day is None and time_UTC is not None:
        logger.info("calculating solar hour of day")
        hour_of_day = calculate_solar_hour_of_day(
            time_UTC=time_UTC,
            geometry=geometry,
            lat=lat,
            lon=lon
        )

    if daylight_hours is None or sunrise_hour is None and day_of_year is not None and lat is not None:
        logger.info("calculating daylight hours and sunrise hour")
        sha_deg = SHA_deg_from_DOY_lat(day_of_year, lat)
        daylight_hours = daylight_from_SHA(sha_deg)
        sunrise_hour = sunrise_from_SHA(sha_deg)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        Rn_daylight = 1.6 * Rn_Wm2 / (np.pi * np.sin(np.pi * (hour_of_day - sunrise_hour) / (daylight_hours)))
    
    return Rn_daylight

# For backward compatibility, but prefer using daylight_Rn_integration_verma
daylight_Rn_integration_verma = daylight_Rn_integration_verma