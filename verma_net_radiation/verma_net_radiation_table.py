"""
Verma Net Radiation Table Utilities
===================================

This module provides a function to process tabular (DataFrame) inputs for the Verma net radiation model.
It computes net radiation and its components for each row of the input DataFrame and appends the results as new columns.

Key Features:
-------------
- Accepts a pandas DataFrame with required input columns for net radiation calculation.
- Computes outgoing/incoming shortwave and longwave radiation, and net radiation.
- Returns a DataFrame with additional columns for all calculated radiation components.

Reference:
----------
Verma, M., Fisher, J. B., Mallick, K., Ryu, Y., Kobayashi, H., Guillaume, A., Moore, G., Ramakrishnan, L., Hendrix, V. C., Wolf, S., Sikka, M., Kiely, G., Wohlfahrt, G., Gielen, B., Roupsard, O., Toscano, P., Arain, A., & Cescatti, A. (2016). Global surface net-radiation at 5 km from MODIS Terra. Remote Sensing, 8, 739. https://api.semanticscholar.org/CorpusID:1517647

Example Usage:
--------------
>>> from verma_net_radiation_table import verma_net_radiation_table
>>> import pandas as pd
>>> df = pd.read_csv('inputs.csv')
>>> df_out = verma_net_radiation_table(df)
"""
import numpy as np
import pandas as pd
from pandas import DataFrame
from rasters import MultiPoint
from geopandas import GeoSeries
import geopandas as gpd
from shapely.geometry import Point

from .constants import *
from .model import verma_net_radiation

def ensure_geometry(df):
    if isinstance(df.geometry.iloc[0], str):
        # Try to parse "x, y" or "POINT (x y)" formats
        def parse_geom(s):
            s = s.strip()
            if s.startswith("POINT"):
                coords = s.replace("POINT", "").replace("(", "").replace(")", "").strip().split()
                return Point(float(coords[0]), float(coords[1]))
            elif "," in s:
                coords = [float(c) for c in s.split(",")]
                return Point(coords[0], coords[1])
            else:
                coords = [float(c) for c in s.split()]
                return Point(coords[0], coords[1])
        df = df.copy()
        df['geometry'] = df['geometry'].apply(parse_geom)
    return df

def verma_net_radiation_table(
        verma_net_radiation_inputs_df: DataFrame,
        upscale_to_daylight: bool = UPSCALE_TO_DAYLIGHT,
        offline_mode: bool = False
        ) -> DataFrame:
    """
    Process a DataFrame containing inputs for Verma net radiation calculations.

    This function takes a DataFrame with columns representing various input parameters
    required for calculating net radiation and its components. It processes the inputs,
    computes the radiation components using the `verma_net_radiation` function,
    and appends the results as new columns to the input DataFrame.

    Parameters:
        verma_net_radiation_inputs_df (DataFrame): A DataFrame containing the following columns:
            - Rg: Incoming shortwave radiation (W/m²).
            - albedo: Surface albedo (unitless, constrained between 0 and 1).
            - ST_C: Surface temperature in Celsius.
            - emissivity: Surface emissivity (unitless, constrained between 0 and 1).
            - Ta_C: Air temperature in Celsius.
            - RH: Relative humidity (fractional, e.g., 0.5 for 50%).

    Returns:
        DataFrame: A copy of the input DataFrame with additional columns for the calculated
        radiation components:
            - SWout: Outgoing shortwave radiation (W/m²).
            - LWin: Incoming longwave radiation (W/m²).
            - LWout: Outgoing longwave radiation (W/m²).
            - Rn: Instantaneous net radiation (W/m²).
    """
    if "time_UTC" not in verma_net_radiation_inputs_df:
        raise ValueError("missing required column: time_UTC")
    
    time_UTC = pd.to_datetime(verma_net_radiation_inputs_df.time_UTC).tolist()
    
    if "geometry" not in verma_net_radiation_inputs_df:
        raise ValueError("missing required column: geometry")
    
    verma_net_radiation_inputs_df = ensure_geometry(verma_net_radiation_inputs_df)
    geometry = MultiPoint(verma_net_radiation_inputs_df.geometry)

    if "SWin_Wm2" not in verma_net_radiation_inputs_df:
        raise ValueError("missing required column: SWin_Wm2")
    
    SWin_Wm2 = np.array(verma_net_radiation_inputs_df.SWin_Wm2)

    if "albedo" not in verma_net_radiation_inputs_df:
        raise ValueError("missing required column: albedo")

    albedo = np.array(verma_net_radiation_inputs_df.albedo)

    if "ST_C" not in verma_net_radiation_inputs_df:
        raise ValueError("missing required column: ST_C")

    ST_C = np.array(verma_net_radiation_inputs_df.ST_C)

    if "emissivity" not in verma_net_radiation_inputs_df:
        raise ValueError("missing required column: emissivity")

    emissivity = np.array(verma_net_radiation_inputs_df.emissivity)

    if "Ta_C" not in verma_net_radiation_inputs_df:
        raise ValueError("missing required column: Ta_C")

    Ta_C = np.array(verma_net_radiation_inputs_df.Ta_C)

    if "RH" not in verma_net_radiation_inputs_df:
        raise ValueError("missing required column: RH")

    RH = np.array(verma_net_radiation_inputs_df.RH)

    results = verma_net_radiation(
        SWin_Wm2=SWin_Wm2,
        albedo=albedo,
        ST_C=ST_C,
        emissivity=emissivity,
        Ta_C=Ta_C,
        RH=RH,
        time_UTC=time_UTC,
        geometry=geometry,
        upscale_to_daylight=upscale_to_daylight,
        offline_mode=offline_mode
    )

    verma_net_radiation_outputs_df = verma_net_radiation_inputs_df.copy()
    for key, value in results.items():
        verma_net_radiation_outputs_df[key] = value

    return verma_net_radiation_outputs_df
