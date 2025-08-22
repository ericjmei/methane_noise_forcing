# src/methane_noise_forcing/utils.py
# -*- coding: utf-8 -*-
"""
Functionality for miscellaneous functions.
"""
import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s")
logger = logging.getLogger(__name__)

def detrend_obs(site_name, data: pd.DataFrame):
    """
    Detrend observed data. "wdc06a" uses a 2nd order polynomial, all others use 1st order.

    Parameters
    ----------
    site_name : str
        The name of the site.
    data : pd.DataFrame
        The observed data to detrend.

    Returns
    -------
    pd.DataFrame
        The detrended data.
    """
    if site_name == "wdc06a":
        order = 2
    else:
        order = 1
    logger.info(f"Detrending data with polynomial of order {order}")
    
    coeffs = np.polyfit(data["gas_age"], data["ch4"], order)

    data_detrended = data.copy()
    data_detrended["ch4"] -= np.polyval(coeffs, data_detrended["gas_age"])

    return data_detrended