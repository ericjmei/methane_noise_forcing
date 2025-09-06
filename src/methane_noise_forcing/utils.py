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

def calculate_mean_and_ci(da, ci=0.95):
    """
    Calculate the mean and confidence interval of a DataArray.

    Parameters
    ----------
    da : xr.DataArray
        Input data array.
    ci : float, optional
        Confidence interval (default is 0.95).

    Returns
    -------
    mean : float
        Mean of the data array.
    ci_low : float
        Lower bound of the confidence interval.
    ci_high : float
        Upper bound of the confidence interval.
    """
    mean = da.mean(dim='ensemble')
    percentile_low = (1 - ci) / 2
    percentile_high = (1 + ci) / 2
    ci_low = da.quantile(percentile_low, dim='ensemble')
    ci_high = da.quantile(percentile_high, dim='ensemble')
    return mean, ci_low, ci_high

def psd_one_sided_cyclic(x, dt):
    """
    One-sided PSD S_f^{(1)}(f) so that var(x) ≈ ∫_0^{f_N} S_f^{(1)}(f) df.

    Parameters
    ----------
    x : array-like
        Input time series data.
    dt : float
        Time step of the input data.

    Returns
    -------
    f : np.ndarray
        Frequencies corresponding to the PSD values.
    S1 : np.ndarray
        One-sided power spectral density values.

    Notes
    -----
    Units: [x]^2 / (1/yr). If x has units ppb/yr, PSD has ppb^2/yr.
    Preserves variance of x in the integral of the PSD over positive frequencies.
    """
    x = np.asarray(x, float) - np.mean(x)
    N = x.size
    X = np.fft.rfft(x)                       # nonnegative freqs
    f = np.fft.rfftfreq(N, d=dt)             # cycles/yr
    S1 = (2.0 * dt / N) * (X.conj() * X).real
    S1[0] *= 0.5           # don’t double DC
    if N % 2 == 0:         # don’t double Nyquist if present
        S1[-1] *= 0.5
    return f, S1