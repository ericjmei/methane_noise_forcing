# src/methane_noise_forcing/utils.py
# -*- coding: utf-8 -*-
"""
Functionality for miscellaneous functions.
"""
import pandas as pd
import numpy as np
import xarray as xr
import logging
from typing import Optional, Tuple, Union

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s")
logger = logging.getLogger(__name__)

ArrayLike = Union[np.ndarray, xr.DataArray, pd.DataFrame, pd.Series]

def _ensure_1d_numeric(a: ArrayLike) -> Tuple[np.ndarray, Optional[Union[xr.DataArray, pd.Series, pd.DataFrame]]]:
    """
    Flattens input to 1D numpy array of floats, returns original structure if possible

    Parameters
    ----------
    a : ArrayLike
        Input array-like object (numpy array, xarray DataArray, pandas DataFrame or Series).

    Returns
    -------
    Tuple[np.ndarray, Optional[Union[xr.DataArray, pd.Series, pd.DataFrame]]]
        A tuple containing:
        - 1D numpy array of floats.
        - The original xarray DataArray, pandas Series, or pandas DataFrame if applicable; None otherwise.
    """
    if isinstance(a, xr.DataArray):
        vals = np.asarray(a.values, dtype=float).ravel()
        return vals, a

    # Handle pandas Series: convert to 1D numpy array
    if isinstance(a, pd.Series):
        return np.asarray(a.values, dtype=float).ravel(), None

    # Handle pandas DataFrame: only allow single-column DataFrame to avoid ambiguity
    if isinstance(a, pd.DataFrame):
        if a.shape[1] == 1:
            # take the single column's values
            vals = np.asarray(a.iloc[:, 0].values, dtype=float).ravel()
            return vals, None
        raise ValueError(
            "DataFrame with multiple columns provided to _ensure_1d_numeric; "
            "please pass a single-column DataFrame or a 1D array-like (e.g., Series or ndarray)."
        )

    # Fallback for numpy arrays and other array-like objects
    return np.asarray(a, dtype=float).ravel(), None

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

def calculate_mean_and_ci(da, ci=0.95, dim='ensemble'):
    """
    Calculate the mean and confidence interval of a DataArray.

    Parameters
    ----------
    da : xr.DataArray
        Input data array.
    ci : float, optional
        Confidence interval (default is 0.95).
    dim : str or Iterable, optional
        Dimension along which to calculate the statistics (default is 'ensemble').

    Returns
    -------
    mean : float
        Mean of the data array.
    ci_low : float
        Lower bound of the confidence interval.
    ci_high : float
        Upper bound of the confidence interval.
    """
    mean = da.mean(dim=dim)
    percentile_low = (1 - ci) / 2
    percentile_high = (1 + ci) / 2
    ci_low = da.quantile(percentile_low, dim=dim)
    ci_high = da.quantile(percentile_high, dim=dim)
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

def average_to_resolution(data, resolution, time_dim="time"):
    """
    Average data along the time dimension to a specified temporal resolution.
    
    This function bins the data along the time dimension and computes the mean
    within each bin. The time coordinate is assumed to be in float years.
    
    Parameters
    ----------
    data : xr.DataArray or xr.Dataset
        Input data to be averaged. Must contain the specified time dimension.
    resolution : float
        Temporal resolution in years for averaging (e.g., 1.0 for annual, 
        0.5 for semi-annual, 10.0 for decadal).
    time_dim : str, optional
        Name of the time dimension to average over. Default is "time".
        
    Returns
    -------
    xr.DataArray or xr.Dataset
        Data averaged to the specified temporal resolution. The time coordinate
        will contain the left edge of each time bin.
        
    Examples
    --------
    >>> # Average monthly data to annual resolution
    >>> annual_data = average_to_resolution(monthly_data, resolution=1.0)
    >>> 
    >>> # Average data to decadal resolution
    >>> decadal_data = average_to_resolution(data, resolution=10.0)
    """
    # Extract time values from the data
    time_values = data[time_dim].values
    
    # Determine the time range for binning
    time_start = time_values.min()
    time_end = time_values.max()
    
    # Create bin edges spanning the full time range
    # Add resolution to ensure the last data point is included
    bin_edges = np.arange(time_start, time_end + resolution, resolution)
    
    # Use the left edge of each bin as the new time coordinate
    new_time_coords = bin_edges[:-1]
    
    # Group data by time bins and compute the mean within each bin
    binned_data = data.groupby_bins(
        time_dim, 
        bins=bin_edges, 
        labels=new_time_coords
    ).mean(dim=time_dim)
    
    # Rename the binned dimension back to the original time dimension name
    output_data = binned_data.rename({f"{time_dim}_bins": time_dim})
    
    return output_data