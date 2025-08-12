# src/methane_noise_forcing/core/sampler.py
# -*- coding: utf-8 -*-
import numpy as np
from scipy.interpolate import interp1d
import xarray as xr


def sample_series_at_core_points(
    series_time: np.ndarray,
    core_time: np.ndarray,
    series_to_sample: np.ndarray,
    core_offset_flag: str = None,
    series_offset: float = 0.0,
):
    """
    Sample a time series at specified core points.

    Parameters
    ----------
    series_time : np.ndarray
        Time array of the series to sample from (usually in years).
    core_time : np.ndarray
        Core time array at which to sample the series (in the same units as series_time).
    series_to_sample : np.ndarray
        The time series data to be sampled. It should have the same length as series_time.
    core_offset_flag : str, optional
        Flag to indicate how to handle core time offset:
        - None: keeps the original core time.
        - "start": core time is offset to start from 0.
    series_offset : float, optional
        Offset for the start the series time (in the same units as series_time), default is 0.0.
        Generally used to remove spinup

    Returns
    -------
    np.ndarray
        Sampled values from the series at the core points.
    """
    # Ensure inputs are numpy arrays
    series_time = np.asarray(series_time)
    core_time = np.asarray(core_time)
    series_to_sample = np.asarray(series_to_sample)

    # Adjust core time to start from 0 at core_offset
    core_offset = _compute_core_offset(core_time, core_offset_flag)
    adjusted_core_time = core_time - core_offset
    # Adjust series time to start from 0 at series_offset
    adjusted_series_time = series_time - series_offset

    # Interpolate the series to sample at the core points
    interpolator = interp1d(
        adjusted_series_time, series_to_sample, bounds_error=False, fill_value=np.nan
    )
    sampled_values = interpolator(adjusted_core_time)

    # Check for NaN values in the sampled values
    if np.any(np.isnan(sampled_values)):
        n_missing = np.sum(np.isnan(sampled_values))
        print(
            f"Warning: {n_missing} NaN values found in the sampled series. Consider checking the input series and core time arrays."
        )

    return sampled_values


def sample_ensemble_series(
    da: xr.DataArray,
    core_time: np.ndarray,
    core_offset_flag: str = None,
    series_offset: float = 0.0,
) -> xr.DataArray:
    """
    Sample each ensemble member in a DataArray at specified core points.

    Parameters
    ----------
    da : xr.DataArray
        Input data with dims ("ensemble", "time").
    core_time : np.ndarray
        Core time points to sample at.
    core_offset_flag : str, optional
        Passed to sample_series_at_core_points ("start" or None).
    series_offset : float, optional
        Offset for the series_time before sampling.

    Returns
    -------
    xr.DataArray
        Sampled data with dims ("ensemble","time") and time coords=
        adjusted core_time. The time‐coordinate attrs store the offsets.
    """
    series_time = da["time"].values

    # compute core_offset exactly as in sample_series_at_core_points
    core_offset = _compute_core_offset(core_time, core_offset_flag)
    adjusted_core_time = core_time - core_offset

    # sample each ensemble member
    sampled_list = []
    for member in da["ensemble"].values:
        vals = da.sel(ensemble=member).values
        sampled = sample_series_at_core_points(
            series_time,
            core_time,
            vals,
            core_offset_flag=core_offset_flag,
            series_offset=series_offset,
        )
        sampled_list.append(sampled)
    sampled_arr = np.stack(sampled_list, axis=0)

    # build output DataArray
    out = xr.DataArray(
        sampled_arr,
        dims=("ensemble", "time"),
        coords={"ensemble": da["ensemble"], "time": core_time},
    )
    # store offsets needed to convert back to core_time
    out["time"].attrs["core_offset"] = float(core_offset)
    out["time"].attrs["series_offset"] = float(series_offset)

    return out


def _compute_core_offset(core_time: np.ndarray, core_offset_flag: str = None) -> float:
    """
    Compute the core offset based on the core time and offset flag.

    Parameters
    ----------
    core_time : np.ndarray
        Core time points.
    core_offset_flag : str, optional
        Flag to indicate how to handle core time offset ("start" or None).

    Returns
    -------
    float
        The computed core offset.
    """
    if core_offset_flag == "start":
        return np.sort(core_time)[0]
    return 0.0
