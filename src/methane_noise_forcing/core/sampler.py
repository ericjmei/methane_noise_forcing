# src/methane_noise_forcing/core/sampler.py
# -*- coding: utf-8 -*-
import numpy as np
from scipy.interpolate import interp1d


def sample_series_at_core_points(
    series_time: np.ndarray,
    core_time: np.ndarray,
    series_to_sample: np.ndarray,
    core_offset: float = 0.0,
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
    core_offset : float, optional
        Offset for the start of sampling of core time (in the same units as series_time), default is 0.0.
    series_offset_years : float, optional
        Offset for the start the series time (in the same units as series_time), default is 0.0.

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
