# src/methane_noise_forcing/io.py
# -*- coding: utf-8 -*-
"""Equations to calculate expected variances of methane forcing or noise realizations."""

import numpy as np


def calculate_kernel_self_lag(kernel: np.ndarray, lags: int) -> np.ndarray:
    """
    Calculate the self-lag of a kernel. If kernel is normalized, this is the autocorrelation function.

    Parameters
    ----------
    kernel : np.ndarray
        The kernel array.
    lags : np.ndarray
        Array of lags at which to calculate the autocorrelation.

    Returns
    -------
    np.ndarray
        Self-lag values for the specified lags.
    """
    assert np.isclose(np.sum(kernel), 1), "Kernel must be normalized to sum to 1."
    assert np.all(np.isclose(lags, np.round(lags))), "Lags must be integers."
    c_taus = np.zeros_like(lags, dtype=float)
    for i, lag in enumerate(lags):
        if lag == 0:
            # Special case for lag=0: sum of squares
            c_tau = np.sum(kernel**2)
        else:
            c_tau = np.sum(kernel[: -int(lag)] * kernel[int(lag) :])
        c_taus[i] = c_tau

    return c_taus


def calculate_variance_ratio_filtered_to_instantaneous(
    c_taus: np.ndarray, instantaneous_autocorrelations: np.ndarray
) -> np.ndarray:
    """
    Calculate the 0-lag variance ratio between the kernel-filtered and instantaneous series.
    General for any prescribed kernel and autocorrelation function.

    Parameters
    ----------
    c_taus : np.ndarray
        The self-lagged values of the filtered signal at increasing lags, starting from 0.
    instantaneous_autocorrelations : np.ndarray
        The instantaneous autocorrelations at increasing lags.
        Ensure that the lags are the same as those used in c_taus.

    Returns
    -------
    np.ndarray
        The variance ratio between filtered and instantaneous series.
    """
    assert len(c_taus) == len(instantaneous_autocorrelations), (
        "c_taus and instantaneous_autocorrelations must have the same length."
    )

    # multiply the self-lagged values by the instantaneous autocorrelations at lag 1 and greater
    c_tau_times_autocorrelations = c_taus[1:] * instantaneous_autocorrelations[1:]

    # calculate the variance ratio
    variance_ratio = c_taus[0] + 2 * np.sum(c_tau_times_autocorrelations)

    return variance_ratio


def calculate_two_timescale_autocorrelation(phi_0, phi_1, lags):
    """
    Calculate the two-timescale autocorrelation function.
    Two-timescale system is an AR(1) process driven by an AR(1) forcing forced by white noise.

    Parameters
    ----------
    phi_0 : float
        Autocorrelation coefficient for the first timescale.
    phi_1 : float
        Autocorrelation coefficient for the second timescale.
    lags : np.ndarray
        Array of integer lags at which to calculate the autocorrelation.

    Returns
    -------
    np.ndarray
        The two-timescale autocorrelation values at the specified lags.
    """
    # ensure lags are integers
    assert np.all(np.isclose(lags, np.round(lags))), "Lags must be integers."
    S_taus = np.array([_calculate_S(lag, phi_0, phi_1) for lag in lags])
    S_0 = _calculate_S(0, phi_0, phi_1)
    return S_taus / S_0


def _calculate_S(lag, phi_0, phi_1):
    """
    Calculate the 'S' function for two-timescale autocorrelation.

    Parameters
    ----------
    lag : float
        The time lag.
    phi_0 : float
        The autocorrelation coefficient for the first timescale.
    phi_1 : float
        The autocorrelation coefficient for the second timescale.

    Returns
    -------
    float
        The value of the 'S' function for the given tau, phi_0, and phi_1.
    """
    if np.isclose(phi_0, phi_1):  # if identical, use other calculation of S(τ)
        return phi_0**lag * (lag + 1 + (1 - lag) * phi_0**2)
    else:
        term_1 = _calculate_term_1_3(lag, phi_1)
        term_2 = _calculate_term_2(lag, phi_0, phi_1)
        term_3 = _calculate_term_1_3(lag, phi_0)
        return term_1 - term_2 + term_3


def _calculate_term_1_3(lag, phi):
    """
    Helper function for the first and third term in the 'S' function for calculating two-timescale autocorrelation.

    Parameters
    ----------
    lag : float
        The time lag.
    phi : float
        The autocorrelation coefficient.

    Returns
    -------
    float
        The value of the first term in the 'S' function.
    """
    return (phi ** (lag + 2)) / (1 - phi**2)


def _calculate_term_2(lag, phi_0, phi_1):
    """
    Helper function for the second term in the 'S' function for calculating two-timescale autocorrelation.

    Parameters
    ----------
    tau : float
        The time lag.
    phi_0 : float
        The autocorrelation coefficient for the first timescale.
    phi_1 : float
        The autocorrelation coefficient for the second timescale.

    Returns
    -------
    float
        The value of the second term in the 'S' function.
    """
    return (phi_1 * phi_0 ** (lag + 1) + phi_0 * phi_1 ** (lag + 1)) / (
        1 - phi_0 * phi_1
    )


def calculate_two_timescale_variance_ratio_methane_to_ar1_forcing(
    tau_methane, tau_forcing
):
    """
    Calculate the variance ratio between AR(1) methane and its AR(1) forcing for a two-timescale system.

    Parameters
    ----------
    tau_methane : float
        e-folding timescale of the methane process.
    tau_forcing : float
        e-folding timescale of the AR(1) forcing process.
    """
    return tau_methane**2 * tau_forcing / (tau_methane + tau_forcing)
