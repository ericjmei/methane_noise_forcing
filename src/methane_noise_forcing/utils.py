# src/methane_noise_forcing/utils.py
# -*- coding: utf-8 -*-
import numpy as np


def generate_ar1_noise(
    phi: float,
    sigma_eps: float,
    n_steps: int,
    rng: np.random.Generator = None
) -> np.ndarray:
    """
    Core AR(1) noise generator.

    Parameters
    ----------
    phi : float
        AR(1) coefficient, must satisfy |phi|<1. For white noise use phi=0.
    sigma_eps : float
        Standard deviation of the innovation (noise) term. Assumes 0 power outside nyquist frequency.
    n_steps : int
        Length of time series to generate.
    rng : np.random.Generator, optional
        Random number generator for reproducibility. If None, a new default_rng() is used.

    Returns
    -------
    x : ndarray of shape (n_steps,)
        The AR(1) noise series.
    """
    # Use provided RNG or create a new one
    if rng is None:
        rng = np.random.default_rng()

    x = np.empty(n_steps, float)
    # compute initial variance for stationary distribution
    if abs(phi) < 1:
        var_init = sigma_eps**2 / (1 - phi**2)
    else:
        var_init = 0.0
    x[0] = rng.normal(0.0, np.sqrt(var_init))
    for i in range(1, n_steps):
        x[i] = phi * x[i - 1] + rng.normal(0.0, sigma_eps)
    return x


def compute_ar1_params_from_tau(
    tau_days: float,
    dt_days: float,
    variance: float
) -> tuple[float, float]:
    """
    Compute AR(1) parameters phi and sigma_eps from timescale and variance.

    Parameters
    ----------
    tau_days : float
        E‐folding autocorrelation timescale (days). If tau_days <= 0, returns phi=0.
    dt_days : float
        Time‐step resolution (days).
    variance : float
        Stationary variance of the process.

    Returns
    -------
    phi : float
        AR(1) coefficient.
    sigma_eps : float
        Standard deviation of the innovation term.
    """
    if tau_days > 0:
        phi = np.exp(-dt_days / tau_days)
    else:
        phi = 0.0
    sigma_eps = np.sqrt(variance * (1 - phi**2))
    return phi, sigma_eps