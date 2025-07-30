# src/methane_noise_forcing/noise/white_noise_ar1.py
# -*- coding: utf-8 -*-
"""Functionality for generating AR(1) noise processes.

This module provides functions to compute AR(1) parameters from a given e-folding
timescale and stationary variance, and to generate AR(1) noise series.
"""

from dataclasses import dataclass
import numpy as np


@dataclass
class WhiteNoiseAR1Params:
    """Container for AR(1) parameters."""

    phi: float  # AR(1) coefficient
    sigma_eps: float  # Standard deviation of the innovation
    sigma_cont: float  # Continuous-time diffusion coefficient


def calculate_white_noise_ar1_params(
    tau_years: float, dt_years: float, variance: float
) -> WhiteNoiseAR1Params:
    """
    Compute AR(1) params from e‐folding timescale and stationary variance.

    Parameters
    ----------
    tau_years : float
        E‐folding autocorrelation timescale (years). If tau_years <= 0,
        we treat it as pure white noise (phi=0).
    dt_years : float
        Time‐step resolution (years).
    variance : float
        Stationary variance of the process (Var[X]).

    Returns
    -------
    WhiteNoiseAR1Params
        Container with phi, sigma_eps, sigma_cont.
    """
    if tau_years > 0:
        # AR(1) persistence
        phi = np.exp(-dt_years / tau_years)
        # continuous‐time diffusion coeff from Var[X] = sigma_cont^2 * tau / 2
        sigma_cont = np.sqrt(2 * variance / tau_years)
        # discrete innovation over dt
        sigma_eps = sigma_cont * np.sqrt(dt_years)
    else:
        # white noise: no memory
        phi = 0.0
        # infer sigma_cont so that sigma_eps = sqrt(variance)
        sigma_eps = np.sqrt(variance)
        sigma_cont = sigma_eps / np.sqrt(dt_years)

    return WhiteNoiseAR1Params(phi=phi, sigma_cont=sigma_cont, sigma_eps=sigma_eps)


def simulate_white_noise_ar1(
    phi: float,
    sigma_eps: float,
    n_steps: int,
    n_ens: int,
    rng: np.random.Generator = None,
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
    n_ens : int
        Number of ensemble members to simulate.
    rng : np.random.Generator, optional
        Random number generator for reproducibility. If None, a new default_rng() is used.

    Returns
    -------
    x_ens : ndarray of shape (n_ens, n_steps)
        The AR(1) noise series.
    forcing_ens : ndarray of shape (n_ens, n_steps)
        The innovations (forcing) series.
    """
    # Use provided RNG or create a new one
    if rng is None:
        rng = np.random.default_rng()

    x_ens = np.empty((n_ens, n_steps), float)
    forcing_ens = rng.normal(0.0, sigma_eps, size=(n_ens, n_steps))  # innovations

    # compute initial variance for stationary distribution
    if abs(phi) < 1:
        var_init = sigma_eps**2 / (1 - phi**2)
    else:
        var_init = 0.0
    x_ens[:, 0] = rng.normal(0.0, np.sqrt(var_init), size=n_ens)

    # Generate AR(1) noise
    for i in range(1, n_steps):
        x_ens[:, i] = phi * x_ens[:, i - 1] + forcing_ens[:, i]
    return x_ens, forcing_ens
