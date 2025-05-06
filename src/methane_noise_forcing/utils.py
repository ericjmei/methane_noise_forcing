# src/methane_noise_forcing/utils.py
# -*- coding: utf-8 -*-
import numpy as np


def generate_ar1_noise(phi, variance, n_steps, seed=None):
    """
    Core AR(1) noise generator.

    Parameters
    ----------
    phi : float
        AR(1) coefficient, must satisfy |phi|<1.  For white noise use phi=0.
    variance : float
        Stationary variance of the process.
    n_steps : int
        Length of time series to generate.
    seed : int, optional
        RNG seed for reproducibility.

    Returns
    -------
    x : ndarray of shape (n_steps,)
        The AR(1) noise series.
    """
    rng = np.random.default_rng(seed)
    # innovations sd so that Var[x] = variance
    sigma_eps = np.sqrt(variance * (1 - phi**2))

    # initialize
    x = np.empty(n_steps, float)
    x[0] = rng.normal(0.0, np.sqrt(variance))
    for i in range(1, n_steps):
        x[i] = phi * x[i - 1] + rng.normal(0.0, sigma_eps)
    return x


def generate_ar1_noise_using_tau(tau_days, dt_days, variance, n_tau_steps, seed=None):
    """
    Wrapper that computes phi = exp(-dt/tau) and calls generate_ar1_noise.
    If you pass tau_days <= 0, it will generate white noise (phi=0).

    Parameters
    ----------
    tau_days : float
        E‐folding autocorrelation timescale (days).  If tau_days<=0,
        we treat it as white noise (phi=0).
    dt_days : float
        Time‐step resolution (days).
    variance : float
        Stationary variance of the noise.
    n_tau_steps : float
        Total simulation length in multiples of tau_days.
    seed : int, optional
        RNG seed.

    Returns
    -------
    x : ndarray
        Noise series of length N = round(n_tau_steps * tau_days / dt_days).
    """
    # compute φ; if tau_days <= 0 → white noise
    phi = np.exp(-dt_days / tau_days) if tau_days > 0 else 0.0

    # number of steps
    N = (
        int(round(n_tau_steps * tau_days / dt_days))
        if tau_days > 0
        else int(round(n_tau_steps))
    )
    # if tau_days <= 0, interpret n_tau_steps directly as n_steps
    return generate_ar1_noise(phi, variance, N, seed)
