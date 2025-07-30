# src/methane_noise_forcing/noise/two_timescale_ar1.py
# -*- coding: utf-8 -*-
"""Functionality for simulating a two-timescale AR(1) noise process.

Two timescale process is defined by:
    dx/dt = A x + B ξ, where x=[f, η]
with timescales τₓ, τη, and Var[f]=variance_x in stationarity and discretized with step dt.
This module provides functions to calculate the necessary parameters and simulate the process.
"""

from dataclasses import dataclass
import numpy as np
from scipy import linalg


@dataclass
class TwoTimescaleAR1Params:
    """Container for two-timescale AR(1) parameters."""

    A: np.ndarray  # continuous‐time system matrix
    B: np.ndarray  # diffusion input matrix
    F: np.ndarray  # discrete‐time transition matrix expm(A*dt)
    M: np.ndarray  # augmented matrix used for covariance integral
    expM: np.ndarray  # expm(M*dt)
    Q0: np.ndarray  # process noise covariance over one step
    L: np.ndarray  # Cholesky factor of Q0
    sigma_f: float  # scaling factor for fast mode
    sigma_eta: float  # scaling factor for slow mode


def calculate_two_timescale_ar1_params(
    tau_x: float,
    tau_eta: float,
    variance_x: float,
    dt: float,
) -> TwoTimescaleAR1Params:
    """
    Compute state‐space & noise parameters for a two‐timescale AR(1) process.

    We have dx/dt = A x + B ξ, where x=[f, η]ᵀ, timescales τₓ, τη,
    and Var[f]=variance_x in stationarity.  We discretize with step dt.

    Parameters
    ----------
    tau_x : float
        e‐folding timescale of the fast component f.
    tau_eta : float
        e‐folding timescale of the slow component η.
    variance_x : float
        Target stationary variance of f.
    dt : float
        Discrete time step.

    Returns
    -------
    TwoTimescaleAR1Params
        Container with A, B, F, M, expM, Q0, L, sigma_f, sigma_eta.
    """
    # Continuous‐time system matrix: df/dt = –f/τₓ + η, dη/dt = –η/τη
    A = np.array([[-1.0 / tau_x, 1.0], [0.0, -1.0 / tau_eta]])

    # Diffusion input only enters η‐equation
    B = np.array([[0.0], [1.0]])

    # Discrete‐time transition matrix
    F = linalg.expm(A * dt)

    # Build augmented matrix for exact covariance integral:
    #   M = [[A, B Bᵀ],
    #        [0, –Aᵀ]]
    M = np.block([[A, B @ B.T], [np.zeros_like(A), -A.T]])

    # Matrix exponential of M dt
    expM = linalg.expm(M * dt)

    # Extract process noise covariance: ∫₀^dt exp(A s) B Bᵀ exp(Aᵀ s) ds
    Q0 = expM[:2, 2:] @ linalg.inv(expM[2:, 2:])

    # Solve discrete‐Lyapunov: F Σ Fᵀ – Σ + Q0 = 0
    Sigma = linalg.solve_discrete_lyapunov(F, Q0)

    # Scale factors so Var[f] = variance_x
    Cff = Sigma[0, 0]
    sigma_f = np.sqrt(variance_x / Cff)
    sigma_eta = sigma_f * np.sqrt(Sigma[1, 1])

    # scale Q0 by sigma_f^2
    Q0 = Q0 * sigma_f**2
    # Cholesky factor of Q0 (for sampling innovations)
    L = np.linalg.cholesky(Q0)

    return TwoTimescaleAR1Params(A, B, F, M, expM, Q0, L, sigma_f, sigma_eta)


def simulate_two_timescale_ar1(tau_x, tau_eta, variance_x, dt, n_steps, n_ens, rng: np.random.Generator = None):
    """
    Simulate a two‐timescale AR(1) process with given parameters.

    Parameters
    ----------
    tau_x : float
        E‐folding timescale of the slow component η.
    tau_eta : float
        E‐folding timescale of the fast component f.
    variance_x : float
        Target stationary variance of x.
    dt : float
        Discrete time step.
    n_steps : int
        Length of the time series to simulate (time / dt).
    n_ens : int
        Number of ensemble members to simulate.
    rng : np.random.Generator, optional
        Random number generator for reproducibility (default: None).

    Returns
    -------
    x_ens : ndarray of shape (n_ens, n_steps)
        Simulated upstream component x for each ensemble member.
    eta_ens : ndarray of shape (n_ens, n_steps)
        Simulated downstream component η for each ensemble member.
    """
    # Use provided RNG or create a new one
    if rng is None:
        rng = np.random.default_rng()

    params = calculate_two_timescale_ar1_params(tau_x, tau_eta, variance_x, dt)
    x_ens = np.zeros((n_ens, n_steps))
    eta_ens = np.zeros((n_ens, n_steps))
    innovations = rng.standard_normal(size=(n_ens, n_steps, 2))

    for k in range(n_ens):
        state = np.zeros((2, n_steps))
        for i in range(1, n_steps):
            state[:, i] = params.F @ state[:, i - 1] + params.L @ innovations[k, i]
        x_ens[k] = state[0]
        eta_ens[k] = state[1]
    return x_ens, eta_ens
