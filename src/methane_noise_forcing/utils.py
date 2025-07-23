# src/methane_noise_forcing/utils.py
# -*- coding: utf-8 -*-
from dataclasses import dataclass
import numpy as np
from scipy import linalg


def generate_ar1_noise(
    phi: float, sigma_eps: float, n_steps: int, rng: np.random.Generator = None
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
    tau_days: float, dt_days: float, variance: float
) -> tuple[float, float, float]:
    """
    Compute AR(1) params from e‐folding timescale and stationary variance.

    Parameters
    ----------
    tau_days : float
        E‐folding autocorrelation timescale (days). If tau_days <= 0,
        we treat it as pure white noise (phi=0).
    dt_days : float
        Time‐step resolution (days).
    variance : float
        Stationary variance of the process (Var[X]).

    Returns
    -------
    phi : float
        AR(1) coefficient.
    sigma_cont : float
        Continuous‐time diffusion coeff (state / sqrt(day)).
    sigma_eps : float
        Discrete‐time innovation std-dev (state units).
    """
    if tau_days > 0:
        # AR(1) persistence
        phi = np.exp(-dt_days / tau_days)
        # continuous‐time diffusion coeff from Var[X] = sigma_cont^2 * tau / 2
        sigma_cont = np.sqrt(2 * variance / tau_days)
        # discrete innovation over dt
        sigma_eps = sigma_cont * np.sqrt(dt_days)
    else:
        # white noise: no memory
        phi = 0.0
        # infer sigma_cont so that sigma_eps = sqrt(variance)
        sigma_eps = np.sqrt(variance)
        sigma_cont = sigma_eps / np.sqrt(dt_days)

    return phi, sigma_cont, sigma_eps


@dataclass
class TwoTimescaleAR1Params:
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


def simulate_two_timescale_ar1(tau_x, tau_eta, variance_x, dt, N, n_ens, seed=42):
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
    N : int
        Length of the time series to simulate (time / dt).
    n_ens : int
        Number of ensemble members to simulate.
    seed : int, optional
        Random seed for reproducibility (default: 42).

    Returns
    -------
    x_ens : ndarray of shape (n_ens, N)
        Simulated upstream component x for each ensemble member.
    eta_ens : ndarray of shape (n_ens, N)
        Simulated downstream component η for each ensemble member.
    """
    params = calculate_two_timescale_ar1_params(tau_x, tau_eta, variance_x, dt)
    x_ens = np.zeros((n_ens, N))
    eta_ens = np.zeros((n_ens, N))
    for k in range(n_ens):
        state = np.zeros((2, N))
        for i in range(1, N):
            state[:, i] = params.F @ state[:, i - 1] + params.L @ np.random.randn(2)
        x_ens[k] = state[0]
        eta_ens[k] = state[1]
    return x_ens, eta_ens
