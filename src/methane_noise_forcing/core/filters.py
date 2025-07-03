# src/methane_noise_forcing/core/filters.py
# -*- coding: utf-8 -*-

import numpy as np
from math import gamma
from scipy.optimize import fsolve


def gamma_kernel(k, theta, t_max=200, dt=1.0):
    """
    Generate a gamma kernel for filtering.

    Parameters
    ----------
    k : float
        Shape parameter of the gamma distribution.
    theta : float
        Scale parameter of the gamma distribution.
    t_max : int, optional
        Maximum time for the kernel, default is 200.
    dt : float, optional
        Time step for the kernel, default is 1.0.

    Returns
    -------
    t : ndarray
        1D array of time points from 0 to t_max in steps of dt.
    g : ndarray
        Normalized gamma PDF values at each t (i.e. ∫ g(t) dt = 1).
    """
    t = np.arange(0.0, t_max + dt, dt)
    g = (t ** (k - 1)) * np.exp(-t / theta) / ((theta**k) * gamma(k))
    g /= np.trapezoid(g, t)
    return t, g


def fit_gamma_params(mode, fwhm, k0=6.5):
    """
    Find gamma parameters k and theta so that the PDF has a given mode and FWHM.

    Parameters
    ----------
    mode : float
        Desired mode (peak location) of the gamma PDF in years.
    fwhm : float
        Desired full width at half maximum (years).
    k0 : float, optional
        Initial guess for the shape parameter k (default: 6.5).  A value
        around 6.5 produces a mode ≈25 yr and FWHM ≈25 yr, typical for
        WAIS Divide firn smoothing.

    Returns
    -------
    k_fit : float
        Fitted shape parameter k (α).
    theta_fit : float
        Fitted scale parameter θ.
    """

    def equations(params):
        k, theta = params
        # 1) mode constraint: (k-1)*θ = mode
        eq1 = (k - 1) * theta - mode

        # 2) FWHM constraint: solve PDF(t) = half‐peak at two points and enforce their difference = fwhm
        t_mode = (k - 1) * theta

        def pdf(t):
            return (t ** (k - 1)) * np.exp(-t / theta) / ((theta**k) * gamma(k))

        peak = pdf(t_mode)
        half = peak / 2
        t1 = fsolve(lambda t: pdf(t) - half, t_mode * 0.5)[0]
        t2 = fsolve(lambda t: pdf(t) - half, t_mode * 1.5)[0]
        eq2 = (t2 - t1) - fwhm

        return [eq1, eq2]

    k_fit, theta_fit = fsolve(equations, x0=[k0, mode / (k0 - 1)])
    return k_fit, theta_fit


def firn_convolve(series, kernel_t, kernel_g, dt_series=1.0):
    """
    Convolve a uniformly sampled time-series with a causal firn-air kernel
    using an FFT-based linear convolution.

    Parameters
    ----------
    series : 1-D array_like, shape (N,)
        The atmospheric or model time-series.  It **must** be sampled at a
        constant interval of ``dt_series`` (e.g., one value per year).
    kernel_t : 1-D array_like, shape (M,)
        Time axis (in the same units as ``dt_series``) for the kernel.
        Typically starts at 0 yr and extends to the age where the kernel
        is effectively zero.  The spacing of ``kernel_t`` does **not** have
        to equal ``dt_series``.
    kernel_g : 1-D array_like, shape (M,)
        Kernel values g(t) evaluated at ``kernel_t``.  Must be **causal**
        ( g(t) = 0 for t < 0 ) and non-negative.  The routine normalises
        it to unit area so that the convolution conserves the mean of
        the input series.
    dt_series : float, optional
        Sampling interval of ``series`` (default = 1.0).  Units must match
        those of ``kernel_t`` (e.g. both in yr).

    Returns
    -------
    smoothed : ndarray, shape (N,)
        The series after convolution with the kernel, aligned to the same
        time grid as the input ``series``.

    Notes
    -----
    * The kernel is first interpolated onto the series grid so that the
      convolution is performed on a single, uniform spacing.
    * The FFT length is chosen as the next power of two ≥ (N + L − 1),
      where L is the interpolated kernel length.  This guarantees linear
      (non-circular) convolution and gives near-optimal FFT speed.
    * Result is mathematically identical to
      ``np.convolve(series, kernel_interp, mode="full")[:N]`` but is
      much faster for long records.
    """
    # ---------- 1. interpolate kernel onto the series grid ------------------
    max_tau = kernel_t[-1]
    n_kernel_series = int(np.floor(max_tau / dt_series)) + 1
    t_uniform = np.arange(n_kernel_series) * dt_series
    kernel_uniform = np.interp(t_uniform, kernel_t, kernel_g, left=0.0, right=0.0)

    # ensure causal ordering and unit area (mass conservation)
    kernel_uniform = np.maximum(kernel_uniform, 0.0)
    kernel_uniform /= kernel_uniform.sum()  # normalise

    # ---------- 2. FFT-based linear convolution ----------------------------
    N = len(series)
    L = len(kernel_uniform)
    n_fft = 1 << int(np.ceil(np.log2(N + L)))  # next power of 2

    S_f = np.fft.rfft(series, n=n_fft)
    K_f = np.fft.rfft(kernel_uniform, n=n_fft)
    smoothed_full = np.fft.irfft(S_f * K_f, n=n_fft)

    # ---------- 3. trim to original length ---------------------------------
    return smoothed_full[:N]


class FirnFilter:
    def __init__(self, kernel: np.ndarray, dt: float = 1.0):
        """
        kernel: array of G(τ) values at τ = 0, dt, 2dt, ...
        dt: time-step of the kernel
        """
        self.kernel = kernel / kernel.sum()
        self.dt = dt

    @classmethod
    def from_gamma(cls, mode: float, fwhm: float, dt: float = 1.0):
        """
        Create a FirnFilter from a gamma kernel with specified mode and FWHM.

        Parameters
        ----------
        mode : float
            Desired mode (peak location) of the gamma PDF in years.
        fwhm : float
            Desired full width at half maximum (years).
        dt : float, optional
            Time step for the kernel, default is 1.0.

        Returns
        -------
        FirnFilter
            An instance of FirnFilter with the gamma kernel.
        """
        k, theta = fit_gamma_params(mode, fwhm)
        t, g = gamma_kernel(k, theta, t_max=200, dt=dt)
        return cls(g, dt)

    def apply(self, series: np.ndarray, dt_series=1.0) -> np.ndarray:
        """
        Apply the firn filter to a time series.

        Parameters
        ----------
        series : 1-D array_like, shape (N,)
            The atmospheric or model time-series. It must be sampled at a
            constant interval of ``dt_series`` (e.g., one value per year).
        dt_series : float, optional
            Sampling interval of ``series`` (default = 1.0). Units must match
            those of the filter kernel.

        Returns
        -------
        smoothed : ndarray, shape (N,)
            The series after convolution with the kernel, aligned to the same
            time grid as the input ``series``.
        """
        return firn_convolve(
            series, np.arange(len(self.kernel)) * self.dt, self.kernel, dt_series
        )
