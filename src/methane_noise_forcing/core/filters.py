# src/methane_noise_forcing/core/filters.py
# -*- coding: utf-8 -*-

import numpy as np
from math import gamma
from scipy.optimize import fsolve
from scipy.signal.windows import tukey

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
    kernel : ndarray
        Normalized gamma PDF values at each t (i.e. ∫ g(t) dt = 1).
    """
    t = np.arange(0.0, t_max + dt, dt)
    kernel = (t ** (k - 1)) * np.exp(-t / theta) / ((theta**k) * gamma(k))
    kernel /= np.trapezoid(kernel, t)

    kernel = taper_kernel(kernel)  # Apply tapering to reduce edge effects
    return t, kernel


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

def log_logistic_kernel(alpha, beta, t_max=200, dt=1.0, offset=0.0, taper_fraction=0.1):
    """
    Generate a log-logistic kernel for firn smoothing, with edge taper
    applied *before* the time shift.

    Parameters
    ----------
    alpha : float
        Scale parameter of the log-logistic distribution.
    beta : float
        Shape parameter of the log-logistic distribution.
    t_max : float, optional
        Maximum time for the kernel, default is 200.
    dt : float, optional
        Time step for the kernel, default is 1.0.
    offset : float, optional
        Time offset to shift the kernel, default is 0.0.
    taper_fraction : float, optional
        Fraction of the kernel length to taper (passed to `taper_kernel`).

    Returns
    -------
    t : np.ndarray
        1D array of time points from 0 to t_max in steps of dt.
    kernel : np.ndarray
        Normalized, tapered, and shifted log-logistic PDF values.
    """
    t = np.arange(0, t_max + dt, dt) # time axis

    # un-shifted log-logistic PDF (causal: zero before t=0)
    kernel = np.zeros_like(t)
    positive = t > 0
    kernel[positive] = (
        (beta / alpha)
        * (t[positive] / alpha) ** (beta - 1)
        / (1 + (t[positive] / alpha) ** beta) ** 2
    )

    # apply taper to both edges of the causal PDF
    kernel = taper_kernel(kernel, taper_fraction=taper_fraction)

    # shift the tapered kernel by 'offset' years
    # compute fractional shift in index-space
    shift_idx = offset / dt
    idx = np.arange(len(kernel))
    # use linear interpolation; anything falling off the ends → 0
    kernel = np.interp(idx - shift_idx, idx, kernel, left=0.0, right=0.0)

    # final taper + renormalization
    kernel = taper_kernel(kernel, taper_fraction=taper_fraction)
    kernel /= np.trapezoid(kernel, t)

    return t, kernel

def fit_log_logistic_params(mode, fwhm, skew, beta0=3.0):
    """
    Find log-logistic parameters α, β, and offset so that the PDF has a given FWHM
    and half-width skew, then shift the mode to the desired location.

    Parameters
    ----------
    mode : float
        Desired mode (peak location) after shifting.
    fwhm : float
        Desired full width at half maximum.
    skew : float
        Ratio of left to right half-widths at half maximum:
        skew = (mode0 - t1) / (t2 - mode0).
    beta0 : float, optional
        Initial guess for β (must be >1).

    Returns
    -------
    alpha_fit : float
        Fitted scale parameter α.
    beta_fit : float
        Fitted shape parameter β.
    offset : float
        Time shift so that the kernel’s peak is at `mode`.
    """
    import numpy as np
    from scipy.optimize import fsolve

    def pdf(t, a, b):
        return (b / a) * (t / a)**(b - 1) / (1 + (t / a)**b)**2

    def equations(params):
        a, b = params
        # 1) intrinsic mode of the un-shifted PDF
        mode0 = a * ((b - 1) / (b + 1))**(1.0 / b)
        peak = pdf(mode0, a, b)
        half = peak / 2

        # 2) find the two half-max points
        t1 = fsolve(lambda tau: pdf(tau, a, b) - half, mode0 * 0.5)[0]
        t2 = fsolve(lambda tau: pdf(tau, a, b) - half, mode0 * 1.5)[0]

        # Equations to match
        eq1 = (t2 - t1) - fwhm  # Equation for FWHM
        eq2 = ((mode0 - t1) / (t2 - mode0)) - skew  # Equation for skew
        return [eq1, eq2]

    # Initial guesses
    alpha0 = mode / (((beta0 - 1) / (beta0 + 1))**(1.0 / beta0))
    a_fit, b_fit = fsolve(equations, x0=[alpha0, beta0])

    # Compute offset to shift mode0 → mode
    mode0 = a_fit * ((b_fit - 1) / (b_fit + 1))**(1.0 / b_fit)
    offset = mode - mode0

    return a_fit, b_fit, offset

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

def taper_kernel(kernel, taper_fraction=0.1):
    """
    Apply a taper to the edges of the kernel to reduce edge effects.

    Parameters
    ----------
    kernel : np.ndarray
        The kernel to be tapered.
    taper_fraction : float, optional
        Fraction of the kernel length to apply the taper (default is 0.1).

    Returns
    -------
    np.ndarray
        Tapered kernel.
    """
    taper = tukey(len(kernel), alpha=taper_fraction)
    kernel_tapered = kernel * taper
    kernel_tapered /= np.trapezoid(kernel_tapered)  # normalize the tapered kernel
    return kernel_tapered