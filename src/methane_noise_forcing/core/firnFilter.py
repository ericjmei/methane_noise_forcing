# src/methane_noise_forcing/core/firnFilter.py
# -*- coding: utf-8 -*-

import numpy as np
from .filters import (
    gamma_kernel,
    fit_gamma_params,
    log_logistic_kernel,
    fit_log_logistic_params,
    firn_convolve,
)


class FirnFilter:
    def __init__(self, kernel: np.ndarray, dt: float = 1.0):
        """
        kernel: array of G(τ) values at τ = 0, dt, 2dt, ...
        dt: time-step of the kernel
        """
        self.kernel = kernel / kernel.sum()
        self.dt = dt

    @classmethod
    def fit_gamma(
        cls,
        mode: float,
        fwhm: float,
        skew: float = 0.7,
        t_max: float = 200,
        dt: float = 1.0,
        k0: float = 6.5,
        **kwargs,
    ):
        """
        Create a FirnFilter from a gamma kernel with specified mode and FWHM.

        Parameters
        ----------
        mode : float
            Desired mode (peak location) of the gamma PDF in years.
        fwhm : float
            Desired full width at half maximum (years).
        skew : float, optional
            Ratio of left to right half-widths at half maximum:
            skew = (mode0 - t1) / (t2 - mode0).
        t_max : float, optional
            Maximum time for the kernel, default is 200.
        dt : float, optional
            Time step for the kernel, default is 1.0.
        k0 : float, optional
            Initial guess for the shape parameter k (default: 6.5). A value
            around 6.5 produces a mode ≈25 yr and FWHM ≈25 yr, typical for
            WAIS Divide firn smoothing.
        **kwargs : dict, optional
            Additional keyword arguments for the gamma kernel.

        Returns
        -------
        FirnFilter
            An instance of FirnFilter with the gamma kernel.
        """
        k, theta, offset = fit_gamma_params(mode, fwhm, skew=skew, k0=k0)
        t, kernel = gamma_kernel(k, theta, t_max=t_max, dt=dt, offset=offset, **kwargs)
        return cls(kernel, dt)

    @classmethod
    def from_gamma_params(
        cls,
        k: float,
        theta: float,
        t_max: float = 200,
        dt: float = 1.0,
        offset: float = 0.0,
        **kwargs,
    ):
        """
        Create a FirnFilter from gamma parameters.

        Parameters
        ----------
        k : float
            Shape parameter of the gamma distribution.
        theta : float
            Scale parameter of the gamma distribution.
        t_max : float, optional
            Maximum time for the kernel, default is 200.
        dt : float, optional
            Time step for the kernel, default is 1.0.
        offset : float, optional
            Time offset to shift the kernel, default is 0.0.
        **kwargs : dict, optional
            Additional keyword arguments for the gamma kernel.

        Returns
        -------
        FirnFilter
            An instance of FirnFilter with the gamma kernel.
        """
        t, g = gamma_kernel(k, theta, t_max=t_max, dt=dt, offset=offset, **kwargs)
        return cls(g, dt)

    @classmethod
    def fit_log_logistic(
        cls,
        mode: float,
        fwhm: float,
        skew: float = 0.7,
        t_max: float = 200,
        dt: float = 1.0,
        **kwargs,
    ):
        """
        Create a FirnFilter from a log-logistic kernel with specified mode, FWHM, and skew.

        Parameters
        ----------
        mode : float
            Desired mode (peak location) of the log-logistic PDF in years.
        fwhm : float
            Desired full width at half maximum (years).
        skew : float
            Ratio of left to right half-widths at half maximum:
            skew = (mode0 - t1) / (t2 - mode0).
        t_max : float, optional
            Maximum time for the kernel, default is 200.
        dt : float, optional
            Time step for the kernel, default is 1.0.
        **kwargs : dict, optional
            Additional keyword arguments for the log-logistic kernel.

        Returns
        -------
        FirnFilter
            An instance of FirnFilter with the log-logistic kernel.
        """
        alpha, beta, offset = fit_log_logistic_params(mode, fwhm, skew, beta0=3.0)
        t, g = log_logistic_kernel(
            alpha, beta, t_max=t_max, dt=dt, offset=offset, **kwargs
        )
        return cls(g, dt)

    @classmethod
    def from_log_logistic_params(
        cls,
        alpha: float,
        beta: float,
        t_max: float = 200,
        dt: float = 1.0,
        offset: float = 0.0,
        **kwargs,
    ):
        """
        Create a FirnFilter from log-logistic parameters.

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
        **kwargs : dict, optional
            Additional keyword arguments for the log-logistic kernel.

        Returns
        -------
        FirnFilter
            An instance of FirnFilter with the log-logistic kernel.
        """
        t, g = log_logistic_kernel(
            alpha, beta, t_max=t_max, dt=dt, offset=offset, **kwargs
        )
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
