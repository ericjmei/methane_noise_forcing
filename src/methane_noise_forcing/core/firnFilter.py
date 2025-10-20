# src/methane_noise_forcing/core/firnFilter.py
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from .filters import (
    gamma_kernel,
    fit_gamma_params,
    log_logistic_kernel,
    fit_log_logistic_params,
    firn_convolve,
)
from pathlib import Path


class FirnFilter:
    def __init__(self, kernel: np.ndarray, dt: float = 1.0):
        """
        kernel: array of G(τ) values at τ = 0, dt, 2dt, ...
        dt: time-step of the kernel
        """

        self.kernel = kernel
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

    @classmethod
    def from_firn_model(cls, file_name: str, path_root: str, **kwargs):
        """
        Load a FirnFilter from a file containing kernel parameters.

        Parameters
        ----------
        path_root : str
            Path to the root directory containing kernel parameter files.
        **kwargs : dict, optional
            Additional keyword arguments for the kernel fitting.

        Returns
        -------
        FirnFilter
            An instance of FirnFilter loaded from the specified path.
        """
        # load the kernel from the file
        # build path
        path = Path(path_root) / file_name
        df = pd.read_csv(path)

        # extract dt, ensure it is constant
        assert "time" in df.columns, "Column 'time' not found in the file."
        time = df["time"].values
        dt = np.mean(np.diff(time))
        assert np.allclose(np.diff(time), dt), "Time intervals are not constant."
        kernel = df.drop(columns=["time"]).values.flatten()
        return cls(kernel, dt)

    @classmethod
    def fit_from_config(cls, kernel_type: str, parameters: dict, **kwargs):
        """
        Create a FirnFilter instance based on the kernel type and parameters.

        Parameters
        ----------
        kernel_type : str
            Type of the kernel to create (e.g., "log_logistic", "gamma", "firn_model").
        parameters : dict
            Parameters required for the specified kernel type.
        **kwargs : dict, optional
            Additional keyword arguments for the kernel fitting.

        Returns
        -------
        FirnFilter
            An instance of FirnFilter created based on the specified kernel type and parameters.
        """
        if kernel_type == "log_logistic":
            return cls.fit_log_logistic(**parameters, **kwargs)
        elif kernel_type == "gamma":
            return cls.fit_gamma(**parameters, **kwargs)
        elif kernel_type == "firn_model":
            return cls.from_firn_model(**parameters, **kwargs)
        else:
            raise ValueError(f"Unknown kernel type: {kernel_type}")

    def apply(self, series: np.ndarray, dt_series=1.0, flip_kernel=False) -> np.ndarray:
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
        flip_kernel : bool, optional
            If True, flip the kernel before applying. Default is False.

        Returns
        -------
        smoothed : ndarray, shape (N,)
            The series after convolution with the kernel, aligned to the same
            time grid as the input ``series``.
        """
        if flip_kernel:
            kernel = self.kernel[::-1]
        else:
            kernel = self.kernel
        return firn_convolve(
            series, np.arange(len(kernel)) * self.dt, kernel, dt_series
        )
