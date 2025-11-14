# src/methane_noise_forcing/stats/validation.py
# -*- coding: utf-8 -*-
"""Tools for validating fitted noise models."""

import numpy as np
from typing import Dict
from scipy.stats import chi2


def acf(x: np.ndarray, nlags: int) -> np.ndarray:
    """
    Unbiased ACF up to nlags (including lag 0 = 1).

    Parameters
    ----------
    x : array_like
        1D time series.
    nlags : int
        Number of lags to compute.

    Returns
    -------
    acf_vals : ndarray
        ACF values from lag 0 to nlags.
    """
    x = np.asarray(x, float)
    x = x - np.nanmean(x)
    n = x.size
    var = np.nanvar(x)
    if var == 0 or not np.isfinite(var):
        return np.concatenate(([1.0], np.zeros(nlags)))
    acf_vals = [1.0]
    for k in range(1, nlags + 1):
        v = np.nanmean((x[:-k] * x[k:]))
        acf_vals.append(v / var)
    return np.asarray(acf_vals)


def ljung_box(resid: np.ndarray, lags: int = 10) -> Dict[str, float]:
    """
    Ljung–Box Q test for residual autocorrelation up to 'lags'.

    Parameters
    ----------
    resid : array_like
        1D array of residuals from fitted model.
    lags : int, default 10
        Number of lags to include in the test.

    Returns
    -------
    results : dict
        Dictionary with keys 'Q', 'df', and 'pvalue' representing the test statistic,
            degrees of freedom, and p-value respectively.
    """
    r = acf(np.asarray(resid, float), nlags=lags)
    n = np.isfinite(resid).sum()
    Q = 0.0
    for k in range(1, lags + 1):
        Q += (r[k] ** 2) / (n - k)
    Q *= n * (n + 2)
    df = lags  # AR(1) parameter already used; conservative to keep lags
    p = 1.0 - chi2.cdf(Q, df=df)
    return {"Q": float(Q), "df": int(df), "pvalue": float(p)}


def durbin_watson(resid: np.ndarray) -> float:
    """
    Durbin–Watson statistic ~ 2 means no AR(1) structure, <2 positive AC, >2 negative.

    Parameters
    ----------
    resid : array_like
        1D array of residuals from fitted model.

    Returns
    -------
    dw_stat : float
        Durbin–Watson statistic.
    """
    e = np.asarray(resid, float)
    num = np.sum(np.diff(e) ** 2)
    den = np.sum(e**2)
    return float(num / den) if den > 0 else np.nan


def bootstrap_bias(tau_hat, tau_bootstrap: np.ndarray) -> float:
    """
    Compute bootstrap bias estimate for AR(1) timescale tau.
    Simple diagnostic of whether there are too few samples or if the autoregressive coefficient is close to 1.
    Autoregressive coefficients close to 1 are either 1) real or 2) biased from coarse sampling resolution

    Parameters
    ----------
    tau_hat : float
        Fitted timescale from original data.
    tau_bootstrap : np.ndarray
        Bootstrap samples of the timescale tau.

    Returns
    -------
    bias : float
        Bootstrap bias estimate.
    """
    bias = np.mean(tau_bootstrap) - tau_hat
    return b