# src/methane_noise_forcing/stats/fit.py
# -*- coding: utf-8 -*-
"""Functionality for fitting noise model parameters to data."""

import numpy as np
import warnings
from typing import Tuple, Optional
from scipy.stats import chi2, norm
from scipy.optimize import minimize_scalar


def _check_length(x: np.ndarray, min_length: int = 3) -> None:
    """Raise ValueError if x has fewer than min_length finite points."""
    if x.size < min_length:
        raise ValueError(f"Series too short (need >= {min_length} finite points).")


def _check_near_unit_root(phi: float, thresh: float = 0.995) -> None:
    """Raise Warning if |phi| >= thresh."""
    if abs(phi) >= thresh:
        warnings.warn(
            f"Fitted AR(1) coefficient phi={phi:.4f} is very close to unit root; "
            "estimates may be unreliable."
        )


def fit_ar1_conditional_ols(x: np.ndarray, demean: bool = True) -> Tuple[float, float]:
    """
    Conditional MLE (== OLS) for AR(1): x_t = phi x_{t-1} + eps_t.

    Parameters
    ----------
    x : array_like
        1D time series with equal spacing.
    demean : bool, default True
        If True, subtract the sample mean before fitting (i.e., zero-mean AR(1)).
        If False, the intercept is not estimated here; pass a demeaned series yourself
        or extend the model to include a mean parameter.

    Returns
    -------
    phi_hat : float
        Conditional MLE of AR(1) coefficient.
    sigma2_hat : float
        Conditional MLE of innovation variance.
    """
    x = np.asarray(x, float)
    mask = np.isfinite(x)
    x = x[mask]
    _check_length(x, min_length=3)
    if demean:
        x = x - x.mean()

    x1 = x[:-1]
    x2 = x[1:]

    denom = np.dot(x1, x1)
    if denom == 0.0:
        # completely flat (after demeaning); autocorr is undefined; set phi=0
        phi = 0.0
        sigma2 = np.var(x2, ddof=1)
        return phi, float(sigma2)

    phi = float(np.dot(x2, x1) / denom)
    resid = x2 - phi * x1
    sigma2 = float(np.dot(resid, resid) / (x.size - 1))  # conditional variance
    _check_near_unit_root(phi)
    return phi, sigma2


def fit_ar1_exact_mle(
    x: np.ndarray,
    demean: bool = True,
    phi_bounds: Tuple[float, float] = (-0.9999, 0.9999),
) -> Tuple[float, float]:
    """
    Exact MLE for AR(1): x_t = phi * x_{t-1} + eps_t, eps_t ~ N(0, sigma^2).
    Includes the marginal distribution of x_1: Var(x_1) = sigma^2 / (1 - phi^2).

    Parameters
    ----------
    x : array_like
        1D time series with equal spacing.
    demean : bool, default True
        If True, subtract the sample mean before fitting (i.e., zero-mean AR(1)).
        If False, the intercept is not estimated here; pass a demeaned series yourself
        or extend the model to include a mean parameter.
    phi_bounds : tuple(float, float)
        Bounds for phi during optimization (must be strictly inside (-1, 1)).

    Returns
    -------
    phi_hat : float
        Exact MLE of AR(1) coefficient.
    sigma2_hat : float
        Exact MLE of innovation variance.

    Notes
    -----
    The exact log-likelihood is optimized using a profile likelihood approach,
    where for each phi, the corresponding sigma^2 that maximizes the likelihood
    is computed in closed form.
    """
    x = np.asarray(x, float)
    mask = np.isfinite(x)
    x = x[mask]
    n = x.size
    _check_length(x, min_length=3)
    if demean:
        x = x - x.mean()

    x1 = x[:-1]
    x2 = x[1:]

    def _obj(phi: float) -> float:
        # Guard against |phi| >= 1
        if not (phi_bounds[0] < phi < phi_bounds[1]):
            return np.inf
        # Innovations and sums
        resid = x2 - phi * x1
        sse = float(np.dot(resid, resid))
        one_minus_phi2 = 1.0 - phi * phi
        if one_minus_phi2 <= 0.0:  # guard against phi approx ±1
            return np.inf
        term0 = one_minus_phi2 * (x[0] ** 2)
        sigma2_hat_phi = (term0 + sse) / n
        # numerical guard (shouldn’t happen unless degenerate)
        if sigma2_hat_phi <= 0.0 or not np.isfinite(sigma2_hat_phi):
            return np.inf
        return n * np.log(sigma2_hat_phi / n) - np.log(one_minus_phi2)

    # 1-D bounded optimization (Brent) on the profile objective
    res = minimize_scalar(_obj, bounds=phi_bounds, method="bounded")
    if not res.success:
        raise RuntimeError(f"Exact MLE optimization failed: {res.message}")

    phi_hat = float(res.x)

    # Recover sigma^2 at phi_hat
    resid_hat = x2 - phi_hat * x1
    sse_hat = float(np.dot(resid_hat, resid_hat))
    sigma2_hat = ((1.0 - phi_hat**2) * (x[0] ** 2) + sse_hat) / n

    _check_near_unit_root(phi_hat)
    return phi_hat, float(sigma2_hat)


def tau_from_phi(phi: float, dt: float) -> Optional[float]:
    """Continuous-time AR(1)/OU e-folding time tau = -dt / ln(phi), defined for 0<phi<1."""
    if 0 < phi < 1:
        return -dt / np.log(phi)
    return np.nan


def bootstrap_tau_parametric(
    phi_hat: float,
    sigma2_hat: float,
    dt: float,
    n_series: int,
    fit_method: str = "exact_mle",
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
):
    """
    Parametric bootstrap for confidence intervals on AR(1) timescale tau.

    Parameters
    ----------
    phi_hat : float
        Fitted AR(1) coefficient from original data.
    sigma2_hat : float
        Fitted innovation variance from original data.
    dt : float
        Time step of the original data.
    n_series : int
        Length of the original time series.
    fit_method : str, default "exact_mle"
        Fitting method to use: "exact_mle" or "conditional_ols".
    n_bootstrap : int, default 1000
        Number of bootstrap samples.
    ci : float, default 0.95
        Confidence interval level.
    seed : int, default 42
        Random seed for reproducibility.

    Returns
    -------
    tau_bootstrap : np.ndarray
        Bootstrap samples of the AR(1) timescale tau.
    tau_ci : tuple(float, float)
        Lower and upper bounds of the confidence interval for tau.
    """
    rng = np.random.default_rng(seed)
    x0_sd = np.sqrt(sigma2_hat / (1 - phi_hat**2))
    tau_bootstrap = np.full(n_bootstrap, np.nan)

    for i in range(n_bootstrap):
        # Simulate AR(1) series with fitted parameters
        eps = rng.normal(loc=0.0, scale=np.sqrt(sigma2_hat), size=n_series)
        x_boot = np.empty(n_series, dtype=float)
        x_boot[0] = rng.normal(loc=0.0, scale=x0_sd)
        for t in range(1, n_series):
            x_boot[t] = phi_hat * x_boot[t - 1] + eps[t]

        # Fit AR(1) to bootstrap sample
        if fit_method == "exact_mle":
            phi_b, _ = fit_ar1_exact_mle(x_boot, demean=True)
        elif fit_method == "conditional_ols":
            phi_b, _ = fit_ar1_conditional_ols(x_boot, demean=True)
        else:
            raise ValueError(f"Unknown fit_method: {fit_method}")

        # Compute tau from fitted phi
        tau_b = tau_from_phi(phi_b, dt)
        tau_bootstrap[i] = tau_b

    tau_bootstrap = tau_bootstrap[~np.isnan(tau_bootstrap)]
    tau_ci = np.quantile(tau_bootstrap, [(1 - ci) / 2, 1 - (1 - ci) / 2])
    return tau_bootstrap, tuple(tau_ci)
