# src/methane_noise_forcing/noise/one_box.py
# -*- coding: utf-8 -*-
"""Functionality for simulating a methane box model with some forcing"""

import numpy as np
from dataclasses import dataclass
from typing import Callable, Optional, Tuple, Union, Literal
from scipy.interpolate import interp1d
from scipy.integrate import solve_ivp

# Types
TimeSeries = Tuple[np.ndarray, np.ndarray]
ScalarOrCallableOrSeries = Union[float, Callable[[float], float], TimeSeries]


@dataclass
class OneBoxResult:
    """
    Container for results from solve_ch4_onebox.

    Attributes
    ----------
    t : np.ndarray
        array of evaluated times
    C : np.ndarray
        array of evaluated concentrations
    success : bool
        True if the solver succeeded, False otherwise
    """

    t: np.ndarray
    C: np.ndarray
    success: bool
    message: str


def _as_timefunc(
    spec: ScalarOrCallableOrSeries,
    *,
    name: str,
    kind: Literal["linear", "nearest", "cubic", "previous"] = "cubic",
    extrapolate: bool = False,
) -> Callable[[float], float]:
    """
    Turn a scalar, callable, or (t, y) pair into a function of time.

    Parameters
    ----------
    spec : ScalarOrCallableOrSeries
        The input to convert into a time function.
    name : str
        The name of the input (for error messages).
    kind : {"linear", "nearest", "cubic", "previous"}
        The kind of interpolation to use for (t, y) pairs.
    extrapolate : bool
        Whether to allow extrapolation beyond the input range.

    Returns
    -------
    Callable[[float], float]
        A function that takes a time value and returns the corresponding
        interpolated value.

    Notes
    -----
    - Scalars -> constant function
    - Callables -> returned as-is
    - (t, y) -> interp1d (with optional step via 'previous')
    """
    if np.isscalar(spec):
        val = float(spec)
        return lambda t: val
    if callable(spec):
        return spec
    # (t, y) pair
    if not (isinstance(spec, (tuple, list)) and len(spec) == 2):
        raise TypeError(f"{name} must be a scalar, callable, or (t, y) pair.")
    t, y = spec
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    if t.ndim != 1 or y.ndim != 1 or t.size != y.size or t.size < 2:
        raise ValueError(f"{name}: (t, y) must be 1D and of equal length >= 2.")
    order = np.argsort(t)
    t = t[order]
    y = y[order]
    if kind == "previous":
        # left-hold step function
        def f(tq):
            tq = np.asarray(tq)
            idx = np.searchsorted(t, tq, side="right") - 1
            if extrapolate:
                idx = np.clip(idx, 0, len(t) - 1)
            else:
                idx = np.where((tq < t[0]) | (tq > t[-1]), -1, idx)
            out = np.where(idx == -1, np.nan, y[idx])
            return out

        return lambda tt: float(f(tt))
    fill = "extrapolate" if extrapolate else (y[0], y[-1])
    f = interp1d(
        t,
        y,
        kind="linear" if kind == "previous" else kind,
        bounds_error=False,
        fill_value=fill,
        assume_sorted=True,
    )
    return lambda tt: float(f(tt))


def solve_ch4_onebox(
    *,
    t_eval: np.ndarray,
    C0: float,
    S: ScalarOrCallableOrSeries,
    # Choose ONE of the following ways to specify λ(t):
    lam: Optional[ScalarOrCallableOrSeries] = None,  # λ(t) directly
    tau: Optional[ScalarOrCallableOrSeries] = None,  # lifetime τ(t) -> λ(t)=1/τ(t)
    k: Optional[float] = None,  # with OH(t): λ(t)=k*OH(t)
    OH: Optional[ScalarOrCallableOrSeries] = None,
    # Interpolation & solver options
    s_kind: Literal["linear", "nearest", "cubic", "previous"] = "cubic",
    l_kind: Literal["linear", "nearest", "cubic", "previous"] = "cubic",
    extrapolate: bool = False,
    method: str = "LSODA",
    enforce_nonnegative: bool = False,
    **kwargs,
) -> OneBoxResult:
    """
    Unified solver for dC/dt = S(t) - λ(t) C(t)

    Parameters
    ----------
    t_eval : np.ndarray
        The time points at which to evaluate the solution.
        Solver begins at t_eval[0] and ends at t_eval[-1].
    C0 : float
        The initial methane concentration at time t=0.
    S : ScalarOrCallableOrSeries
        The source term as a function of time.
    lam : Optional[ScalarOrCallableOrSeries]
        The decay rate as a function of time.
    tau : Optional[ScalarOrCallableOrSeries]
        The lifetime as a function of time.
    k : Optional[float]
        The scaling factor for the OH term.
    OH : Optional[ScalarOrCallableOrSeries]
        The OH concentration as a function of time.
    s_kind : Literal["linear", "nearest", "cubic", "previous"]
        The interpolation kind for the source term S.
    l_kind : Literal["linear", "nearest", "cubic", "previous"]
        The interpolation kind for the decay rate λ.
    extrapolate : bool
        Whether to extrapolate the source and decay rate functions beyond their defined range.
        If True, the functions will be extended beyond their original range using the specified interpolation method.
        If False, any requests for values outside the original range will return NaN.
    method : str
        The integration method to use for the ODE solver.
    enforce_nonnegative : bool
        Whether to enforce non-negativity of the solution C(t).
    **kwargs
        Additional keyword arguments passed to scipy.solve_ivp().

    Returns
    -------
    OneBoxResult
        The result of the one-box model simulation.

    Notes
    -----
    Supply λ(t) via exactly ONE of:
      1) lam               -> scalar/callable/(t, y)
      2) tau               -> λ(t) = 1 / τ(t)
      3) k AND OH          -> λ(t) = k * OH(t)

    S can be scalar, callable, or (t, y).

    Units:
      - Pick a time unit (e.g., years) for t_eval and all time series.
      - If C is ppb, then:
          S in ppb / time
          λ in 1 / time  (so k*OH or 1/τ must evaluate to 1/time)
      - If C is mol / mol, then:
          S is in mol / mol / time
          λ is in 1 / time  (so k*OH or 1/τ must evaluate to 1/time)
    """
    t_eval = np.asarray(t_eval, dtype=float)
    if t_eval.ndim != 1 or t_eval.size < 2 or not np.all(np.diff(t_eval) > 0):
        raise ValueError("t_eval must be 1D, strictly increasing, length >= 2.")

    # Build S(t)
    S_of_t = _as_timefunc(S, name="S", kind=s_kind, extrapolate=extrapolate)

    # Build λ(t)
    provided = sum(
        [lam is not None, tau is not None, (k is not None and OH is not None)]
    )
    if provided != 1:
        raise ValueError("Specify exactly ONE of lam, tau, or (k and OH).")

    if lam is not None:
        lam_of_t = _as_timefunc(lam, name="lam", kind=l_kind, extrapolate=extrapolate)
    elif tau is not None:
        tau_of_t = _as_timefunc(tau, name="tau", kind=l_kind, extrapolate=extrapolate)
        lam_of_t = lambda t: 1.0 / tau_of_t(t)
    else:
        if not np.isscalar(k):
            raise TypeError("k must be a scalar when using (k, OH).")
        OH_of_t = _as_timefunc(OH, name="OH", kind=l_kind, extrapolate=extrapolate)
        lam_of_t = lambda t: k * OH_of_t(t)

    # RHS
    def rhs(t, C):
        return S_of_t(t) - lam_of_t(t) * C

    sol = solve_ivp(
        rhs,
        t_span=(t_eval[0], t_eval[-1]),
        y0=[C0],
        t_eval=t_eval,
        method=method,
        **kwargs,
    )

    C = sol.y[0]
    if enforce_nonnegative:
        C = np.maximum(C, 0.0)

    return OneBoxResult(t=sol.t, C=C, success=sol.success, message=sol.message)


def solve_ch4_anomaly_lifetime(
    t_eval: np.ndarray,
    C0_anom: float,
    S_anom: ScalarOrCallableOrSeries,  # source-sink imbalance anomaly (ppb/time)
    tau: ScalarOrCallableOrSeries = 10.0,  # years by default
    **kwargs,
) -> OneBoxResult:
    """
    Convenience wrapper for anomaly mode of solve_ch4_onebox.

    Parameters
    ----------
    t_eval : np.ndarray
        Time points at which to evaluate the solution.
    C0_anom : float
        Initial condition for the anomaly concentration.
    S_anom : ScalarOrCallableOrSeries
        Source-sink imbalance anomaly (ppb/time).
    tau : ScalarOrCallableOrSeries, optional
        Characteristic lifetime (years) for the anomaly. Default is 10.0 years.
    **kwargs
        Additional keyword arguments passed to scipy.solve_ivp().

    Returns
    -------
    OneBoxResult
        The result of the one-box model simulation.

    Notes
    -----
    dC'/dt = S'(t) - (1/τ) C'
    For anomalies, negatives are physically meaningful – we do not clip by default.
    If S is a sink term, multiply by -1 to enforce physicality.
    """
    return solve_ch4_onebox(
        t_eval=t_eval,
        C0=C0_anom,
        S=S_anom,
        tau=tau,
        enforce_nonnegative=False,
        **kwargs,
    )
