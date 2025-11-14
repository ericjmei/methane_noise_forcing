# src/methane_noise_forcing/isotope.py
# -*- coding: utf-8 -*-
"""
Functionality for isotope-related calculations.
"""

import numpy as np

# ---- Reference standards (override if you prefer different evaluated values) ----
# 13C/12C ratio of VPDB
R13C_VPDB = 0.0112372  # ~1.12372e-2
# D/H ratio of VSMOW
RD_VSMOW = 155.76e-6  # ~1.5576e-4


# -----------------------------------------------------------------------------
# Core ratio <-> delta helpers
# -----------------------------------------------------------------------------
def ratio_from_delta(delta, R_standard, scale=1000.0):
    """
    Convert delta (per mil by default) to isotope ratio R.

    Parameters
    ----------
    delta : array-like or float
        δ value (e.g., in ‰ if scale=1000).
    R_standard : float
        Reference ratio (e.g., R13C_VPDB, RD_VSMOW).
    scale : float, default 1000.0
        Scaling for delta (‰ => 1000; % => 100).

    Returns
    -------
    R : like(delta)
        Isotope ratio of the sample.
    """
    return R_standard * (1.0 + np.asarray(delta) / scale)


def delta_from_ratio(R, R_standard, scale=1000.0):
    """
    Convert isotope ratio R to delta (per mil by default).

    Parameters
    ----------
    R : array-like or float
        Isotope ratio of the sample.
    R_standard : float
        Reference ratio (e.g., R13C_VPDB, RD_VSMOW).
    scale : float, default 1000.0
        Scaling for delta (‰ => 1000; % => 100).

    Returns
    -------
    delta : like(R)
        δ value (e.g., in ‰ if scale=1000).
    """
    R = np.asarray(R)
    return scale * (R / R_standard - 1.0)


# -----------------------------------------------------------------------------
# From bulk + rare isotopologue to isotope ratio R (rare/abundant)
# -----------------------------------------------------------------------------
def ratio_from_bulk_rare(
    bulk,
    rare,
    *,
    bulk_includes_rare: bool = True,
    combinatorial: float = 1.0,
):
    """
    Compute the isotope ratio R = n_rare / n_abundant (up to a combinatorial factor)
    from bulk and rare isotopologue concentrations.

    - In isotope geochemistry, 'bulk' methane is typically the *total* methane,
      which includes both abundant and rare isotopologues. If that's your case,
      set bulk_includes_rare=True (default).
    - If your 'bulk' excludes the rare isotopologue (i.e., it already *is* the
      abundant pool), set bulk_includes_rare=False.

    For δ13C in methane:
      R(13C/12C) = [CH4 with 13C] / [CH4 with 12C]
      => combinatorial = 1.0

    For δD in methane (using CH3D as the rare species and CH4 as abundant):
      R(D/H) ≈ [CH3D] / [4 * CH4]  (low D approximation; ignores multi-D species)
      => set combinatorial = 4.0

    Parameters
    ----------
    bulk : array-like or float
        Bulk methane concentration (units arbitrary but consistent).
    rare : array-like or float
        Concentration of the 'rare' isotopologue (e.g., 13CH4 or CH3D).
    bulk_includes_rare : bool, default True
        Whether `bulk` is the total including `rare`.
    combinatorial : float, default 1.0
        Factor mapping the isotopologue ratio to the isotope-atom ratio.
        Use 1.0 for 13C/12C; 4.0 for D/H in methane with CH3D as 'rare'.

    Returns
    -------
    R : like(bulk, rare)
        Isotope ratio (rare/abundant) adjusted by `combinatorial`.
    """
    bulk = np.asarray(bulk)
    rare = np.asarray(rare)

    abundant = (bulk - rare) if bulk_includes_rare else bulk
    # small numerical guard to avoid division by zero (keeps xarray dtypes)
    eps = np.finfo(float).tiny
    abundant_safe = np.where(abundant == 0, eps, abundant)

    return rare / (combinatorial * abundant_safe)


def delta_from_bulk_rare(
    bulk,
    rare,
    *,
    R_standard: float,
    bulk_includes_rare: bool = True,
    combinatorial: float = 1.0,
    scale: float = 1000.0,
):
    """
    Directly compute δ from bulk + rare.

    Parameters
    ----------
    bulk, rare : array-like or float
        See `ratio_from_bulk_rare`.
    R_standard : float
        Reference ratio (e.g., R13C_VPDB, RD_VSMOW).
    bulk_includes_rare : bool, default True
    combinatorial : float, default 1.0
        See `ratio_from_bulk_rare`.
    scale : float, default 1000.0
        δ scaling (‰ by default).

    Returns
    -------
    delta : like(bulk, rare)
        δ value.
    """
    R = ratio_from_bulk_rare(
        bulk,
        rare,
        bulk_includes_rare=bulk_includes_rare,
        combinatorial=combinatorial,
    )
    return delta_from_ratio(R, R_standard, scale=scale)


# -----------------------------------------------------------------------------
# Inverse: from bulk + delta to rare isotopologue concentration
# -----------------------------------------------------------------------------
def rare_from_bulk_delta(
    bulk,
    delta,
    *,
    R_standard: float,
    bulk_includes_rare: bool = True,
    combinatorial: float = 1.0,
    scale: float = 1000.0,
):
    """
    Solve for the rare isotopologue concentration from (bulk, δ).

    We first get R from δ. Then:
      If bulk_includes_rare:
         For general combinatorial factor `g`:
           R = rare / (g * (bulk - rare))
           => rare = (g * R / (1 + g * R)) * bulk
      If bulk_excludes_rare:
           R = rare / (g * bulk) => rare = g * R * bulk

    Parameters
    ----------
    bulk : array-like or float
        Bulk concentration (total if bulk_includes_rare=True; otherwise abundant).
    delta : array-like or float
        δ value (‰ if scale=1000).
    R_standard : float
        Reference isotope ratio.
    bulk_includes_rare : bool, default True
    combinatorial : float, default 1.0
        1.0 for 13C/12C; 4.0 for D/H in methane (CH3D rare).
    scale : float, default 1000.0

    Returns
    -------
    rare : like(bulk, delta)
        Rare isotopologue concentration.
    """
    bulk = np.asarray(bulk)
    R = ratio_from_delta(delta, R_standard, scale=scale)
    g = combinatorial

    if bulk_includes_rare:
        # rare = (g*R / (1 + g*R)) * bulk
        return (g * R / (1.0 + g * R)) * bulk
    else:
        # bulk is abundant
        return (g * R) * bulk


# -----------------------------------------------------------------------------
# Convenient methane-specific wrappers
# -----------------------------------------------------------------------------
def delta13C_from_bulk_rare(
    bulk_CH4,
    rare_13CH4,
    *,
    bulk_includes_rare: bool = True,
    R_standard: float = R13C_VPDB,
    scale: float = 1000.0,
):
    """
    δ13C-CH4 (‰ VPDB by default) from total CH4 and 13CH4 concentrations.
    """
    return delta_from_bulk_rare(
        bulk_CH4,
        rare_13CH4,
        R_standard=R_standard,
        bulk_includes_rare=bulk_includes_rare,
        combinatorial=1.0,
        scale=scale,
    )


def rare_13CH4_from_bulk_delta13C(
    bulk_CH4,
    delta13C,
    *,
    bulk_includes_rare: bool = True,
    R_standard: float = R13C_VPDB,
    scale: float = 1000.0,
):
    """
    Inverse: 13CH4 concentration from (total CH4, δ13C).
    """
    return rare_from_bulk_delta(
        bulk_CH4,
        delta13C,
        R_standard=R_standard,
        bulk_includes_rare=bulk_includes_rare,
        combinatorial=1.0,
        scale=scale,
    )


def deltaD_from_bulk_rare_CH3D(
    bulk_CH4,
    rare_CH3D,
    *,
    bulk_includes_rare: bool = True,
    R_standard: float = RD_VSMOW,
    scale: float = 1000.0,
    hydrogen_sites: int = 4,
):
    """
    δD-CH4 (‰ VSMOW by default) from total CH4 and CH3D concentrations.

    Uses the low-D approximation: R(D/H) ≈ [CH3D] / (n_H * CH4_abundant),
    with n_H = 4 for methane. Multi-deuterated species are neglected.
    """
    return delta_from_bulk_rare(
        bulk_CH4,
        rare_CH3D,
        R_standard=R_standard,
        bulk_includes_rare=bulk_includes_rare,
        combinatorial=float(hydrogen_sites),
        scale=scale,
    )


def rare_CH3D_from_bulk_deltaD(
    bulk_CH4,
    deltaD,
    *,
    bulk_includes_rare: bool = True,
    R_standard: float = RD_VSMOW,
    scale: float = 1000.0,
    hydrogen_sites: int = 4,
):
    """
    Inverse: CH3D concentration from (total CH4, δD) under the low-D approximation.

    rare = (n_H * R / (1 + n_H * R)) * bulk, if bulk includes rare.
    """
    return rare_from_bulk_delta(
        bulk_CH4,
        deltaD,
        R_standard=R_standard,
        bulk_includes_rare=bulk_includes_rare,
        combinatorial=float(hydrogen_sites),
        scale=scale,
    )
