"""Calculates instantaneous variance of methane and forcing at various forcing timescales."""

import json
from omegaconf import DictConfig, OmegaConf
import hydra
from pathlib import Path
import numpy as np
import xarray as xr
from methane_noise_forcing import (
    closed_form,
)
from methane_noise_forcing.core import FirnFilter
import logging


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def calculate_methane_variance(cfg: DictConfig):
    """
    Calculate the instantaneous variance of methane at various forcing timescales.

    Parameters
    ----------
    cfg : omegaconf.DictConfig
        Configuration object containing parameters for variance calculation.
        Includes:
        - variance_calculation: target smoothed variance, timescale ranges,
          step sizes, and lagged covariance settings.
        - firn_kernel: firn filter settings (type, parameters).
        - paths: directories such as firn_filters_root and data_dir.

    Returns
    -------
    None
        Writes a NetCDF file to:
        <cfg.paths.data_dir>/variance_calculation/methane_variance_timescales.nc

    Examples
    --------
    Run with default configuration from the project root:
    $ pixi run python scripts/calculate_two_timescale_variance.py

    Override selected parameters with Hydra syntax:
    $ pixi run python scripts/calculate_two_timescale_variance.py \
        '+firn_kernel=firn_model_wais'
    """
    logger.info("Starting methane variance calculation")

    # Retrieve input parameters
    variance_methane_smooth = cfg.variance_calculation.get(
        "variance_methane_smooth", 100
    )  ## ppb^2
    tau_methane_min = cfg.variance_calculation.get("tau_methane_min", 7)  # years
    tau_methane_max = cfg.variance_calculation.get("tau_methane_max", 13)  # years
    dt_tau_methane = cfg.variance_calculation.get("dt_tau_methane", 0.2)  # years
    tau_forcing_min = cfg.variance_calculation.get("tau_forcing_min", 0.01)  # years
    tau_forcing_max = cfg.variance_calculation.get("tau_forcing_max", 1000)  # years
    n_tau_forcing = cfg.variance_calculation.get(
        "n_tau_forcing", 10
    )  # number of timescales per decade
    n_lag_years = cfg.variance_calculation.get("n_lag_years", 100)  # years
    logger.info(f"Variance of firn-smoothed methane: {variance_methane_smooth} ppb^2")
    logger.info(
        f"Methane timescale range: {tau_methane_min} to {tau_methane_max} years, step {dt_tau_methane} years"
    )
    logger.info(
        f"Forcing timescale range: {tau_forcing_min} to {tau_forcing_max} years, {n_tau_forcing} timescales per decade"
    )

    # Calculate timescale arrays
    taus_methane = np.arange(
        tau_methane_min, tau_methane_max + dt_tau_methane, dt_tau_methane
    )
    a, b = np.log10(tau_forcing_min), np.log10(tau_forcing_max)
    taus_forcing = np.logspace(a, b, num=int((b - a) * n_tau_forcing) + 1)
    logger.info(
        f"Calculated {len(taus_methane)} methane timescales and {len(taus_forcing)} forcing timescales."
    )

    # Import firn filter
    firn_filter = FirnFilter.fit_from_config(
        kernel_type=cfg.firn_kernel.type,
        parameters=cfg.firn_kernel.parameters,
        path_root=cfg.paths.firn_filters_root,
    )
    logger.info(
        f"Loaded firn filter {cfg.firn_kernel.name} of type {cfg.firn_kernel.type} for variance calculation."
    )

    # Calculate firn self-overlap
    dt = firn_filter.dt  # years
    lags = np.arange(0, n_lag_years / dt, 1)  # years
    kernel_normalized = firn_filter.kernel / np.sum(firn_filter.kernel)
    c_taus = closed_form.calculate_kernel_self_lag(
        kernel=kernel_normalized,
        lags=lags,
    )
    logger.info("Calculated firn filter self-overlap.")

    # Generate autocorrelation at range of timescales
    autocorrelation_two_timescale = xr.DataArray(
        np.zeros((len(taus_methane), len(taus_forcing), len(lags))),
        dims=["taus_methane", "taus_forcing", "lags"],
        coords={
            "taus_methane": taus_methane,
            "taus_forcing": taus_forcing,
            "lags": lags,
        },
    )
    for tau_methane in taus_methane:
        for tau_forcing in taus_forcing:
            autocorrelation = closed_form.calculate_two_timescale_autocorrelation(
                phi_0=np.exp(-dt / tau_methane),
                phi_1=np.exp(-dt / tau_forcing),
                lags=lags,
            )
            autocorrelation_two_timescale.loc[
                dict(
                    taus_methane=tau_methane,
                    taus_forcing=tau_forcing,
                )
            ] = autocorrelation
    logger.info("Generated autocorrelation functions for all timescale combinations.")

    # Calculate variance of methane and forcing at each timescale
    atmospheric_methane_variances = xr.DataArray(
        np.zeros((len(taus_methane), len(taus_forcing))),
        dims=["taus_methane", "taus_forcing"],
        coords={
            "taus_methane": taus_methane,
            "taus_forcing": taus_forcing,
        },
    )
    forcing_variances = xr.DataArray(
        np.zeros((len(taus_methane), len(taus_forcing))),
        dims=["taus_methane", "taus_forcing"],
        coords={
            "taus_methane": taus_methane,
            "taus_forcing": taus_forcing,
        },
    )
    for tau_methane in taus_methane:
        for tau_forcing in taus_forcing:
            variance_ratio_filtered_to_atmospheric = (
                closed_form.calculate_variance_ratio_filtered_to_instantaneous(
                    c_taus=c_taus,
                    instantaneous_autocorrelations=autocorrelation_two_timescale.loc[
                        dict(
                            taus_methane=tau_methane,
                            taus_forcing=tau_forcing,
                        )
                    ],
                )
            )
            variance_atmospheric_methane = (
                variance_methane_smooth / variance_ratio_filtered_to_atmospheric
            )
            atmospheric_methane_variances.loc[
                dict(
                    taus_methane=tau_methane,
                    taus_forcing=tau_forcing,
                )
            ] = variance_atmospheric_methane

            variance_ratio_methane_to_forcing = closed_form.calculate_two_timescale_variance_ratio_methane_to_ar1_forcing(
                tau_methane=tau_methane,
                tau_forcing=tau_forcing,
            )
            variance_forcing = (
                variance_atmospheric_methane / variance_ratio_methane_to_forcing
            )
            forcing_variances.loc[
                dict(
                    taus_methane=tau_methane,
                    taus_forcing=tau_forcing,
                )
            ] = variance_forcing
    logger.info(
        "Calculated atmospheric methane variance and forcing variance for all timescale combinations."
    )

    # Save results: methane variance and forcing variance (taus_methane, taus_forcing)
    ds_save = xr.Dataset(
        {
            "atmospheric_methane_variance": atmospheric_methane_variances,
            "forcing_variance": forcing_variances,
        }
    )
    ds_save.attrs["cfgs"] = json.dumps(
        OmegaConf.to_container(cfg.variance_calculation, resolve=True)
    )  # include variance calculation config
    ds_save.atmospheric_methane_variance.attrs["units"] = "ppb^2"
    ds_save.forcing_variance.attrs["units"] = "(ppb/yr)^2"
    output_dir = Path(cfg.paths.data_dir) / "variance_calculation"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "methane_variance_timescales.nc"
    ds_save.to_netcdf(output_path)
    logger.info(f"Saved methane variance results to {output_path}")


if __name__ == "__main__":
    calculate_methane_variance()
