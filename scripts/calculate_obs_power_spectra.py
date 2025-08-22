"""Script to generate normalized and unnormalized power spectra for detrended and observed observational."""

from omegaconf import DictConfig
import hydra
from pathlib import Path
import numpy as np
import xarray as xr
from pathlib import Path
from astropy.timeseries import LombScargle
from methane_noise_forcing.io import load_observational_data
from methane_noise_forcing import detrend_obs
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s")
logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def generate_obs_power_spectra(cfg: DictConfig):
    """
    Generate normalized and unnormalized power spectra for observational data.

    Parameters
    ----------
    cfg : DictConfig
        Configuration object containing parameters for power spectrum generation.
    """
    logger.info("Starting power spectrum generation for observational data")
    # extract the site from the configuration
    assert len(cfg.sites) == 1, (
        "Only one site should be specified in the configuration."
    )
    site = cfg.sites[0]

    # Resolve paths
    output_dir = Path(cfg.paths.observations_power_spectra)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load observational data
    obs_data = load_observational_data(site, **cfg.io[site].input)

    # Calculate power spectra with original data (raw)
    logger.info("Calculating power spectrum for raw data")
    frequencies, power_spectrum_raw_unnormalized = _calculate_power_spectrum(
        obs_data["gas_age"], obs_data["ch4"]
    )
    _, power_spectrum_raw_normalized = _calculate_normalized_power_spectrum(
        obs_data["gas_age"], obs_data["ch4"]
    )

    # Calculate power spectra with detrended data
    obs_data_detrended = detrend_obs(site, obs_data)
    logger.info("Calculating power spectrum for detrended data")
    _, power_spectrum_detrended_unnormalized = _calculate_power_spectrum(
        obs_data_detrended["gas_age"], obs_data_detrended["ch4"]
    )
    _, power_spectrum_detrended_normalized = _calculate_normalized_power_spectrum(
        obs_data_detrended["gas_age"], obs_data_detrended["ch4"]
    )

    # Create xarray Dataset
    ds = xr.Dataset(
        {
            "power_spectrum_raw_unnormalized": (
                ["frequency"],
                power_spectrum_raw_unnormalized,
            ),
            "power_spectrum_raw_normalized": (
                ["frequency"],
                power_spectrum_raw_normalized,
            ),
            "power_spectrum_detrended_unnormalized": (
                ["frequency"],
                power_spectrum_detrended_unnormalized,
            ),
            "power_spectrum_detrended_normalized": (
                ["frequency"],
                power_spectrum_detrended_normalized,
            ),
        },
        coords={
            "frequency": frequencies,
        },
    )

    # Set attributes
    ds.attrs["description"] = "Power spectra of observational data"
    ds.attrs["site"] = site
    ds.attrs["mask_year"] = cfg.io[site].input.mask_year
    ds.attrs["gas_age_units"] = "years"
    ds.attrs["ch4_units"] = "ppb"
    if site == "wdc06a":
        order = 2
    else:
        order = 1
    ds.attrs["detrending_polynomial_order"] = order

    # Save the dataset
    output_path = Path(cfg.io[site].power_spectra_path)
    ds.to_netcdf(output_path, mode="w")
    logger.info(f"Power spectra saved to {output_path}")


def _calculate_normalized_power_spectrum(gas_age: np.ndarray, ch4: np.ndarray):
    """
    Use astropy's LombScargle to calculate the normalized power spectrum.

    Parameters
    ----------
    gas_age : np.ndarray
        Array of gas ages.
    ch4 : np.ndarray
        Array of CH4 concentrations.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Frequencies and normalized power spectrum.
    """
    frequencies, power_spectrum = LombScargle(gas_age, ch4).autopower(
        normalization="standard", samples_per_peak=1
    )
    return frequencies, power_spectrum


def _calculate_power_spectrum(gas_age: np.ndarray, ch4: np.ndarray):
    """
    Use astropy's LombScargle to calculate the power spectrum.

    Parameters
    ----------
    gas_age : np.ndarray
        Array of gas ages.
    ch4 : np.ndarray
        Array of CH4 concentrations.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Frequencies and power spectrum.
    """
    frequencies, power_spectrum = LombScargle(gas_age, ch4).autopower(
        normalization="psd", samples_per_peak=1
    )
    return frequencies, power_spectrum


if __name__ == "__main__":
    generate_obs_power_spectra()
