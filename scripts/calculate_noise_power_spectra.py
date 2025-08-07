"""Script to calculate noise power spectra."""

from omegaconf import DictConfig, OmegaConf
import hydra
from pathlib import Path
import numpy as np
import xarray as xr
from astropy.timeseries import LombScargle
from methane_noise_forcing.naming import (
    generate_noise_realization_name,
    generate_firn_smoothing_name,
    generate_sample_name,
)
import logging


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def calculate_noise_power_spectra(cfg: DictConfig):
    """
    Calculate noise power spectra based on the provided configuration.

    Args:
        cfg (DictConfig): Configuration object containing parameters for power spectrum calculation.
        Must pass in cfg.noise, cfg.firn_kernel, and cfg.sampling.
    """
    logger.info("Starting noise power spectrum calculation")

    # Resolve paths
    noise_name = generate_noise_realization_name(cfg)
    firn_name = generate_firn_smoothing_name(cfg)
    sampled_name = generate_sample_name(cfg)
    input_dir = (
        Path(cfg.paths.noise_realizations_root) / noise_name / firn_name / sampled_name
    )
    output_dir = input_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Input directory: {input_dir}")
    logger.info(f"Output directory: {output_dir}")

    # Load noise data
    logger.info(
        f"Loading noise data from {input_dir / f'sampled_smoothed_{noise_name}.nc'}"
    )
    noise = xr.open_dataset(input_dir / f"sampled_smoothed_{noise_name}.nc")

    # Calculate power spectra
    logger.info("Calculating power spectra for noise realizations")
    psds = []
    normalization_factors = []
    for idx_ens in noise.ensemble.values:
        frequencies, psd = LombScargle(
            noise.sel(ensemble=idx_ens)["time"].values,
            noise.sel(ensemble=idx_ens)["ch4"].values,
        ).autopower(normalization="psd", samples_per_peak=1)
        psds.append(psd)

        # calculate normalization factor (2 / sum(residuals^2))
        residuals = noise.sel(ensemble=idx_ens)["ch4"].values - np.mean(
            noise.sel(ensemble=idx_ens)["ch4"].values
        )
        normalization_factor = 2 / np.sum(residuals**2)
        normalization_factors.append(normalization_factor)
    psds = np.array(psds)
    normalization_factors = np.array(normalization_factors)

    # Create a new dataset to store the results
    ds = xr.Dataset(
        {
            "power_spectrum": (["ensemble", "frequency"], psds),
            "normalization_factor": (["ensemble"], normalization_factors),
        },
        coords={"frequency": frequencies, "ensemble": noise.ensemble.values},
    )
    ds.attrs = noise.attrs.copy()
    ds["power_spectrum"].attrs["units"] = "ppb^2 year"
    ds["normalization_factor"].attrs["description"] = (
        "Multiplicative factor of the power spectrum to normalize it to the variance of the noise realization."
    )

    # Save the dataset
    output_path = output_dir / f"power_spectra.nc"
    ds.to_netcdf(output_path, mode="w")

    logger.info(f"Power spectra saved to {output_path}")


if __name__ == "__main__":
    calculate_noise_power_spectra()
