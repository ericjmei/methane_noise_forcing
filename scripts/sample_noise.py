"""Script to pseudo-sample noise realizations similar to observations."""

from omegaconf import DictConfig, OmegaConf
import hydra
from pathlib import Path
import xarray as xr
import numpy as np
import json
from methane_noise_forcing.core import sample_ensemble_series
from methane_noise_forcing.naming import (
    generate_noise_realization_name,
    generate_firn_smoothing_name,
    generate_sample_name,
)
from methane_noise_forcing.io import (
    load_observational_data,
)
import logging


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def sample_noise(cfg: DictConfig):
    """
    Sample noise realizations based on the provided configuration.

    Args:
        cfg (DictConfig): Configuration object containing parameters for noise sampling.
    """
    logger.info("Starting noise sampling process")

    # Resolve paths
    noise_name = generate_noise_realization_name(cfg)
    firn_name = generate_firn_smoothing_name(cfg)
    input_dir = Path(cfg.paths.noise_realizations_root) / noise_name / firn_name
    output_dir = input_dir / generate_sample_name(cfg)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Input directory: {input_dir}")
    logger.info(f"Output directory: {output_dir}")

    # Load noise data
    logger.info(f"Loading noise data from {input_dir / f'smoothed_{noise_name}.nc'}")
    noise = xr.open_dataset(input_dir / f"smoothed_{noise_name}.nc")

    # Load observational data for sampling times
    obs_data = load_observational_data(
        cfg.sampling.site, **cfg.io[cfg.sampling.site].input
    )
    core_time = obs_data["gas_age"].values
    logger.info(
        f"Loaded observational data for {cfg.sampling.site} with {len(core_time)} core points."
    )

    # Sample noise at core points
    logger.info("Sampling noise at core points...")
    sampled_noise = sample_ensemble_series(
        da=noise["ch4"],
        core_time=core_time,
        core_offset_flag=cfg.sampling.core_offset_flag,
        series_offset=cfg.sampling.series_offset,
    )
    sampled_noise = sampled_noise.rename("ch4").to_dataset()
    sampled_noise.attrs = noise.attrs  # Preserve original attributes
    sampled_noise.attrs["sampling"] = json.dumps(
        OmegaConf.to_container(cfg.sampling, resolve=True)
    )  # include sampling config

    # save sampled noise
    sampled_noise.to_netcdf(output_dir / f"sampled_smoothed_{noise_name}.nc")
    logger.info(
        f"Sampled noise saved to {output_dir / f'sampled_smoothed_{noise_name}.nc'}"
    )


if __name__ == "__main__":
    sample_noise()
