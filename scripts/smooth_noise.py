"""Script to smooth noise realizations using a firn filter."""

from omegaconf import DictConfig, OmegaConf
import hydra
from pathlib import Path
import xarray as xr
import numpy as np
import json
from methane_noise_forcing.core.firnFilter import FirnFilter
from methane_noise_forcing.naming import (
    generate_firn_smoothing_name,
    generate_noise_realization_name,
)
import logging


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def smooth_noise(cfg: DictConfig):
    """
    Smooth noise realizations using a firn filter based on the provided configuration.

    Parameters
    ----------
    cfg : omegaconf.DictConfig
        Configuration object containing parameters for noise smoothing.
        Includes:
        - noise: noise realization parameters (tau_ch4, tau_forcing, dt, etc.).
        - firn_kernel: firn filter settings (type, parameters, name).
        - paths: directories such as noise_realizations_root and firn_filters_root.

    Returns
    -------
    None
        Writes a NetCDF file to:
        <cfg.paths.noise_realizations_root>/<noise_name>/<firn_smoothing_name>/smoothed_<noise_name>.nc

    Examples
    --------
    Run with default configuration from the project root:
    $ pixi run python scripts/smooth_noise.py

    Override selected parameters with Hydra syntax:
    $ pixi run python ./scripts/smooth_noise.py \
        '+noise=white_noise_ar1,two_timescale_ar1_0-1,two_timescale_ar1_1,two_timescale_ar1_10,two_timescale_ar1_100' \
        '+firn_kernel=firn_model_wais' -m
    """
    logger.info("Starting noise smoothing process")

    # Resolve paths
    noise_name = generate_noise_realization_name(cfg)
    input_dir = Path(cfg.paths.noise_realizations_root) / noise_name
    output_dir = input_dir / generate_firn_smoothing_name(cfg)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Input directory: {input_dir}")
    logger.info(f"Output directory: {output_dir}")

    # Load noise data
    ds = xr.open_dataset(input_dir / f"{noise_name}.nc")

    # Create FirnFilter instance
    logger.info(f"Creating FirnFilter instance of {cfg.firn_kernel.type}...")
    firn_filter = FirnFilter.fit_from_config(
        kernel_type=cfg.firn_kernel.type,
        parameters=cfg.firn_kernel.parameters,
        path_root=cfg.paths.firn_filters_root
    )

    # Apply the filter to the noise data
    logger.info("Applying firn filter to noise data...")
    smoothed_data = []
    for idx_ens in ds.ensemble.values:
        smoothed_data.append(
            firn_filter.apply(
                ds["ch4"].isel(ensemble=idx_ens).values, 
                dt_series=cfg.noise.dt,
                flip_kernel=True
            )
        )
    smoothed_data = np.array(smoothed_data)

    # Save the smoothed data
    logger.info("Saving smoothed data to NetCDF...")
    smoothed_ds = xr.Dataset(
        {f"ch4": (["ensemble", "time"], smoothed_data)},
        coords={
            "ensemble": ds.ensemble,
            "time": ds.time,
        },
    )
    smoothed_ds.attrs = ds.attrs  # inherit attributes from original dataset
    smoothed_ds.attrs["firn_filter"] = json.dumps(
        OmegaConf.to_container(cfg.firn_kernel, resolve=True)
    )  # include firn filter parameters

    smoothed_ds.to_netcdf(output_dir / f"smoothed_{noise_name}.nc")

    logger.info(
        f"Smoothed noise data saved to {output_dir / f'smoothed_{noise_name}.nc'}"
    )


if __name__ == "__main__":
    smooth_noise()
