"""Script to generate two-timescale AR(1) noise realizations for methane forcing studies."""

from omegaconf import DictConfig
import hydra
from pathlib import Path
import xarray as xr
import numpy as np
from methane_noise_forcing.noise import simulate_two_timescale_ar1
from methane_noise_forcing.naming import generate_noise_realization_name
import logging


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def generate_two_timescale(cfg: DictConfig):
    """
    Generate noise with two timescales based on the provided configuration.

    Args:
        cfg (DictConfig): Configuration object containing parameters for noise generation.
    """
    logger.info("Starting two-timescale AR(1) noise generation")
    logger.info(f"Configuration: tau_ch4={cfg.noise.tau_ch4}, tau_forcing={cfg.noise.tau_forcing}, "
                f"variance_ch4={cfg.noise.variance_ch4}, duration={cfg.noise.duration_timeseries}yr, "
                f"n_ens={cfg.noise.n_ens}")
    
    # Resolve paths
    noise_name = generate_noise_realization_name(cfg)
    output_dir = Path(cfg.paths.noise_realizations_root) / noise_name
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    # Generate noise data
    n_steps = int(cfg.noise.duration_timeseries / cfg.noise.dt)
    logger.info(f"Generating {n_steps} time steps with dt={cfg.noise.dt}")
    
    time = np.arange(n_steps) * cfg.noise.dt
    logger.info("Simulating two-timescale AR(1) process...")
    
    ch4_ens, forcing_ens = simulate_two_timescale_ar1(
        tau_x=cfg.noise.tau_ch4,
        tau_eta=cfg.noise.tau_forcing,
        variance_x=cfg.noise.variance_ch4,
        dt=cfg.noise.dt,
        n_steps=n_steps,
        n_ens=cfg.noise.n_ens,
    )
    logger.info(f"Generated ensemble arrays with shape: {ch4_ens.shape}")

    # Create xarray Dataset
    logger.info("Creating xarray Dataset...")
    ds = xr.Dataset(
        {
            "ch4": (["ensemble", "time"], ch4_ens),
            "forcing": (["ensemble", "time"], forcing_ens),
        },
        coords={
            "ensemble": np.arange(cfg.noise.n_ens),
            "time": time,
        },
    )
    # Set attributes
    ds.attrs["tau_ch4"] = cfg.noise.tau_ch4
    ds.attrs["tau_forcing"] = cfg.noise.tau_forcing
    ds.attrs["variance_ch4"] = cfg.noise.variance_ch4
    ds.attrs["dt"] = cfg.noise.dt
    ds.attrs["time_units"] = cfg.noise.time_units
    ds.attrs["description"] = (
        "Two timescale AR(1) noise realization for instantaneous methane"
    )

    # Save the dataset
    ext = "nc" if cfg.noise.output.format == "netcdf" else "h5"
    filename = noise_name + f".{ext}"
    output_path = output_dir / filename
    logger.info(f"Saving dataset to: {output_path}")
    
    try:
        if cfg.noise.output.format == "netcdf":
            ds.to_netcdf(output_path, mode="w")
        elif cfg.noise.output.format == "hdf5":
            ds.to_hdf5(output_path, mode="w")
        else:
            logger.error(f"Unsupported output format: {cfg.noise.output.format}")
            raise ValueError(f"Unsupported output format: {cfg.noise.output.format}")
        
        logger.info(f"Successfully saved dataset with {cfg.noise.output.format} format")
        logger.info(f"File size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")
        
    except Exception as e:
        logger.error(f"Failed to save dataset: {e}")
        raise
    
    logger.info("Two-timescale AR(1) noise generation completed successfully")


if __name__ == "__main__":
    generate_two_timescale()