"""Script to generate white noise AR(1) realizations for methane forcing studies."""

from omegaconf import DictConfig
import hydra
from pathlib import Path
import xarray as xr
import numpy as np
from methane_noise_forcing.noise import simulate_white_noise_ar1, calculate_white_noise_ar1_params
import logging


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def generate_white_noise_ar1(cfg: DictConfig):
    """
    Generate white noise AR(1) noise realizations based on the provided configuration.

    Args:
        cfg (DictConfig): Configuration object containing parameters for noise generation.
    """
    logger.info("Starting white noise AR(1) noise generation")
    logger.info(f"Configuration: tau_ch4={cfg.noise.tau_ch4}, variance_ch4={cfg.noise.variance_ch4}, "
                f"duration={cfg.noise.duration_timeseries}yr, n_ens={cfg.noise.n_ens}")
    
    # Resolve paths
    output_dir = Path(cfg.paths.noise_realizations_raw)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    # Generate noise data
    n_steps = int(cfg.noise.duration_timeseries / cfg.noise.dt)
    logger.info(f"Generating {n_steps} time steps with dt={cfg.noise.dt}")
    
    time = np.arange(n_steps) * cfg.noise.dt
    
    logger.info("Calculating AR(1) parameters...")
    params = calculate_white_noise_ar1_params(
        tau_years=cfg.noise.tau_ch4,
        dt_years=cfg.noise.dt,
        variance=cfg.noise.variance_ch4,
    )
    logger.info(f"AR(1) parameters: phi={params.phi:.4f}, sigma_eps={params.sigma_eps:.4f}")
    
    logger.info("Simulating white noise AR(1) process...")
    ch4_ens, forcing_ens = simulate_white_noise_ar1(
        phi=params.phi,
        sigma_eps=params.sigma_eps,
        n_steps=n_steps,
        n_ens=cfg.noise.n_ens,
    )
    logger.info(f"Generated ensemble arrays with shape: {ch4_ens.shape}")

    # Create xarray Dataset
    logger.info("Creating xarray Dataset...")
    ds = xr.Dataset(
        {
            "ch4_ens": (["ensemble", "time"], ch4_ens),
            "forcing_ens": (["ensemble", "time"], forcing_ens),
        },
        coords={
            "ensemble": np.arange(cfg.noise.n_ens),
            "time": time,
        },
    )
    
    # Set attributes
    ds.attrs.update({
        "tau_ch4": cfg.noise.tau_ch4,
        "variance_ch4": cfg.noise.variance_ch4,
        "dt": cfg.noise.dt,
        "time_units": cfg.noise.time_units,
        "description": "White noise AR(1) noise realization for methane forcing"
    })

    # Save the dataset
    filename = _generate_filename(cfg)
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
    
    logger.info("White noise AR(1) noise generation completed successfully")


def _generate_filename(cfg: DictConfig) -> str:
    """Generate descriptive filename from config parameters."""
    ext = "nc" if cfg.noise.output.format == "netcdf" else "h5"

    filename = (
        f"white_noise_ar1_"
        f"tau-ch4-{cfg.noise.tau_ch4}_"
        f"var-ch4-{cfg.noise.variance_ch4}_"
        f"{cfg.noise.duration_timeseries}yr_"
        f"{cfg.noise.n_ens}ens.{ext}"
    )
    
    logger.debug(f"Generated filename: {filename}")
    return filename


if __name__ == "__main__":
    generate_white_noise_ar1()