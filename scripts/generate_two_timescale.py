"""Script to generate two-timescale AR(1) noise realizations for methane forcing studies."""

from omegaconf import DictConfig
import hydra
from pathlib import Path
import xarray as xr
import numpy as np
from methane_noise_forcing import simulate_two_timescale_ar1


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def generate_two_timescale(cfg: DictConfig):
    """
    Generate noise with two timescales based on the provided configuration.

    Args:
        cfg (DictConfig): Configuration object containing parameters for noise generation.
    """
    # Resolve paths
    project_root = Path(cfg.paths.project_root)
    output_dir = project_root / cfg.paths.noise_realizations_raw
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate noise data
    N = int(cfg.noise.duration_timeseries / cfg.noise.dt)
    time = np.arange(N) * cfg.noise.dt
    ch4_ens, forcing_ens = simulate_two_timescale_ar1(
        tau_x=cfg.noise.tau_ch4,
        tau_eta=cfg.noise.tau_forcing,
        variance_x=cfg.noise.variance_ch4,
        dt=cfg.noise.dt,
        N=N,
        n_ens=cfg.noise.n_ens,
    )

    # Create xarray Dataset
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
    ds.attrs["tau_ch4"] = cfg.noise.tau_ch4
    ds.attrs["tau_forcing"] = cfg.noise.tau_forcing
    ds.attrs["variance_ch4"] = cfg.noise.variance_ch4
    ds.attrs["dt"] = cfg.noise.dt
    ds.attrs["time_units"] = cfg.noise.time_units
    ds.attrs["description"] = (
        "Two timescale AR(1) noise realization for instantaneous methane"
    )

    # Save the dataset
    filename = _generate_filename(cfg)
    output_path = output_dir / filename
    if cfg.noise.output.format == "netcdf":
        ds.to_netcdf(output_path, mode="w")
    elif cfg.noise.output.format == "hdf5":
        ds.to_hdf5(output_path, mode="w")


def _generate_filename(cfg: DictConfig) -> str:
    """Generate descriptive filename from config parameters."""
    ext = "nc" if cfg.noise.output.format == "netcdf" else "h5"

    return (
        f"two_timescale_ar1_"
        f"tau{cfg.noise.tau_ch4}_"
        f"tau{cfg.noise.tau_forcing}_"
        f"var{cfg.noise.variance_ch4}_"
        f"{cfg.noise.duration_timeseries}yr_"
        f"{cfg.noise.n_ens}.{ext}"
    )


if __name__ == "__main__":
    generate_two_timescale()
