# src/methane_noise_forcing/naming.py
# -*- coding: utf-8 -*-
"""Functionality for generating file and directory names based on simulation parameters.

This module provides functions to create standardized file names for noise simulations,
firn smoothing, and other related processes. The file names are constructed
using the parameters provided in the configuration object, ensuring consistency
"""

import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def generate_noise_realization_name(cfg) -> str:
    if cfg.noise.type == "white_noise_ar1":
        return generate_white_noise_ar1_name(cfg)
    elif cfg.noise.type == "two_timescale_ar1":
        return generate_two_timescale_ar1_name(cfg)
    else:
        logger.error(f"Unsupported noise type: {cfg.noise.type}")
        raise ValueError(f"Unsupported noise type: {cfg.noise.type}")


def generate_white_noise_ar1_name(cfg) -> str:
    """
    Generate a file or directory name for white noise AR(1) noise simulation.

    Parameters
    ----------
    cfg : Config
        Configuration object containing noise parameters.

    Returns
    -------
    str
        Formatted file name for the white noise AR(1) noise simulation.
    """
    name = (
        f"white_noise_ar1_"
        f"tau-ch4-{cfg.noise.tau_ch4}_"
        f"var-ch4-{cfg.noise.variance_ch4}_"
        f"{cfg.noise.duration_timeseries}yr_"
        f"{cfg.noise.n_ens}ens"
    )

    return name


def generate_two_timescale_ar1_name(cfg) -> str:
    """
    Generate a file or directory name for two-timescale AR(1) noise simulation.

    Parameters
    ----------
    cfg : Config
        Configuration object containing noise parameters.

    Returns
    -------
    str
        Formatted file name for the two-timescale AR(1) noise simulation.
    """
    name = (
        f"two_timescale_ar1_"
        f"tau-ch4-{cfg.noise.tau_ch4}_"
        f"tau-forcing-{cfg.noise.tau_forcing}_"
        f"var-ch4-{cfg.noise.variance_ch4}_"
        f"{cfg.noise.duration_timeseries}yr_"
        f"{cfg.noise.n_ens}ens"
    )

    return name


def generate_firn_smoothing_name(cfg) -> str:
    """
    Generate a file or directory name for firn smoothing.

    Parameters
    ----------
    cfg : Config
        Configuration object containing firn smoothing parameters.

    Returns
    -------
    str
        Formatted file name for the firn smoothing.
    """
    if cfg.firn_kernel.type == "firn_model":
        name = f"firn_kernel-{cfg.firn_kernel.type}_name-{cfg.firn_kernel.name}"
    else:
        name = (
            f"firn_"
            f"kernel-{cfg.firn_kernel.type}_"
            f"name-{cfg.firn_kernel.name}_"
            f"mode-{cfg.firn_kernel.parameters.mode}_"
            f"fwhm-{cfg.firn_kernel.parameters.fwhm}"
        )

    return name


def generate_sample_name(cfg) -> str:
    """
    Generate a file or directory name for sampled noise realizations.

    Parameters
    ----------
    cfg : Config
        Configuration object containing sampling parameters.

    Returns
    -------
    str
        Formatted file name for the sampled noise realizations.
    """
    name = (
        f"sampled_"
        f"site-{cfg.sampling.site}_"
        f"core-offset-{cfg.sampling.core_offset_flag}_"
        f"series-offset-{cfg.sampling.series_offset}"
    )

    return name
