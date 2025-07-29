# src/methane_noise_forcing/noise/__init__.py
# -*- coding: utf-8 -*-
"""Functionality for generating and simulating noise processes."""

from .two_timescale_ar1 import (
    TwoTimescaleAR1Params,
    calculate_two_timescale_ar1_params,
    simulate_two_timescale_ar1,
)
from .white_noise_ar1 import (
    WhiteNoiseAR1Params,
    calculate_white_noise_ar1_params,
    simulate_white_noise_ar1,
)

__all__ = [
    "TwoTimescaleAR1Params",
    "calculate_two_timescale_ar1_params",
    "simulate_two_timescale_ar1",
    "WhiteNoiseAR1Params",
    "calculate_white_noise_ar1_params",
    "simulate_white_noise_ar1",
]
