# src/methane_noise_forcing/io.py
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from pathlib import Path


def _mask_data_after_year(df: pd.DataFrame, year: float) -> pd.DataFrame:
    """
    Mask the DataFrame to exclude data after a specified year.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing a 'gas_age' column.
    year : float
        Year after which data will be excluded.

    Returns
    -------
    pd.DataFrame
        DataFrame with data masked after the specified year.
    """
    return df[df["gas_age"] <= year].copy()


def load_mitchell2013(file_path: Path, mask_year: float) -> pd.DataFrame:
    """
    Load the data from Mitchell et al. 2013 dataset (WDC05A, WDC06A, GISP2) from a CSV file.

    Parameters
    ----------
    file_path : Path
        Path to the Mitchell 2013 CSV file.
    mask_year : float
        Year to mask the data. Data before this year will be excluded.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the WDC05A, WDC06A, or GISP2 data.
    """
    df = pd.read_csv(file_path)
    df = df.rename(
        columns={
            "gas_age_ce": "gas_age",
            "mean_concentration_ppb": "ch4",
        }
    )
    df = _mask_data_after_year(df, mask_year)
    return df


def load_rhodesandbrook2019(file_path: Path, mask_year: float) -> pd.DataFrame:
    """
    Load the data from Rhodes and Brook 2019 dataset (NEEM) from a CSV file.
    Data is also from Rhodes et al. 2013.

    Parameters
    ----------
    file_path : Path
        Path to the Rhodes and Brook 2019 CSV file.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the NEEM data.
    """
    with open(file_path) as f:  # remove header rows
        for idx, line in enumerate(f):
            if line.startswith("Depth ice/snow"):
                header_row = idx
                break
    df = pd.read_csv(
        file_path,
        sep="\t",
        skiprows=header_row,  # skip everything before the header
        header=0,
    )
    df = df.rename(
        columns={
            "Age [a AD/CE] (Gas age (yr CE) (constant del...)": "gas_age",
            "CH4 [ppbv] (outliers removed, see abstract)": "ch4",
        }
    )
    df = df[["gas_age", "ch4"]].copy()
    df = _mask_data_after_year(df, mask_year)  # mask data after 1800
    return df
