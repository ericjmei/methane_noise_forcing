# src/methane_noise_forcing/plotting/constants.py
# -*- coding: utf-8 -*-
"""
Constants (colors, settings) for plotting.
"""

# rcparams
large_font = 8.5
small_font = 8.5
matplotlib_rcparams = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial"],
    "font.size": large_font,
    "axes.titlesize": large_font,
    "axes.labelsize": large_font,
    "xtick.labelsize": small_font,
    "ytick.labelsize": small_font,
    "legend.fontsize": small_font,
}

# figure width (inches)
fig_width_options = {
    "one_column": 3.42,
    "two_column_small": 4.49,
    "two_column_large": 7
}

# colors (using bright and vibrant schemes from https://sronpersonalpages.nl/~pault/)
colors_bright = {
    "blue": "#4477AA",
    "cyan": "#66CCEE",
    "green": "#228833",
    "yellow": "#CCBB44",
    "red": "#EE6677",
    "purple": "#AA3377",
    "grey": "#BBBBBB",
}
colors_vibrant = {
    "blue": "#0077BB",
    "cyan": "#33BBEE",
    "teal": "#009988",
    "orange": "#EE7733",
    "red": "#CC3311",
    "magenta": "#EE3377",
    "grey": "#BBBBBB",
}