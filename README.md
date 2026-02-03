# Methane Noise Forcing
This repository contains all code for analyzing and visualizing unforced methane variability on decadal to centennial scales.

## Getting started

### Setting up the repository
Clone the repository
```bash
git clone https://github.com/ericjmei/methane_noise_forcing
```

Install dependencies using pixi:
```bash
pixi install
```
TODO: make a requirements.txt for pip and conda

Configs and file paths are managed by [hydra](https://hydra.cc/) in the `configs/` directory (See the [hydra documentation](https://hydra.cc/docs/intro/) for more information). Change the file paths in `configs/config.yaml` to match your directory structure:
```yaml
# example
paths:
  project_root: "/home/disk/p/emei/repos/methane_noise_forcing" # change this path to the cloned project directory
  data_dir: "${paths.project_root}/data"
  configs_dir: "${paths.project_root}/configs"

  figures_dir: "/home/disk/hermes3/emei/methane_noise_forcing/figures" # change this path
  
  noise_realizations_root: "/home/disk/hermes3/emei/methane_noise_forcing/noise_realizations" # change this path
  
  observations_root: "${paths.data_dir}/observations"
  observations_cultivated: "${paths.observations_root}/cultivated"
  observations_power_spectra: "${paths.observations_root}/power_spectra"

  firn_filters_root: "${paths.data_dir}/firn_filters"
```
The most common paths to change are `project_root/`, `figures_dir/`, and `noise_realizations_root/`. Small data files can be stored in `data_dir/`, but figures and large datasets (noise_realizations) should be stored elsewhere unless you modify the `.gitignore` file to include them.

## Running the forward simulations
Analysis has been conducted for ice core methane records from from wdc05a (Michell et al. 2011), wdc06a (Mitchell et al. 2013), GISP2 (Michell et al. 2013), and NEEM (Rhodes et al. 2013, Rhodes et al. 2019). wdc05a and wdc06a are both from WAIS.

Forward model simulations can be generated for an AR(1) forced by white noise (`white_noise_ar1`) or an AR(1) that is itself forced by an AR(1) forced by white noise (`two_timescale_ar1`). Simulations can then be smoothed by their respective firn filters and sampled to produce pseudo-observations that resemble the observations.

The code blocks below show how to generate this analysis for the wdc06a core for the following simulations:
- AR(1) forced by white noise
- AR(1) forced by AR(1)s of 0.1, 1, 10, and 100 year decorrelation timescales.

The hydra package makes it really easy to run these experiments via the "multirun" feature enabled by the `-m` flag. If running the analysis with other cores, replace instances of `wdc06a` or `wais` with the appropriate core name.

If not using pixi, you can run the analysis with the same code by omitting `pixi run` from the beginning of each of the prompts. You may need to remove the quotes from the hydra config arguments.

### Generating noise realizations
Global arguments for generating noise realizations, such as the time step and the number of ensemble members, are found in `configs/config.yaml`. The methane lifetime, forcing timescales, and the target atmospheric (instantaneous) methane variance, along with other miscellaneous parameters can be specified in the configs in `configs/noise/`. To generate noise realizations, run:

```bash
pixi run python ./scripts/generate_white_noise_ar1.py \
'+noise=white_noise_ar1'
pixi run python ./scripts/generate_two_timescale_ar1.py \
'+noise=two_timescale_ar1_0-1,two_timescale_ar1_1,two_timescale_ar1_10,two_timescale_ar1_100' -m
```

Each noise realization is stored in its own directory in the path specified by `cfg.paths.noise_realizations_root` in `configs/config.yaml`.

### Smoothing and sampling
Configs for smoothing via firn filtering can be found in `configs/firn_kernel/`. The configs that start with `firn_model_` direct the scripts to look for the empirical firn filters in the `data/firn_filters/` directory. Configs in `configs/sampling/` direct the scripts to look for the sampling structure of the observations. These actions can be performed by:

```bash
pixi run python ./scripts/smooth_noise.py \
'+noise=white_noise_ar1,two_timescale_ar1_0-1,two_timescale_ar1_1,two_timescale_ar1_10,two_timescale_ar1_100' \
'+firn_kernel=firn_model_wais' -m
pixi run python ./scripts/sample_noise.py \
'+noise=white_noise_ar1,two_timescale_ar1_0-1,two_timescale_ar1_1,two_timescale_ar1_10,two_timescale_ar1_100' \
'+firn_kernel=firn_model_wais' '+sampling=wdc06a' -m
```

Quick plug: notice how hydra automatically sweeps all permutations of the configs with the `-m` flag!

## Figures
Again, figures are stored in `cfg.paths.figures_dir` in `configs/config.yaml`. Figures are mostly generated in the `notebooks/figures_for_publication/` directory. They may rely on configs to read in data from specific forward model runs. For example:

```python
filter_name = "wais"
site_name = "wdc06a"

with initialize(version_base=None, config_path="../../configs"):
    cfg = compose(config_name="config", overrides=[
        f"+noise=white_noise_ar1",
        f"+firn_kernel=firn_model_{filter_name}",
        f"+sampling={site_name}"
    ])
```

The `overrides` argument in `compose()` acts similarly to the command line arguments passed to the scripts.

ALEX: I will put your figures in `notebooks/figures_for_PCC/` in the branch `PCC_figures` whenever I make them.
