# GP21cmFg

An Gaussian Process based 21-cm foreground separation code, including codes for sampling, separation and evaluation.

## Introduction

## Presiquities

[JAX](https://jax.readthedocs.io), [numpyro](https://num.pyro.ai), [ArviZ](https://python.arviz.org/), [GPR4IM](https://github.com/paulassoares/gpr4im) (For evaluation only) and [Kymatio](https://www.kymat.io) (For evaluation only).

## Usage

### Data

This code has some specifically designed reading function to read the data in [this link](https://www.dropbox.com/sh/9zftczeypu7xgt3/AABiiBw_0SBPrLgSHsjiISz8a?dl=0), as is used in [Soares et al. 2022](http://dx.doi.org/10.1093/mnras/stab2594). If you're using this dataset, remember to change the data directory in each notebook or script.

### Sampling and Component Separation

For **CP** (conventional) method, the sampling and component separation are integrated into one notebook ([CP_sampling_prediction.ipynb](https://github.com/dkn16/GP21cmFg/blob/main/CP_sampling_prediction.ipynb)), as it is fast.

For **HGP** and **NP**, sampling codes are in the `Sampling` folder. After running the scripts there, one is supposed to get a pickled numpyro MCMC object. Be careful about the **directory** to saving the object.

After getting the object, using notebooks in `Prediction` folder to do the component separation.

suffix `_nopol` means only two kernels are adopted, one for modeling the foreground and one for modeling the 21 cm emission. `_pol` means three kernels are used, for foreground, polarization leakage and 21 cm, respectively.

bias correction. (pending)

### Post analysis and evaluation

See notebooks in `Evaluation` folder.

- `Visualization`: draw images of recovered 21 cm signal.

- `Residuals`:

- `Powerspectrum`:

- `ST`:
