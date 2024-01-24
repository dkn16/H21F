# GP21cmFg

An Gaussian Process based 21-cm foreground separation code, including codes for sampling, separation and evaluation.

## Introduction

## Presiquities

JAX, numpyro, ArviZ, GP4IM (For evaluation only) and Kymatio (For evaluation only).

## Usage

### Data

This code has some specifically designed reading function to read the data in [this link](https://www.dropbox.com/sh/9zftczeypu7xgt3/AABiiBw_0SBPrLgSHsjiISz8a?dl=0), as is used in [Soares et al. 2022](http://dx.doi.org/10.1093/mnras/stab2594). If you're using this dataset, remember to change the data directory in each notebook or script.

### Sampling and Component Separation

For **CP** (conventional) method, the sampling and component separation are integrated into one notebook ([CP_sampling_prediction.ipynb](https://github.com/dkn16/GP21cmFg/blob/main/CP_sampling_prediction.ipynb)), as it is fast.

For **HGP** and **NP**, sampling codes are in the `Sampling` folder. After getting posterier samples, using notebooks in `Prediction` folder to do the component separation.

### Post analysis and evaluation

See notebooks in `Evaluation` folder.
