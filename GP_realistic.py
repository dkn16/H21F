#first import dependencies, mainly numpyro
import argparse
import os
import time
from jax.config import config
config.update("jax_enable_x64", True)
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import jax
from jax import vmap
import jax.numpy as jnp
import jax.random as random

import numpyro
import numpyro.distributions as dist
from numpyro.infer import (
    MCMC,
    NUTS,
    init_to_feasible,
    init_to_median,
    init_to_sample,
    init_to_uniform,
    init_to_value,
)
import pandas as pd
data = pd.read_pickle('/home/dkn16/data.pkl')
FGnopol = data.beam.foregrounds.all_nopol
FGpol = data.beam.foregrounds.pleak
HI = data.beam.HI
noise = data.beam.noise
freqs = data.freqs
#matplotlib.use("Agg")  # noqa: E402
# create artificial regression dataset
def get_data(pol=False,dim=256,x0=0,y0=0,freqs = 285):
    #idx=np.random.randint(1,6)
    #loc_idx=np.random.randint(48)
    
    #foreground
    if pol:
        sky = jnp.array(FGnopol+FGpol)[x0:x0+dim,y0:y0+dim,0:freqs].astype(jnp.float64)
    else:
        sky = jnp.array(FGnopol)[x0:x0+dim,y0:y0+dim,0:freqs].astype(jnp.float64)
    
    #HI signal
    cosmos = jnp.array(HI+noise)[x0:x0+dim,y0:y0+dim,0:freqs].astype(jnp.float64)
    
    sky = sky - jnp.mean(sky,axis=(0,1))
    cosmos = cosmos - jnp.mean(cosmos,axis=(0,1))
    
    sky=sky.reshape((dim*dim,freqs))/1000
    cosmos=cosmos.reshape((dim*dim,freqs))/1000
    
    np.random.seed(0)
    X = jnp.linspace(0., 1., sky.shape[1])#,dtype = jnp.float64)
    #Y = X + 0.2 * jnp.power(X, 3.0) + 0.5 * jnp.power(0.5 + X, 2.0) * jnp.sin(4.0 * X)
    #Y += sigma_obs * np.random.randn(N)
    #Y -= jnp.mean(Y)
    #Y /= jnp.std(Y)

    #assert X.shape == (N,)
    #assert Y.shape == (N,)

    X_test = jnp.linspace(0., 1.,sky.shape[1])#,dtype = jnp.float64)

    return X, sky+cosmos, X_test
# squared exponential kernel with diagonal noise term
def kernel(X, Z, var, length,var_HI,length_HI, noise, jitter=1.0e-16,is_noise=True):
    deltaXsq = jnp.power((X[:, None] - Z) / length, 2.0)
    deltaHI = jnp.abs((X[:, None] - Z) / length_HI)
    
    k_fg = var * jnp.exp(-0.5 * deltaXsq)
    k_HI = 1.0e-10*var_HI * jnp.exp(-0.5 * deltaHI)
    #k = var * jnp.exp(-0.5 * deltaXsq)
    if is_noise:
        k_HI += (noise*noise*1.0e-14 + jitter) * jnp.eye(X.shape[0])
        k_fg += k_HI
    return k_fg

def kernel_pol(X, Z, var, length,var_pol,length_pol,var_HI,length_HI, noise, jitter=1.0e-16,is_noise=True):
    deltaXsq = jnp.power((X[:, None] - Z) / length, 2.0)
    deltapol = jnp.power((X[:, None] - Z) / length_pol, 2.0)
    deltaHI = jnp.abs((X[:, None] - Z) *100/ length_HI)
    
    k_fg = 1.0e-2*var * jnp.exp(-0.5 * deltaXsq)
    k_pol = 1.0e-6*var_pol * jnp.exp(-0.5 * deltapol)
    k_HI = 1.0e-9*var_HI * jnp.exp(-0.5 * deltaHI)
    #k = var * jnp.exp(-0.5 * deltaXsq)
    if is_noise:
        k_HI += (noise*noise*1.0e-14 + jitter) * jnp.eye(X.shape[0])
        k_fg += k_HI
    return k_fg

def model(X, Y):
    # set uninformative log-normal priors on our three kernel hyperparameters
    var_fg_std = numpyro.sample("varfg_std", dist.HalfNormal(1))
    
    length_fg_alpha = numpyro.sample("length_fg_alpha",dist.LogNormal(1,3))
    length_fg_beta = numpyro.sample("length_fg_beta",dist.LogNormal(0,3))
    
    #var = 100
    noise = numpyro.sample("kernel_noise", dist.HalfNormal(1))
    
    var_HI_std = numpyro.sample("varHI_std", dist.HalfNormal(1))
    length_HI_std = numpyro.sample("lengthHI_std", dist.HalfNormal(2))
    
    
    var_pol_std = numpyro.sample("varpol_std", dist.HalfNormal(5))
    
    length_pol_alpha = numpyro.sample("length_pol_alpha",dist.LogNormal(2,3))
    length_pol_beta = numpyro.sample("length_pol_beta",dist.LogNormal(0,3))
    
    var_HI = numpyro.sample("kernel_varHI", dist.HalfNormal(jnp.ones(Y.shape[0])*var_HI_std))
    length_HI = numpyro.sample("kernel_lengthHI",dist.HalfNormal(jnp.ones(Y.shape[0])*length_HI_std))
    var_fg = numpyro.sample("kernel_var", dist.HalfNormal(jnp.ones(Y.shape[0])*var_fg_std))
    length_fg = numpyro.sample("kernel_length", dist.InverseGamma(jnp.ones(Y.shape[0])*length_fg_alpha,jnp.ones(Y.shape[0])*length_fg_beta))
    var_pol = numpyro.sample("kernel_varpol", dist.HalfNormal(jnp.ones(Y.shape[0])*var_pol_std))
    length_pol = numpyro.sample("kernel_lengthpol", dist.InverseGamma(jnp.ones(Y.shape[0])*length_pol_alpha,jnp.ones(Y.shape[0])*length_pol_beta))

    # compute kernel
    X=jnp.repeat(jnp.array([X]),Y.shape[0],axis=0)
    noise=jnp.repeat(jnp.array([noise]),Y.shape[0],axis=0)
    # compute kernel
    vmap_args = (
        X,X,var_fg,length_fg,var_pol,length_pol,var_HI,length_HI,noise
    )
    
    k = vmap(
        lambda X,  Z,var_fg,length_fg,var_pol,length_pol,var_HI,length_HI,noise: kernel_pol(
            X, Z,var_fg,length_fg,var_pol,length_pol,var_HI,length_HI,noise
        )
    )(*vmap_args)
    #k = kernel_pol(X, X, var_fg, length_fg,var_pol,length_pol,var_HI,length_HI, noise)

    # sample Y according to the standard gaussian process formula
    #print(k.shape)
    numpyro.sample(
        "Y",
        dist.MultivariateNormal(loc=jnp.zeros((Y.shape[0],Y.shape[1])), covariance_matrix=k),
        obs=Y,
    )


# helper function for doing hmc inference
def run_inference(model,init_strategy, rng_key, X, Y):
    start = time.time()
    # demonstrate how to use different HMC initialization strategies
    if init_strategy == "value":
        init_strategy = init_to_value(
            values={"kernel_var": 1.0, "kernel_noise": 0.05, "kernel_length": 0.5}
        )
    elif init_strategy == "median":
        init_strategy = init_to_median(num_samples=10)
    elif init_strategy == "feasible":
        init_strategy = init_to_feasible()
    elif init_strategy == "sample":
        init_strategy = init_to_sample()
    elif init_strategy == "uniform":
        init_strategy = init_to_uniform(radius=1)
    kernel = NUTS(model, init_strategy=init_strategy,target_accept_prob=0.8,max_tree_depth=8)
    mcmc = MCMC(
        kernel,
        num_warmup=1000,
        num_samples=1000,
        num_chains=1,
        thinning=1,
        progress_bar= True,
    )
    mcmc.run(rng_key, X, Y)
    mcmc.print_summary()
    print("\nMCMC elapsed time:", time.time() - start)
    return mcmc.get_samples()


# do GP prediction for a given set of hyperparameters. this makes use of the well-known
# formula for gaussian process predictions
def predict(rng_key, X, Y, X_test, var, length,var_pol, length_pol,var_HI, length_HI, noise):
    # compute kernels between train and test data, etc.
    k_pp = kernel_pol(X_test, X_test, var, length,var_pol, length_pol,var_HI, length_HI, noise,is_noise=False)
    k_pX = kernel_pol(X_test, X, var, length,var_pol, length_pol,var_HI, length_HI, noise,is_noise=False)
    k_XX = kernel_pol(X, X, var, length,var_pol, length_pol,var_HI, length_HI, noise,is_noise=True)
    K_xx_inv = jnp.linalg.inv(k_XX)
    K = k_pp - jnp.matmul(k_pX, jnp.matmul(K_xx_inv, jnp.transpose(k_pX)))
    #sigma_noise = jnp.sqrt(jnp.clip(jnp.diag(K), a_min=0.0)) * jax.random.normal(
    #    rng_key, X_test.shape[:1]
    #)
    mean = jnp.matmul(k_pX, jnp.matmul(K_xx_inv, Y.T)).T
    # we return both the mean function and a sample from the posterior predictive for the
    # given set of hyperparameters
    return mean#, mean + sigma_noise




def main():
    X, Y, X_test = get_data(pol=True,dim=64,x0=183,y0=0,freqs=150)

    # do inference
    rng_key, rng_key_predict = random.split(random.PRNGKey(42))
    samples = run_inference(model, "median", rng_key, X, Y)
    return samples


numpyro.set_platform('gpu')
numpyro.set_host_device_count(1)
samples = main()
jnp.save('samples.npy',samples)
#mean_prediction,percentiles = make_predictions(samples)
#jnp.save('mean.npy',mean_prediction)
#jnp.save('percentile.npy',percentiles)