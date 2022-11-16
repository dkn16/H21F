import argparse
import os
import time

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from jax.config import config
config.update("jax_enable_x64", True)

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
# create artificial regression dataset
def get_data():
    #idx=np.random.randint(1,6)
    #loc_idx=np.random.randint(48)
    
    #foreground
    sky=jnp.load(f'/work/dante/data/fg_rm_data/train_data/data_3/sky/SKY_idx_1_loc_0.npy').astype(jnp.float64)[:,0:64,192:256].transpose(2,1,0)
    
    #HI signal
    cosmos=jnp.load(f'/work/dante/data/fg_rm_data/train_data/data_3/cosmos/HI_idx_1_loc_0.npy').astype(jnp.float64)[:,0:64,192:256].transpose(2,1,0)
    
    sky = sky - jnp.mean(sky,axis=(0,1))
    cosmos = cosmos - jnp.mean(cosmos,axis=(0,1))
    
    sky=sky.reshape((64*64,150))/10000
    cosmos=cosmos.reshape((64*64,150))/10000
    np.random.seed(0)
    X = jnp.linspace(0., 1., sky.shape[1],dtype = jnp.float64)
    #Y = X + 0.2 * jnp.power(X, 3.0) + 0.5 * jnp.power(0.5 + X, 2.0) * jnp.sin(4.0 * X)
    #Y += sigma_obs * np.random.randn(N)
    #Y -= jnp.mean(Y)
    #Y /= jnp.std(Y)

    #assert X.shape == (N,)
    #assert Y.shape == (N,)

    X_test = jnp.linspace(0., 1.,sky.shape[1],dtype = jnp.float64)

    return X, sky+cosmos, X_test

# squared exponential kernel with diagonal noise term
def kernel(X, Z, var, length, noise, jitter=1.0e-14,is_noise=True):
    deltaXsq = jnp.power((X[:, None] - Z) / length, 2.0)
    k = 10*var * jnp.exp(-0.5 * deltaXsq)
    #k = var * jnp.exp(-0.5 * deltaXsq)
    if is_noise:
        k += (noise*noise*1.0e-10 + jitter) * jnp.eye(X.shape[0])
    return k


def model(X, Y):
    # set weakly informative priors on our three kernel hyperparameters
    var_mean = numpyro.sample("var_mean", dist.HalfNormal(1))
    var_std = numpyro.sample("var_std", dist.HalfNormal(1))

    
    length_mean = numpyro.sample("length_mean", dist.InverseGamma(10,2))
    length_std = numpyro.sample("length_std", dist.HalfNormal(1))
    
    noise_mean = numpyro.sample("noise_mean", dist.HalfNormal(1))
    noise_std = numpyro.sample("noise_std", dist.HalfNormal(1))
    
    #sample for the var and length
    var_ksi = numpyro.sample("kernel_var",dist.Normal(loc=jnp.zeros(Y.shape[0])))
    length_ksi = numpyro.sample("kernel_length",dist.Normal(loc=jnp.zeros(Y.shape[0])))
    noise_ksi = numpyro.sample("kernel_noise",dist.Normal(loc=jnp.zeros(Y.shape[0])))
    
    var = var_std*var_ksi+var_mean
    length = length_std*length_ksi+length_mean
    noise = noise_std*noise_ksi+noise_mean
    
    X=jnp.repeat(jnp.array([X]),Y.shape[0],axis=0)
    # compute kernel
    vmap_args = (
        X,X,var,length,noise
    )
    
    k = vmap(
        lambda X, Z, var, length, noise: kernel(
            X, Z,var, length, noise
        )
    )(*vmap_args)
    #print(k.shape)
    # sample Y according to the standard gaussian process formula
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
    kernel = NUTS(model, init_strategy=init_strategy,target_accept_prob=0.7,max_tree_depth=7)
    mcmc = MCMC(
        kernel,
        num_warmup=1000,
        num_samples=2000,
        num_chains=1,
        thinning=10,
        progress_bar= True,
    )
    mcmc.run(rng_key, X, Y)
    mcmc.print_summary()
    print("\nMCMC elapsed time:", time.time() - start)
    return mcmc.get_samples()


# do GP prediction for a given set of hyperparameters. this makes use of the well-known
# formula for gaussian process predictions
@jax.jit
def predict(rng_key, X, Y, X_test, var, length, noise):
    X=jnp.repeat(jnp.array([X]),Y.shape[0],axis=0)
    # compute kernel of size (4096*150)
    vmap_args = (
        X,X,var,length,noise
    )
    
    k_pp = vmap(
        lambda X, Z, var, length, noise: kernel(
            X, Z,var, length, noise,is_noise=False
        )
    )(*vmap_args)
    k_pX = vmap(
        lambda X, Z, var, length, noise: kernel(
            X, Z,var, length, noise,is_noise=False
        )
    )(*vmap_args)
    k_XX = vmap(
        lambda X, Z, var, length, noise: kernel(
            X, Z,var, length, noise,is_noise=True
        )
    )(*vmap_args)
    
    #compute some useful matrix
    K_xx_inv = jnp.linalg.inv(k_XX)
    K = k_pp - jax.lax.batch_matmul(k_pX, jax.lax.batch_matmul(K_xx_inv, jnp.transpose(k_pX,axes=(0,2,1))))
    
    vmap_args = (
        K_xx_inv,Y
    )
    cache = vmap(lambda A,B: jnp.matmul(A,B))(*vmap_args)
    vmap_args = (
        k_pX,cache
    )
    mean = vmap(lambda A,B: jnp.matmul(A,B))(*vmap_args)
    # we return both the mean function and a sample from the posterior predictive for the
    # given set of hyperparameters
    return mean




def main():
    X, Y, X_test = get_data()

    # do inference
    rng_key, rng_key_predict = random.split(random.PRNGKey(42))
    samples = run_inference(model, "median", rng_key, X, Y)
    return samples

def make_predictions(samples):
    X, Y, X_test = get_data()
    rng_key, rng_key_predict = random.split(random.PRNGKey(42))
    # do prediction
    kernel_var = samples["kernel_var"].T*samples["var_std"]+samples["var_mean"]
    kernel_noise = samples["kernel_noise"].T*samples["noise_std"]+samples["noise_mean"]
    kernel_length = samples["kernel_length"].T*samples["length_std"]+samples["length_mean"]
    #print(kernel_var.shape)
    #vmap_args = (
    #    random.split(rng_key_predict, samples["var_std"].shape[0]),
    #    kernel_var.T,
    #    kernel_length.T,
    #    kernel_noise.T,
    #)
    #means, predictions = vmap(
    #    lambda rng_key, var, length, noise: predict(
    #       rng_key, X, Y, X_test, var, length, noise
    #    )
    #)(*vmap_args)
    means = []
    for i in range(samples["var_std"].shape[0]):
        means.append(predict(
           rng_key, X, Y, X_test, kernel_var.T[i], kernel_noise.T[i], kernel_noise.T[i]
        ))
    #print(means）
    mean_prediction = np.mean(means, axis=0)
    percentiles = np.percentile(means, [5.0, 95.0], axis=0)
    
    return mean_prediction,percentiles


numpyro.set_platform('gpu')
numpyro.set_host_device_count(1)
samples = main()
jnp.save('samples.npy',samples)
mean_prediction,percentiles = make_predictions(samples)
jnp.save('mean.npy',mean_prediction)
jnp.save('percentile.npy',percentiles)