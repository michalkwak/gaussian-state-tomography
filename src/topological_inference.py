"""
Bayesian inference of the squeezing parameter r from Betti curves.

The Betti curves of a point cloud have no closed-form likelihood, so we use a
synthetic likelihood (Wood, Nature 466, 1102 (2010)):
  1. simulate n_train datasets with r drawn from the prior, through the same
     measurement chain as the data (simulate.sample_measurements);
  2. summarise each by its Betti curves on fixed grids (homology.betti_features);
  3. standardise and compress the curves with PCA to z (n_components values);
  4. model z | r ~ N(m(r), C), with m(r) a polynomial in r fitted by least
     squares and C the pooled residual covariance.
The fitted BettiEmulator then plays the role of the likelihood in a numpyro
model, so the posterior of r given only the topological summary is sampled
with NUTS exactly like the raw-quadrature models.

The detector noise is not inferred here: the emulator is trained at a fixed
noise level, in practice the calibration estimate of sigma.
"""
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import numpy as np
import jax
import jax.numpy as jnp
from jax import random
import numpyro
import numpyro.distributions as dist

import simulate
import homology


def simulate_clouds(r_values, n_samples, noise_level, measurement_type, key) -> list:
    """One simulated (n_samples, 2) point cloud per value in r_values, as numpy arrays."""
    clouds = []
    for r, k in zip(np.asarray(r_values), random.split(key, len(r_values))):
        V_A = simulate.reduce_to_accessible(simulate.generate_covariance_matrix(float(r)))
        clouds.append(np.asarray(simulate.sample_measurements(V_A, n_samples, noise_level, measurement_type, k)))
    return clouds


def persistence_diagrams(clouds, n_workers=None) -> list:
    """
    homology.compute_persistence_diagrams (H0, H1) for many point clouds, in
    parallel worker processes: ripser takes ~1 s per 1000-point cloud.
    """
    n_workers = min(n_workers or os.cpu_count(), len(clouds))
    if n_workers <= 1:
        return [homology.compute_persistence_diagrams(c) for c in clouds]
    with ProcessPoolExecutor(n_workers) as pool:
        return list(pool.map(homology.compute_persistence_diagrams, clouds,
                             chunksize=max(1, len(clouds) // (4 * n_workers))))


def betti_grids(diagrams, n_grid=60, quantile=0.95) -> list:
    """
    One filtration grid per homology dimension, from 0 to a high quantile
    (over the training clouds) of each cloud's largest finite death, so the
    tail of the curves - where the scale information sits - is not cut off.
    """
    grids = []
    for dim in range(len(diagrams[0])):
        max_deaths = []
        for d in diagrams:
            deaths = d[dim][:, 1]
            deaths = deaths[np.isfinite(deaths)]
            if len(deaths):
                max_deaths.append(deaths.max())
        grids.append(np.linspace(0.0, np.quantile(max_deaths, quantile), n_grid))
    return grids


@dataclass
class BettiEmulator:
    """Synthetic likelihood z | r ~ N(m(r), C) of the PCA-compressed Betti curves."""
    grids: list
    feature_mean: np.ndarray   # (n_features,) standardisation
    feature_scale: np.ndarray  # (n_features,)
    components: np.ndarray     # (n_components, n_features) PCA directions
    coefs: jax.Array           # (degree + 1, n_components): m(r) = sum_j coefs[j] r^j
    cov: jax.Array             # (n_components, n_components) pooled residual covariance
    n_samples: int
    noise_level: float
    measurement_type: str

    def summarise(self, diagrams) -> np.ndarray:
        """Compressed summary z of one dataset, from its persistence diagrams."""
        f = homology.betti_features(diagrams, self.grids)
        return ((f - self.feature_mean) / self.feature_scale) @ self.components.T

    def mean(self, r):
        """m(r); differentiable in r, so usable inside a NUTS model."""
        powers = r ** jnp.arange(self.coefs.shape[0])
        return powers @ self.coefs

    def log_likelihood(self, r, z):
        return dist.MultivariateNormal(self.mean(r), covariance_matrix=self.cov).log_prob(z)


def fit_emulator(measurement_type, n_samples, noise_level, key, n_train=1200,
                 r_max=1.0, n_components=3, degree=3, n_grid=60):
    """
    Trains a BettiEmulator on n_train simulated datasets with r ~ Uniform(0, r_max)
    (the notebook's prior). Returns (emulator, r_train, z_train); the training
    summaries are returned for diagnostic plots.

    Only the leading principal component tracks r closely; the higher ones are
    mostly noise whose spread varies with r, which the pooled covariance cannot
    capture. Keeping 3 components gave calibrated 90% intervals in tests
    (heterodyne, N = 1000, sigma = 1: 93% coverage), whereas 6 components gave
    overconfident ones (83%).
    """
    r_key, data_key = random.split(key)
    r_train = np.asarray(random.uniform(r_key, (n_train,), maxval=r_max))
    diagrams = persistence_diagrams(simulate_clouds(r_train, n_samples, noise_level, measurement_type, data_key))

    grids = betti_grids(diagrams, n_grid)
    F = np.stack([homology.betti_features(d, grids) for d in diagrams]).astype(float)
    feature_mean = F.mean(axis=0)
    feature_scale = F.std(axis=0)
    feature_scale[feature_scale == 0] = 1.0  # constant features (e.g. beta_0(0) = n_samples)
    Fs = (F - feature_mean) / feature_scale
    components = np.linalg.svd(Fs, full_matrices=False)[2][:n_components]
    Z = Fs @ components.T

    X = r_train[:, None] ** np.arange(degree + 1)
    coefs = np.linalg.lstsq(X, Z, rcond=None)[0]
    resid = Z - X @ coefs
    cov = resid.T @ resid / (n_train - degree - 1)

    emulator = BettiEmulator(grids, feature_mean, feature_scale, components,
                             jnp.asarray(coefs), jnp.asarray(cov),
                             n_samples, noise_level, measurement_type)
    return emulator, r_train, Z


def make_betti_model(emulator, r_max=1.0):
    """numpyro model for r given the compressed Betti summary z_obs of one dataset."""
    def model(z_obs):
        r = numpyro.sample("r", dist.Uniform(0.0, r_max))
        numpyro.sample("z_obs", dist.MultivariateNormal(emulator.mean(r), covariance_matrix=emulator.cov),
                       obs=z_obs)
    return model


def grid_posterior(log_lik, r_grid) -> dict:
    """
    Mean, median, sd and 90% central interval of a 1-D posterior under a
    uniform prior on r_grid, from log-likelihood values on that grid. A fast
    exact alternative to MCMC for repeated-dataset studies, since r is 1-D.
    """
    log_lik = np.asarray(log_lik, dtype=float)
    w = np.exp(log_lik - log_lik.max())
    w /= w.sum()
    mean = float(np.sum(w * r_grid))
    cdf = np.cumsum(w)
    return {"mean": mean,
            "median": float(np.interp(0.5, cdf, r_grid)),
            "sd": float(np.sqrt(np.sum(w * (r_grid - mean) ** 2))),
            "low": float(np.interp(0.05, cdf, r_grid)),
            "high": float(np.interp(0.95, cdf, r_grid))}


def quadrature_log_likelihood(samples, r_grid, noise_level, measurement_type):
    """Raw-quadrature log-likelihood on r_grid at fixed noise, via simulate.observed_covariance."""
    def ll(r):
        cov = simulate.observed_covariance(r, noise_level, measurement_type)
        return dist.MultivariateNormal(jnp.zeros(2), covariance_matrix=cov).log_prob(samples).sum()
    return jax.vmap(ll)(jnp.asarray(r_grid))
