import jax
import jax.numpy as jnp
from jax import random
import matplotlib.pyplot as plt
import numpy as np
import ripser

import simulate

# homology.py
#
# Persistent-homology analysis of the synthetic measurement point cloud
# produced by simulate.sample_measurements(). Consumes the same (n_samples, 2)
# arrays (or (n_samples, 4) for the idealized joint-access comparison,
# see simulate.sample_idealized_joint_access) as the rest of the pipeline --
# no separate data generation logic lives here.


def compute_persistence_diagrams(samples: jax.Array, maxdim: int = 1) -> list:
    """
    Computes Vietoris-Rips persistence diagrams directly from a point cloud.

    ripser computes pairwise Euclidean distances internally, so no explicit
    distance matrix needs to be constructed here -- passing a covariance
    matrix (as in the original script) is NOT a valid substitute: a
    covariance matrix has none of the properties required of a distance
    matrix (symmetric zero diagonal, non-negativity, triangle inequality).

    maxdim=1 (H0, H1) is fast even for n_samples in the thousands. Going to
    higher maxdim (per the H0-H2 study described in the paper)
    is combinatorially more expensive -- test on a small n_samples (50-100)
    before scaling up.

    Returns the list of persistence diagrams, diagrams[0] = H0, diagrams[1] = H1, etc.
    """
    return ripser.ripser(jax.device_get(samples), maxdim=maxdim)["dgms"]


def compute_betti_curve(diagram: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """
    Computes the Betti curve beta_k(t) of a single persistence diagram on a grid.

    beta_k(t) is the number of persistence pairs (b, d) alive at filtration
    value t, i.e. b <= t < d. Infinite deaths (the one essential H0 class)
    are handled naturally, since t < inf is always True.
    """
    births = diagram[:, 0]
    deaths = diagram[:, 1]
    alive = (births[None, :] <= grid[:, None]) & (grid[:, None] < deaths[None, :])
    return alive.sum(axis=1)


def plot_betti_curves(samples: jax.Array, diagrams: list, R: float = 2.0, n_grid: int = 500):
    """Plots the point cloud alongside its H0 and H1 Betti curves."""
    samples = jax.device_get(samples)
    grid = np.linspace(0.0, R, n_grid)
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(20, 5))

    plt.sca(axes[0])
    plt.title("Point Cloud")
    plt.scatter(samples[:, 0], samples[:, 1], alpha=0.5, s=10)

    plt.sca(axes[1])
    plt.title("0-dim Betti Curve")
    plt.step(grid, compute_betti_curve(diagrams[0], grid), where="post")
    plt.xlabel("Filtration value")
    plt.ylabel(r"$\beta_0$")
    plt.xlim(0, R)

    plt.sca(axes[2])
    plt.title("1-dim Betti Curve")
    plt.step(grid, compute_betti_curve(diagrams[1], grid), where="post")
    plt.xlabel("Filtration value")
    plt.ylabel(r"$\beta_1$")
    plt.xlim(0, R)

    plt.show()


def main():
    # Example usage
    r = 0.5  # Squeezing parameter
    V = simulate.generate_covariance_matrix(r)
    V_A = simulate.reduce_to_accessible(V)

    n_samples = 1000
    noise_level = 0.1
    measurement_type = 'heterodyne'  # or 'joint'

    key = random.key(42)  

    samples = simulate.sample_measurements(V_A, n_samples, noise_level, measurement_type, key)
    diagrams = compute_persistence_diagrams(samples, maxdim=1)
    plot_betti_curves(samples, diagrams)


if __name__ == "__main__":
    main()