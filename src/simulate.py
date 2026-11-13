import jax
import jax.numpy as jnp
from jax import random
import matplotlib.pyplot as plt

def unruh_occupation(omega, a, c: float = 1.0):
    """Planckian Rindler occupation n_Omega = 1 / (exp(2 pi c Omega / a) - 1), eq. 2.5.2."""
    return 1.0 / jnp.expm1(2 * jnp.pi * c * omega / a)


def r_from_unruh(omega, a, c: float = 1.0):
    """Squeezing parameter of the Minkowski vacuum seen by an observer with
    acceleration a, at Rindler frequency omega: sinh^2(r) = n_Omega, eq. 2.6.2."""
    return jnp.arcsinh(jnp.sqrt(unruh_occupation(omega, a, c)))


def generate_covariance_matrix(r: float) -> jax.Array:
    """Returns 4x4 V(r) for the two-mode squeezed vacuum (joint state,
    both wedges). Used only for the idealized joint-access comparison —
    the physically realistic pipeline uses reduce_to_accessible(V) below."""
    cov_matrix = jnp.array([[jnp.cosh(2*r), 0, jnp.sinh(2*r), 0],
                            [0, jnp.cosh(2*r), 0, -jnp.sinh(2*r)],
                            [jnp.sinh(2*r), 0, jnp.cosh(2*r), 0],
                            [0, -jnp.sinh(2*r), 0, jnp.cosh(2*r)]])
    return cov_matrix


def reduce_to_accessible(V: jax.Array) -> jax.Array:
    """Top-left 2x2 block — the one-wedge (right-wedge) marginal state
    V_R(r) = cosh(2r) I_2, i.e. what an outside observer actually has
    access to. This is the physically realistic starting point for the
    main experiment; recovering the full joint state from this alone is
    not possible, since many joint states share the same marginal."""
    top_left_block = V[:2, :2]
    return top_left_block


def observed_covariance(r, noise_level, measurement_type) -> jax.Array:
    """
    Covariance of the data that sample_measurements produces for squeezing r,
    eq. (Sobs) of Section 3.1:
        joint:      V_A(r) + noise_level^2 I_2
        heterodyne: V_A(r) + (1 + noise_level^2) I_2
    The vacuum calibration run of sample_calibration is the r = 0 case.
    Differentiable in r and noise_level, so it can be used directly as the
    likelihood covariance in a numpyro model.
    """
    if measurement_type == 'joint':
        offset = 0.0
    elif measurement_type == 'heterodyne':
        offset = 1.0
    else:
        raise ValueError(f"Unknown measurement type: {measurement_type}")
    V_A = reduce_to_accessible(generate_covariance_matrix(r))
    return V_A + (offset + noise_level ** 2) * jnp.eye(2)


def sample_measurements(V_A, n_samples, noise_level, measurement_type, key):
    """
    Simulates detector outcomes for the one-wedge accessible mode V_A (2x2).

    This is a two-step model, kept explicitly separate:
      1. State sampling: x ~ N(0, V_A) via Cholesky — samples the Gaussian
         state's Wigner function. This alone is NOT a measurement outcome.
      2. Measurement protocol: converts state samples into what a given
         detector scheme would actually report.
           - joint:     both quadratures observed exactly (idealized
                        benchmark, not physically realizable since X and P
                        do not commute)
           - heterodyne: both quadratures observed every shot, but with an
                        unavoidable +1 unit of vacuum noise added
                        (Sigma_het = V_A + I_2, i.e. samples of the Husimi Q
                        function), per Weedbrook et al. Sec II.5
    Independent Gaussian detector noise (sigma_noise) is added on top of
    the ideal measurement outcome in all cases, to emulate real apparatus
    imperfection.

    Returns (n_samples, 2) array.
    """
    # --- Step 1: state sampling (Wigner-function sampling, not a measurement yet) ---
    state_key, heterodyne_key, detector_key = random.split(key, 3)
    L = jnp.linalg.cholesky(V_A)  # lower triangular matrix such that V_A = L @ L.T
    z = random.normal(state_key, shape=(n_samples, 2))
    samples = z @ L.T  # samples ~ N(0, V_A)

    # --- Step 2: measurement protocol ---
    if measurement_type == 'joint':
        pass  # idealized: both quadratures observed exactly, nothing to add here
    elif measurement_type == 'heterodyne':
        # fixed +1 unit of vacuum noise (Sigma_het = V_A + I_2), NOT scaled
        # by noise_level -- this is a fundamental measurement penalty, not
        # a technical/detector imperfection
        samples += random.normal(heterodyne_key, shape=(n_samples, 2))
    else:
        raise ValueError(f"Unknown measurement type: {measurement_type}")

    # --- detector noise, on top of the measurement outcome, all schemes ---
    detector_noise = noise_level * random.normal(detector_key, shape=(n_samples, 2))
    samples += detector_noise

    return samples


def sample_calibration(n_calibration, noise_level, measurement_type, key, source="vacuum"):
    """
    Simulates a detector calibration run, taken with the signal blocked, so
    the detector noise sigma can be inferred independently of the squeezing r.
    Without it, r and sigma are confounded: the signal data only constrain
    cosh(2r) + sigma^2.

    source:
      - "vacuum" (shot-noise reference, the standard optics calibration): the
        signal port is blocked, so the detector measures the vacuum state,
        i.e. the r = 0 state V = I_2, through the same measurement scheme.
        Observed variance per quadrature: 1 + sigma^2 (joint) or
        2 + sigma^2 (heterodyne).
      - "dark": the detector is also unilluminated, so only the electronic
        noise is recorded. Observed variance: sigma^2, scheme-independent.

    Use a key independent of the signal run's key. Returns (n_calibration, 2).
    """
    if source == "vacuum":
        return sample_measurements(jnp.eye(2), n_calibration, noise_level, measurement_type, key)
    if source == "dark":
        return noise_level * random.normal(key, shape=(n_calibration, 2))
    raise ValueError(f"Unknown calibration source: {source}")


def sample_idealized_joint_access(V, n_samples, noise_level, key):
    """Idealized COMPARISON case only: samples both modes (right + left
    wedge) as if an observer had access to the full joint state. This is
    not physically realistic (the left-wedge/partner mode is inaccessible
    behind the horizon) -- it exists purely to quantify, by contrast, how
    much reconstruction accuracy is lost when restricted to one wedge.
    V must be the full 4x4 joint covariance matrix. Returns (n_samples, 4).
    """
    state_key, noise_key = random.split(key)
    L = jnp.linalg.cholesky(V)
    z = random.normal(state_key, shape=(n_samples, 4))
    samples = z @ L.T
    noise = noise_level * random.normal(noise_key, shape=(n_samples, 4))
    samples += noise
    return samples


def plot_samples(samples):
    """Plots the samples in phase space."""
    samples = jax.device_get(samples)
    plt.figure(figsize=(6, 6))
    plt.scatter(samples[:, 0], samples[:, 1], alpha=0.5, s=10)
    plt.title("Phase Space Samples")
    plt.xlabel("X Quadrature")
    plt.ylabel("P Quadrature")
    plt.axis('equal')
    plt.grid(True)
    plt.show()


def main():
    # Example usage: physically realistic one-wedge pipeline
    r = 0.5  # Squeezing parameter
    V = generate_covariance_matrix(r)
    V_A = reduce_to_accessible(V)  # one-wedge marginal, the physical case

    n_samples = 1000
    noise_level = 0.1
    key = random.key(42)

    for measurement_type in ('joint', 'heterodyne'):
        key, sample_key = random.split(key)
        samples = sample_measurements(V_A, n_samples, noise_level, measurement_type, sample_key)
        print(measurement_type, samples[:5])

    # Detector calibration run (signal blocked), independent key
    key, calibration_key = random.split(key)
    calibration = sample_calibration(n_samples, noise_level, 'joint', calibration_key)
    print("calibration (vacuum):", calibration[:5])

    # Idealized joint-access comparison (both wedges) - not the physical case
    joint_samples = sample_idealized_joint_access(V, n_samples, noise_level, key)
    print("idealized joint access:", joint_samples[:5])


if __name__ == "__main__":
    main()