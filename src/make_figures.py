"""
Figures for Section 3.1 (synthetic measurement simulation).

Run from the repo root:
    .venv/Scripts/python src/make_figures.py
Writes PDF (for LaTeX) and PNG (for previewing) to figures/.
"""
from pathlib import Path

import numpy as np
import jax.numpy as jnp
from jax import random
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import simulate
import estimate
import metrics

OUT = Path(__file__).resolve().parent.parent / "figures"
SEED = 42
N_TRIALS = 200

# fixed series identity across every figure (colour + marker, never colour alone)
STYLE = {
    "joint":      dict(color="#2a78d6", marker="o", label="one wedge, joint (ideal)"),
    "heterodyne": dict(color="#eb6834", marker="s", label="one wedge, heterodyne"),
    "two_wedge":  dict(color="#1baf7a", marker="^", label="two wedges, joint (ideal)"),
}
INK, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7"

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9, "legend.fontsize": 8,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "text.color": INK, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "lines.linewidth": 1.5, "lines.markersize": 5,
    "legend.frameon": False, "savefig.bbox": "tight", "savefig.dpi": 300,
})


def accessible_cov(r):
    return simulate.reduce_to_accessible(simulate.generate_covariance_matrix(r))


def ellipse(cov, n_sigma=2.0, n_pts=200):
    vals, vecs = np.linalg.eigh(np.asarray(cov))
    t = np.linspace(0, 2 * np.pi, n_pts)
    return (vecs @ (n_sigma * np.sqrt(vals)[:, None] * np.stack([np.cos(t), np.sin(t)]))).T


def r_error(r, n, sigma, scheme, key):
    """r_hat - r for one simulated dataset of the given scheme."""
    if scheme == "two_wedge":
        x = simulate.sample_idealized_joint_access(simulate.generate_covariance_matrix(r), n, sigma, key)
        return float(estimate.estimate_squeezing_joint_access(x, sigma)) - r
    x = simulate.sample_measurements(accessible_cov(r), n, sigma, scheme, key)
    return float(metrics.estimate_squeezing(estimate.estimate_covariance(x, scheme, sigma))) - r


def rmse(r, n, sigma, scheme, n_trials=N_TRIALS):
    errs = [r_error(r, n, sigma, scheme, random.fold_in(random.key(SEED), t)) for t in range(n_trials)]
    return float(np.sqrt(np.mean(np.square(errs))))


def crb(r, n, sigma, scheme):
    if scheme == "two_wedge":
        return float(metrics.squeezing_crb_joint_access(r, n, sigma))
    return float(metrics.squeezing_crb(r, n, sigma, scheme))


def save(fig, name):
    OUT.mkdir(exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}")
    plt.close(fig)
    print(f"wrote figures/{name}.pdf")


def fig_reduced_state(r=0.5, n=2000):
    """Joint state is correlated across wedges; one wedge alone is a round thermal blob."""
    x = np.asarray(simulate.sample_idealized_joint_access(
        simulate.generate_covariance_matrix(r), n, 0.0, random.key(SEED)))
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 3.0), sharex=True, sharey=True)
    panels = [(0, 2, "$X_A$", "$X_B$", "(a) both wedges: $X_A$ vs $X_B$"),
              (0, 1, "$X_A$", "$P_A$", "(b) right wedge only: $X_A$ vs $P_A$")]
    for ax, (i, j, xl, yl, title) in zip(axes, panels):
        ax.scatter(x[:, i], x[:, j], s=3, alpha=0.35, color=STYLE["joint"]["color"], linewidths=0)
        e = ellipse(np.asarray(simulate.generate_covariance_matrix(r))[np.ix_([i, j], [i, j])])
        ax.plot(e[:, 0], e[:, 1], color=INK, lw=1.2, label=r"$2\sigma$ contour of the exact covariance")
        ax.set(xlabel=xl, ylabel=yl, title=title, aspect="equal")
    fig.legend(*axes[1].get_legend_handles_labels(), loc="lower center", bbox_to_anchor=(0.5, -0.12))
    fig.suptitle(f"$r = {r}$, $N = {n}$, no detector noise", y=1.0, color=MUTED, fontsize=8)
    save(fig, "fig_reduced_state")


def fig_measurement_schemes(rs=(0.25, 0.5, 1.0), n=1000, sigma=0.1):
    """Point clouds y_i for each scheme, with the 2-sigma contour of Sigma_obs and of V_A."""
    schemes = ("joint", "heterodyne")
    lim = 2.6 * np.sqrt(np.asarray(simulate.observed_covariance(max(rs), sigma, "heterodyne"))[0, 0])
    fig, axes = plt.subplots(len(schemes), len(rs), figsize=(6.3, 4.4), sharex=True, sharey=True)
    for row, scheme in enumerate(schemes):
        for col, r in enumerate(rs):
            ax = axes[row, col]
            key = random.fold_in(random.key(SEED), 10 * row + col)
            y = np.asarray(simulate.sample_measurements(accessible_cov(r), n, sigma, scheme, key))
            ax.scatter(y[:, 0], y[:, 1], s=3, alpha=0.35, color=STYLE[scheme]["color"], linewidths=0)
            e_obs, e_state = ellipse(np.asarray(simulate.observed_covariance(r, sigma, scheme))), ellipse(accessible_cov(r))
            ax.plot(e_obs[:, 0], e_obs[:, 1], color=INK, lw=1.2, label=r"$\Sigma_\mathrm{obs}$ ($2\sigma$)")
            ax.plot(e_state[:, 0], e_state[:, 1], color=INK, lw=1.0, ls="--", label=r"$V_A$ ($2\sigma$)")
            ax.set(xlim=(-lim, lim), ylim=(-lim, lim), aspect="equal")
            if row == 0:
                ax.set_title(f"$r = {r}$")
            if row == len(schemes) - 1:
                ax.set_xlabel("$x$")
            if col == 0:
                ax.set_ylabel(f"{scheme}\n$p$")
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="lower center", ncols=2,
               bbox_to_anchor=(0.5, -0.04))
    fig.suptitle(f"$N = {n}$, $\\sigma_\\mathrm{{noise}} = {sigma}$", y=1.0, color=MUTED, fontsize=8)
    save(fig, "fig_measurement_schemes")


def fig_accuracy(r=0.5, sigma=0.1, n=1000,
                 n_grid=(50, 100, 300, 1000, 3000, 10000),
                 sigma_grid=(0.0, 0.1, 0.2, 0.5, 1.0, 2.0)):
    """RMSE of r_hat (markers) against the Cramer-Rao bound (lines), vs N and vs sigma."""
    fig, (ax_n, ax_s) = plt.subplots(1, 2, figsize=(6.3, 2.8), sharey=True)
    for scheme, st in STYLE.items():
        ax_n.plot(n_grid, [rmse(r, m, sigma, scheme) for m in n_grid], ls="none", **st)
        ax_n.plot(n_grid, [crb(r, m, sigma, scheme) for m in n_grid], color=st["color"], lw=1.2)
        ax_s.plot(sigma_grid, [rmse(r, n, s, scheme) for s in sigma_grid], ls="none", **st)
        ax_s.plot(sigma_grid, [crb(r, n, s, scheme) for s in sigma_grid], color=st["color"], lw=1.2)
    ax_n.set(xscale="log", yscale="log", xlabel="samples $N$", ylabel=r"RMSE of $\hat r$",
             title=f"(a) vs $N$ ($\\sigma_\\mathrm{{noise}} = {sigma}$)")
    ax_s.set(yscale="log", xlabel=r"detector noise $\sigma_\mathrm{noise}$",
             title=f"(b) vs noise ($N = {n}$)")
    ax_n.plot([], [], color=MUTED, lw=1.2, label="Cramér–Rao bound")
    fig.legend(*ax_n.get_legend_handles_labels(), loc="lower center", ncols=2,
               bbox_to_anchor=(0.5, -0.2))
    fig.suptitle(f"$r = {r}$, {N_TRIALS} simulated datasets per point", y=1.02, color=MUTED, fontsize=8)
    save(fig, "fig_accuracy")


def fig_wedge_loss(sigma=0.1, n=1000, r_points=(0.1, 0.25, 0.5, 1.0, 1.5)):
    """How much one-wedge access costs: bound (lines) and simulation (markers) vs r."""
    r_curve = np.linspace(0.02, 2.0, 200)
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    for scheme, st in STYLE.items():
        ax.plot(r_curve, [crb(r, n, sigma, scheme) for r in r_curve], color=st["color"], lw=1.2)
        ax.plot(r_points, [rmse(r, n, sigma, scheme, N_TRIALS // 2) for r in r_points], ls="none", **st)
    ax.plot([], [], color=MUTED, lw=1.2, label="Cramér–Rao bound")
    ax.set(yscale="log", xlabel="squeezing $r$", ylabel=r"RMSE of $\hat r$",
           title=f"$N = {n}$, $\\sigma_\\mathrm{{noise}} = {sigma}$")
    ax.legend(loc="upper right")
    save(fig, "fig_wedge_loss")


def main():
    fig_reduced_state()
    fig_measurement_schemes()
    fig_accuracy()
    fig_wedge_loss()


if __name__ == "__main__":
    main()
