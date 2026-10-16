"""
plot_dense_mass_comparison.py
==============================

Generates the two figures used in docs/mcmc_implementation.md: a direct,
real before/after comparison of NUTS with the default diagonal mass matrix
versus ``dense_mass=True`` (see CLAUDE.md decision #17) on the same
synthetic dataset -- a histogram of leapfrog steps per sample, and a
scatter of ``sigma_drw``/``tau_drw`` posterior draws with the two mass
matrices' implied step-direction ellipses overlaid, showing why a diagonal
mass matrix struggles with this correlated pair specifically.

This is a documentation/figure-generation script, not a test -- run it
directly to regenerate the PNGs under docs/images/ if the model's
posterior geometry or the default settings ever change materially:

    poetry run python scripts/plot_dense_mass_comparison.py

Takes a couple of minutes (runs two real, if short, NUTS chains).
"""

import matplotlib

matplotlib.use("Agg")

import numpy as np
import jax
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from numpyro.infer import MCMC, NUTS

from discotau.synthetic import generate_synthetic_dataset
from discotau.echofit import EchoFit
from discotau.model import reverberation_model

OUT_DIR = "docs/images"


def _run(dense_mass: bool, seed: int, num_warmup: int, num_samples: int):
    data = generate_synthetic_dataset(
        M_BH=1.0e9, log_mdot_true=0.0, inclination_true=35.0,
        sigma_drw_true=0.3, tau_drw_true=30.0,
        bands={"g": 4770.0, "i": 7625.0}, t_span=200.0, n_obs_per_band=50,
        n_freq=15, n_tau=100, tau_max=50.0, noise_level=0.05, seed=0,
    )
    # drw_prior=True: pinned so this reproduces the exact numbers already
    # documented in CLAUDE.md decision #17 (measured before decision #19
    # made the random-walk prior the default) -- not a claim that DRW is
    # still the default, or that the same dense_mass benefit hasn't been
    # re-checked under the random-walk prior.
    ef = EchoFit(M_BH=data["truth"]["M_BH"], drw_prior=True)
    for name, d in data["bands"].items():
        ef.add_lightcurve(name, wavelength=d["wavelength"], t=d["t"], y=d["y"], yerr=d["yerr"])
    ef.build_grid(n_freq=15, n_tau=100)

    kernel = NUTS(reverberation_model, target_accept_prob=0.85, dense_mass=dense_mass)
    mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples, num_chains=1, progress_bar=False)
    mcmc.run(jax.random.PRNGKey(seed), extra_fields=("num_steps", "diverging"), **ef._model_kwargs())
    return mcmc.get_samples(), mcmc.get_extra_fields()


def _cov_ellipse(ax, x, y, use_correlation: bool, **kwargs):
    """A 1-sigma ellipse for the mass matrix a NUTS kernel would build from
    (x, y): the full covariance if use_correlation, else a diagonal
    (axis-aligned) approximation using only the two variances."""
    cov = np.cov(x, y)
    if not use_correlation:
        cov = np.diag(np.diag(cov))
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    width, height = 2 * np.sqrt(vals)
    ell = Ellipse((x.mean(), y.mean()), width, height, angle=angle, fill=False, linewidth=2.5, **kwargs)
    ax.add_patch(ell)


def plot_num_steps_histogram(diag_extra, dense_extra, path):
    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.linspace(0, 1023, 40)
    ax.hist(diag_extra["num_steps"], bins=bins, alpha=0.6, label="diagonal mass matrix (default)", color="tab:orange")
    ax.hist(dense_extra["num_steps"], bins=bins, alpha=0.6, label="dense_mass=True", color="tab:blue")
    ax.set_xlabel("leapfrog steps per sample")
    ax.set_ylabel("number of samples")
    ax.set_title("NUTS trajectory length per sample: diagonal vs dense mass matrix")
    ax.legend()
    ax.grid(alpha=0.6)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_correlated_pair_scatter(diag_samples, dense_samples, path):
    # sigma_drw vs S_g (a band's driver-amplitude scaling) is one of the more
    # strongly correlated pairs in this model's posterior (r ~ -0.68,
    # checked directly) -- distinct from the S_g/S_i *exact* r=1.000
    # amplitude degeneracy that CLAUDE.md decision #13 already covers, so
    # this illustrates decision #17's mass-matrix story on its own.
    #
    # Both runs sample the *same* posterior (only sampler efficiency
    # differs), so plotting their draws in two side-by-side panels made two
    # near-identical-looking scatter clouds -- correct (that's what a
    # working sampler should do), but it buried the actual point. The
    # point isn't "the data looks different", it's "the two mass matrices
    # *assume* different shapes for the same data" -- so both ellipses go
    # on one panel, on top of the pooled draws, where the mismatch between
    # the orange circle and the visibly tilted point cloud is immediate.
    x = np.concatenate([np.asarray(diag_samples["S_g"]), np.asarray(dense_samples["S_g"])])
    y = np.concatenate([np.asarray(diag_samples["sigma_drw"]), np.asarray(dense_samples["sigma_drw"])])

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    ax.scatter(x, y, s=10, alpha=0.35, color="0.35", label="posterior draws (pooled, both runs)")
    _cov_ellipse(
        ax, x, y, use_correlation=False, edgecolor="tab:orange",
        label="diagonal mass matrix's assumed shape\n(a circle -- misses the tilt)",
    )
    _cov_ellipse(
        ax, x, y, use_correlation=True, edgecolor="tab:blue",
        label="dense mass matrix's assumed shape\n(tilted -- matches the data)",
    )
    ax.set_xlabel("S_g")
    ax.set_ylabel("sigma_drw")
    ax.set_title("The posterior's real shape vs. what each mass matrix assumes")
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
    ax.grid(alpha=0.6)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    print("Running NUTS with the default diagonal mass matrix...")
    diag_samples, diag_extra = _run(dense_mass=False, seed=1, num_warmup=800, num_samples=300)
    print(f"  mean steps/sample: {np.asarray(diag_extra['num_steps']).mean():.1f}, "
          f"divergences: {int(np.asarray(diag_extra['diverging']).sum())}/300")

    print("Running NUTS with dense_mass=True...")
    dense_samples, dense_extra = _run(dense_mass=True, seed=1, num_warmup=800, num_samples=300)
    print(f"  mean steps/sample: {np.asarray(dense_extra['num_steps']).mean():.1f}, "
          f"divergences: {int(np.asarray(dense_extra['diverging']).sum())}/300")

    plot_num_steps_histogram(diag_extra, dense_extra, f"{OUT_DIR}/dense_mass_num_steps.png")
    plot_correlated_pair_scatter(diag_samples, dense_samples, f"{OUT_DIR}/dense_mass_correlation_ellipse.png")
    print(f"\nSaved figures to {OUT_DIR}/")


if __name__ == "__main__":
    main()
