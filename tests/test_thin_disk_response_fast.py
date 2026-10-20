"""
Tests for the precomputed-template fast path (forward_model.build_thin_disk_response_table
/ build_thin_disk_response_fast): the same precompute-a-grid-then-interpolate-
and-stretch trick used in the author's PhD-era CREAM Fortran code, so a fit
doesn't have to re-run thin_disk_response's O(n_phi * n_tau) disk integral
on every NUTS step (see docs/thin_disk_response.md).

Tables here use much smaller n_phi/n_u/incl_grid than the production
defaults -- accuracy at production resolution is what
scripts/plot_thin_disk_response_scalings.py's figures already demonstrate;
these tests only need to be fast and check the mechanism is sound.
"""

import time

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from discotau.forward_model import (
    typical_log_mdot,
    thin_disk_response,
    build_thin_disk_response_table,
    thin_disk_response_from_table,
    build_thin_disk_response_fast,
)

_np_trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz

M_BH = 1.0e8

_SMALL_TABLE_KWARGS = dict(
    incl_grid=jnp.arange(0.0, 90.1, 15.0), n_phi=40, n_u=300,
)


def _mean_lag(tau_grid, psi):
    tau_np = np.asarray(tau_grid)
    return float(_np_trapz(np.asarray(psi) * tau_np, tau_np))


def test_fast_response_is_causal_and_normalised():
    fast = build_thin_disk_response_fast(M_BH, **_SMALL_TABLE_KWARGS)
    tau_grid = jnp.linspace(-2.0, 15.0, 300)
    psi = np.asarray(fast(tau_grid, typical_log_mdot(M_BH), 5000.0, 30.0, M_BH))
    tau_np = np.asarray(tau_grid)
    assert not np.isnan(psi).any()
    assert np.all(psi[tau_np < 0.0] == 0.0)
    assert np.isclose(_np_trapz(psi, tau_np), 1.0, atol=1e-2)


def test_fast_response_mean_lag_matches_exact_reasonably_well():
    """Near the table's own reference point (typical_log_mdot(M_BH), wavelength=5000,
    matching build_thin_disk_response_table's defaults), the stretch factor
    is close to 1 and interpolation is between adjacent templates -- this is
    the "easy" regime, so the mean lag should match closely, not just be in
    the right ballpark."""
    fast = build_thin_disk_response_fast(M_BH, **_SMALL_TABLE_KWARGS)
    tau_grid = jnp.linspace(-2.0, 15.0, 300)
    for above_typical, wavelength, inclination in [(0.0, 5000.0, 20.0), (0.2, 5000.0, 45.0)]:
        log_mdot_msun_yr = typical_log_mdot(M_BH) + above_typical
        psi_slow = thin_disk_response(tau_grid, log_mdot_msun_yr, wavelength, inclination, M_BH)
        psi_fast = fast(tau_grid, log_mdot_msun_yr, wavelength, inclination, M_BH)
        lag_slow = _mean_lag(tau_grid, psi_slow)
        lag_fast = _mean_lag(tau_grid, psi_fast)
        assert abs(lag_fast - lag_slow) / lag_slow < 0.1, (
            f"log_mdot_msun_yr={log_mdot_msun_yr} wavelength={wavelength} inclination={inclination}: "
            f"mean lag exact={lag_slow:.3f} fast={lag_fast:.3f}"
        )


def test_fast_response_approximation_degrades_away_from_reference():
    """Documents, as a running check rather than just prose in
    docs/thin_disk_response.md, that the self-similar-stretching
    approximation is honestly worse far from the table's reference point
    (here: high inclination, a wavelength far from the default 5000 A
    reference) than close to it -- not a bug, an inherent property of
    stretching a disk whose inner radius doesn't itself stretch."""
    fast = build_thin_disk_response_fast(M_BH, **_SMALL_TABLE_KWARGS)
    tau_grid = jnp.linspace(-2.0, 15.0, 300)

    psi_slow_near = np.asarray(thin_disk_response(tau_grid, typical_log_mdot(M_BH), 5000.0, 30.0, M_BH))
    psi_fast_near = np.asarray(fast(tau_grid, typical_log_mdot(M_BH), 5000.0, 30.0, M_BH))
    near_diff = np.max(np.abs(psi_slow_near - psi_fast_near))

    psi_slow_far = np.asarray(thin_disk_response(tau_grid, typical_log_mdot(M_BH) - 0.5, 7000.0, 85.0, M_BH))
    psi_fast_far = np.asarray(fast(tau_grid, typical_log_mdot(M_BH) - 0.5, 7000.0, 85.0, M_BH))
    far_diff = np.max(np.abs(psi_slow_far - psi_fast_far))

    assert far_diff > near_diff


def test_fast_response_has_nonzero_gradients():
    """Same discipline as tophat_response_free / thin_disk_response: a
    lookup-table response could in principle have zero gradient if built
    wrong (e.g. a hard nearest-neighbour lookup instead of jnp.interp), so
    this is checked directly rather than assumed from jnp.interp's
    reputation. Checked both exactly on an inclination grid point (30.0 is
    in _SMALL_TABLE_KWARGS's incl_grid) and off it (41.3 is not), since a
    naive implementation could special-case exact grid hits."""
    fast = build_thin_disk_response_fast(M_BH, **_SMALL_TABLE_KWARGS)
    tau_grid = jnp.linspace(0.0, 15.0, 300)

    def mean_lag_mdot(log_mdot_msun_yr):
        return jnp.sum(fast(tau_grid, log_mdot_msun_yr, 5000.0, 30.0, M_BH) * tau_grid)

    def mean_lag_incl(inclination):
        return jnp.sum(fast(tau_grid, typical_log_mdot(M_BH), 5000.0, inclination, M_BH) * tau_grid)

    assert float(jax.grad(mean_lag_mdot)(typical_log_mdot(M_BH))) != 0.0
    assert float(jax.grad(mean_lag_incl)(30.0)) != 0.0
    assert float(jax.grad(mean_lag_incl)(41.3)) != 0.0


def test_fast_response_is_faster_than_the_exact_disk_integral():
    """Timed through jax.jit, not eager repeated calls -- eager timing
    measures mostly per-call Python dispatch overhead, not the real cost
    NUTS actually pays (NumPyro jax.jit-compiles the whole log-density
    function once; every leapfrog step reuses that compiled executable).
    Confirmed directly this matters a lot: at n_tau=600, an eager call to
    thin_disk_response took ~134ms; the same call through jax.jit took
    ~2.6ms. Earlier versions of this test (and CLAUDE.md/docs quoting its
    numbers) used eager timing and got a materially wrong picture as a
    result -- properly measured, the fast path was found to be a larger
    win at one point in this feature's history than the eager numbers
    had claimed (~13x vs. a claimed ~4x), not just a smaller one, i.e. the
    eager/jit gap doesn't even bias consistently in one direction. See
    CLAUDE.md decision #9's "methodology correction" paragraph."""
    fast = build_thin_disk_response_fast(M_BH, **_SMALL_TABLE_KWARGS)
    tau_grid = jnp.linspace(0.0, 15.0, 400)

    def _bench(fn):
        fn_jit = jax.jit(fn)
        fn_jit(typical_log_mdot(M_BH), 30.0).block_until_ready()  # compile once, outside timing
        t0 = time.time()
        for _ in range(100):
            fn_jit(typical_log_mdot(M_BH), 30.0).block_until_ready()
        return (time.time() - t0) / 100

    slow_time = _bench(lambda log_mdot_msun_yr, inclination: thin_disk_response(tau_grid, log_mdot_msun_yr, 5000.0, inclination, M_BH))
    fast_time = _bench(lambda log_mdot_msun_yr, inclination: fast(tau_grid, log_mdot_msun_yr, 5000.0, inclination, M_BH))

    assert fast_time < slow_time, (
        f"expected the templated lookup to be at least somewhat faster than "
        f"the disk integral, got slow={slow_time:.5f}s fast={fast_time:.5f}s"
    )


def test_table_and_fast_wrapper_agree():
    """build_thin_disk_response_fast is documented as a thin convenience
    wrapper around build_thin_disk_response_table + thin_disk_response_from_table
    -- check that's actually true, not just described that way."""
    table = build_thin_disk_response_table(M_BH, **_SMALL_TABLE_KWARGS)
    fast = build_thin_disk_response_fast(M_BH, **_SMALL_TABLE_KWARGS)
    tau_grid = jnp.linspace(0.0, 15.0, 300)

    direct = thin_disk_response_from_table(table, tau_grid, typical_log_mdot(M_BH), 5000.0, 30.0)
    via_wrapper = fast(tau_grid, typical_log_mdot(M_BH), 5000.0, 30.0, M_BH)
    np.testing.assert_array_equal(np.asarray(direct), np.asarray(via_wrapper))
    assert fast.table is not None


@pytest.mark.fit
def test_fast_response_is_usable_via_the_existing_swap_mechanism():
    """Matches the same (tau_grid, log_mdot_msun_yr, wavelength, inclination, M_BH)
    contract as response_function/thin_disk_response, so it plugs into the
    existing discotau.model.response_function swap point (CLAUDE.md decision
    #5) with no further wiring -- run a (tiny, fast-table) real fit through it."""
    import discotau.model as model_mod
    from discotau.synthetic import generate_synthetic_dataset
    from discotau.echofit import EchoFit

    original = model_mod.response_function
    try:
        model_mod.response_function = build_thin_disk_response_fast(M_BH, **_SMALL_TABLE_KWARGS)

        data = generate_synthetic_dataset(
            M_BH=M_BH, bands={"g": 4770.0}, n_obs_per_band=15,
            n_freq=6, n_tau=60, tau_max=40.0, noise_level=0.08, seed=0,
        )
        ef = EchoFit(M_BH=M_BH)
        for name, d in data["bands"].items():
            ef.add_lightcurve(name, wavelength=d["wavelength"], t=d["t"], y=d["y"], yerr=d["yerr"])
        ef.build_grid(n_freq=6, n_tau=60)
        ef.fit(rng_seed=0, num_warmup=10, num_samples=10, progress_bar=False)
    finally:
        model_mod.response_function = original
