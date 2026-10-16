"""
profile_pipeline.py
====================

One-off performance profile of the whole ``discotau`` pipeline, organised
around the one question that actually matters for deciding what to
optimise: does this cost get paid **once per run** (fixed, however long you
fit for), or **every NUTS step** (scales with total samples, so it's what
actually slows a long run down)?

One-off: grid building, the thin-disk fast response's one-off template
table build, NUTS's own JIT-compile+warmup, and report/plot generation.

Per-iteration (paid every leapfrog step -- total added cost is roughly
this times the total number of leapfrog steps across the whole run): each
response-function family's per-call cost, the closed-form convolution step
(``transfer_coeffs`` + ``compute_echo``), a full-model potential-energy/
gradient evaluation, and NUTS's own measured steady-state per-sample cost
(the real, end-to-end number -- the others are a breakdown of what's
inside it, and depend on which response function you've chosen).

Writes one chart per category, a results table (with a one-off/
per-iteration column), and a small ``report.html`` tying them together to
``--outdir`` -- meant to focus future performance work on real numbers
rather than guesses.

Every per-call timing here is measured *after* a warm-up call under
``jax.jit``, not eager Python -- see ``CLAUDE.md`` decision #9: an eager
call to ``thin_disk_response`` was found to cost ~50x what the same call
costs once actually jit-compiled, which is the only way NumPyro ever runs
it during a real fit, so an eager measurement would badly mislead which
part of the pipeline actually matters.

Usage
-----
    python scripts/profile_pipeline.py
    python scripts/profile_pipeline.py --outdir /tmp/discotau_profile

Takes a few minutes -- NUTS's own JIT compilation dominates the wall time
of the NUTS-timing section, which is rather the point: separating that
one-off cost from the steady-state per-sample cost it's trying to measure.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

from discotau import forward_model
from discotau.echofit import EchoFit
from discotau.model import reverberation_model
from discotau.synthetic import generate_synthetic_dataset

# Which measurements are paid once per run (independent of how long you
# fit for) versus every NUTS step (scales with total samples -- this is
# what actually slows a long run down). Everything not listed here is
# per-iteration; see the module docstring.
ONE_OFF_KEYS = {
    "grid_build",
    "thin_disk_fast_table_build_once",
    "nuts_one_off_compile_and_warmup",
    "report_generation",
}


def _time_jit(fn, *args, n_repeat=50, **kwargs):
    """Return ``(first_call_seconds, steady_state_seconds_per_call)`` for
    ``jax.jit(fn)(*args, **kwargs)``. ``first_call`` includes the one-off
    trace+compile cost; ``steady_state`` is the mean of ``n_repeat`` calls
    to the now-compiled function, i.e. what NUTS actually pays per call
    once warmed up."""
    jfn = jax.jit(fn)
    t0 = time.perf_counter()
    out = jax.block_until_ready(jfn(*args, **kwargs))
    first_call_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    for _ in range(n_repeat):
        out = jfn(*args, **kwargs)
    jax.block_until_ready(out)
    steady_state_s = (time.perf_counter() - t0) / n_repeat
    return first_call_s, steady_state_s


def _build_reference_echofit(seed: int) -> tuple[EchoFit, dict]:
    """A representative 2-band synthetic setup, matching the scale
    ``scripts/smoke_test.py`` and ``tests/test_recovery.py`` already use --
    small enough to run several times in a few minutes, large enough to be
    a realistic stand-in for a real campaign. Deliberately stops short of
    ``build_grid()`` so callers can time that step on its own, without
    ``generate_synthetic_dataset``'s own (much larger) cost folded in."""
    data = generate_synthetic_dataset(
        M_BH=1.0e9, log_mdot_true=0.0, inclination_true=35.0,
        sigma_drw_true=0.3, tau_drw_true=30.0,
        bands={"g": 4770.0, "i": 7625.0}, t_span=200.0, n_obs_per_band=50,
        n_freq=15, n_tau=100, tau_max=50.0, noise_level=0.05, seed=seed,
    )
    # drw_prior=True: pinned so this reproduces the exact numbers already
    # documented in CLAUDE.md decisions #9/#17 (measured before decision
    # #19 made the random-walk prior the default), not a claim DRW is
    # still the default.
    ef = EchoFit(M_BH=data["truth"]["M_BH"], drw_prior=True)
    for name, d in data["bands"].items():
        ef.add_lightcurve(name, wavelength=d["wavelength"], t=d["t"], y=d["y"], yerr=d["yerr"])
    return ef, data


def profile_response_functions(ef: EchoFit, n_repeat: int) -> dict:
    tau_grid, log_mdot, wavelength, inclination, M_BH = (
        ef.tau_grid, 0.1, 5000.0, 45.0, ef.M_BH,
    )
    results = {}

    _, results["skew_normal"] = _time_jit(
        forward_model.response_function, tau_grid, log_mdot, wavelength, inclination, M_BH,
        n_repeat=n_repeat,
    )
    _, results["thin_disk_default_smoothing"] = _time_jit(
        forward_model.thin_disk_response, tau_grid, log_mdot, wavelength, inclination, M_BH,
        n_repeat=n_repeat,
    )
    _, results["thin_disk_no_smoothing"] = _time_jit(
        forward_model.thin_disk_response, tau_grid, log_mdot, wavelength, inclination, M_BH,
        n_repeat=n_repeat, smoothing_days=0.0,
    )

    t0 = time.perf_counter()
    fast_response = forward_model.build_thin_disk_response_fast(M_BH=M_BH)
    table_build_s = time.perf_counter() - t0
    _, results["thin_disk_fast_templated"] = _time_jit(
        fast_response, tau_grid, log_mdot, wavelength, inclination, M_BH,
        n_repeat=n_repeat,
    )
    results["thin_disk_fast_table_build_once"] = table_build_s
    return results


def profile_convolution(ef: EchoFit, n_repeat: int) -> dict:
    tau_grid, freqs = ef.tau_grid, ef.freqs
    psi = forward_model.response_function(tau_grid, 0.1, 5000.0, 45.0, ef.M_BH)
    S = np.ones(len(freqs))
    C = np.zeros(len(freqs))
    t_obs = next(iter(ef.bands.values()))["t"]

    def convolve(tau_grid, psi, freqs, S, C, t_obs):
        A, B = forward_model.transfer_coeffs(tau_grid, psi, freqs)
        return forward_model.compute_echo(S, C, freqs, A, B, t_obs)

    _, steady = _time_jit(convolve, tau_grid, psi, freqs, S, C, t_obs, n_repeat=n_repeat)
    return {"transfer_coeffs_plus_compute_echo": steady}


def profile_full_model(ef: EchoFit, n_repeat: int) -> dict:
    """Time one full-model potential-energy evaluation and its gradient --
    exactly what NUTS computes at every leapfrog step -- via NumPyro's own
    ``initialize_model`` rather than hand-listing every sample site, so
    this doesn't need updating if model.py's sites ever change."""
    from numpyro.infer.util import initialize_model

    kwargs = ef._model_kwargs()
    init_info, potential_fn, _, _ = initialize_model(
        jax.random.PRNGKey(0), reverberation_model, model_kwargs=kwargs,
    )
    z = init_info.z

    _, potential_s = _time_jit(potential_fn, z, n_repeat=n_repeat)
    _, grad_s = _time_jit(lambda z: jax.grad(potential_fn)(z), z, n_repeat=n_repeat)
    return {"potential_energy_eval": potential_s, "potential_energy_grad": grad_s}


def profile_nuts_and_report(ef: EchoFit, data: dict, outdir: Path) -> dict:
    """Two short fits at different sample counts isolate NUTS's one-off
    JIT-compile+warmup overhead from its steady-state per-sample cost: with
    warmup fixed, the *difference* in total wall time between a short and a
    long run, divided by the difference in sample count, is the pure
    post-warmup per-sample cost (compile time and warmup cost are identical
    in both runs and cancel out)."""
    warmup = 50
    n_short, n_long = 50, 250

    t0 = time.perf_counter()
    ef.fit(rng_seed=0, num_warmup=warmup, num_samples=n_short, num_chains=1, progress_bar=False)
    t_short = time.perf_counter() - t0

    t0 = time.perf_counter()
    ef.fit(rng_seed=0, num_warmup=warmup, num_samples=n_long, num_chains=1, progress_bar=False)
    t_long = time.perf_counter() - t0

    per_sample_s = (t_long - t_short) / (n_long - n_short)
    compile_and_warmup_s = t_short - per_sample_s * n_short

    from discotau import reporting
    t0 = time.perf_counter()
    reporting.generate_report(ef, outdir / "example_fit_report", fit_seconds=t_long, truth=data["truth"])
    report_s = time.perf_counter() - t0

    return {
        f"fit_{warmup}warmup_{n_short}samples_total": t_short,
        f"fit_{warmup}warmup_{n_long}samples_total": t_long,
        "nuts_per_sample_post_warmup": per_sample_s,
        "nuts_one_off_compile_and_warmup": compile_and_warmup_s,
        "report_generation": report_s,
    }


def _plot_one_off_costs(all_results: dict, path: Path):
    stages = [
        ("Grid build", all_results["grid_build"]),
        ("Thin-disk fast response:\none-off template table build", all_results["response"]["thin_disk_fast_table_build_once"]),
        ("NUTS: one-off\nJIT compile + warmup", all_results["nuts"]["nuts_one_off_compile_and_warmup"]),
        ("Report generation\n(plots + html)", all_results["nuts"]["report_generation"]),
    ]
    labels = [s[0] for s in stages]
    values_s = [s[1] for s in stages]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(labels, values_s, color="tab:gray")
    ax.set_xscale("log")
    ax.set_xlabel("time (seconds, log scale)")
    ax.set_title("One-off costs -- paid once, don't scale with run length")
    ax.grid(alpha=0.6, axis="x")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_per_iteration_costs(all_results: dict, path: Path):
    stages = [
        ("Response fn: skew-normal\n(default response_function)", all_results["response"]["skew_normal"], "tab:blue"),
        ("Response fn: thin_disk_response\n(default smoothing)", all_results["response"]["thin_disk_default_smoothing"], "tab:blue"),
        ("Response fn: thin_disk_response\n(smoothing_days=0)", all_results["response"]["thin_disk_no_smoothing"], "tab:blue"),
        ("Response fn: thin_disk\n(fast/templated)", all_results["response"]["thin_disk_fast_templated"], "tab:blue"),
        ("Convolution\n(transfer_coeffs+compute_echo)", all_results["convolution"]["transfer_coeffs_plus_compute_echo"], "tab:blue"),
        ("Full model potential\nenergy evaluation", all_results["full_model"]["potential_energy_eval"], "tab:blue"),
        ("Full model potential\nenergy gradient", all_results["full_model"]["potential_energy_grad"], "tab:blue"),
        ("NUTS: measured per-sample cost\n(real, end-to-end)", all_results["nuts"]["nuts_per_sample_post_warmup"], "tab:red"),
    ]
    labels = [s[0] for s in stages]
    values_ms = [s[1] * 1000 for s in stages]
    colors = [s[2] for s in stages]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(labels, values_ms, color=colors)
    ax.set_xscale("log")
    ax.set_xlabel("time per call, jax-jitted steady state (ms, log scale)")
    ax.set_title("Per-iteration costs -- scale with total run length")
    ax.grid(alpha=0.6, axis="x")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _write_html_report(outdir: Path, all_results: dict):
    flat = [("grid", "grid_build", all_results["grid_build"])]
    for stage, section in all_results.items():
        if isinstance(section, dict):
            flat.extend((stage, name, value) for name, value in section.items())

    def _rows(category, scale):
        return "\n".join(
            f"<tr><td>{stage}</td><td>{name}</td><td>{value * scale:.3f}</td></tr>"
            for stage, name, value in flat
            if not name.startswith("fit_") and (name in ONE_OFF_KEYS) == (category == "one-off")
        )

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>discotau pipeline profile</title>
<style>
body {{ font-family: sans-serif; max-width: 900px; margin: 2em auto; }}
table {{ border-collapse: collapse; width: 100%; margin-bottom: 2em; }}
th, td {{ border: 1px solid #ccc; padding: 4px 8px; text-align: right; }}
th:first-child, td:first-child, th:nth-child(2), td:nth-child(2) {{ text-align: left; }}
img {{ max-width: 100%; margin-bottom: 2em; }}
</style>
</head>
<body>
<h1>discotau pipeline profile</h1>
<p>Generated by <code>scripts/profile_pipeline.py</code>. All per-call
timings are jax-jitted steady state (post-compile) -- see
<code>CLAUDE.md</code> decision #9 for why eager timing would badly
mislead here. Split into <strong>one-off</strong> costs (paid once, however
long you fit for) and <strong>per-iteration</strong> costs (paid every NUTS
step, so total added cost is roughly the per-iteration number times the
total number of leapfrog steps across the whole run) -- it's the
per-iteration costs that determine how a long run scales, not the one-off
ones.</p>

<img src="one_off_costs.png" alt="one-off costs">
<img src="per_iteration_costs.png" alt="per-iteration costs">

<p>The red "NUTS: measured per-sample cost" bar is the real, end-to-end
number (one NUTS sample = one or more leapfrog steps, each a potential
energy + gradient evaluation) -- it's naturally bigger than a single
potential-energy-gradient call above it; the other blue bars are a
breakdown of what's inside it, or alternative response-function choices
that would change it.</p>

<h2>One-off costs (seconds)</h2>
<table>
<tr><th>stage</th><th>measurement</th><th>seconds</th></tr>
{_rows("one-off", 1.0)}
</table>

<h2>Per-iteration costs (ms)</h2>
<table>
<tr><th>stage</th><th>measurement</th><th>ms</th></tr>
{_rows("per-iteration", 1000.0)}
</table>

<p>See <code>results.json</code> in this folder for the same numbers in
seconds, machine-readable -- it also has the two raw <code>fit_*_total</code>
wall times the one-off/per-iteration split above was derived from (each
mixes both, so they're left out of the tables here).</p>
</body>
</html>
"""
    (outdir / "report.html").write_text(html)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--outdir", type=Path, default=Path("profiling_output"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-repeat", type=int, default=50, help="Repeated calls per jitted timing.")
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    # JAX/XLA backend initialisation happens lazily on the first array op in
    # the process and can itself take a second or more -- do it here, before
    # any timing starts, so grid_build below measures actual grid-building
    # work rather than one-time process startup cost.
    jax.block_until_ready(jnp.zeros(1) + 1.0)

    print("Building reference dataset + EchoFit...")
    ef, data = _build_reference_echofit(args.seed)

    t0 = time.perf_counter()
    ef.build_grid(n_freq=15, n_tau=100)
    grid_build_s = time.perf_counter() - t0

    print("Profiling response functions...")
    response_results = profile_response_functions(ef, args.n_repeat)

    print("Profiling the convolution step (transfer_coeffs + compute_echo)...")
    convolution_results = profile_convolution(ef, args.n_repeat)

    print("Profiling a full model potential-energy evaluation + gradient...")
    full_model_results = profile_full_model(ef, args.n_repeat)

    print("Profiling NUTS one-off overhead vs steady-state per-sample cost "
          "(two short fits -- this is the slow part)...")
    nuts_results = profile_nuts_and_report(ef, data, args.outdir)

    all_results = {
        "grid_build": grid_build_s,
        "response": response_results,
        "convolution": convolution_results,
        "full_model": full_model_results,
        "nuts": nuts_results,
    }

    flat = [("grid_build", grid_build_s)]
    for section in ("response", "convolution", "full_model", "nuts"):
        flat.extend(all_results[section].items())

    print("\n=== One-off costs (paid once, don't scale with run length) -- ms ===")
    for name, value in flat:
        if name in ONE_OFF_KEYS:
            print(f"{name:<45}{value * 1000:>12.3f}")

    print("\n=== Per-iteration costs (paid every NUTS step, scale with run length) -- ms ===")
    for name, value in flat:
        if name not in ONE_OFF_KEYS and not name.startswith("fit_"):
            print(f"{name:<45}{value * 1000:>12.3f}")

    print("\n(raw diagnostics -- mix both categories, see results.json)")
    for name, value in flat:
        if name.startswith("fit_"):
            print(f"{name:<45}{value:>12.3f}s")

    print("\nRendering charts...")
    _plot_one_off_costs(all_results, args.outdir / "one_off_costs.png")
    _plot_per_iteration_costs(all_results, args.outdir / "per_iteration_costs.png")
    _write_html_report(args.outdir, all_results)

    with open(args.outdir / "results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nSaved charts, results.json, and report.html to {args.outdir}/")
    print(f"Open {args.outdir / 'report.html'} to view everything in one page.")


if __name__ == "__main__":
    main()
