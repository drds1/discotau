"""
inference.py
============

A small, clean wrapper around NumPyro's NUTS sampler.
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import jax
from numpyro.infer import MCMC, NUTS


def _build_kernel(
    model: Callable, target_accept_prob: float, max_tree_depth: Optional[int],
    init_strategy=None, dense_mass: bool = False,
):
    nuts_kwargs = dict(target_accept_prob=target_accept_prob, dense_mass=dense_mass)
    if max_tree_depth is not None:
        nuts_kwargs["max_tree_depth"] = max_tree_depth
    if init_strategy is not None:
        nuts_kwargs["init_strategy"] = init_strategy
    return NUTS(model, **nuts_kwargs)


def run_mcmc(
    model: Callable,
    model_kwargs: dict,
    rng_key: jax.Array,
    num_warmup: int = 1000,
    num_samples: int = 1000,
    num_chains: int = 1,
    target_accept_prob: float = 0.85,
    max_tree_depth: Optional[int] = None,
    chain_method: str = "parallel",
    progress_bar: bool = True,
    init_strategy=None,
    dense_mass: bool = False,
) -> MCMC:
    """Run NUTS on ``model(**model_kwargs)`` and return the fitted MCMC object.

    Parameters
    ----------
    model : callable
        A NumPyro model function, e.g. ``discotau.model.reverberation_model``.
    model_kwargs : dict
        Keyword arguments passed through to ``model`` on every call
        (frequency grid, tau grid, fixed M_BH, band data, ...).
    rng_key : jax.Array
        A ``jax.random.PRNGKey``.
    num_warmup, num_samples, num_chains : int
        Standard NUTS/MCMC controls.
    target_accept_prob : float
        NUTS target acceptance probability.
    max_tree_depth : int, optional
        Caps each NUTS trajectory at ``2**max_tree_depth - 1`` leapfrog
        steps (NumPyro's own default is 10, i.e. up to 1023 steps). If the
        posterior geometry is difficult (e.g. hierarchical funnels), NUTS
        can spend most samples at that ceiling; lowering this bounds the
        worst-case per-sample cost at the expense of exploration efficiency.
        Left as NumPyro's default when not given.
    dense_mass : bool
        Use a full covariance-based mass matrix (estimated during warmup)
        instead of NumPyro's default diagonal one. See
        ``scripts/profile_pipeline.py``'s docstring and CLAUDE.md decision
        #17: this model's parameters are correlated enough that a diagonal
        mass matrix makes NUTS spend nearly every sample pinned at
        ``max_tree_depth``'s ceiling (confirmed directly: mean ~858 leapfrog
        steps/sample, essentially always at or near 1023); ``dense_mass=True``
        cut that ~7.5x (to ~114 steps/sample) with no loss of recovery
        accuracy, at the cost of needing a longer warmup for the (much
        larger, O(P^2) vs O(P)) mass matrix to adapt properly -- a
        short-warmup dense-mass run showed a real divergence rate (6%) that
        a longer warmup brought back down to a healthy 1.5%. Off by default
        to keep existing behaviour unchanged; worth turning on for any
        run where the sampling phase dominates wall time (most real runs).
    chain_method : str
        ``"parallel"`` (NumPyro's default, one process/device per chain),
        ``"vectorized"`` (batch chains via ``vmap`` on a single device --
        useful for getting multiple chains for R-hat/ESS diagnostics
        without paying near-linear extra wall time when a run is dominated
        by fixed per-call overhead rather than compute), or ``"sequential"``.
    progress_bar : bool
        Whether to show NumPyro's sampling progress bar.
    init_strategy : callable, optional
        NumPyro init strategy, e.g. ``numpyro.infer.init_to_value(values={...})``
        for data-anchored starting guesses (``EchoFit`` uses this for each
        band's ``S_{band}``/``C_{band}``, see ``EchoFit._init_strategy``).
        Left as NumPyro's own default (``init_to_uniform``) when not given.

    Returns
    -------
    mcmc : numpyro.infer.MCMC
        Fitted MCMC object; use ``mcmc.get_samples()`` for posterior draws,
        ``mcmc.get_extra_fields()["potential_energy"]`` for the
        Badness-of-Fit trace (``plotting.plot_bof``), and
        ``mcmc.print_summary()`` / ``arviz`` for diagnostics.
    """
    kernel = _build_kernel(model, target_accept_prob, max_tree_depth, init_strategy, dense_mass)
    mcmc = MCMC(
        kernel,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        chain_method=chain_method,
        progress_bar=progress_bar,
    )
    # potential_energy (NUTS's own -log joint density) is what
    # plotting.plot_bof shows as the Badness-of-Fit trace -- not requested
    # by default (NumPyro only returns "diverging" unless asked).
    mcmc.run(rng_key, extra_fields=("potential_energy",), **model_kwargs)
    return mcmc


def run_mcmc_chunked(
    model: Callable,
    model_kwargs: dict,
    rng_key: jax.Array,
    num_warmup: int,
    num_samples: int,
    checkpoint_every: int,
    target_accept_prob: float = 0.85,
    max_tree_depth: Optional[int] = None,
    progress_bar: bool = True,
    init_last_state=None,
    n_already_done: int = 0,
    on_chunk_done: Optional[Callable] = None,
    init_strategy=None,
    dense_mass: bool = False,
):
    """Run NUTS in checkpointable chunks of up to ``checkpoint_every`` samples
    each, single chain only. Resumes from ``init_last_state`` (a previous
    chunk's ``mcmc.last_state``) if given, skipping warmup entirely --
    this is NumPyro's documented pattern for sequentially drawing samples
    (``mcmc.post_warmup_state = mcmc.last_state``, see ``MCMC.post_warmup_state``'s
    docstring). ``n_already_done`` should be the number of post-warmup
    samples already collected in prior chunks (from ``init_last_state``'s
    run), so this only samples the remainder of ``num_samples``.

    ``on_chunk_done(mcmc, last_state, n_done_total)`` is called after every
    chunk (including the warmup-containing first one) -- use it to persist
    the chunk's samples/extra_fields/state to disk for resuming later.

    Returns
    -------
    samples, samples_by_chain, extra_fields : dict
        Accumulated across all chunks (this call's + any already done before
        it, via ``init_last_state``); ``samples_by_chain`` has a leading
        ``(1, n_total_samples)`` shape for compatibility with
        ``plot_mcmc_diagnostics``.
    last_state
        The final chunk's ``mcmc.last_state``, for further resuming.
    """
    last_state = init_last_state
    n_done = n_already_done
    samples_chunks, samples_by_chain_chunks, extra_chunks = [], [], []

    if n_done >= num_samples:
        # Already fully sampled (e.g. resuming a run that had already
        # completed) -- nothing left to do.
        return {}, {}, {}, last_state

    while n_done < num_samples:
        this_chunk = min(checkpoint_every, num_samples - n_done)
        kernel = _build_kernel(model, target_accept_prob, max_tree_depth, init_strategy, dense_mass)
        mcmc = MCMC(
            kernel,
            num_warmup=0 if last_state is not None else num_warmup,
            num_samples=this_chunk,
            num_chains=1,
            progress_bar=progress_bar,
        )
        if last_state is not None:
            mcmc.post_warmup_state = last_state
            mcmc.run(last_state.rng_key, extra_fields=("potential_energy",), **model_kwargs)
        else:
            mcmc.run(rng_key, extra_fields=("potential_energy",), **model_kwargs)

        samples_chunks.append(mcmc.get_samples())
        samples_by_chain_chunks.append(mcmc.get_samples(group_by_chain=True))
        extra_chunks.append(mcmc.get_extra_fields())
        last_state = mcmc.last_state
        n_done += this_chunk

        if on_chunk_done is not None:
            on_chunk_done(mcmc, last_state, n_done)

    samples = {k: np.concatenate([c[k] for c in samples_chunks], axis=0) for k in samples_chunks[0]}
    samples_by_chain = {
        k: np.concatenate([c[k] for c in samples_by_chain_chunks], axis=1)
        for k in samples_by_chain_chunks[0]
    }
    extra_fields = {k: np.concatenate([c[k] for c in extra_chunks], axis=0) for k in extra_chunks[0]}
    return samples, samples_by_chain, extra_fields, last_state
