"""The lazy sample dicts a direct solve leaves (CLAUDE.md decision #30): the
exact linear draws are made once, on first need, and shared by ``samples`` and
``_samples_by_chain``; reading a nonlinear parameter never triggers them. A
stand-in replaces the (slow) draw, so this needs no fit."""

import numpy as np

from discotau.echofit import _LazySamples, _LinearDraws


class _FakeEchoFit:
    marginalise_linear = False

    def __init__(self):
        self.calls = 0

    def _add_linear_draws(self, nonlinear_by_chain, rng_seed):
        self.calls += 1
        assert self.marginalise_linear  # the draws are made with the marginalised model
        n = next(iter(nonlinear_by_chain.values())).shape[1]
        return dict(nonlinear_by_chain, S=np.full((1, n, 3), 2.0), y_pred_g=np.full((1, n, 5), 7.0))


def _lazy():
    ef = _FakeEchoFit()
    nonlinear = {"log_mdot": np.arange(4.0)[None, :]}
    resolver = _LinearDraws(ef, nonlinear, rng_seed=0, sites=["S", "y_pred_g"])
    by_chain = _LazySamples(nonlinear, resolver, by_chain=True)
    flat = _LazySamples({k: v[0] for k, v in nonlinear.items()}, resolver, by_chain=False)
    return ef, flat, by_chain


def test_nonlinear_reads_and_membership_do_not_draw():
    ef, flat, _ = _lazy()
    np.testing.assert_array_equal(flat["log_mdot"], np.arange(4.0))
    assert "S" in flat and "y_pred_g" in flat and "nope" not in flat
    assert ef.calls == 0


def test_a_linear_read_draws_once_for_both_dicts():
    ef, flat, by_chain = _lazy()
    assert flat["S"].shape == (4, 3)  # chain axis dropped
    assert by_chain["S"].shape == (1, 4, 3)
    assert by_chain["y_pred_g"].shape == (1, 4, 5)
    assert ef.calls == 1
    assert ef.marginalise_linear is False  # restored afterwards


def test_walking_the_dict_draws_and_includes_every_site():
    ef, flat, _ = _lazy()
    assert set(flat) == {"log_mdot", "S", "y_pred_g"}
    assert len(flat) == 3 and set(dict(flat.items())) == set(flat.copy())
    assert ef.calls == 1
