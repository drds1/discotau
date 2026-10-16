"""
Two tiers of tests:

- the quick tier, ``pytest -m "not fit"`` (what CI runs on every push and pull
  request): unit tests that fit nothing, a couple of minutes in all;
- the fitting tier, ``pytest -m fit``: every test that runs NUTS or a direct
  solve (``optimise()``, ``nested_laplace``), each paying its own JAX compile.
  Run it by hand before a release or after a change to the model or solvers.

A test that fits must be marked ``@pytest.mark.fit`` (or the module given
``pytestmark = pytest.mark.fit``). The guard below enforces it: in a test
without the mark, any sampler or solver call fails at once, so a fitting test
can't slip into the quick tier unnoticed.
"""

import numpyro.infer
import pytest

import discotau.echofit
import discotau.nested_laplace


def _blocked(name):
    def raise_(*args, **kwargs):
        raise pytest.fail.Exception(
            f"{name} called in a test without @pytest.mark.fit: mark the test (it fits a model, "
            "so it belongs in the fitting tier, not the quick tier CI runs on every push)")

    return raise_


@pytest.fixture(autouse=True)
def _guard_unmarked_fits(request, monkeypatch):
    if request.node.get_closest_marker("fit") is not None:
        return
    monkeypatch.setattr(numpyro.infer.MCMC, "run", _blocked("MCMC.run (NUTS)"))
    monkeypatch.setattr(discotau.echofit.EchoFit, "_optimise_laplace", _blocked("optimise(method='laplace')"))
    monkeypatch.setattr(discotau.echofit.EchoFit, "nested_laplace", _blocked("nested_laplace()"))
    monkeypatch.setattr(discotau.nested_laplace, "run", _blocked("nested_laplace.run()"))
