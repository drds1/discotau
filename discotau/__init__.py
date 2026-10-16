"""
discotau
========

Bayesian modelling of AGN reverberation-mapping light curves as a delayed,
smoothed echo of an unobserved driving (lamppost) light curve.

The public entry point is the :class:`EchoFit` class::

    from discotau import EchoFit

    ef = EchoFit(M_BH=1e8)
    ef.add_lightcurve("g", wavelength=4770.0, t=t_g, y=y_g, yerr=yerr_g)
    ef.add_lightcurve("i", wavelength=7625.0, t=t_i, y=y_i, yerr=yerr_i)
    ef.build_grid()
    ef.fit(rng_seed=0)
    ef.plot_lightcurve_fits()
"""

from importlib.metadata import PackageNotFoundError, version as _version

from .echofit import EchoFit
from .forward_model import lag_scaling, response_function, compute_echo
from .synthetic import generate_synthetic_dataset

__all__ = [
    "EchoFit",
    "lag_scaling",
    "response_function",
    "compute_echo",
    "generate_synthetic_dataset",
]

# Read from the installed package's metadata, so pyproject.toml is the one
# place the version is set (see docs/releasing.md).
try:
    __version__ = _version("discotau")
except PackageNotFoundError:  # imported from a source tree that was never installed
    __version__ = "0+unknown"
