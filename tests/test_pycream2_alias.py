"""The deprecated ``pycream2`` alias of ``discotau`` (renamed October 2026):
the old name still imports, warns, and points at the very same modules, so a
swap made through either name (CLAUDE.md decision #5) affects both."""

import importlib
import sys

import pytest

import discotau
import discotau.model


def _fresh_import(name):
    for key in [k for k in sys.modules if k == "pycream2" or k.startswith("pycream2.")]:
        del sys.modules[key]
    return importlib.import_module(name)


def test_import_pycream2_warns_and_is_discotau():
    with pytest.warns(DeprecationWarning, match="renamed discotau"):
        pycream2 = _fresh_import("pycream2")
    assert pycream2 is discotau
    assert pycream2.EchoFit is discotau.EchoFit


def test_pycream2_submodules_are_the_discotau_modules():
    with pytest.warns(DeprecationWarning):
        model = _fresh_import("pycream2.model")
    assert model is discotau.model
    from pycream2.forward_model import thin_disk_response
    from discotau.forward_model import thin_disk_response as new

    assert thin_disk_response is new


def test_swapping_through_the_old_name_affects_the_new_one():
    with pytest.warns(DeprecationWarning):
        old_model = _fresh_import("pycream2.model")
    original = discotau.model.response_function
    try:
        sentinel = object()
        old_model.response_function = sentinel
        assert discotau.model.response_function is sentinel
    finally:
        discotau.model.response_function = original
