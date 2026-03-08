"""Tests for jHBV plugin registration."""

import pytest


def test_register_function_exists():
    """jhbv module should have a register() function."""
    import jhbv
    assert hasattr(jhbv, 'register')
    assert callable(jhbv.register)


def test_entry_point_discoverable():
    """The hbv entry point should be discoverable."""
    from importlib.metadata import entry_points
    eps = entry_points(group='symfluence.plugins')
    names = [ep.name for ep in eps]
    assert 'hbv' in names


def test_register_creates_config_adapter():
    """Calling register() should register HBV config adapter."""
    import jhbv
    jhbv.register()

    from symfluence.core.registries import R
    assert 'HBV' in R.config_adapters
