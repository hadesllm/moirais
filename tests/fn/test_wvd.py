"""Tests for wvd.py - Wigner-Ville distribution."""
import numpy as np
import pytest
from moirais.fn.wvd import wigner_ville_fn, wvd


def test_wigner_ville_returns_descriptive_result():
    x = np.random.default_rng(42).standard_normal(32)
    result = wigner_ville_fn(x)
    assert result.name == "wigner_ville"
    assert "wvd" in result.extra


def test_wigner_ville_output_2d():
    x = np.random.default_rng(42).standard_normal(32)
    result = wigner_ville_fn(x)
    assert result.extra["wvd"].ndim == 2


def test_wigner_ville_finite():
    x = np.random.default_rng(42).standard_normal(32)
    result = wigner_ville_fn(x)
    assert np.all(np.isfinite(result.extra["wvd"]))


def test_wvd_alias():
    x = np.random.default_rng(42).standard_normal(32)
    result = wvd(x)
    assert result.name == "wigner_ville"
