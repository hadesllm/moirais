"""Tests for prcor.py - partial autocorrelation coefficients."""
import numpy as np
import pytest
from morie.fn.prcor import parcor_fn, prcor


def test_parcor_returns_descriptive_result():
    x = np.random.default_rng(42).standard_normal(256)
    result = parcor_fn(x)
    assert result.name == "parcor_coefficients"
    assert "parcor" in result.extra


def test_parcor_count():
    x = np.random.default_rng(42).standard_normal(256)
    result = parcor_fn(x, order=10)
    assert len(result.extra["parcor"]) > 0


def test_parcor_bounded():
    x = np.random.default_rng(42).standard_normal(256)
    result = parcor_fn(x, order=8)
    k = result.extra["parcor"]
    assert np.all(np.abs(k) <= 1.0 + 1e-9)


def test_prcor_alias():
    x = np.random.default_rng(42).standard_normal(64)
    result = prcor(x, order=5)
    assert result.name == "parcor_coefficients"
