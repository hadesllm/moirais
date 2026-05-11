"""Tests for refco.py - reflection coefficients."""
import numpy as np
import pytest
from morie.fn.refco import reflection_coeff_fn, refco


def test_reflection_coefficients_returns_descriptive_result():
    x = np.random.default_rng(42).standard_normal(256)
    result = reflection_coeff_fn(x)
    assert result.name == "reflection_coefficients"
    assert "reflection_coefficients" in result.extra


def test_reflection_coefficients_count():
    x = np.random.default_rng(42).standard_normal(256)
    result = reflection_coeff_fn(x, order=10)
    assert len(result.extra["reflection_coefficients"]) > 0


def test_reflection_coefficients_bounded():
    x = np.random.default_rng(42).standard_normal(256)
    result = reflection_coeff_fn(x, order=8)
    k = result.extra["reflection_coefficients"]
    assert np.all(np.abs(k) <= 1.0 + 1e-9)


def test_refco_alias():
    x = np.random.default_rng(42).standard_normal(64)
    result = refco(x, order=5)
    assert result.name == "reflection_coefficients"
