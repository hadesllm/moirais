"""Tests for rflcf.py - Reflection coefficients."""
import numpy as np
from moirais.fn.rflcf import reflection_coefficients_fn, rflcf


def test_rflcf_returns_result():
    ar = np.array([0.5, -0.3, 0.1])
    result = reflection_coefficients_fn(ar)
    assert result.name == "reflection_coefficients"
    assert "reflection_coeffs" in result.extra
    assert len(result.extra["reflection_coeffs"]) == 3


def test_rflcf_stability():
    ar = np.array([0.3, -0.2])
    result = reflection_coefficients_fn(ar)
    assert isinstance(result.extra["stable"], bool)


def test_rflcf_alias():
    ar = np.array([0.5, -0.3])
    result = rflcf(ar)
    assert result.name == "reflection_coefficients"
