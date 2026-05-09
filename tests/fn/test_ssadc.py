"""Tests for ssadc.py - SSA decomposition."""
import numpy as np
from moirais.fn.ssadc import ssa_decompose_fn, ssadc


def test_ssadc_returns_result():
    x = np.random.default_rng(42).standard_normal(100)
    result = ssa_decompose_fn(x)
    assert result.name == "ssa_decompose"
    assert "singular_values" in result.extra
    assert "components" in result.extra


def test_ssadc_singular_values_descending():
    x = np.random.default_rng(42).standard_normal(100)
    result = ssa_decompose_fn(x, L=30)
    sv = result.extra["singular_values"]
    assert np.all(sv[:-1] >= sv[1:] - 1e-10)


def test_ssadc_alias():
    x = np.random.default_rng(42).standard_normal(50)
    result = ssadc(x, L=10)
    assert result.name == "ssa_decompose"
