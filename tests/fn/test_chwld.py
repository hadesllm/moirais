"""Tests for chwld.py - Choi-Williams distribution."""
import numpy as np
import pytest
from moirais.fn.chwld import choi_williams_fn, chwld


def test_choi_williams_returns_descriptive_result():
    x = np.random.default_rng(42).standard_normal(32)
    result = choi_williams_fn(x)
    assert result.name == "choi_williams"
    assert "cwd" in result.extra


def test_choi_williams_output_2d():
    x = np.random.default_rng(42).standard_normal(32)
    result = choi_williams_fn(x)
    assert result.extra["cwd"].ndim == 2


def test_choi_williams_finite():
    x = np.random.default_rng(42).standard_normal(32)
    result = choi_williams_fn(x)
    assert np.all(np.isfinite(result.extra["cwd"]))


def test_chwld_alias():
    x = np.random.default_rng(42).standard_normal(32)
    result = chwld(x, sigma=0.5)
    assert result.name == "choi_williams"
