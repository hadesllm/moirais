"""Tests for rlslt.py - RLS lattice adaptive filter."""
import numpy as np
from morie.fn.rlslt import rls_lattice_filter_fn, rlslt


def test_rlslt_returns_signal_result():
    rng = np.random.default_rng(42)
    x = rng.standard_normal(128)
    d = x + 0.1 * rng.standard_normal(128)
    result = rls_lattice_filter_fn(x, d, order=4)
    assert result.name == "rls_lattice_filter"
    assert result.n_samples == 128


def test_rlslt_error_in_extra():
    rng = np.random.default_rng(42)
    x = rng.standard_normal(64)
    d = x.copy()
    result = rls_lattice_filter_fn(x, d, order=4)
    assert "error" in result.extra
    assert len(result.extra["error"]) == 64


def test_rlslt_output_length():
    rng = np.random.default_rng(42)
    n = 200
    x = rng.standard_normal(n)
    d = rng.standard_normal(n)
    result = rls_lattice_filter_fn(x, d, order=8)
    assert len(result.filtered) == n


def test_rlslt_alias():
    rng = np.random.default_rng(42)
    x = rng.standard_normal(64)
    d = rng.standard_normal(64)
    result = rlslt(x, d, order=2)
    assert result.name == "rls_lattice_filter"
