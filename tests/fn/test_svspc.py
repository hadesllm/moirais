"""Tests for svspc.py - Subspace decomposition."""
import numpy as np
from moirais.fn.svspc import subspace_decompose_fn, svspc


def test_svspc_returns_result():
    x = np.random.default_rng(42).standard_normal(100)
    result = subspace_decompose_fn(x)
    assert result.name == "subspace_decompose"
    assert "signal_basis" in result.extra
    assert "noise_basis" in result.extra


def test_svspc_with_dim():
    x = np.random.default_rng(42).standard_normal(100)
    result = subspace_decompose_fn(x, dim=3)
    assert result.extra["signal_dim"] == 3
    assert result.extra["signal_basis"].shape[1] == 3


def test_svspc_alias():
    x = np.random.default_rng(42).standard_normal(50)
    result = svspc(x, dim=2)
    assert result.name == "subspace_decompose"
