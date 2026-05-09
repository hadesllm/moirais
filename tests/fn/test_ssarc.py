"""Tests for ssarc.py - SSA reconstruction."""
import numpy as np
from moirais.fn.ssarc import ssa_reconstruct_fn, ssarc


def test_ssarc_returns_result():
    x = np.random.default_rng(42).standard_normal(100)
    result = ssa_reconstruct_fn(x, groups=[0])
    assert result.name == "ssa_reconstruct"
    assert "reconstructed" in result.extra
    assert "residual" in result.extra


def test_ssarc_length_matches():
    x = np.random.default_rng(42).standard_normal(100)
    result = ssa_reconstruct_fn(x, L=30, groups=[0, 1])
    assert len(result.extra["reconstructed"]) == 100
    assert len(result.extra["residual"]) == 100


def test_ssarc_alias():
    x = np.random.default_rng(42).standard_normal(50)
    result = ssarc(x, L=10, groups=[0])
    assert result.name == "ssa_reconstruct"
