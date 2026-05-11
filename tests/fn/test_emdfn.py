"""Tests for emdfn.py - Empirical Mode Decomposition."""
import numpy as np
import pytest
from morie.fn.emdfn import emd_fn, emdfn


def test_emd_returns_descriptive_result():
    x = np.random.default_rng(42).standard_normal(256)
    result = emd_fn(x)
    assert result.name == "emd"
    assert isinstance(result.value, int)
    assert "imfs" in result.extra


def test_emd_at_least_one_imf():
    x = np.random.default_rng(42).standard_normal(256)
    result = emd_fn(x)
    assert result.value >= 1
    assert len(result.extra["imfs"]) >= 1


def test_emd_imfs_same_length():
    x = np.random.default_rng(42).standard_normal(256)
    result = emd_fn(x)
    for imf in result.extra["imfs"]:
        assert len(imf) == len(x)


def test_emdfn_alias():
    x = np.random.default_rng(42).standard_normal(128)
    result = emdfn(x, max_imfs=5)
    assert result.name == "emd"
