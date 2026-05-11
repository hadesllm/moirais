"""Tests for eemd.py - Ensemble Empirical Mode Decomposition."""
import numpy as np
import pytest
from morie.fn.eemd import eemd_fn, eemd


def test_eemd_returns_descriptive_result():
    t = np.linspace(0, 1, 256)
    x = np.sin(2 * np.pi * 5 * t) + np.sin(2 * np.pi * 20 * t)
    result = eemd_fn(x, n_ensembles=10)
    assert result.name == "ensemble_emd"
    assert isinstance(result.value, int)
    assert "imfs" in result.extra


def test_eemd_at_least_one_imf():
    t = np.linspace(0, 1, 256)
    x = np.sin(2 * np.pi * 5 * t) + np.sin(2 * np.pi * 20 * t)
    result = eemd_fn(x, n_ensembles=10)
    assert result.value >= 1
    assert len(result.extra["imfs"]) >= 1


def test_eemd_imfs_same_length():
    t = np.linspace(0, 1, 256)
    x = np.sin(2 * np.pi * 5 * t)
    result = eemd_fn(x, n_ensembles=5)
    for imf in result.extra["imfs"]:
        assert len(imf) == len(x)


def test_eemd_alias():
    x = np.random.default_rng(42).standard_normal(128)
    result = eemd(x, n_ensembles=5)
    assert result.name == "ensemble_emd"
