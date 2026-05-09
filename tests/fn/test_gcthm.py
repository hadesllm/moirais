"""Tests for moirais.fn.gcthm — Glivenko-Cantelli test."""

import numpy as np
import pytest

from moirais.fn.gcthm import gcthm


def test_normal_converges():
    rng = np.random.default_rng(42)
    x = rng.standard_normal(5000)
    result = gcthm(x)
    assert result["converges"] is True


def test_ks_stat_positive():
    rng = np.random.default_rng(7)
    x = rng.standard_normal(100)
    result = gcthm(x)
    assert result["ks_stat"] > 0


def test_wrong_distribution():
    rng = np.random.default_rng(42)
    x = rng.exponential(1.0, size=500)
    result = gcthm(x)
    assert result["ks_stat"] > result["critical_value"]


def test_empty_raises():
    with pytest.raises(ValueError, match="non-empty"):
        gcthm(np.array([]))


def test_bad_alpha():
    with pytest.raises(ValueError, match="alpha"):
        gcthm(np.array([1.0, 2.0]), alpha=0.0)
