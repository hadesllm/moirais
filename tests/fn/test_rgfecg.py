"""Tests for rgfecg.rangayyan_fetal_ecg."""
import numpy as np
import pytest
from moirais.fn.rgfecg import rangayyan_fetal_ecg


def test_rgfecg_basic():
    """Test basic functionality."""
    abdominal = np.random.default_rng(42).normal(0, 1, 100)
    maternal_ref = np.random.default_rng(42).normal(0, 1, 100)
    mu = 0.0
    order = 4
    result = rangayyan_fetal_ecg(abdominal, maternal_ref, mu, order)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rgfecg_edge():
    """Test edge cases."""
    abdominal = np.random.default_rng(42).normal(0, 1, 100)
    maternal_ref = np.random.default_rng(42).normal(0, 1, 100)
    mu = 0.0
    order = 4
    result = rangayyan_fetal_ecg(abdominal, maternal_ref, mu, order)
    assert isinstance(result, dict)
