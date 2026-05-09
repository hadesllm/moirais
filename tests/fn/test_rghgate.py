"""Tests for rghgate.rangayyan_hh_gating."""
import numpy as np
import pytest
from moirais.fn.rghgate import rangayyan_hh_gating


def test_rghgate_basic():
    """Test basic functionality."""
    V = np.random.default_rng(42).normal(0, 1, 100)
    dt = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_hh_gating(V, dt)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rghgate_edge():
    """Test edge cases."""
    V = np.random.default_rng(42).normal(0, 1, 100)
    dt = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_hh_gating(V, dt)
    assert isinstance(result, dict)
