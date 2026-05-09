"""Tests for pgdsdg.projected_gd."""
import numpy as np
import pytest
from moirais.fn.pgdsdg import projected_gd


def test_pgdsdg_basic():
    """Test basic functionality."""
    f = np.random.default_rng(42).normal(0, 1, 100)
    grad_f = np.random.default_rng(42).normal(0, 1, 100)
    project = np.random.default_rng(42).normal(0, 1, 100)
    x0 = np.random.default_rng(42).normal(0, 1, 100)
    lr = np.random.default_rng(42).normal(0, 1, 100)
    result = projected_gd(f, grad_f, project, x0, lr)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_pgdsdg_edge():
    """Test edge cases."""
    f = np.random.default_rng(42).normal(0, 1, 100)
    grad_f = np.random.default_rng(42).normal(0, 1, 100)
    project = np.random.default_rng(42).normal(0, 1, 100)
    x0 = np.random.default_rng(42).normal(0, 1, 100)
    lr = np.random.default_rng(42).normal(0, 1, 100)
    result = projected_gd(f, grad_f, project, x0, lr)
    assert isinstance(result, dict)
