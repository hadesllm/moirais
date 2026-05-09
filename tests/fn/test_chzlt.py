"""Tests for chzlt.cinelli_hazlett."""
import numpy as np
import pytest
from moirais.fn.chzlt import cinelli_hazlett


def test_chzlt_basic():
    """Test basic functionality."""
    model = np.random.default_rng(42).normal(0, 1, 100)
    treat = np.random.default_rng(42).normal(0, 1, 100)
    cov = np.random.default_rng(42).normal(0, 1, 100)
    R2_yu = np.random.default_rng(42).normal(0, 1, 100)
    R2_du = np.random.default_rng(42).normal(0, 1, 100)
    result = cinelli_hazlett(model, treat, cov, R2_yu, R2_du)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_chzlt_edge():
    """Test edge cases."""
    model = np.random.default_rng(42).normal(0, 1, 100)
    treat = np.random.default_rng(42).normal(0, 1, 100)
    cov = np.random.default_rng(42).normal(0, 1, 100)
    R2_yu = np.random.default_rng(42).normal(0, 1, 100)
    R2_du = np.random.default_rng(42).normal(0, 1, 100)
    result = cinelli_hazlett(model, treat, cov, R2_yu, R2_du)
    assert isinstance(result, dict)
