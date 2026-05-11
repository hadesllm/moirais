"""Tests for matrl.ma_three_level."""
import numpy as np
import pytest
from morie.fn.matrl import ma_three_level


def test_matrl_basic():
    """Test basic functionality."""
    yi = np.random.default_rng(42).normal(0, 1, 100)
    vi = np.random.default_rng(42).normal(0, 1, 100)
    cluster_id = np.random.default_rng(42).normal(0, 1, 100)
    study_id = np.random.default_rng(42).normal(0, 1, 100)
    result = ma_three_level(yi, vi, cluster_id, study_id)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_matrl_edge():
    """Test edge cases."""
    yi = np.random.default_rng(42).normal(0, 1, 100)
    vi = np.random.default_rng(42).normal(0, 1, 100)
    cluster_id = np.random.default_rng(42).normal(0, 1, 100)
    study_id = np.random.default_rng(42).normal(0, 1, 100)
    result = ma_three_level(yi, vi, cluster_id, study_id)
    assert isinstance(result, dict)
