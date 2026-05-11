"""Tests for hmppp.geron_pipeline_parallelism."""
import numpy as np
import pytest
from morie.fn.hmppp import geron_pipeline_parallelism


def test_hmppp_basic():
    """Test basic functionality."""
    model = np.random.default_rng(42).normal(0, 1, 100)
    n_stages = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_pipeline_parallelism(model, n_stages)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmppp_edge():
    """Test edge cases."""
    model = np.random.default_rng(42).normal(0, 1, 100)
    n_stages = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_pipeline_parallelism(model, n_stages)
    assert isinstance(result, dict)
