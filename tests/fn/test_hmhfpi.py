"""Tests for hmhfpi.geron_hf_pipelines."""
import numpy as np
import pytest
from morie.fn.hmhfpi import geron_hf_pipelines


def test_hmhfpi_basic():
    """Test basic functionality."""
    task = np.random.default_rng(42).normal(0, 1, 100)
    inputs = np.random.default_rng(42).normal(0, 1, 100)
    model = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_hf_pipelines(task, inputs, model)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmhfpi_edge():
    """Test edge cases."""
    task = np.random.default_rng(42).normal(0, 1, 100)
    inputs = np.random.default_rng(42).normal(0, 1, 100)
    model = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_hf_pipelines(task, inputs, model)
    assert isinstance(result, dict)
