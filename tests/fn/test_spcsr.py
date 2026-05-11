"""Tests for spcsr.schabenberger_csr_def."""
import numpy as np
import pytest
from morie.fn.spcsr import schabenberger_csr_def


def test_spcsr_basic():
    """Test basic functionality."""
    points = np.random.default_rng(42).normal(0, 1, 100)
    region = (0.0, 1.0, 0.0, 1.0)
    result = schabenberger_csr_def(points, region)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_spcsr_edge():
    """Test edge cases."""
    points = np.random.default_rng(42).normal(0, 1, 100)
    region = (0.0, 1.0, 0.0, 1.0)
    result = schabenberger_csr_def(points, region)
    assert isinstance(result, dict)
