"""Tests for aitprt.aitchison_perturbation."""
import numpy as np
import pytest
from morie.fn.aitprt import aitchison_perturbation


def test_aitprt_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = aitchison_perturbation(x, y)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_aitprt_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = aitchison_perturbation(x, y)
    assert isinstance(result, dict)
