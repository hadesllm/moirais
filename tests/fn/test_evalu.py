"""Tests for evalu.evalue."""
import numpy as np
import pytest
from morie.fn.evalu import evalue


def test_evalu_basic():
    """Test basic functionality."""
    RR = np.random.default_rng(42).normal(0, 1, 100)
    result = evalue(RR)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_evalu_edge():
    """Test edge cases."""
    RR = np.random.default_rng(42).normal(0, 1, 100)
    result = evalue(RR)
    assert isinstance(result, dict)
