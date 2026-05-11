"""Tests for chipsq.chip_seq_peak."""
import numpy as np
import pytest
from morie.fn.chipsq import chip_seq_peak


def test_chipsq_basic():
    """Test basic functionality."""
    reads = np.random.default_rng(42).normal(0, 1, 100)
    control = np.random.default_rng(42).normal(0, 1, 100)
    result = chip_seq_peak(reads, control)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_chipsq_edge():
    """Test edge cases."""
    reads = np.random.default_rng(42).normal(0, 1, 100)
    control = np.random.default_rng(42).normal(0, 1, 100)
    result = chip_seq_peak(reads, control)
    assert isinstance(result, dict)
