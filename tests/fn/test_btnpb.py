"""Tests for btnpb.boot_nonoverlap_block."""
import numpy as np
import pytest
from morie.fn.btnpb import boot_nonoverlap_block


def test_btnpb_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    block_len = np.random.default_rng(42).normal(0, 1, 100)
    stat = np.random.default_rng(42).normal(0, 1, 100)
    B = np.random.default_rng(43).normal(0, 1, (10, 10))
    result = boot_nonoverlap_block(x, block_len, stat, B)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_btnpb_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    block_len = np.random.default_rng(42).normal(0, 1, 100)
    stat = np.random.default_rng(42).normal(0, 1, 100)
    B = np.random.default_rng(43).normal(0, 1, (10, 10))
    result = boot_nonoverlap_block(x, block_len, stat, B)
    assert isinstance(result, dict)
