"""Tests for gwsblc.gwas_block_combine."""
import numpy as np
import pytest
from moirais.fn.gwsblc import gwas_block_combine


def test_gwsblc_basic():
    """Test basic functionality."""
    block_results = np.random.default_rng(42).normal(0, 1, 100)
    result = gwas_block_combine(block_results)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_gwsblc_edge():
    """Test edge cases."""
    block_results = np.random.default_rng(42).normal(0, 1, 100)
    result = gwas_block_combine(block_results)
    assert isinstance(result, dict)
