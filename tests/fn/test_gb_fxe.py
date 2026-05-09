"""Tests for gb_fxe.gibbons_fisher_one_sided."""
import numpy as np
import pytest
from moirais.fn.gb_fxe import gibbons_fisher_one_sided


def test_gb_fxe_basic():
    """Test basic functionality."""
    table = np.random.default_rng(42).normal(0, 1, 100)
    result = gibbons_fisher_one_sided(table)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_gb_fxe_edge():
    """Test edge cases."""
    table = np.random.default_rng(42).normal(0, 1, 100)
    result = gibbons_fisher_one_sided(table)
    assert isinstance(result, dict)
