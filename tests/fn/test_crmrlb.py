"""Tests for crmrlb.cramer_rao_bound."""
import numpy as np
import pytest
from moirais.fn.crmrlb import cramer_rao_bound


def test_crmrlb_basic():
    """Test basic functionality."""
    fisher_info = np.random.default_rng(42).normal(0, 1, 100)
    result = cramer_rao_bound(fisher_info)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_crmrlb_edge():
    """Test edge cases."""
    fisher_info = np.random.default_rng(42).normal(0, 1, 100)
    result = cramer_rao_bound(fisher_info)
    assert isinstance(result, dict)
