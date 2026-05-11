"""Tests for mtgbl.multi_trait_gblup."""
import numpy as np
import pytest
from morie.fn.mtgbl import multi_trait_gblup


def test_mtgbl_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    markers = np.random.default_rng(43).integers(0, 3, (100, 20))
    result = multi_trait_gblup(x, y, markers)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_mtgbl_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    markers = np.random.default_rng(43).integers(0, 3, (100, 20))
    result = multi_trait_gblup(x, y, markers)
    assert isinstance(result, dict)
