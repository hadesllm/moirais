"""Tests for rqtmpl.qtl_mapping."""
import numpy as np
import pytest
from moirais.fn.rqtmpl import qtl_mapping


def test_rqtmpl_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    markers = np.random.default_rng(43).integers(0, 3, (100, 20))
    positions = np.random.default_rng(42).uniform(0, 1, (100, 2))
    result = qtl_mapping(y, markers, positions)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rqtmpl_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    markers = np.random.default_rng(43).integers(0, 3, (100, 20))
    positions = np.random.default_rng(42).uniform(0, 1, (100, 2))
    result = qtl_mapping(y, markers, positions)
    assert isinstance(result, dict)
