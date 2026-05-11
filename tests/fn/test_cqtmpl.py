"""Tests for cqtmpl.cim_qtl."""
import numpy as np
import pytest
from morie.fn.cqtmpl import cim_qtl


def test_cqtmpl_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    markers = np.random.default_rng(43).integers(0, 3, (100, 20))
    positions = np.random.default_rng(42).uniform(0, 1, (100, 2))
    cofactors = np.random.default_rng(42).normal(0, 1, 100)
    result = cim_qtl(y, markers, positions, cofactors)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_cqtmpl_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    markers = np.random.default_rng(43).integers(0, 3, (100, 20))
    positions = np.random.default_rng(42).uniform(0, 1, (100, 2))
    cofactors = np.random.default_rng(42).normal(0, 1, 100)
    result = cim_qtl(y, markers, positions, cofactors)
    assert isinstance(result, dict)
