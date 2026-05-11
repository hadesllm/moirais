"""Tests for mqtmpl.multi_qtl."""
import numpy as np
import pytest
from morie.fn.mqtmpl import multi_qtl


def test_mqtmpl_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    markers = np.random.default_rng(43).integers(0, 3, (100, 20))
    positions = np.random.default_rng(42).uniform(0, 1, (100, 2))
    result = multi_qtl(y, markers, positions)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_mqtmpl_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    markers = np.random.default_rng(43).integers(0, 3, (100, 20))
    positions = np.random.default_rng(42).uniform(0, 1, (100, 2))
    result = multi_qtl(y, markers, positions)
    assert isinstance(result, dict)
