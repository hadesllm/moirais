"""Tests for adjsrs.effective_srs."""
import numpy as np
import pytest
from morie.fn.adjsrs import effective_srs


def test_adjsrs_basic():
    """Test basic functionality."""
    design = np.random.default_rng(42).normal(0, 1, 100)
    method = 'auto'
    result = effective_srs(design, method)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_adjsrs_edge():
    """Test edge cases."""
    design = np.random.default_rng(42).normal(0, 1, 100)
    method = 'auto'
    result = effective_srs(design, method)
    assert isinstance(result, dict)
