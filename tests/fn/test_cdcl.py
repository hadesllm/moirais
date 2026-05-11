"""Tests for cdcl.cdcl."""
import numpy as np
import pytest
from morie.fn.cdcl import cdcl


def test_cdcl_basic():
    """Test basic functionality."""
    cnf = np.random.default_rng(42).normal(0, 1, 100)
    result = cdcl(cnf)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_cdcl_edge():
    """Test edge cases."""
    cnf = np.random.default_rng(42).normal(0, 1, 100)
    result = cdcl(cnf)
    assert isinstance(result, dict)
