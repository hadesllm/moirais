"""Tests for strtfd.stratified_design."""
import numpy as np
import pytest
from moirais.fn.strtfd import stratified_design


def test_strtfd_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    stratum = np.random.default_rng(42).normal(0, 1, 100)
    Nh = np.random.default_rng(42).normal(0, 1, 100)
    result = stratified_design(y, stratum, Nh)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_strtfd_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    stratum = np.random.default_rng(42).normal(0, 1, 100)
    Nh = np.random.default_rng(42).normal(0, 1, 100)
    result = stratified_design(y, stratum, Nh)
    assert isinstance(result, dict)
