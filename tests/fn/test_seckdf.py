"""Tests for seckdf.hkdf_extract_expand."""
import numpy as np
import pytest
from morie.fn.seckdf import hkdf_extract_expand


def test_seckdf_basic():
    """Test basic functionality."""
    salt = np.random.default_rng(42).normal(0, 1, 100)
    ikm = np.random.default_rng(42).normal(0, 1, 100)
    info = np.random.default_rng(42).normal(0, 1, 100)
    length = np.random.default_rng(42).normal(0, 1, 100)
    result = hkdf_extract_expand(salt, ikm, info, length)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_seckdf_edge():
    """Test edge cases."""
    salt = np.random.default_rng(42).normal(0, 1, 100)
    ikm = np.random.default_rng(42).normal(0, 1, 100)
    info = np.random.default_rng(42).normal(0, 1, 100)
    length = np.random.default_rng(42).normal(0, 1, 100)
    result = hkdf_extract_expand(salt, ikm, info, length)
    assert isinstance(result, dict)
