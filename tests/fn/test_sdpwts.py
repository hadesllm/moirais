"""Tests for sdpwts.semidefinite_program."""
import numpy as np
import pytest
from morie.fn.sdpwts import semidefinite_program


def test_sdpwts_basic():
    """Test basic functionality."""
    C = np.random.default_rng(42).normal(0, 1, 100)
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    b = np.random.default_rng(42).normal(0, 1, 100)
    result = semidefinite_program(C, A, b)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_sdpwts_edge():
    """Test edge cases."""
    C = np.random.default_rng(42).normal(0, 1, 100)
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    b = np.random.default_rng(42).normal(0, 1, 100)
    result = semidefinite_program(C, A, b)
    assert isinstance(result, dict)
