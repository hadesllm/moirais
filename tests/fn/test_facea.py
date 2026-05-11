"""Tests for facea.face_smooth."""
import numpy as np
import pytest
from morie.fn.facea import face_smooth


def test_facea_basic():
    """Test basic functionality."""
    Y = np.random.default_rng(43).normal(0, 1, 100)
    argvals = np.random.default_rng(42).normal(0, 1, 100)
    result = face_smooth(Y, argvals)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_facea_edge():
    """Test edge cases."""
    Y = np.random.default_rng(43).normal(0, 1, 100)
    argvals = np.random.default_rng(42).normal(0, 1, 100)
    result = face_smooth(Y, argvals)
    assert isinstance(result, dict)
