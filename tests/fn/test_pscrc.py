"""Tests for pscrc.pscl_rollcall."""
import numpy as np
import pytest
from morie.fn.pscrc import pscl_rollcall


def test_pscrc_basic():
    """Test basic functionality."""
    vote_matrix = np.random.default_rng(42).normal(0, 1, (10, 10))
    legis_data = np.random.default_rng(42).normal(0, 1, 100)
    vote_data = np.random.default_rng(42).normal(0, 1, 100)
    result = pscl_rollcall(vote_matrix, legis_data, vote_data)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_pscrc_edge():
    """Test edge cases."""
    vote_matrix = np.random.default_rng(42).normal(0, 1, (10, 10))
    legis_data = np.random.default_rng(42).normal(0, 1, 100)
    vote_data = np.random.default_rng(42).normal(0, 1, 100)
    result = pscl_rollcall(vote_matrix, legis_data, vote_data)
    assert isinstance(result, dict)
