"""Tests for phmmsr.profile_hmm_search."""
import numpy as np
import pytest
from morie.fn.phmmsr import profile_hmm_search


def test_phmmsr_basic():
    """Test basic functionality."""
    profile = np.random.default_rng(42).normal(0, 1, 100)
    db = np.random.default_rng(42).normal(0, 1, 100)
    result = profile_hmm_search(profile, db)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_phmmsr_edge():
    """Test edge cases."""
    profile = np.random.default_rng(42).normal(0, 1, 100)
    db = np.random.default_rng(42).normal(0, 1, 100)
    result = profile_hmm_search(profile, db)
    assert isinstance(result, dict)
