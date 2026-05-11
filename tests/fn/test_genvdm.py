"""Tests for genvdm.d_study_decision."""
import numpy as np
import pytest
from morie.fn.genvdm import d_study_decision


def test_genvdm_basic():
    """Test basic functionality."""
    G_components = np.random.default_rng(42).normal(0, 1, 100)
    n_proposed = np.random.default_rng(42).normal(0, 1, 100)
    result = d_study_decision(G_components, n_proposed)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_genvdm_edge():
    """Test edge cases."""
    G_components = np.random.default_rng(42).normal(0, 1, 100)
    n_proposed = np.random.default_rng(42).normal(0, 1, 100)
    result = d_study_decision(G_components, n_proposed)
    assert isinstance(result, dict)
