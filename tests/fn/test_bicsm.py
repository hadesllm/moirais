"""Tests for bicsm.bic_score_dag."""
import numpy as np
import pytest
from morie.fn.bicsm import bic_score_dag


def test_bicsm_basic():
    """Test basic functionality."""
    dag = {'A': [], 'B': ['A'], 'C': ['B']}
    data = np.random.default_rng(42).normal(0, 1, 100)
    result = bic_score_dag(dag, data)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_bicsm_edge():
    """Test edge cases."""
    dag = {'A': [], 'B': ['A'], 'C': ['B']}
    data = np.random.default_rng(42).normal(0, 1, 100)
    result = bic_score_dag(dag, data)
    assert isinstance(result, dict)
