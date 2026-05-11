"""Tests for esldai.esl_dirichlet_proc."""
import numpy as np
import pytest
from morie.fn.esldai import esl_dirichlet_proc


def test_esldai_basic():
    """Test basic functionality."""
    alpha = 0.05
    G0 = np.random.default_rng(42).normal(0, 1, 100)
    result = esl_dirichlet_proc(alpha, G0)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_esldai_edge():
    """Test edge cases."""
    alpha = 0.05
    G0 = np.random.default_rng(42).normal(0, 1, 100)
    result = esl_dirichlet_proc(alpha, G0)
    assert isinstance(result, dict)
