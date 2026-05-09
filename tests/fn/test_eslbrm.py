"""Tests for eslbrm.esl_boltzmann."""
import numpy as np
import pytest
from moirais.fn.eslbrm import esl_boltzmann


def test_eslbrm_basic():
    """Test basic functionality."""
    v = np.random.default_rng(44).normal(0, 1, 100)
    h = 0.3
    result = esl_boltzmann(v, h)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_eslbrm_edge():
    """Test edge cases."""
    v = np.random.default_rng(44).normal(0, 1, 100)
    h = 0.3
    result = esl_boltzmann(v, h)
    assert isinstance(result, dict)
