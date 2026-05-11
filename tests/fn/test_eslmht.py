"""Tests for eslmht.esl_holm_bonferroni."""
import numpy as np
import pytest
from morie.fn.eslmht import esl_holm_bonferroni


def test_eslmht_basic():
    """Test basic functionality."""
    pvalues = np.random.default_rng(42).normal(0, 1, 100)
    alpha = 0.05
    result = esl_holm_bonferroni(pvalues, alpha)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_eslmht_edge():
    """Test edge cases."""
    pvalues = np.random.default_rng(42).normal(0, 1, 100)
    alpha = 0.05
    result = esl_holm_bonferroni(pvalues, alpha)
    assert isinstance(result, dict)
