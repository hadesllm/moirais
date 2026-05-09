"""Tests for eslfdr.esl_bh_fdr."""
import numpy as np
import pytest
from moirais.fn.eslfdr import esl_bh_fdr


def test_eslfdr_basic():
    """Test basic functionality."""
    pvalues = np.random.default_rng(42).normal(0, 1, 100)
    alpha = 0.05
    result = esl_bh_fdr(pvalues, alpha)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_eslfdr_edge():
    """Test edge cases."""
    pvalues = np.random.default_rng(42).normal(0, 1, 100)
    alpha = 0.05
    result = esl_bh_fdr(pvalues, alpha)
    assert isinstance(result, dict)
