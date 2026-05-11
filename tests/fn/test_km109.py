"""Tests for km109.kamath_ch6_perplexity_leakage."""
import numpy as np
import pytest
from morie.fn.km109 import kamath_ch6_perplexity_leakage


def test_km109_basic():
    """Test basic functionality."""
    S_uniq = np.random.default_rng(42).normal(0, 1, 100)
    PP_public = np.random.default_rng(42).normal(0, 1, 100)
    PP_lm = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_ch6_perplexity_leakage(S_uniq, PP_public, PP_lm)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_km109_edge():
    """Test edge cases."""
    S_uniq = np.random.default_rng(42).normal(0, 1, 100)
    PP_public = np.random.default_rng(42).normal(0, 1, 100)
    PP_lm = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_ch6_perplexity_leakage(S_uniq, PP_public, PP_lm)
    assert isinstance(result, dict)
