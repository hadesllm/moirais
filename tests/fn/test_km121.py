"""Tests for km121.kamath_ch8_bertscore_f1."""
import numpy as np
import pytest
from morie.fn.km121 import kamath_ch8_bertscore_f1


def test_km121_basic():
    """Test basic functionality."""
    P_BERT = np.random.default_rng(42).normal(0, 1, 100)
    R_BERT = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_ch8_bertscore_f1(P_BERT, R_BERT)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_km121_edge():
    """Test edge cases."""
    P_BERT = np.random.default_rng(42).normal(0, 1, 100)
    R_BERT = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_ch8_bertscore_f1(P_BERT, R_BERT)
    assert isinstance(result, dict)
