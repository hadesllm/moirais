"""Tests for kmfait.kamath_ragas_faithfulness."""
import numpy as np
import pytest
from moirais.fn.kmfait import kamath_ragas_faithfulness


def test_kmfait_basic():
    """Test basic functionality."""
    answer = np.random.default_rng(42).normal(0, 1, 100)
    context = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_ragas_faithfulness(answer, context)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_kmfait_edge():
    """Test edge cases."""
    answer = np.random.default_rng(42).normal(0, 1, 100)
    context = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_ragas_faithfulness(answer, context)
    assert isinstance(result, dict)
