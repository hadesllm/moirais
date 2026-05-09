"""Tests for kmarel.kamath_ragas_answer_relevance."""
import numpy as np
import pytest
from moirais.fn.kmarel import kamath_ragas_answer_relevance


def test_kmarel_basic():
    """Test basic functionality."""
    answer = np.random.default_rng(42).normal(0, 1, 100)
    original_question = np.random.default_rng(42).normal(0, 1, 100)
    model = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_ragas_answer_relevance(answer, original_question, model)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_kmarel_edge():
    """Test edge cases."""
    answer = np.random.default_rng(42).normal(0, 1, 100)
    original_question = np.random.default_rng(42).normal(0, 1, 100)
    model = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_ragas_answer_relevance(answer, original_question, model)
    assert isinstance(result, dict)
