"""Tests for alllmj.alammar_llm_as_judge."""
import numpy as np
import pytest
from morie.fn.alllmj import alammar_llm_as_judge


def test_alllmj_basic():
    """Test basic functionality."""
    responses = np.random.default_rng(42).normal(0, 1, 100)
    rubric = np.random.default_rng(42).normal(0, 1, 100)
    judge_model = np.random.default_rng(42).normal(0, 1, 100)
    result = alammar_llm_as_judge(responses, rubric, judge_model)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_alllmj_edge():
    """Test edge cases."""
    responses = np.random.default_rng(42).normal(0, 1, 100)
    rubric = np.random.default_rng(42).normal(0, 1, 100)
    judge_model = np.random.default_rng(42).normal(0, 1, 100)
    result = alammar_llm_as_judge(responses, rubric, judge_model)
    assert isinstance(result, dict)
