"""Tests for kmpens.kamath_prompt_ensemble."""
import numpy as np
import pytest
from morie.fn.kmpens import kamath_prompt_ensemble


def test_kmpens_basic():
    """Test basic functionality."""
    prompt_logits = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_prompt_ensemble(prompt_logits)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_kmpens_edge():
    """Test edge cases."""
    prompt_logits = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_prompt_ensemble(prompt_logits)
    assert isinstance(result, dict)
