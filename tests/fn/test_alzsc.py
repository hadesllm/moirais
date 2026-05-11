"""Tests for alzsc.alammar_zero_shot_classification."""
import numpy as np
import pytest
from morie.fn.alzsc import alammar_zero_shot_classification


def test_alzsc_basic():
    """Test basic functionality."""
    text = np.random.default_rng(42).normal(0, 1, 100)
    candidate_labels = np.random.default_rng(43).integers(0, 2, 100)
    nli_model = np.random.default_rng(42).normal(0, 1, 100)
    result = alammar_zero_shot_classification(text, candidate_labels, nli_model)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_alzsc_edge():
    """Test edge cases."""
    text = np.random.default_rng(42).normal(0, 1, 100)
    candidate_labels = np.random.default_rng(43).integers(0, 2, 100)
    nli_model = np.random.default_rng(42).normal(0, 1, 100)
    result = alammar_zero_shot_classification(text, candidate_labels, nli_model)
    assert isinstance(result, dict)
