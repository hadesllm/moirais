"""Tests for benRea.named_entity."""
import numpy as np
import pytest
from morie.fn.benRea import named_entity


def test_benRea_basic():
    """Test basic functionality."""
    sentence = np.random.default_rng(42).normal(0, 1, 100)
    model = np.random.default_rng(42).normal(0, 1, 100)
    result = named_entity(sentence, model)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_benRea_edge():
    """Test edge cases."""
    sentence = np.random.default_rng(42).normal(0, 1, 100)
    model = np.random.default_rng(42).normal(0, 1, 100)
    result = named_entity(sentence, model)
    assert isinstance(result, dict)
