"""Tests for elmo.elmo."""
import numpy as np
import pytest
from moirais.fn.elmo import elmo


def test_elmo_basic():
    """Test basic functionality."""
    sentence = np.random.default_rng(42).normal(0, 1, 100)
    model = np.random.default_rng(42).normal(0, 1, 100)
    result = elmo(sentence, model)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_elmo_edge():
    """Test edge cases."""
    sentence = np.random.default_rng(42).normal(0, 1, 100)
    model = np.random.default_rng(42).normal(0, 1, 100)
    result = elmo(sentence, model)
    assert isinstance(result, dict)
