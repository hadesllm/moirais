"""Tests for alchat.alammar_chat_template."""
import numpy as np
import pytest
from morie.fn.alchat import alammar_chat_template


def test_alchat_basic():
    """Test basic functionality."""
    turns = np.random.default_rng(42).normal(0, 1, 100)
    template_tokens = np.random.default_rng(42).normal(0, 1, 100)
    result = alammar_chat_template(turns, template_tokens)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_alchat_edge():
    """Test edge cases."""
    turns = np.random.default_rng(42).normal(0, 1, 100)
    template_tokens = np.random.default_rng(42).normal(0, 1, 100)
    result = alammar_chat_template(turns, template_tokens)
    assert isinstance(result, dict)
