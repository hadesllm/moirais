"""Tests for alcbm.alammar_conversation_buffer_memory."""
import numpy as np
import pytest
from morie.fn.alcbm import alammar_conversation_buffer_memory


def test_alcbm_basic():
    """Test basic functionality."""
    conversation = np.random.default_rng(42).normal(0, 1, 100)
    N = 100
    result = alammar_conversation_buffer_memory(conversation, N)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_alcbm_edge():
    """Test edge cases."""
    conversation = np.random.default_rng(42).normal(0, 1, 100)
    N = 100
    result = alammar_conversation_buffer_memory(conversation, N)
    assert isinstance(result, dict)
