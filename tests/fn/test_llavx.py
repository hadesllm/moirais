"""Tests for llavx.llava_visual_chat."""
import numpy as np
import pytest
from morie.fn.llavx import llava_visual_chat


def test_llavx_basic():
    """Test basic functionality."""
    image = np.random.default_rng(42).normal(0, 1, 100)
    instruction = np.random.default_rng(42).normal(0, 1, 100)
    llm = np.random.default_rng(42).normal(0, 1, 100)
    result = llava_visual_chat(image, instruction, llm)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_llavx_edge():
    """Test edge cases."""
    image = np.random.default_rng(42).normal(0, 1, 100)
    instruction = np.random.default_rng(42).normal(0, 1, 100)
    llm = np.random.default_rng(42).normal(0, 1, 100)
    result = llava_visual_chat(image, instruction, llm)
    assert isinstance(result, dict)
