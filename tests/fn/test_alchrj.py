"""Tests for alchrj.alammar_chosen_rejected_template."""
import numpy as np
import pytest
from morie.fn.alchrj import alammar_chosen_rejected_template


def test_alchrj_basic():
    """Test basic functionality."""
    prompts = np.random.default_rng(42).normal(0, 1, 100)
    chosen = np.random.default_rng(42).normal(0, 1, 100)
    rejected = np.random.default_rng(42).normal(0, 1, 100)
    result = alammar_chosen_rejected_template(prompts, chosen, rejected)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_alchrj_edge():
    """Test edge cases."""
    prompts = np.random.default_rng(42).normal(0, 1, 100)
    chosen = np.random.default_rng(42).normal(0, 1, 100)
    rejected = np.random.default_rng(42).normal(0, 1, 100)
    result = alammar_chosen_rejected_template(prompts, chosen, rejected)
    assert isinstance(result, dict)
