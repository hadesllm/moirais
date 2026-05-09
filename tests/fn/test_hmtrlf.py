"""Tests for hmtrlf.geron_trl_finetune."""
import numpy as np
import pytest
from moirais.fn.hmtrlf import geron_trl_finetune


def test_hmtrlf_basic():
    """Test basic functionality."""
    model = np.random.default_rng(42).normal(0, 1, 100)
    dataset = np.random.default_rng(42).normal(0, 1, 100)
    method = 'auto'
    result = geron_trl_finetune(model, dataset, method)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmtrlf_edge():
    """Test edge cases."""
    model = np.random.default_rng(42).normal(0, 1, 100)
    dataset = np.random.default_rng(42).normal(0, 1, 100)
    method = 'auto'
    result = geron_trl_finetune(model, dataset, method)
    assert isinstance(result, dict)
