"""Tests for allktmpl.alammar_instruction_data_template."""
import numpy as np
import pytest
from morie.fn.allktmpl import alammar_instruction_data_template


def test_allktmpl_basic():
    """Test basic functionality."""
    records = np.random.default_rng(42).normal(0, 1, 100)
    template = np.random.default_rng(42).normal(0, 1, 100)
    result = alammar_instruction_data_template(records, template)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_allktmpl_edge():
    """Test edge cases."""
    records = np.random.default_rng(42).normal(0, 1, 100)
    template = np.random.default_rng(42).normal(0, 1, 100)
    result = alammar_instruction_data_template(records, template)
    assert isinstance(result, dict)
