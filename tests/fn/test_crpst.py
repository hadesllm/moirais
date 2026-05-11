"""Tests for crpst: Chinese restaurant process."""

import numpy as np
import pytest

from morie.fn.crpst import chinese_restaurant_process


def test_crpst_basic():
    """Test basic CRP simulation."""
    result = chinese_restaurant_process(n=100, alpha=1.0)

    assert len(result["table_assignments"]) == 100
    assert result["n_tables"] > 0
    assert result["n_tables"] <= 100


def test_crpst_concentration():
    """Test effect of concentration on number of tables."""
    result_low = chinese_restaurant_process(n=100, alpha=0.1)
    result_high = chinese_restaurant_process(n=100, alpha=10.0)

    # Higher alpha should give more tables on average
    assert result_high["n_tables"] > result_low["n_tables"]


def test_crpst_table_counts():
    """Test consistency of table counts."""
    result = chinese_restaurant_process(n=50, alpha=1.0)

    # Sum of table counts should equal n
    total = np.sum(result["table_counts"])
    assert total == 50


def test_crpst_small_n():
    """Test with small n."""
    result = chinese_restaurant_process(n=5, alpha=1.0)

    assert result["n_tables"] >= 1
    assert result["n_tables"] <= 5
    assert len(result["table_assignments"]) == 5


def test_crpst_invalid_alpha():
    """Test error handling."""
    with pytest.raises(ValueError):
        chinese_restaurant_process(n=10, alpha=-1.0)

    with pytest.raises(ValueError):
        chinese_restaurant_process(n=0, alpha=1.0)
