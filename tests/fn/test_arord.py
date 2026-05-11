"""Tests for arord.py - AR order selection."""
import numpy as np
import pytest
from morie.fn.arord import ar_order_select, arord


def test_ar_order_select_returns_descriptive_result():
    x = np.random.default_rng(42).standard_normal(256)
    result = ar_order_select(x)
    assert result.name == "ar_order_select"
    assert isinstance(result.value, int)
    assert "scores" in result.extra


def test_ar_order_select_valid_range():
    x = np.random.default_rng(42).standard_normal(256)
    result = ar_order_select(x, max_order=20)
    assert 1 <= result.value <= 20


def test_ar_order_select_bic():
    x = np.random.default_rng(42).standard_normal(256)
    result = ar_order_select(x, criterion="bic")
    assert result.extra["criterion"] == "bic"


def test_arord_alias():
    x = np.random.default_rng(42).standard_normal(128)
    result = arord(x, max_order=10)
    assert result.name == "ar_order_select"
