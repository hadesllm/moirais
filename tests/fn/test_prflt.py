"""Tests for prflt.py - Polyphase filter."""
import numpy as np
from moirais.fn.prflt import polyphase_filter, prflt


def test_prflt_returns_descriptive_result():
    h = np.array([1.0, 2.0, 3.0, 4.0])
    result = polyphase_filter(h, M=2)
    assert result.name == "polyphase_filter"
    assert "components" in result.extra


def test_prflt_component_count():
    h = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    result = polyphase_filter(h, M=3)
    assert len(result.extra["components"]) == 3


def test_prflt_alias():
    h = np.array([1.0, 2.0, 3.0, 4.0])
    result = prflt(h)
    assert result.name == "polyphase_filter"
