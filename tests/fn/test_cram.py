"""Tests for moirais.fn.cram — alias for Cramer's V."""
import numpy as np

from moirais.fn.cram import cram


def test_cram_is_callable():
    table = [[10, 20], [30, 40]]
    result = cram(table)
    assert isinstance(result, float)
    assert 0 <= result <= 1


def test_cram_same_as_cramv():
    from moirais.fn.cramv import cramers_v
    table = [[5, 15], [25, 35]]
    assert cram(table) == cramers_v(table)
