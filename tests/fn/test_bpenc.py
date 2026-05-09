"""Tests for bits per character."""
import pytest
from moirais.fn.bpenc import bits_per_char, bpenc


def test_single_char():
    r = bits_per_char("aaaa")
    assert r.estimate == 0.0


def test_two_chars():
    r = bits_per_char("abababab")
    assert r.estimate == pytest.approx(1.0, abs=0.01)


def test_alias():
    assert bpenc is bits_per_char


def test_empty_raises():
    with pytest.raises(ValueError):
        bits_per_char("")
