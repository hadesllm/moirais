"""Tests for blsum (BLOSUM alignment score)."""
from moirais.fn.blsum import blosum_score


def test_blosum_identical():
    r = blosum_score("ACGT", "ACGT")
    assert r.value > 0


def test_blosum_mismatch():
    r_match = blosum_score("AAAA", "AAAA")
    r_mis = blosum_score("AAAA", "CCCC")
    assert r_match.value > r_mis.value
