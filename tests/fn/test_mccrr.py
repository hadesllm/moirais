"""Tests for mcc_score."""
import pytest
from morie.fn.mccrr import mcc_score, mccrr


def test_perfect():
    r = mcc_score([1, 0, 1, 0], [1, 0, 1, 0])
    assert abs(r.estimate - 1.0) < 1e-10


def test_inverse():
    r = mcc_score([1, 0, 1, 0], [0, 1, 0, 1])
    assert abs(r.estimate - (-1.0)) < 1e-10


def test_alias():
    assert mccrr is mcc_score
