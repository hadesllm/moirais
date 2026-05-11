"""Tests for f_score."""
import pytest
from morie.fn.fscor import f_score, fscor


def test_beta1():
    r = f_score([1, 0, 1, 0], [1, 0, 1, 0], beta=1.0)
    assert abs(r.estimate - 1.0) < 1e-10


def test_alias():
    assert fscor is f_score


def test_beta2():
    r = f_score([1, 1, 0], [1, 0, 0], beta=2.0)
    assert 0 < r.estimate < 1
