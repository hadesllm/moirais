"""Test susyq."""
from morie.fn.susyq import susy_algebra


def test_susyq_n1d4():
    r = susy_algebra(N=1, d=4)
    assert r.value == 4.0


def test_susyq_n2():
    r = susy_algebra(N=2, d=4)
    assert r.value == 8.0
