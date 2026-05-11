"""Tests for kr21 — KR-21."""
import numpy as np
from morie.fn.kr21 import kr21

def test_kr21_basic(mapq_binary_df):
    result = kr21(mapq_binary_df)
    assert isinstance(result, float)
    assert -1 <= result <= 1


def test_cheatsheet():
    from morie.fn.kr21 import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
