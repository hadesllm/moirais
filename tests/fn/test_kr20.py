"""Tests for kr20 — KR-20."""
import numpy as np
from morie.fn.kr20 import kr20

def test_kr20_basic(mapq_binary_df):
    result = kr20(mapq_binary_df)
    assert isinstance(result, float)
    assert -1 <= result <= 1


def test_cheatsheet():
    from morie.fn.kr20 import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
