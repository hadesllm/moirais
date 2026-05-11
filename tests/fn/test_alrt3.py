"""Tests for alrt3 — suicide watch alert."""
import pandas as pd
from morie.fn.alrt3 import alrt_sw

def test_alrt3_basic(otis_df):
    result = alrt_sw(otis_df)
    assert isinstance(result, pd.DataFrame)


def test_cheatsheet():
    from morie.fn.alrt3 import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
