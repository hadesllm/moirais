"""Tests for alrt2 — suicide risk alert."""
import pandas as pd
from moirais.fn.alrt2 import alrt_sr

def test_alrt2_basic(otis_df):
    result = alrt_sr(otis_df)
    assert isinstance(result, pd.DataFrame)


def test_cheatsheet():
    from moirais.fn.alrt2 import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
