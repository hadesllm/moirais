"""Tests for alrt1 — mental health alert."""
import pandas as pd
from moirais.fn.alrt1 import alrt_mh

def test_alrt1_basic(otis_df):
    result = alrt_mh(otis_df)
    assert isinstance(result, pd.DataFrame)
    assert len(result) > 0


def test_cheatsheet():
    from moirais.fn.alrt1 import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
