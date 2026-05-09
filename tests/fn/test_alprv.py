"""Tests for alprv — alert prevalence by group."""
import pandas as pd
from moirais.fn.alprv import alprev

def test_alprv_basic(otis_df):
    result = alprev(otis_df)
    assert isinstance(result, pd.DataFrame)


def test_cheatsheet():
    from moirais.fn.alprv import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
