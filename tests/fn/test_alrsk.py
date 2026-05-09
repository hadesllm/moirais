"""Tests for alrsk — alert risk score."""
import pandas as pd
from moirais.fn.alrsk import alrisk

def test_alrsk_basic(otis_df):
    result = alrisk(otis_df)
    assert isinstance(result, pd.DataFrame)


def test_cheatsheet():
    from moirais.fn.alrsk import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
