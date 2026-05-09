"""Tests for alinc — alert incidence."""
import pandas as pd
from moirais.fn.alinc import alincd

def test_alinc_basic(otis_df):
    result = alincd(otis_df)
    assert isinstance(result, pd.DataFrame)


def test_cheatsheet():
    from moirais.fn.alinc import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
