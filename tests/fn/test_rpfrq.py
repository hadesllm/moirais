"""Tests for rpfrq — repeat placement frequency."""
import pandas as pd
from moirais.fn.rpfrq import rplace_frequency

def test_rpfrq_basic(otis_df):
    result = rplace_frequency(otis_df)
    assert isinstance(result, pd.DataFrame)


def test_cheatsheet():
    from moirais.fn.rpfrq import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
