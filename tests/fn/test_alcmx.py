"""Tests for alcmx — alert complexity."""
import pandas as pd
from morie.fn.alcmx import alcmpx

def test_alcmx_basic(otis_df):
    result = alcmpx(otis_df)
    assert isinstance(result, pd.DataFrame)


def test_cheatsheet():
    from morie.fn.alcmx import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
