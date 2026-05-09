"""Tests for alesc — alert escalation."""
import pandas as pd
from moirais.fn.alesc import alescl

def test_alesc_basic(otis_df):
    result = alescl(otis_df)
    assert isinstance(result, (pd.DataFrame, dict))


def test_cheatsheet():
    from moirais.fn.alesc import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
