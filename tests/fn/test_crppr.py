"""Tests for moirais.fn.crppr -- Chinese restaurant process."""

from moirais.fn.crppr import chinese_restaurant_process


def test_returns_dict():
    result = chinese_restaurant_process(50)
    assert isinstance(result, dict)
    assert "n_tables" in result


def test_all_assigned():
    result = chinese_restaurant_process(100)
    assert len(result["assignments"]) == 100


def test_more_tables_with_high_alpha():
    r1 = chinese_restaurant_process(100, alpha=0.1)
    r2 = chinese_restaurant_process(100, alpha=10.0)
    assert r2["n_tables"] >= r1["n_tables"]
