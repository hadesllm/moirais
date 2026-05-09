import importlib


def test_package_import_exposes_stable_top_level_api():
    moirais = importlib.import_module("moirais")

    assert moirais.__version__ == "0.2.0"
    assert hasattr(moirais, "DatasetRegistry")
    assert hasattr(moirais, "estimate_ate")
    assert not hasattr(moirais, "estimate_double_ml")
