# moirais.fn — function file (hadesllm/moirais)
"""Random Forest classifier robustness evaluation."""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report


def eval_robustness(
    X: pd.DataFrame,
    y: pd.Series,
    test_X: pd.DataFrame,
    test_y: pd.Series,
) -> dict[str, Any]:
    """Evaluate predictive accuracy of a Random Forest classifier.

    Parameters
    ----------
    X : DataFrame
        Training features.
    y : Series
        Training labels.
    test_X : DataFrame
        Testing features.
    test_y : Series
        Testing labels.

    Returns
    -------
    dict
        Classification report metrics (precision, recall, f1, support).
    """
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X, y)
    preds = clf.predict(test_X)
    report = classification_report(test_y, preds, output_dict=True)
    assert isinstance(report, dict)
    return report


robust = eval_robustness


def cheatsheet() -> str:
    return "eval_robustness({}) -> Random Forest classifier robustness evaluation."
