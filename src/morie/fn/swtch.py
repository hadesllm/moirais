"""Label / one-hot encoding. 'Not like this. Not like this.' -- Switch"""

from __future__ import annotations

import pandas as pd

from ._containers import DescriptiveResult


def encode_labels(
    data: pd.DataFrame,
    *,
    column: str = "category",
    method: str = "onehot",
    drop_first: bool = False,
) -> DescriptiveResult:
    """Encode categorical column as numeric labels or one-hot dummies.

    Parameters
    ----------
    data : DataFrame
        Input data.
    column : str
        Name of the categorical column to encode.
    method : str
        ``"label"`` for integer encoding, ``"onehot"`` for dummy variables.
    drop_first : bool
        If True and method is ``"onehot"``, drop the first category column
        to avoid multicollinearity.

    Returns
    -------
    DescriptiveResult
        ``value`` is the encoded DataFrame; ``extra`` has the mapping.
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    if method not in ("label", "onehot"):
        raise ValueError(f"method must be 'label' or 'onehot', got '{method}'")

    series = data[column].astype(str)
    categories = sorted(series.unique())
    cat_map = {cat: i for i, cat in enumerate(categories)}

    if method == "label":
        encoded = series.map(cat_map).astype(int)
        result_df = data.copy()
        result_df[column] = encoded
        return DescriptiveResult(
            name="Label Encoding",
            value=result_df,
            extra={"mapping": cat_map, "n_categories": len(categories)},
        )

    dummies = pd.DataFrame(0, index=data.index, columns=categories)
    for cat in categories:
        dummies[cat] = (series == cat).astype(int)
    if drop_first:
        dummies = dummies.iloc[:, 1:]

    result_df = pd.concat([data.drop(columns=[column]), dummies], axis=1)
    return DescriptiveResult(
        name="One-Hot Encoding",
        value=result_df,
        extra={
            "categories": categories,
            "n_categories": len(categories),
            "drop_first": drop_first,
        },
    )


swtch = encode_labels


def cheatsheet() -> str:
    return "encode_labels({}) -> Label / one-hot encoding. 'Not like this. Not like this.' --"
