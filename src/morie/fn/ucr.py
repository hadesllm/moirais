"""No man ever steps in the same river twice. — Heraclitus"""

from __future__ import annotations

import pandas as pd

from ._containers import DescriptiveResult

_UCR_PART_I = {
    "murder": ("Criminal homicide", "Part I"),
    "homicide": ("Criminal homicide", "Part I"),
    "manslaughter": ("Criminal homicide", "Part I"),
    "rape": ("Forcible rape", "Part I"),
    "sexual assault": ("Forcible rape", "Part I"),
    "robbery": ("Robbery", "Part I"),
    "assault": ("Aggravated assault", "Part I"),
    "aggravated assault": ("Aggravated assault", "Part I"),
    "burglary": ("Burglary", "Part I"),
    "break and enter": ("Burglary", "Part I"),
    "larceny": ("Larceny-theft", "Part I"),
    "theft": ("Larceny-theft", "Part I"),
    "motor vehicle theft": ("Motor vehicle theft", "Part I"),
    "auto theft": ("Motor vehicle theft", "Part I"),
    "arson": ("Arson", "Part I"),
}

_UCR_PART_II_KEYWORDS = {
    "fraud": ("Fraud", "Part II"),
    "forgery": ("Forgery and counterfeiting", "Part II"),
    "embezzlement": ("Embezzlement", "Part II"),
    "vandalism": ("Vandalism", "Part II"),
    "mischief": ("Vandalism", "Part II"),
    "weapon": ("Weapons", "Part II"),
    "prostitution": ("Prostitution", "Part II"),
    "drug": ("Drug abuse violations", "Part II"),
    "gambling": ("Gambling", "Part II"),
    "dui": ("DUI", "Part II"),
    "trespass": ("Trespass", "Part II"),
    "disorderly": ("Disorderly conduct", "Part II"),
    "vagrancy": ("Vagrancy", "Part II"),
}


def ucr_classify(offenses: list[str]) -> DescriptiveResult:
    """Classify offense descriptions into UCR Part I / Part II categories.

    Parameters
    ----------
    offenses : list[str]
        Offense description strings.

    Returns
    -------
    DescriptiveResult
        ``value`` is a DataFrame with columns ``offense``, ``category``, ``part``.
    """
    rows = []
    for off in offenses:
        low = off.strip().lower()
        matched = False
        for key, (cat, part) in _UCR_PART_I.items():
            if key in low:
                rows.append({"offense": off, "category": cat, "part": part})
                matched = True
                break
        if not matched:
            for key, (cat, part) in _UCR_PART_II_KEYWORDS.items():
                if key in low:
                    rows.append({"offense": off, "category": cat, "part": part})
                    matched = True
                    break
        if not matched:
            rows.append({"offense": off, "category": "Unknown", "part": "Unclassified"})
    return DescriptiveResult(
        name="UCR classification",
        value=pd.DataFrame(rows),
    )


ucr = ucr_classify


def cheatsheet() -> str:
    return "ucr_classify({}) -> UCR offense classification. 'I find your lack of faith distu"
