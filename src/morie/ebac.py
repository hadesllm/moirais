"""
eBAC calculations.
Ported from internal routines for computing continuous eBAC and legal threshold binaries.
"""


def calculate_ebac(drinks: float, weight_lbs: float, hours: float, gender_constant: float) -> float:
    """
    Compute the continuous estimated Blood Alcohol Concentration (eBAC) using the standard Widmark formula.

    :param drinks: The number of standard drinks consumed (1 drink = 14 grams of alcohol).
    :type drinks: float
    :param weight_lbs: The weight of the individual in pounds.
    :type weight_lbs: float
    :param hours: Hours elapsed since drinking began.
    :type hours: float
    :param gender_constant: Widmark gender multiplier (standard is 0.73 for men, 0.66 for women).
    :type gender_constant: float
    :return: The estimated Blood Alcohol Concentration level (non-negative).
    :rtype: float
    """
    if weight_lbs <= 0:
        return 0.0

    # 1 standard drink roughly = 14 grams of alcohol
    # Formula: (Drinks * 5.14) / (Weight * gender_constant) - 0.015 * hours
    ebac = (drinks * 5.14) / (weight_lbs * gender_constant) - (0.015 * hours)
    return max(0.0, ebac)


def is_over_legal_limit(ebac: float, limit: float = 0.08) -> int:
    """
    Determine whether an eBAC exceeds the specified legal driving limit.

    :param ebac: The calculated eBAC value to evaluate.
    :type ebac: float
    :param limit: The legal blood alcohol concentration limit, defaults to 0.08.
    :type limit: float, optional
    :return: 1 if the eBAC is greater than or equal to the limit, 0 otherwise.
    :rtype: int
    """
    return 1 if ebac >= limit else 0
