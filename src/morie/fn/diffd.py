# morie.fn — function file (hadesllm/morie)
"""Difference-in-Differences with R-style verbose result."""


def diffd(y_treated_pre: float, y_treated_post: float,
          y_control_pre: float, y_control_post: float):
    """Canonical Difference-in-Differences estimator (2x2)."""
    from ._richresult import RichResult
    treated_change = y_treated_post - y_treated_pre
    control_change = y_control_post - y_control_pre
    dd = treated_change - control_change
    return RichResult(
        title="Difference-in-Differences (canonical 2x2)",
        summary_lines=[
            ("DD estimate", dd),
            ("Treated: pre", y_treated_pre),
            ("Treated: post", y_treated_post),
            ("Treated change", treated_change),
            ("Control: pre", y_control_pre),
            ("Control: post", y_control_post),
            ("Control change", control_change),
        ],
        warnings=[
            "Identifying assumption: parallel trends - control's change is a "
            "valid counterfactual for what treated would have experienced "
            "absent treatment. Verify with pre-period plots if possible.",
        ],
        interpretation=(f"DD = ({treated_change:+.4g}) - ({control_change:+.4g}) "
                        f"= {dd:+.4g}. Sign indicates direction of treatment effect."),
        payload={"value": dd, "estimate": dd,
                 "treated_change": treated_change,
                 "control_change": control_change},
    )
