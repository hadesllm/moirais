# describe('from_impact_to_action_final_report_into_anti_black_racism_by8u5') — from_impact_to_action_final_report_into_anti_black_racism_by_chapter_8_unnumbered_5

## WHAT IT DOES

GeneralStatistics expression (auto-extracted; see ref).

## WHEN TO USE

This callable applies when you have the inputs (x) and want
the outputs (value). See the FORMULA section for the assumed
parametric form.

## WHEN NOT TO USE

- The data violates the formula's domain assumptions (e.g. zero variance,
  perfectly collinear inputs).
- A more specialised version of this method exists for your data shape.
- Sample size is too small for the asymptotics this estimator relies on.

## ASSUMPTIONS

- Inputs are real-valued and free of NaN/Inf.
- Observations are independent unless the method explicitly handles
  clustering.
- Distributional assumptions vary; see the formula and reference for
  specifics.

## FORMULA

[62] TPSB, “Minutes of the Public Meeting: October 11, 2022” (11 October 2022), online (pdf): https://tpsb.ca/jdownloads-categories?task=download.send&id=754&catid=62&m=0.

## INPUTS / OUTPUTS

Inputs: x
Outputs: a RichResult whose payload exposes value.
You can read fields by attribute (`r.statistic`), index (`r["statistic"]`),
or `.get(...)`.

## WORKED EXAMPLE

```python
from moirais.fn.from_impact_to_action_final_report_into_anti_black_racism_by8u5 import from_impact_to_action_final_report_into_anti_black_racism_by_chapter_8_unnumbered_5
import numpy as np
result = from_impact_to_action_final_report_into_anti_black_racism_by_chapter_8_unnumbered_5(np.random.default_rng(42).normal(0, 1, 100))
print(result)              # full multi-section render
result.payload             # raw dict
```

## COMMON MISTAKES

- Treating the result as a plain dict — it's a RichResult; `isinstance(r, dict)`
  is False but `'statistic' in r` and `r['statistic']` both work.
- Ignoring the warnings/interpretation block when sample sizes are small.
- Confusing this with a similarly-named callable in a different family
  (check `cheatsheet()` for disambiguation).

## REFERENCES

From Impact to Action- Final report into anti-Black racism by the Toronto Police Service, ch.8 (unnumbered)
