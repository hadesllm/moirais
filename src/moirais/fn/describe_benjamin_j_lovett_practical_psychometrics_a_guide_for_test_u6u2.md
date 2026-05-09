# describe('benjamin_j_lovett_practical_psychometrics_a_guide_for_test_u6u2') — benjamin_j_lovett_practical_psychometrics_a_guide_for_test_u_chapter_6_unnumbered_2

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

(CFI) = .99, Tucker– Lewis Index (TLI) = .99, RMSEA = .067. In addition, the

## INPUTS / OUTPUTS

Inputs: x
Outputs: a RichResult whose payload exposes value.
You can read fields by attribute (`r.statistic`), index (`r["statistic"]`),
or `.get(...)`.

## WORKED EXAMPLE

```python
from moirais.fn.benjamin_j_lovett_practical_psychometrics_a_guide_for_test_u6u2 import benjamin_j_lovett_practical_psychometrics_a_guide_for_test_u_chapter_6_unnumbered_2
import numpy as np
result = benjamin_j_lovett_practical_psychometrics_a_guide_for_test_u_chapter_6_unnumbered_2(np.random.default_rng(42).normal(0, 1, 100))
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

Benjamin J. Lovett - Practical Psychometrics  A Guide for Test Users, ch.6 (unnumbered)
