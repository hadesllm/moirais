# describe('bookadvanced_elementsofstatisticallearning13e7') — bookadvanced_elementsofstatisticallearning_chapter_13_equation_7

## WHAT IT DOES

Dispersion equation extracted from BookAdvanced elementsofstatisticallearning.

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

are assumed to be scalar: Σr =σ I∀r = 1,...,R , and σ is a fixed parameter. Discuss the analogy between the K-means clustering algorithm and

## INPUTS / OUTPUTS

Inputs: x
Outputs: a RichResult whose payload exposes value.
You can read fields by attribute (`r.statistic`), index (`r["statistic"]`),
or `.get(...)`.

## WORKED EXAMPLE

```python
from morie.fn.bookadvanced_elementsofstatisticallearning13e7 import bookadvanced_elementsofstatisticallearning_chapter_13_equation_7
import numpy as np
result = bookadvanced_elementsofstatisticallearning_chapter_13_equation_7(np.random.default_rng(42).normal(0, 1, 100))
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

BookAdvanced elementsofstatisticallearning, ch.13 eq.13.7
