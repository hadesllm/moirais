# describe('bookadvanced_elementsofstatisticallearning6u859') — bookadvanced_elementsofstatisticallearning_chapter_6_unnumbered_859

## WHAT IT DOES

Spatial expression (auto-extracted; see ref).

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

Ex. 13.5 Let Bi,i = 1, 2,...,N be square p× p positive semi-definite matrices and let ¯B = (1/N ) ∑Bi. Write the eigen-decomposition of ¯B as∑p

## INPUTS / OUTPUTS

Inputs: x
Outputs: a RichResult whose payload exposes value.
You can read fields by attribute (`r.statistic`), index (`r["statistic"]`),
or `.get(...)`.

## WORKED EXAMPLE

```python
from moirais.fn.bookadvanced_elementsofstatisticallearning6u859 import bookadvanced_elementsofstatisticallearning_chapter_6_unnumbered_859
import numpy as np
result = bookadvanced_elementsofstatisticallearning_chapter_6_unnumbered_859(np.random.default_rng(42).normal(0, 1, 100))
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

BookAdvanced elementsofstatisticallearning, ch.6 (unnumbered)
