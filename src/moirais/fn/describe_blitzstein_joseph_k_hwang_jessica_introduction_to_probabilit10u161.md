# describe('blitzstein_joseph_k_hwang_jessica_introduction_to_probabilit10u161') — blitzstein_joseph_k_hwang_jessica_introduction_to_probabilit_chapter_10_unnumbered_161

## WHAT IT DOES

CentralTendency expression (auto-extracted; see ref).

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

in the Binomial PMF to get P (X = ijN = i +j) and the Poisson PMF to get

## INPUTS / OUTPUTS

Inputs: x
Outputs: a RichResult whose payload exposes value.
You can read fields by attribute (`r.statistic`), index (`r["statistic"]`),
or `.get(...)`.

## WORKED EXAMPLE

```python
from moirais.fn.blitzstein_joseph_k_hwang_jessica_introduction_to_probabilit10u161 import blitzstein_joseph_k_hwang_jessica_introduction_to_probabilit_chapter_10_unnumbered_161
import numpy as np
result = blitzstein_joseph_k_hwang_jessica_introduction_to_probabilit_chapter_10_unnumbered_161(np.random.default_rng(42).normal(0, 1, 100))
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

Blitzstein, Joseph K.  Hwang, Jessica - Introduction to probability, ch.10 (unnumbered)
