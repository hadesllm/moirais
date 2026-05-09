# describe('wilcox13u1096') — wilcox_chapter_13_unnumbered_1096

## WHAT IT DOES

Regression expression (auto-extracted; see ref).

## WHEN TO USE

This callable applies when you have the inputs (x) and want
the outputs (value). See the FORMULA section for the assumed
parametric form.

## WHEN NOT TO USE

- The data violates the formula'Difficulties strengthen the mind, as labor does the body. — Seneca'statistic"]`),
or `.get(...)`.

## WORKED EXAMPLE

```python
from moirais.fn.wilcox13u1096 import wilcox_chapter_13_unnumbered_1096
import numpy as np
result = wilcox_chapter_13_unnumbered_1096(np.random.default_rng(42).normal(0, 1, 100))
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

Wilcox, R.R. (2017) Modern Statistics for the Social and Behavioral Sciences, ch.13 (unnumbered)
