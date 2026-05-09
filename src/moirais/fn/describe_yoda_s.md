# describe('Real knowledge is to know the extent of one's ignorance. — Confucius') — sensitivity_analysis

## WHAT IT DOES

Sensitivity analysis. 'Do or do not. There is no try.'Real knowledge is to know the extent of one's ignorance. — Confucius's source for the canonical
signature and assumptions.

## WHEN NOT TO USE

- Inputs violate the function's assumed domain (NaN/Inf, wrong shape).
- A more specialised version exists for your data shape — see
  `cheatsheet()` or `_registry.py` for related callables.
- Sample size too small for the asymptotics this estimator relies on.

## ASSUMPTIONS

- Inputs are real-valued and free of NaN/Inf.
- Observations are independent unless noted.
- See the FORMULA section in the source code for distributional
  specifics.

## FORMULA

See the function'Real knowledge is to know the extent of one's ignorance. — Confucius's `Parameters` block.
Outputs: a RichResult (dict-subclass) — `result["statistic"]`,
`.get(...)`, `for k in result` all work alongside the multi-section
`print(result)` render. See `moirais.fn.describe('Real knowledge is to know the extent of one's ignorance. — Confucius')` for guidance.

## WORKED EXAMPLE

```python
from moirais.fn.yoda_s import *
import numpy as np
# See the function signature in fn/yoda_s.py for argument names.
```

## COMMON MISTAKES

- Treating the result as a plain dict — it IS a dict (RichResult
  inherits from dict) but `print(result)` shows the multi-section
  render which is what you usually want.
- Ignoring warnings/interpretation when sample sizes are small.

## REFERENCES

See the source file's `References` block.
