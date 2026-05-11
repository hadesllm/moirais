# describe('ghs027') — ghosal_ch3_tailfree_strong_support_event

## WHAT IT DOES

Lower bound on the prior probability of approximating a target density p_0 in L1, used to establish that the total-variation support contains all densities absolutely continuous with respect to mu..

## WHEN TO USE

This callable applies when you have inputs of the appropriate shape and
want the documented output. Read the function's source for the canonical
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

See the function's docstring `Formula:` line in
`fn/ghs027.py`.

## INPUTS / OUTPUTS

Inputs: documented in the function's `Parameters` block.
Outputs: a RichResult (dict-subclass) — `result["statistic"]`,
`.get(...)`, `for k in result` all work alongside the multi-section
`print(result)` render. See `morie.fn.describe('ghs027')` for guidance.

## WORKED EXAMPLE

```python
from morie.fn.ghs027 import *
import numpy as np
# See the function signature in fn/ghs027.py for argument names.
```

## COMMON MISTAKES

- Treating the result as a plain dict — it IS a dict (RichResult
  inherits from dict) but `print(result)` shows the multi-section
  render which is what you usually want.
- Ignoring warnings/interpretation when sample sizes are small.

## REFERENCES

See the source file's `References` block.
