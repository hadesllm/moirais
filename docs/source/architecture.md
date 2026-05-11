# MORIE architecture

*Part of {doc}`index` — high-level structural overview of the package.*

This page gives a single-glance map of MORIE's class structure and
the contracts between components. It complements the prose in the
{doc}`methods/index` reference and the {doc}`api/index`.

## The result-container spine

Every analysis function in MORIE returns a `RichResult`. Estimator
hierarchies are organised around a `BaseEstimator` abstract class that
declares the call contract; concrete estimators specialise it for
particular causal / survey / spectral methods.

```{mermaid}
classDiagram
  class RichResult {
    +str title
    +list summary_lines
    +list tables
    +list warnings
    +str interpretation
    +Any payload
    +__str__()
    +to_json()
  }

  class BaseEstimator {
    <<abstract>>
    +DataFrame data
    +str treatment
    +str outcome
    +list covariates
    +fit() RichResult
    +describe()
    +_check_inputs()*
    +_estimate()*
  }

  class IPWEstimator {
    +bool stabilised
    +_estimate() RichResult
  }
  class AIPWEstimator {
    +SuperLearner outcome_model
    +SuperLearner propensity_model
    +_estimate() RichResult
  }
  class DMLEstimator {
    +int n_folds
    +str score
    +_estimate() RichResult
  }
  class MatchingEstimator {
    +str method
    +int caliper
    +_estimate() RichResult
  }
  class HawkesEstimator {
    +str kernel
    +str baseline
    +_estimate() RichResult
  }

  BaseEstimator <|-- IPWEstimator
  BaseEstimator <|-- AIPWEstimator
  BaseEstimator <|-- DMLEstimator
  BaseEstimator <|-- MatchingEstimator
  BaseEstimator <|-- HawkesEstimator

  BaseEstimator ..> RichResult : returns
```

The diagram captures the central pattern: every estimator inherits a
common call contract (`fit() -> RichResult`) and shares a common
`_check_inputs` validation; the differences live in `_estimate`,
which encapsulates the method-specific algorithm.

## The data layer

The `DatasetRegistry` decouples loaders from analysis code. Estimators
never know which physical store a dataset came from, only the column
contract.

```{mermaid}
classDiagram
  class DatasetRegistry {
    +list datasets()
    +DatasetInfo info(slug)
    +DataFrame load(slug)
    +ColumnProfile profile(slug, col)
    +list suggest_roles(slug)
  }

  class LocalSQLite {
    +Path path
    +load(slug) DataFrame
  }
  class RemoteSQL {
    +str base_url
    +load(slug) DataFrame
  }
  class BundledSQLite {
    +str pkg_data_path
    +load(slug) DataFrame
  }

  DatasetRegistry <|.. LocalSQLite
  DatasetRegistry <|.. RemoteSQL
  DatasetRegistry <|.. BundledSQLite
```

## The MRM module group

The MRM (McNamara-Ruhela-Medina) framework is implemented as a
coordinated set of modules built on top of `BaseEstimator`. Each MRM
module composes ten causal estimators on a single (treatment,
outcome, covariates) design and returns a single aggregate
`RichResult`.

```{mermaid}
classDiagram
  class MRMModule {
    <<abstract>>
    +DataFrame data
    +str treatment
    +str outcome
    +list covariates
    +run() RichResult
  }

  class PerRowMRM {
    +run() RichResult
  }
  class AggregateMRM {
    +str family
    +run() RichResult
  }
  class DoobChiSquare {
    +ndarray table
    +run() RichResult
  }
  class MandelaClassifier {
    +str jurisdiction
    +run() RichResult
  }

  MRMModule <|-- PerRowMRM
  MRMModule <|-- AggregateMRM
  MRMModule <|-- DoobChiSquare
  MRMModule <|-- MandelaClassifier

  PerRowMRM o-- IPWEstimator
  PerRowMRM o-- AIPWEstimator
  PerRowMRM o-- DMLEstimator
  PerRowMRM o-- MatchingEstimator
```

`PerRowMRM` is the ten-estimator ensemble: each instance composes
ten `BaseEstimator` subclasses (IPW Hájek, AIPW, g-computation, PSM
1:1 NN, PSM 5-strata subclass, IRM-DML, PSM→IRM-DML, ATC AIPW,
PLR-DML, SuperLearner-stacked AIPW) and reports them in a single
`RichResult`.

## How to read the diagrams

- **Filled triangle (▷)** = inheritance: the child class IS-A
  parent (substitutable).
- **Dashed arrow (..>)** = dependency: the source uses the target
  but does not own it.
- **Diamond (◇)** = aggregation: the container holds and uses
  instances of the contained class.
- The `<<abstract>>` stereotype marks classes with at least one
  abstract method (`fit_predict()`, `_estimate()`, `run()`). Concrete
  subclasses fill in those methods.
