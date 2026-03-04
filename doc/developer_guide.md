# Developer Guide

This document provides an overview of the internal architecture of PEPH
and guidelines for contributing to the codebase.

The goal of this guide is to make it easy to:

-   understand how the statistical model maps to the code
-   extend the package (e.g., new priors, new metrics, non-PH support)
-   modify the pipeline safely
-   maintain performance and reproducibility

------------------------------------------------------------------------

# 1. High-Level Architecture

PEPH is organized around a **configuration-driven modeling pipeline**.

Core flow:

1.  Load YAML config → validate via `RunConfig`
2.  Load dataset (wide survival format)
3.  Subject-level train/test split
4.  Expand training data to long format
5.  Fit model (PH or spatial frailty)
6.  Generate predictions
7.  Compute metrics and diagnostics
8.  Save artifacts

Everything is orchestrated through:

    peph.pipeline.run.run_pipeline

CLI entry point:

    python -m peph.cli.main run --config path/to/config.yml

------------------------------------------------------------------------

# 2. Module Structure

    src/peph/

      cli/
          main.py                # CLI entrypoint

      config/
          schema.py              # Pydantic config schema + loader

      data/
          long.py                # Wide → long PE expansion

      model/
          design.py              # Design matrix construction
          fit.py                 # PH model fitting
          fit_leroux.py          # Spatial MAP fitting
          leroux_objective.py    # Negative log-posterior
          predict.py             # Survival / risk prediction

      spatial/
          graph.py               # SpatialGraph dataclass
          weights.py             # Area weight construction

      metrics/
          auc.py
          brier.py
          calibration.py

      diagnostics/
          residuals.py

      pipeline/
          run.py                 # End-to-end workflow

      sim/
          peph.py
          spatial.py             # Simulation utilities (testing)

    tests/
    configs/
    models/

------------------------------------------------------------------------

# 3. Core Design Principles

### 3.1 Configuration-Driven

All runs are controlled via YAML. No modeling parameters should be
hardcoded in the pipeline.

The config schema lives in:

    peph.config.schema

Changes to the schema should:

-   update `RunConfig`
-   update docs (`config_reference.md`)
-   update example configs
-   preserve backward compatibility when possible

------------------------------------------------------------------------

### 3.2 Explicit Design Matrix Construction

Design matrix logic lives in:

    peph.model.design

Key responsibilities:

-   numeric covariate extraction
-   categorical one-hot encoding
-   reference level enforcement
-   strict prediction-time validation
-   feature name tracking

Avoid embedding design logic in fitting functions.

------------------------------------------------------------------------

### 3.3 Separation of PH and Spatial Logic

`fit.py` implements classical PE-PH.

`fit_leroux.py` wraps PH logic and augments with:

-   spatial frailty parameters
-   MAP optimization
-   CAR precision construction

The PH implementation must remain usable independently.

------------------------------------------------------------------------

### 3.4 Sparse Spatial Operations

Spatial graphs use CSR matrices:

-   `SpatialGraph.W()` returns sparse adjacency
-   `SpatialGraph.leroux_Q(rho)` builds sparse precision

Avoid dense matrix construction for large graphs.

------------------------------------------------------------------------

### 3.5 Reproducibility

Artifacts are saved to JSON / Parquet.

Model JSON must contain:

-   parameter estimates
-   baseline hazards
-   spatial parameters (if present)
-   encoding metadata
-   config snapshot (recommended future improvement)

Round-trip serialization is covered by tests.

------------------------------------------------------------------------

# 4. How the Statistical Model Maps to Code

  Statistical Object   Code Location
  -------------------- -----------------------
  Baseline α_k         `fit.py`
  β coefficients       `fit.py`
  Long expansion       `data/long.py`
  Poisson likelihood   `statsmodels GLM`
  Frailty vector u     `fit_leroux.py`
  Q(ρ) precision       `spatial/graph.py`
  Centering operator   `leroux_objective.py`
  MAP objective        `leroux_objective.py`

------------------------------------------------------------------------

# 5. Adding New Features

## 5.1 Non-Proportional Hazards

Likely extension path:

-   allow interval-specific β_k
-   modify design matrix to include interactions with interval
    indicators
-   extend prediction logic accordingly

Recommended location:

    peph.model.fit_nonph.py

------------------------------------------------------------------------

## 5.2 Alternative Spatial Priors (e.g., BYM2)

Add:

-   new precision construction
-   new hyperparameter transform
-   update MAP objective

Avoid modifying Leroux code directly --- create a new backend.

------------------------------------------------------------------------

## 5.3 New Metrics

Add file under:

    peph.metrics/

Metrics should:

-   operate on test-wide data
-   accept predicted risks
-   not depend on fitting internals

Register metric in pipeline run logic.

------------------------------------------------------------------------

# 6. Testing Strategy

PEPH uses two classes of tests:

### Fast tests

-   unit tests
-   design matrix correctness
-   serialization round-trip
-   pipeline smoke tests

### Slow tests (`@pytest.mark.slow`)

-   parameter recovery
-   spatial recovery
-   simulation-based validation

Before merging a major modeling change, run:

    pytest
    pytest -m slow

------------------------------------------------------------------------

# 7. Performance Considerations

SEER-scale data may include:

-   100k--500k subjects
-   30k+ ZIP codes

Critical areas:

-   long-format expansion memory usage
-   sparse precision matrix operations
-   repeated factorizations in MAP optimization
-   prediction vectorization

Avoid:

-   dense G×G matrices
-   repeated graph construction
-   repeated design matrix building in loops

------------------------------------------------------------------------

# 8. Coding Standards

-   Prefer explicit variable names over compact notation
-   Keep statistical and engineering logic separated
-   Write tests for every new feature
-   Avoid silent fallback behaviors
-   Fail loudly on invalid config or unseen categories (unless
    explicitly allowed)

------------------------------------------------------------------------

# 9. Suggested Future Refactors

-   Add model artifact schema versioning
-   Cache sparse Cholesky factorizations in Leroux optimization
-   Add marginal frailty prediction
-   Add baseline hazard smoothing option
-   Improve memory efficiency in long expansion

------------------------------------------------------------------------

# 10. Contribution Workflow

1.  Create feature branch
2.  Implement change
3.  Add/adjust tests
4.  Run full test suite
5.  Update documentation
6.  Submit pull request

Major statistical changes should include simulation validation.

------------------------------------------------------------------------

# 11. Philosophy

PEPH prioritizes:

-   statistical correctness
-   reproducibility
-   transparency
-   modular extensibility

The goal is not to be a general-purpose survival package, but to provide
a **robust, production-grade implementation of PE-PH models suitable for
epidemiologic research.**
