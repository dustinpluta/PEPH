# PEPH Developer Guide

This document provides a comprehensive guide to the internal architecture,
design philosophy, and development workflow of the PEPH project.

The goal of this guide is to make it possible for a new developer to:

- understand how the statistical model maps to the code
- extend the modeling framework safely
- modify the pipeline
- add diagnostics, metrics, and reporting
- maintain reproducibility and performance

PEPH is designed as a production-grade implementation of **piecewise
exponential proportional hazards (PEPH) models** with optional
**spatial frailty via the Leroux CAR model**.

The codebase prioritizes:

- statistical correctness
- modular architecture
- reproducible artifacts
- extensibility for epidemiologic research.

------------------------------------------------------------

# 1. Conceptual Overview

The package implements survival analysis using a **piecewise exponential
proportional hazards model**, optionally augmented with **spatial frailty**.

The workflow is organized around a **configuration-driven modeling pipeline**.

High-level workflow:

1. Load YAML configuration
2. Validate configuration via Pydantic schema
3. Load dataset (wide survival format)
4. Perform subject-level train/test split
5. Expand training data to long format
6. Fit the survival model
7. Generate predictions for the test set
8. Compute metrics and diagnostics
9. Save artifacts and plots

The entire process is orchestrated by a single pipeline function.

Pipeline entrypoint:

    peph.pipeline.run.run_pipeline

Command line entrypoint:

    python -m peph.cli.main run --config path/to/config.yml

------------------------------------------------------------

# 2. Repository Structure

Source code is located in:

    src/peph/

Top-level structure:

    cli/
        main.py

    config/
        schema.py

    data/
        long.py

    model/
        design.py
        fit.py
        fit_leroux.py
        leroux_objective.py
        predict.py
        result.py

    spatial/
        graph.py
        weights.py

    metrics/
        auc.py
        brier.py
        calibration.py

    diagnostics/
        residuals.py

    plots/
        calibration.py
        diagnostics.py
        spatial.py

    pipeline/
        run.py

    report/
        cli.py
        tables.py

    sim/
        ph.py
        peph.py
        spatial.py

Additional directories:

    tests/
    configs/
    models/

------------------------------------------------------------

# 3. Configuration System

PEPH uses a strict **configuration-driven design**.

All modeling behavior is controlled by YAML configuration files.

Example invocation:

    python -m peph.cli.main run --config configs/run1.yml

The configuration schema is defined in:

    peph.config.schema.RunConfig

Responsibilities of the schema:

- validate input parameters
- enforce allowed model backends
- define modeling options
- ensure reproducibility

Major configuration sections include:

    data
    data_schema
    model
    fit
    prediction
    metrics
    diagnostics
    output

When modifying configuration:

1. Update RunConfig
2. Update documentation
3. Update example configs
4. Update tests

------------------------------------------------------------

# 4. Data Representation

PEPH uses two primary data representations.

Wide format:

One row per subject.

Required columns:

    id
    time
    event

Additional covariates are defined in the config.

Long format:

After expansion, each subject contributes rows for each time interval.

Columns include:

    id
    interval index
    exposure time
    event indicator
    covariates

Long-format expansion is implemented in:

    peph.data.long.expand_long

Responsibilities:

- apply breakpoints
- compute exposure time
- assign events to intervals
- replicate covariates

The interval convention is:

    left-closed, right-open

Events occurring exactly on a breakpoint are assigned to the interval
starting at that breakpoint.

------------------------------------------------------------

# 5. Model Implementation

## 5.1 Piecewise Exponential PH Model

The baseline model is a piecewise exponential proportional hazards model.

Hazard:

    h_i(t) = exp(α_k + x_i β)

Where:

    α_k = baseline log-hazard in interval k
    β = regression coefficients

Likelihood is implemented using the **Poisson trick**.

Long-format observations are treated as Poisson counts with log exposure offsets.

Model fitting is performed with:

    statsmodels GLM (Poisson)

Code location:

    peph.model.fit

Outputs:

- coefficient estimates
- standard errors
- baseline hazards
- inference statistics

------------------------------------------------------------

## 5.2 Spatial Frailty Model

Spatial frailty is implemented using the **Leroux CAR model**.

Extended hazard:

    h_i(t) = exp(α_k + x_i β + u_{area(i)})

Frailty vector:

    u ~ N(0, τ Q(ρ)^(-1))

Precision matrix:

    Q(ρ) = (1 − ρ) I + ρ (D − W)

Where:

    W = adjacency matrix
    D = degree matrix
    ρ = spatial dependence parameter
    τ = precision parameter

Code locations:

    peph.model.fit_leroux
    peph.model.leroux_objective
    peph.spatial.graph

Estimation uses MAP optimization.

------------------------------------------------------------

# 6. Spatial Graph Representation

Spatial graphs are represented by:

    SpatialGraph dataclass

Location:

    peph.spatial.graph

Fields include:

    zips
    edges
    W_csr_data
    components

Key methods:

    W()            → adjacency matrix
    degree()       → node degrees
    leroux_Q(rho)  → precision matrix
    component_ids()

Graphs are stored using sparse CSR format.

------------------------------------------------------------

# 7. Prediction System

Prediction logic is implemented in:

    peph.model.predict

Predictions include:

- survival probabilities
- cumulative hazard
- risk at specified horizons

Prediction requires:

- fitted model artifact
- test design matrix
- baseline hazard

Predictions are stored under:

    predictions/

## Time-to-Treatment Prediction Semantics

PR-C adds prediction support for models that include the time-dependent treatment indicator `treated_td`.

### Supported prediction mode

Current TTT prediction uses each subject’s **observed treatment history** from the wide data.

For a subject with treatment time `treatment_time`:

- `treated_td(t) = 0` for `t < treatment_time`
- `treated_td(t) = 1` for `t >= treatment_time`

If `treatment_time` is missing, the subject is treated as **never treated during predicted follow-up**.

Prediction therefore integrates the piecewise exponential hazard across two phases when treatment occurs before the prediction horizon:

1. untreated phase before treatment
2. treated phase after treatment

For a fitted model of the form

\[
h_i(t) = h_0(t)\exp\{x_i^\top\beta + \gamma \, treated_i(t) + u_{zip(i)}\},
\]

the cumulative hazard used in prediction is

\[
H_i(t)
=
e^{\eta_{0i}} H_0(\min(t, s_i))
+
e^{\eta_{0i}+\gamma}\max\{H_0(t)-H_0(s_i), 0\},
\]

where:

- \( s_i \) is the subject-specific treatment time
- \( \eta_{0i} \) is the baseline linear predictor excluding `treated_td`
- \( u_{zip(i)} \) is the optional Leroux frailty contribution

Survival and risk are then computed as usual:

\[
S_i(t) = \exp(-H_i(t)), \qquad R_i(t) = 1 - S_i(t).
\]

### Important interpretation

These predictions are **observed-history predictions**, not counterfactual treatment-policy predictions.

That means:

- if a subject was observed to receive treatment at day 80, the prediction uses a hazard switch at day 80
- if a subject never had observed treatment, prediction uses untreated hazard throughout follow-up
- the pipeline does **not** currently alter or simulate treatment timing at prediction time

### Current API behavior

For models containing `treated_td` in `x_td_numeric`:

- `predict_cumhaz()` supports TTT-aware prediction
- `predict_survival()` supports TTT-aware prediction
- `predict_risk()` supports TTT-aware prediction

These functions require access to the treatment-time column in the wide prediction data.

`predict_linear_predictor()` is **not defined** for TTT models and intentionally raises an error. A single static linear predictor is not well-defined when the covariate path changes over time.

### Pipeline behavior

For TTT-enabled runs with nonempty prediction horizons, the pipeline writes standard prediction artifacts, including:

- `predictions/test_predictions.parquet`
- horizon-specific columns such as:
  - `surv_t365`
  - `risk_t365`
  - `cumhaz_t365`

For TTT-enabled models, the predictions artifact does **not** currently include an `eta` column.

### What is not yet supported

PR-C does not yet implement counterfactual prediction modes such as:

- never-treated prediction
- treatment-at-time-\( s \) prediction
- user-specified treatment schedules

Those are planned for a later PR.

### Practical implication

Current TTT prediction answers the question:

> Given the fitted model and the subject’s observed treatment time, what are predicted survival, risk, and cumulative hazard at the requested horizons?

It does **not** yet answer:

> What would predicted survival have been if this subject had been treated earlier, later, or never?
------------------------------------------------------------

# 8. Metrics and Diagnostics

Metrics are computed on the test set.

Implemented metrics include:

Time-dependent discrimination:

    time-dependent AUC
    concordance index

Calibration:

    calibration-in-the-large
    calibration slope

Accuracy:

    Brier score

Diagnostics:

    Cox-Snell residuals

Metric modules live under:

    peph.metrics

Diagnostic code lives under:

    peph.diagnostics

------------------------------------------------------------

# 9. Output Artifacts

Each pipeline run produces a directory under:

    models/

Example contents:

    baseline_table.parquet
    coef_table.parquet
    frailty_table.parquet
    metrics.json
    model.json
    spatial_autocorr.json
    predictions/
    plots/

Artifacts are designed for reproducibility.

Model artifacts can be reloaded using:

    FittedPEPHModel.load()

------------------------------------------------------------

# 10. Simulation Framework

Simulation tools are located in:

    peph.sim

Components:

    ph.py
    spatial.py
    peph.py

Simulation supports:

- baseline PH models
- spatial frailty models
- parameter recovery tests

Simulations are used for:

- validation
- testing
- methodological experiments

------------------------------------------------------------

# 11. Reporting Utilities

Reporting CLI is implemented under:

    peph.report

Capabilities include:

- coefficient tables
- metrics summaries
- frailty summaries
- spatial risk shift tables

Example command:

    python -m peph.cli.main report coef --run-dir models/run123

Tables can be printed to console or exported to CSV.

------------------------------------------------------------

# 12. Testing Strategy

Tests are located in:

    tests/

Two classes of tests exist.

Fast tests:

- unit tests
- serialization tests
- design matrix tests
- pipeline smoke tests

Slow tests:

- simulation recovery
- spatial recovery
- end-to-end validation

Run all tests:

    pytest

Run slow tests:

    pytest -m slow

------------------------------------------------------------

# 13. Performance Considerations

Typical target datasets include SEER-Medicare cohorts.

Scale:

    100k – 500k subjects
    thousands of ZIP codes

Critical performance areas:

- long-format expansion
- sparse matrix operations
- MAP optimization
- prediction vectorization

Avoid:

- dense spatial matrices
- repeated graph construction
- unnecessary dataframe copies

------------------------------------------------------------

# 14. Coding Standards

General guidelines:

- use explicit variable names
- isolate statistical logic from pipeline code
- maintain modular functions
- fail loudly on invalid configurations
- write tests for every new feature

All new modeling features should include:

- simulation validation
- unit tests
- documentation updates

------------------------------------------------------------

# 15. Planned Extensions

Upcoming modeling features include:

Non-proportional hazards

Smoothed baseline hazards

Time-to-treatment effects

Additional spatial priors (BYM2)

Improved frailty inference

Memory-optimized long expansion

------------------------------------------------------------

# 16. Development Workflow

Recommended workflow:

1. create feature branch
2. implement feature
3. add tests
4. run full test suite
5. update documentation
6. submit pull request

Major modeling changes should include simulation validation.

------------------------------------------------------------

# 17. Project Philosophy

PEPH is intended to be:

- statistically rigorous
- reproducible
- transparent
- extensible

It is not intended to be a general-purpose survival library.

Instead, it focuses on providing a **robust implementation of
piecewise exponential survival models for epidemiologic research**.