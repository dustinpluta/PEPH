# PEPH: Piecewise Exponential Proportional Hazards Models in Python

PEPH is a production-oriented Python package for fitting and evaluating **piecewise exponential proportional hazards (PE-PH) survival models**. The package supports both standard proportional hazards models and **spatial frailty extensions** using conditional autoregressive (CAR) priors.

The implementation is designed for large-scale epidemiologic datasets, with particular focus on **SEER–Medicare cancer survival data**, where cohort sizes can reach hundreds of thousands of subjects and spatial structure (e.g., ZIP-code–level effects) may be present.

The package provides an end-to-end modeling pipeline including:

- data preparation
- piecewise exponential model fitting
- spatial frailty modeling
- survival and risk prediction
- model diagnostics
- calibration and discrimination metrics
- reproducible configuration-driven workflows

All modeling steps are controlled via YAML configuration files, enabling reproducible analyses and easy integration into larger data pipelines.

---

# Statistical Model

PEPH implements the **piecewise exponential proportional hazards model**

\[
h(t \mid x) = h_0(t)\exp(x^\top \beta)
\]

where the baseline hazard is assumed constant within user-defined time intervals:

\[
h_0(t) = \nu_k \quad \text{for } t \in [\tau_k, \tau_{k+1})
\]

This formulation enables efficient estimation using the **Poisson likelihood trick**, converting survival likelihoods into generalized linear models.

The model supports:

- right-censored survival data
- arbitrary covariates (numeric and categorical)
- user-defined time intervals
- standard errors and Wald inference
- robust (clustered) covariance estimation

---

# Spatial Frailty Extension

PEPH also supports **Leroux conditional autoregressive (CAR) spatial frailty models** for areal data such as ZIP codes.

The hazard model becomes

\[
h_i(t) = h_0(t)\exp(x_i^\top \beta + u_{g(i)})
\]

where

- \(u_g\) is a spatial random effect
- \(g(i)\) indexes the spatial unit for subject \(i\)

Frailties follow the **Leroux CAR model**

\[
u \sim N\left(0,\;(\tau Q(\rho))^{-1}\right)
\]

with precision matrix

\[
Q(\rho) = (1-\rho)I + \rho(D-W)
\]

where

- \(W\) is the adjacency matrix
- \(D\) is the degree matrix
- \(0 \le \rho < 1\) controls spatial dependence
- \(\tau\) controls overall precision

Frailty parameters are estimated via **maximum a posteriori (MAP) optimization**.

---

# Core Features

## Data Pipeline

The package includes utilities to prepare survival datasets for PE modeling:

- subject-level train/test splits
- conversion from wide survival format to **long counting-process format**
- categorical encoding with strict prediction-time validation
- administrative censoring support

---

## Model Fitting

Two model types are currently supported:

### Piecewise Exponential PH

Estimated using a **Poisson GLM formulation** implemented via `statsmodels`.

Features include:

- baseline hazard estimation
- Wald confidence intervals
- cluster-robust covariance
- fast estimation for large datasets

---

### Spatial Frailty Model (Leroux CAR)

Adds spatially structured random effects with:

- Leroux CAR prior
- connected-component centering constraints
- MAP estimation
- compatibility with arbitrary spatial graphs

Spatial structure is defined through a **`SpatialGraph`** object containing:

- node labels (e.g., ZIP codes)
- adjacency relationships
- connected component structure

---

## Prediction

The package supports prediction of:

- survival probability \(S(t)\)
- cumulative hazard \(H(t)\)
- event risk \(1 - S(t)\)

Predictions can incorporate frailty estimates when spatial models are used.

---

## Model Evaluation

PEPH includes a suite of survival model evaluation tools:

### Discrimination

- time-dependent AUC
- concordance index (C-index)

### Calibration

- calibration curves
- calibration slope
- calibration in the large
- Brier score

### Diagnostics

- Cox–Snell residuals
- interval hazard diagnostics
- risk calibration plots

These metrics can be computed at multiple prediction horizons.

---

# Configuration-Driven Pipeline

Analyses are executed through a configuration file rather than command-line arguments.

Example workflow:

```bash
python -m peph.cli.main run --config configs/run.yml
