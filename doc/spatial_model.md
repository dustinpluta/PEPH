
# Spatial Frailty Model: Leroux CAR (MAP Estimation)

This document describes the spatial frailty extension implemented in
**PEPH** using the **Leroux conditional autoregressive (CAR)** prior and
**maximum a posteriori (MAP)** estimation.

The intended application is ZIP‑code–level frailty for SEER–Medicare
analyses, but the model applies to any areal units with a user‑supplied
adjacency graph.

---

# 1. Survival Model with Spatial Frailty

Start with the piecewise exponential proportional hazards (PE‑PH) model:

h_i(t) = h_0(t) * exp(x_i^T β)

Add a spatial frailty term u_{g(i)} for subject i, where g(i) maps each
subject to an areal unit (e.g., ZIP code):

h_i(t) = h_0(t) * exp(x_i^T β + u_{g(i)})

Under a piecewise constant baseline hazard with breaks

0 = τ₀ < τ₁ < ... < τ_K

h_0(t) = ν_k for t ∈ [τ_k, τ_{k+1})

with α_k = log(ν_k).

---

# 2. Poisson Likelihood Representation (Frailty Included)

After expanding to long form with intervals k = 0,…,K−1, the standard
PE–Poisson equivalence yields

Y_{ik} ~ Poisson(μ_{ik})

log μ_{ik} = log(e_{ik}) + α_k + x_i^T β + u_{g(i)}

where

- e_{ik} is the exposure time for subject i in interval k
- Y_{ik} ∈ {0,1} indicates whether the event occurs in interval k

Thus, frailty enters linearly in the GLM predictor exactly like an
additional covariate with group‑specific values.

---

# 3. Leroux CAR Prior

Let there be G areal units and let

u ∈ R^G

be the vector of frailties.

Define

W : symmetric adjacency matrix (w_ij ∈ {0,1})  
D : diagonal degree matrix with d_ii = Σ_j w_ij  
I : identity matrix

The **Leroux precision matrix** is

Q(ρ) = (1 − ρ) I + ρ (D − W)

with

0 ≤ ρ < 1

The Leroux prior is

u | τ, ρ ~ N(0, (τ Q(ρ))^{-1})

where

τ > 0 : global precision parameter  
ρ : spatial dependence parameter

Interpretation:

ρ ≈ 0 → independent frailties  
ρ → 1 → strong spatial smoothing (approaches intrinsic CAR)

PEPH estimates ρ inside (0,1) with boundary clipping for numerical
stability.

---

# 4. Hyperparameter Priors (MAP Regularization)

MAP estimation requires priors on hyperparameters.

Default weakly‑informative choices:

log τ ~ Normal(0, σ²_{log τ})

ρ ~ Beta(a_ρ, b_ρ)

These priors primarily stabilize optimization and prevent degenerate
solutions.

---

# 5. Identifiability Constraints (Centering)

Frailty terms have an intercept‑like ambiguity. Without constraints a
shift in u can be absorbed by the baseline hazard.

PEPH enforces **component‑wise centering**:

Σ_{g ∈ c} w_g u_g = 0

for each connected component c of the spatial graph.

Implementation details:

- `SpatialGraph` stores connected components
- a projection operator removes the component‑wise mean during
  optimization

---

# 6. MAP Objective

Let

θ = (α, β, u, log τ, logit ρ)

be the unconstrained parameter vector used during optimization.

The negative log posterior is

L(θ) =
    − log p(Y | α, β, u)
    − log p(u | τ, ρ)
    − log p(τ)
    − log p(ρ)

## Likelihood term

Using the Poisson representation

− log p(Y | ·) =
Σ_{i,k} ( μ_{ik} − Y_{ik} log μ_{ik} )

where

μ_{ik} = e_{ik} exp( α_k + x_i^T β + u_{g(i)} )

---

# 7. Prediction with Frailty

For subject i in area g

η_i = x_i^T β + u_g

Predicted quantities:

Cumulative hazard

H_i(t) = exp(η_i) H_0(t)

Survival

S_i(t) = exp(−H_i(t))

Risk

R_i(t) = 1 − S_i(t)

## Unseen areas

If a new dataset contains an unseen area label:

Default behavior: **hard fail**

Optional behavior: set u_g = 0

Controlled by `allow_unseen_area`.

---

# 8. Spatial Diagnostics and Outputs (PR9)

When a Leroux spatial frailty model is fitted, PEPH produces additional
spatial diagnostics and artifacts.

## Frailty estimates

frailty_table.parquet

Columns include

zip  
u_hat  
component_id  
degree

frailty_summary.json

Contains summary statistics of the frailty distribution.

## Spatial autocorrelation

spatial_autocorr.json contains:

Moran’s I statistic  
Expected value under spatial randomness  
Variance and approximate z‑score

## Diagnostic plots

plots/frailty_caterpillar.png

Sorted visualization of frailty estimates.

plots/morans_scatter_u.png

Moran scatterplot showing spatial lag vs frailty.

## Frailty‑stratified calibration

Calibration by frailty decile is produced for each evaluation horizon.

tables/calibration_by_frailty_decile_t{h}.parquet

plots/calibration_by_frailty_decile_t{h}.png

These plots help determine whether spatial heterogeneity affects model
calibration.

---

# 9. Practical Notes for Large Graphs

ZIP graphs in SEER–Medicare analyses can contain tens of thousands of
nodes.

PEPH uses the following design choices:

Sparse CSR storage for adjacency (`W_csr_data`)  
Sparse precision matrix construction  
Precomputed connected components

Potential future improvements:

Sparse Cholesky reuse  
Approximate log‑determinant calculations  
Iterative linear solvers

---

# 10. References

Leroux, B. G., Lei, X., & Breslow, N. (2000).
Estimation of disease rates in small areas: a new mixed model for spatial dependence.

Besag, J., York, J., & Mollié, A. (1991).
Bayesian image restoration, with two applications in spatial statistics.

Rue, H., & Held, L. (2005).
Gaussian Markov Random Fields: Theory and Applications.
