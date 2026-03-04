# Statistical Model: Piecewise Exponential Proportional Hazards (PE-PH)

This document describes the statistical foundation of the Piecewise
Exponential Proportional Hazards (PE-PH) model implemented in PEPH.

------------------------------------------------------------------------

# 1. Proportional Hazards Model

The proportional hazards model specifies the hazard function

h(t \| x) = h₀(t) exp(xᵀβ)

where:

-   h₀(t) is the baseline hazard function
-   x is a covariate vector
-   β is the regression coefficient vector

The model assumes proportional hazards:

h(t \| x₁) / h(t \| x₂) = exp((x₁ − x₂)ᵀβ)

------------------------------------------------------------------------

# 2. Piecewise Exponential Baseline Hazard

In the PE model, the baseline hazard is assumed constant within
pre-defined intervals:

Let breakpoints be

0 = τ₀ \< τ₁ \< ... \< τ_K

Then

h₀(t) = ν_k for t ∈ \[τ_k, τ\_{k+1})

where ν_k \> 0.

This yields cumulative baseline hazard:

H₀(t) = Σ_k ν_k × exposure_k(t)

where exposure_k(t) is the time spent in interval k.

------------------------------------------------------------------------

# 3. Survival and Likelihood

For subject i with covariates x_i:

H_i(t) = exp(x_iᵀβ) H₀(t)

Survival:

S_i(t) = exp(-H_i(t))

For right-censored data, the likelihood contribution is:

L_i = \[h_i(T_i)\]\^{δ_i} × S_i(T_i)

where:

-   T_i is observed time
-   δ_i is event indicator

------------------------------------------------------------------------

# 4. The Poisson Trick

The key computational insight is that the PE likelihood can be rewritten
as a Poisson likelihood.

After expanding each subject into interval-specific observations:

Y_ik \~ Poisson(μ_ik)

with

log μ_ik = log(exposure_ik) + α_k + x_iᵀβ

where:

-   α_k = log(ν_k)
-   exposure_ik is time at risk in interval k

This transforms survival estimation into a generalized linear model
(GLM) with log link and offset log(exposure).

This formulation:

-   yields identical maximum likelihood estimates
-   allows use of standard GLM software
-   provides standard errors via Fisher information

------------------------------------------------------------------------

# 5. Inference

Parameters estimated:

-   Baseline interval log-hazards α_k
-   Regression coefficients β

Standard errors are derived from the observed information matrix.

Optional cluster-robust covariance can be computed at the subject level.

Wald confidence intervals:

β̂ ± 1.96 × SE(β̂)

------------------------------------------------------------------------

# 6. Prediction

Given estimated parameters:

Cumulative hazard:

Ĥ\_i(t) = exp(x_iᵀβ̂) Ĥ₀(t)

Survival:

Ŝ\_i(t) = exp(-Ĥ\_i(t))

Risk at horizon t:

R̂\_i(t) = 1 − Ŝ\_i(t)

------------------------------------------------------------------------

# 7. Advantages of the PE Formulation

Compared to the Cox model:

-   Baseline hazard is explicitly estimated
-   Direct likelihood-based inference
-   Natural integration with GLM framework
-   Straightforward extension to frailty models
-   Efficient for large datasets

------------------------------------------------------------------------

# 8. Connection to Spatial Frailty

With spatial frailty u_g:

h_i(t) = h₀(t) exp(x_iᵀβ + u_g)

This modifies the linear predictor in the Poisson GLM.

Frailty estimation requires penalized likelihood / MAP optimization.

------------------------------------------------------------------------

# 9. References

Holford (1980). Analysis of rates and survivorship using log-linear
models.

Laird & Olivier (1981). Covariance analysis of censored survival data
using log-linear techniques.
