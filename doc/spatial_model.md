# Spatial Frailty Model: Leroux CAR (MAP Estimation)

This document describes the spatial frailty extension implemented in
PEPH using the **Leroux conditional autoregressive (CAR)** prior and
**maximum a posteriori (MAP)** estimation.

The intended application is ZIP-code--level frailty for SEER--Medicare
analyses, but the model applies to any areal units with a user-supplied
adjacency graph.

------------------------------------------------------------------------

# 1. Survival Model with Spatial Frailty

Start with the piecewise exponential proportional hazards (PE-PH) model:

\[ h_i(t) = h_0(t)`\exp`{=tex}(x_i\^`\top `{=tex}`\beta`{=tex}) \]

Add a spatial frailty term (u\_{g(i)}) for subject (i), where (g(i))
maps each subject to an areal unit (e.g., ZIP code):

\[ h_i(t) = h_0(t)`\exp`{=tex}(x_i\^`\top `{=tex}`\beta `{=tex}+
u\_{g(i)}) \]

Under a piecewise constant baseline hazard with breaks
(0=`\tau`{=tex}\_0\<`\cdots`{=tex}\<`\tau`{=tex}\_K),

\[ h_0(t) =
`\nu`{=tex}\_k,`\quad `{=tex}t`\in`{=tex}\[`\tau`{=tex}*k,`\tau`{=tex}*{k+1})
\]

with (`\alpha`{=tex}\_k=`\log`{=tex}`\nu`{=tex}\_k).

------------------------------------------------------------------------

# 2. Poisson Likelihood Representation (Frailty Included)

After expanding to long form with intervals (k=0,`\dots`{=tex},K-1), the
standard PE--Poisson equivalence yields:

\[ Y\_{ik} `\sim `{=tex}`\mathrm{Poisson}`{=tex}(`\mu`{=tex}\_{ik}) \]

\[ `\log `{=tex}`\mu`{=tex}*{ik} = `\log`{=tex}(e*{ik}) +
`\alpha`{=tex}*k + x_i\^`\top`{=tex}`\beta `{=tex}+ u*{g(i)} \]

where

-   (e\_{ik}) is the exposure time for subject (i) in interval (k),
-   (Y\_{ik}`\in`{=tex}{0,1}) indicates whether the event occurs in
    interval (k).

Thus, frailty enters linearly in the GLM predictor exactly like an
additional covariate with group-specific values.

------------------------------------------------------------------------

# 3. Leroux CAR Prior

Let there be (G) areal units, and let
(u`\in`{=tex}`\mathbb{R}`{=tex}\^G) be the vector of frailties.

Define:

-   (W): symmetric adjacency matrix with entries
    (w\_{ij}`\in`{=tex}{0,1}), zero diagonal.
-   (D): diagonal degree matrix with (d\_{ii} = `\sum`{=tex}*j w*{ij}).
-   (I): identity matrix.

The **Leroux** precision (up to scale) is

\[ Q(`\rho`{=tex}) = (1-`\rho`{=tex})I + `\rho`{=tex}(D-W) \]

with (0 `\le `{=tex}`\rho `{=tex}\< 1).

The Leroux prior is

\[ u
`\mid `{=tex}`\tau`{=tex},`\rho `{=tex}`\sim `{=tex}`\mathcal{N}`{=tex}`\left`{=tex}(0,;(`\tau `{=tex}Q(`\rho`{=tex}))\^{-1}`\right`{=tex})
\]

where

-   (`\tau`{=tex}\>0) is a global precision parameter,
-   (`\rho`{=tex}) controls spatial structure:
    -   (`\rho `{=tex}`\approx 0`{=tex}): nearly independent (u_g)
    -   (`\rho `{=tex}`\rightarrow 1`{=tex}): intrinsic CAR-like
        smoothing (improper at (`\rho=1`{=tex}))

In practice PEPH estimates (`\rho`{=tex}) in ((0,1)), with a small clip
away from the boundaries for numerical stability.

------------------------------------------------------------------------

# 4. Priors / Regularization for Hyperparameters

PEPH uses MAP estimation. This requires priors on hyperparameters. A
common choice is:

-   prior on (`\log`{=tex}`\tau`{=tex}): Gaussian \[
    `\log`{=tex}`\tau `{=tex}`\sim `{=tex}`\mathcal{N}`{=tex}(0,`\sigma`{=tex}\_{`\log`{=tex}`\tau`{=tex}}\^2)
    \]
-   prior on (`\rho`{=tex}): Beta \[
    `\rho `{=tex}`\sim `{=tex}`\mathrm{Beta}`{=tex}(a\_`\rho`{=tex},b\_`\rho`{=tex})
    \]

These priors are weakly informative by default but stabilize
optimization.

------------------------------------------------------------------------

# 5. Identifiability Constraints (Centering)

Frailty terms have an intercept-like ambiguity. Without constraints, a
shift in (u) can be absorbed into the baseline/intercept, leading to
non-identifiability and unstable estimates.

PEPH enforces **component-wise centering**:

For each connected component (c) of the spatial graph, impose

\[ `\sum`{=tex}\_{g`\in `{=tex}c} w_g u_g = 0 \]

where (w_g`\ge 0`{=tex}) are weights (default: equal weights, or
proportional to number of subjects per area).

This constraint is applied **per connected component**, which is
necessary when the graph is not fully connected.

Implementation detail: - the `SpatialGraph` stores `components`
(component IDs per node), - a projection operator removes the weighted
mean within each component.

------------------------------------------------------------------------

# 6. MAP Objective

Let (`\theta `{=tex}=
(`\alpha`{=tex},`\beta`{=tex},u,`\log`{=tex}`\tau`{=tex},`\mathrm{logit}`{=tex}(`\rho`{=tex})))
be the unconstrained parameterization used in optimization.

The negative log-posterior is:

\[ `\mathcal{L}`{=tex}(`\theta`{=tex}) =
-`\log `{=tex}p(Y`\mid `{=tex}`\alpha`{=tex},`\beta`{=tex},u) -
`\log `{=tex}p(u`\mid `{=tex}`\tau`{=tex},`\rho`{=tex}) -
`\log `{=tex}p(`\tau`{=tex}) - `\log `{=tex}p(`\rho`{=tex}) \]

### 6.1 Likelihood term

Using the Poisson likelihood for the long-form data:

\[ -`\log `{=tex}p(Y`\mid`{=tex}`\cdot`{=tex}) = `\sum`{=tex}*{i,k}
`\left`{=tex}(`\mu`{=tex}*{ik} -
Y\_{ik}`\log`{=tex}`\mu`{=tex}\_{ik}`\right`{=tex}) +
`\mathrm{const}`{=tex} \]

with

\[ `\mu`{=tex}*{ik} = e*{ik}`\exp`{=tex}(`\alpha`{=tex}*k +
x_i\^`\top`{=tex}`\beta `{=tex}+ u*{g(i)}) \]

### 6.2 Frailty prior term

Ignoring constants:

\[ -`\log `{=tex}p(u`\mid`{=tex}`\tau`{=tex},`\rho`{=tex}) =
`\frac{\tau}{2}`{=tex}u\^`\top `{=tex}Q(`\rho`{=tex})u -
`\frac{1}{2}`{=tex}`\log`{=tex}`\det`{=tex}(`\tau `{=tex}Q(`\rho`{=tex}))
\]

\[ = `\frac{\tau}{2}`{=tex}u\^`\top `{=tex}Q(`\rho`{=tex})u -
`\frac{G}{2}`{=tex}`\log`{=tex}`\tau `{=tex}-
`\frac{1}{2}`{=tex}`\log`{=tex}`\det`{=tex}(Q(`\rho`{=tex})) \]

### 6.3 Hyperpriors

-   Gaussian prior on (`\log`{=tex}`\tau`{=tex}) contributes: \[
    `\frac{(\log\tau)^2}{2\sigma_{\log\tau}^2}`{=tex} \]
-   Beta prior on (`\rho`{=tex}) contributes: \[
    -(a\_`\rho-1`{=tex})`\log`{=tex}`\rho `{=tex}-
    (b\_`\rho-1`{=tex})`\log`{=tex}(1-`\rho`{=tex}) \]

Optimization is performed over unconstrained variables: -
(`\log`{=tex}`\tau`{=tex}`\in`{=tex}`\mathbb{R}`{=tex}) -
(`\rho `{=tex}=
`\mathrm{sigmoid}`{=tex}(`\eta`{=tex}\_`\rho`{=tex})`\in`{=tex}(0,1))

------------------------------------------------------------------------

# 7. Prediction with Frailty

For a subject in area (g), the linear predictor is:

\[ `\eta`{=tex}\_i = x_i\^`\top`{=tex}`\beta `{=tex}+ u_g \]

PEPH supports multiple frailty modes in prediction:

-   **none**: ignore frailty ((u_g=0))
-   **conditional**: use fitted (u_g) directly
-   **auto**: use conditional if the fitted model includes frailty, else
    none

The key outputs at horizon (t) are:

-   cumulative hazard: \[ `\hat `{=tex}H_i(t) =
    `\exp`{=tex}(`\eta`{=tex}\_i)`\hat `{=tex}H_0(t) \]
-   survival: \[ `\hat `{=tex}S_i(t) =
    `\exp`{=tex}(-`\hat `{=tex}H_i(t)) \]
-   risk: \[ `\hat `{=tex}R_i(t) = 1-`\hat `{=tex}S_i(t) \]

### Unseen areas at prediction time

If a new dataset contains an area label not present during training:

-   default behavior is **hard fail**
-   optional behavior allows unseen areas by setting (u_g=0) for those
    rows

This behavior is configured via `allow_unseen_area`.

------------------------------------------------------------------------

# 8. Practical Notes for Large Graphs

For SEER--Medicare analyses, the ZIP graph can be large (tens of
thousands of nodes). Production implementations should avoid dense
matrix operations.

Design choices used in PEPH:

-   adjacency stored as sparse CSR (`W_csr_data`)
-   `SpatialGraph.leroux_Q(rho)` constructs sparse (Q(`\rho`{=tex}))
-   connected components stored to enable stable centering constraints

Future scaling enhancements may include:

-   sparse Cholesky / factorization reuse during optimization
-   approximate log-determinant calculations
-   iterative solvers (CG) with preconditioning

------------------------------------------------------------------------

# 9. References

Leroux, B. G., Lei, X., & Breslow, N. (2000).\
*Estimation of disease rates in small areas: a new mixed model for
spatial dependence.*\
Statistical Models in Epidemiology.

Besag, J., York, J., & Mollié, A. (1991).\
*Bayesian image restoration, with two applications in spatial
statistics.*\
Annals of the Institute of Statistical Mathematics.

Rue, H., & Held, L. (2005).\
*Gaussian Markov Random Fields: Theory and Applications.*
