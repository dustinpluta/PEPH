# Configuration Reference (`RunConfig` YAML)

This document describes the YAML configuration schema used by the PEPH
pipeline. All runs are driven by a single YAML file that is validated
against `peph.config.schema.RunConfig`.

Run the pipeline via:

``` bash
python -m peph.cli.main run --config path/to/run.yml
```

The configuration is validated on load. Unknown keys or missing required
fields will raise an error.

------------------------------------------------------------------------

# 1. Top-level Structure

A typical configuration file has the following top-level keys:

-   `run_name` *(string)* --- prefix for run output directory names
-   `data` *(object)* --- input dataset location and format
-   `output` *(object)* --- output root directory for artifacts
-   `data_schema` *(object)* --- column names and covariate definitions
-   `time` *(object)* --- time scale and interval breaks
-   `fit` *(object)* --- model fitting options (PH or Leroux)
-   `spatial` *(object, optional)* --- spatial model specification
    (required for Leroux)
-   `split` *(object)* --- train/test split settings
-   `predict` *(object)* --- prediction horizons and frailty mode
-   `metrics` *(object)* --- evaluation metric toggles
-   `plots` *(object)* --- plot toggles

------------------------------------------------------------------------

# 2. Full Example

``` yaml
run_name: seer_crc_peph_v1

data:
  path: data/seer_crc_example.csv
  format: csv

output:
  root_dir: models

data_schema:
  id_col: id
  time_col: time
  event_col: event
  x_numeric:
    - age_per10_centered
    - cci
    - tumor_size_log
    - ses
  x_categorical:
    - sex
    - stage
  categorical_reference_levels:
    sex: F
    stage: I

time:
  breaks: [0, 30, 90, 180, 365, 730, 1825]

fit:
  backend: statsmodels_glm_poisson
  covariance: classical

split:
  test_size: 0.25
  seed: 0

predict:
  horizons_days: [365, 730, 1825]
  frailty_mode: auto

metrics:
  discrimination:
    c_index: true
    time_dependent_auc: true
  calibration:
    brier_score: true
    calibration_in_the_large: true
    calibration_slope: true
  residuals:
    cox_snell: true

plots:
  cox_snell: true
  calibration_risk: true
```

------------------------------------------------------------------------

# 3. `data`

Defines the input dataset location and file format.

``` yaml
data:
  path: <string>
  format: csv|parquet
```

-   `path` *(required)* --- file path to input dataset
-   `format` *(optional, default `csv`)* --- one of:
    -   `csv`
    -   `parquet`

The dataset is expected in **wide survival format** with one row per
subject.

------------------------------------------------------------------------

# 4. `output`

Controls where artifacts are written.

``` yaml
output:
  root_dir: <string>
```

-   `root_dir` *(optional, default `models`)* --- directory where run
    folders are created

Run outputs are created under:

    {output.root_dir}/{run_name}_{timestamp}/

------------------------------------------------------------------------

# 5. `data_schema`

Defines canonical column names and covariates used to build the design
matrix.

``` yaml
data_schema:
  id_col: <string>
  time_col: <string>
  event_col: <string>
  x_numeric: [<string>, ...]
  x_categorical: [<string>, ...]
  categorical_reference_levels:
    <cat_col>: <reference_level>
```

### Fields

-   `id_col` *(required)* --- unique subject identifier column
-   `time_col` *(required)* --- observed time-to-event or censoring time
    (days)
-   `event_col` *(required)* --- event indicator (1=event, 0=censored)
-   `x_numeric` *(required)* --- list of numeric covariate columns
-   `x_categorical` *(required)* --- list of categorical covariate
    columns
-   `categorical_reference_levels` *(required)* --- mapping from each
    categorical column to its reference level

### Categorical encoding

Categoricals are one-hot encoded with the reference level dropped. The
expanded feature name convention is:

-   For categorical column `sex` with level `M` (reference `F`) →
    feature name `sexM`

### Prediction-time behavior

If prediction data contains an unseen level for any categorical
covariate, PEPH **hard fails** by default.

------------------------------------------------------------------------

# 6. `time`

Defines the time discretization for the piecewise exponential baseline
hazard.

``` yaml
time:
  breaks: [<float>, <float>, ...]
```

-   `breaks` *(required)* --- strictly increasing list of time cut
    points in **days**

Interval convention:

-   intervals are **left-closed, right-open**: `[a,b)`
-   an event exactly at a break is assigned to the interval starting at
    that break

------------------------------------------------------------------------

# 7. `fit`

Defines fitting backend and inference options.

``` yaml
fit:
  backend: statsmodels_glm_poisson|map_leroux
  max_iter: <int>
  tol: <float>
  covariance: classical|cluster_id
  leroux_max_iter: <int>
  leroux_ftol: <float>
  rho_clip: <float>
  q_jitter: <float>
  prior_logtau_sd: <float>
  prior_rho_a: <float>
  prior_rho_b: <float>
```

### Fields

-   `backend` *(optional, default `statsmodels_glm_poisson`)*

    -   `statsmodels_glm_poisson` --- standard PE-PH via Poisson GLM
        (fast)
    -   `map_leroux` --- PE-PH + Leroux CAR spatial frailty via MAP
        optimization

-   `max_iter` *(optional)* --- PH optimization iterations
    (backend-dependent)

-   `tol` *(optional)* --- PH optimization tolerance (backend-dependent)

-   `covariance` *(optional, default `classical`)*

    -   `classical` --- model-based covariance
    -   `cluster_id` --- subject-level clustered sandwich covariance

Leroux-only options (used when `backend: map_leroux`):

-   `leroux_max_iter` --- MAP optimizer iterations
-   `leroux_ftol` --- convergence tolerance
-   `rho_clip` --- clipping value to keep rho away from 0 and 1
-   `q_jitter` --- diagonal jitter for numerical stability
-   `prior_logtau_sd` --- prior SD for log(tau)
-   `prior_rho_a`, `prior_rho_b` --- Beta prior parameters for rho

------------------------------------------------------------------------

# 8. `spatial` (optional)

Required when `fit.backend: map_leroux`. Defines the spatial graph and
areal unit mapping.

``` yaml
spatial:
  area_col: <string>
  zips_path: <string>
  edges_path: <string>
  edges_u_col: <string>
  edges_v_col: <string>
  allow_unseen_area: <bool>
```

### Fields

-   `area_col` *(optional, default `zip`)* --- column in the dataset
    giving areal unit labels
-   `zips_path` *(required)* --- CSV listing allowed areal unit labels
    (one per row)
-   `edges_path` *(required)* --- CSV edge list describing adjacency
-   `edges_u_col`, `edges_v_col` *(optional)* --- column names for
    adjacency endpoints
-   `allow_unseen_area` *(optional, default `false`)* --- controls
    behavior when prediction data contains unseen areal units

Unseen area behavior:

-   `false` (default): error if an area label is not in the training
    graph
-   `true`: allow unseen areas and treat frailty as 0 for those rows

------------------------------------------------------------------------

# 9. `split`

Subject-level train/test split configuration.

``` yaml
split:
  test_size: <float>
  seed: <int>
```

-   `test_size` *(optional, default 0.25)* --- fraction of subjects in
    test set
-   `seed` *(optional, default 0)* --- RNG seed for reproducibility

Splitting is performed at the **subject ID** level.

------------------------------------------------------------------------

# 10. `predict`

Controls prediction horizons and frailty prediction behavior.

``` yaml
predict:
  horizons_days: [<float>, ...]
  frailty_mode: auto|none|conditional|marginal
```

-   `horizons_days` *(optional, default \[365, 730, 1825\])* ---
    horizons in days
-   `frailty_mode` *(optional, default `auto`)*:
    -   `auto` --- conditional frailty if model includes spatial
        effects, else none
    -   `none` --- ignore frailty even if present
    -   `conditional` --- use fitted frailty values directly
    -   `marginal` --- reserved for future support (integrate over
        frailty)

------------------------------------------------------------------------

# 11. `metrics`

Enables/disables evaluation metrics.

``` yaml
metrics:
  discrimination:
    c_index: <bool>
    time_dependent_auc: <bool>
  calibration:
    brier_score: <bool>
    calibration_in_the_large: <bool>
    calibration_slope: <bool>
  residuals:
    cox_snell: <bool>
```

All metric flags default to `true` unless specified otherwise in your
config defaults.

------------------------------------------------------------------------

# 12. `plots`

Enables/disables plot generation.

``` yaml
plots:
  cox_snell: <bool>
  calibration_risk: <bool>
```

-   `cox_snell` --- Cox--Snell residual diagnostic plot
-   `calibration_risk` --- calibration risk-by-quantile plot at each
    horizon

Plots can be disabled for faster smoke tests.

------------------------------------------------------------------------

# 13. Notes on Backward Compatibility

The configuration schema is validated strictly. If you change field
names (e.g., `schema` → `data_schema`), old configs must be updated
accordingly.

To reduce breakage in the future, consider adding a schema version field
and supporting migration logic in `load_run_config`.
