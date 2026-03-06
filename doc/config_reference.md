# PEPH Configuration Reference

This document describes the YAML configuration schema used by the PEPH
modeling pipeline. All pipeline runs are driven by a single YAML file
validated against the `RunConfig` schema in:

peph.config.schema

Run the pipeline via:

python -m peph.cli.main run --config path/to/run.yml

The configuration file controls every stage of the modeling workflow:

• data loading  
• design matrix construction  
• model fitting  
• prediction  
• metrics and diagnostics  
• artifact generation  

Invalid keys or missing required fields will raise a validation error.

---------------------------------------------------------------------

# 1. Top-Level Structure

A typical configuration file contains the following sections:

run_name  
data  
data_schema  
time  
split  
fit  
spatial (optional)  
predict  
metrics  
plots  
output

Example configuration structure:

run_name: seer_crc_peph_v1

data:
  path: data/simulated_seer_crc.csv
  format: csv

data_schema:
  ...

time:
  ...

split:
  ...

fit:
  ...

spatial:
  ...

predict:
  ...

metrics:
  ...

plots:
  ...

output:
  root_dir: models

---------------------------------------------------------------------

# 2. run_name

run_name: <string>

A descriptive name for the pipeline run.

Used to generate the run directory:

{output.root_dir}/{run_name}_{timestamp}

Example:

models/seer_crc_peph_v1_2026-03-05_1124/

---------------------------------------------------------------------

# 3. data

Specifies the dataset location and file format.

data:
  path: <string>
  format: csv | parquet

Fields:

path  
Path to the dataset file.

format  
Input format. Supported values:

csv  
parquet

The dataset must be in **wide survival format**, meaning one row per subject.

---------------------------------------------------------------------

# 4. data_schema

Defines the dataset column names and covariates used to build the design matrix.

data_schema:

  id_col: <string>  
  time_col: <string>  
  event_col: <string>  

  x_numeric:
    - <column>
    - ...

  x_categorical:
    - <column>
    - ...

  categorical_reference_levels:
    <column>: <reference_level>

Fields:

id_col  
Unique subject identifier.

time_col  
Observed survival or censoring time (days).

event_col  
Event indicator (1 = event, 0 = censored).

x_numeric  
List of numeric covariate columns.

x_categorical  
List of categorical covariate columns.

categorical_reference_levels  
Mapping specifying the baseline level for each categorical variable.

Categorical variables are encoded using one-hot encoding with the reference level dropped.

Example:

sex: F  
stage: I

Expanded design matrix feature names follow the pattern:

sexM  
stageII  
stageIII  

Prediction-time behavior:

If unseen categorical levels appear in prediction data,
PEPH raises a hard error unless explicitly allowed.

---------------------------------------------------------------------

# 5. time

Defines the piecewise exponential baseline hazard intervals.

time:

  scale: days  
  breaks: [<float>, ...]  
  interval_closed: left  
  interval_open: right  

Fields:

scale  
Time unit. Currently informational (typically "days").

breaks  
Strictly increasing list of interval boundaries.

Example:

[0, 30, 90, 180, 365, 730, 1825]

Intervals are interpreted as:

[a, b)

Events occurring exactly at a breakpoint are assigned to the interval beginning at that breakpoint.

interval_closed / interval_open  
Document the interval convention.

---------------------------------------------------------------------

# 6. split

Controls the train/test split.

split:

  strategy: subject_random  
  test_size: <float>  
  seed: <int>

Fields:

strategy  
Method for splitting subjects.

Current option:

subject_random

Splitting occurs at the **subject ID level** to avoid data leakage.

test_size  
Fraction of subjects assigned to the test set.

seed  
Random seed used for reproducibility.

---------------------------------------------------------------------

# 7. fit

Defines the model backend and estimation parameters.

fit:

  backend: statsmodels_glm_poisson | map_leroux

  leroux_max_iter: <int>  
  leroux_ftol: <float>  

  rho_clip: <float>  
  q_jitter: <float>  

  prior_logtau_sd: <float>  
  prior_rho_a: <float>  
  prior_rho_b: <float>

Fields:

backend

statsmodels_glm_poisson  
Standard piecewise exponential PH model.

map_leroux  
Piecewise exponential PH with Leroux spatial frailty.

---------------------------------------------------------------------

Leroux optimization parameters:

leroux_max_iter  
Maximum MAP optimizer iterations.

leroux_ftol  
Function tolerance for convergence.

rho_clip  
Prevents rho from approaching 0 or 1 exactly.

q_jitter  
Small diagonal value added to the precision matrix for numerical stability.

---------------------------------------------------------------------

Leroux priors:

prior_logtau_sd  
Standard deviation for Gaussian prior on log(tau).

prior_rho_a  
Beta prior parameter for rho.

prior_rho_b  
Beta prior parameter for rho.

---------------------------------------------------------------------

# 8. spatial

Required when using the Leroux backend.

spatial:

  area_col: <string>

  zips_path: <string>  
  edges_path: <string>

  edges_u_col: <string>  
  edges_v_col: <string>

  allow_unseen_area: <bool>

Fields:

area_col  
Dataset column specifying the spatial unit.

Example:

zip

zips_path  
CSV file listing valid spatial units.

edges_path  
CSV edge list defining adjacency relationships.

edges_u_col  
First column of the adjacency edge.

edges_v_col  
Second column of the adjacency edge.

allow_unseen_area

false (default)  
Error if prediction data contains areas not in the graph.

true  
Allow unseen areas and treat frailty as zero.

---------------------------------------------------------------------

# 9. predict

Controls prediction horizons and prediction grid.

predict:

  horizons_days: [<float>, ...]

  grid:
    use_breaks: <bool>
    extra_times: [<float>, ...]

Fields:

horizons_days  
List of time horizons (in days) for risk prediction.

Example:

[365, 730, 1825]

grid.use_breaks

true  
Prediction grid includes the baseline hazard breakpoints.

grid.extra_times

Additional custom prediction times.

Example:

extra_times: [60, 120]

---------------------------------------------------------------------

# 10. metrics

Controls evaluation metrics.

metrics:

  compute_on: test

  discrimination:
    c_index: <bool>
    time_dependent_auc: <bool>

  calibration:
    calibration_in_the_large: <bool>
    calibration_slope: <bool>
    brier_score: <bool>

  residuals:
    cox_snell: <bool>

compute_on

Which dataset to compute metrics on.

Options:

test  
train

Most runs should use:

compute_on: test

---------------------------------------------------------------------

# 11. plots

Controls diagnostic plot generation.

plots:

  calibration_risk: <bool>
  cox_snell: <bool>

Plots are saved to:

plots/

calibration_risk

Produces calibration curves across risk quantiles.

cox_snell

Produces Cox–Snell residual diagnostics.

Plots use matplotlib.

---------------------------------------------------------------------

# 12. output

Defines where run artifacts are written.

output:

  root_dir: <string>

Example:

root_dir: models

The pipeline writes outputs to:

{root_dir}/{run_name}_{timestamp}/

Artifacts include:

model.json  
coef_table.parquet  
baseline_table.parquet  
metrics.json  
frailty_table.parquet  
predictions/  
plots/

---------------------------------------------------------------------

# 13. Configuration Best Practices

Use explicit reference levels for categorical variables.

Always set a random seed for reproducibility.

Avoid modifying configs after model runs.

Store configs alongside model artifacts.

Prefer descriptive run names reflecting dataset and model version.

---------------------------------------------------------------------

# 14. Future Configuration Extensions

Upcoming features may add configuration sections for:

non_proportional_hazards  
baseline_smoothing  
treatment_effects  
frailty_marginalization  

These will extend the current schema while preserving compatibility.
