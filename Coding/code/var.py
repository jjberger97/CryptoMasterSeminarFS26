#%% DESC

"""
Vector autoregression analysis for crypto-market and credit-spread dynamics.

Input:
    data_table.csv imported from GitHub

Main analysis:
    Six crypto-indicator VARs with HY and IG credit spreads

Output:
    CSV result tables saved in the var_results directory
"""


#%%------- IMPORT PACKAGES

import pandas as pd
import numpy as np
from urllib.parse import quote
from pathlib import Path
from statsmodels.stats.multitest import multipletests
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller


#%%------- SETTINGS

# github path
github_url = "https://raw.githubusercontent.com/jjberger97/CryptoMasterSeminarFS26/main/Coding/code"
data_frame_file = "data_table.csv"

# create output folder for VAR results
script_dir = Path(__file__).resolve().parent
output_dir = script_dir / "var_results"
output_dir.mkdir(exist_ok=True)

# full sample used in paper
sample_start = pd.to_datetime("2020-01-01")
sample_end = pd.to_datetime("2025-12-31")

# split-sample period
split_stage1_end = pd.to_datetime("2023-12-31")

# maximum lag length in lag selection
maxlags = 10

# deterministic terms in VAR
# "c" = constant only, no deterministic trend
trend = "c"

# VAR diagnostics
whiteness_test_horizons = [5, 10, 20]
minimum_observations = 150

# generalized impulse responses
irf_periods = 20
bootstrap_reps = 500
bootstrap_seed = 20260730
bootstrap_block_length = 10


#%%------- HELPER FUNC: General

# creates github url
def raw_github_path(base_url, filename):
    return f"{base_url}/{quote(filename)}"

# makes dates under date col to dates and sort by date asc
def standardize_date_column(df):
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df = df.sort_values("date").drop_duplicates("date").reset_index(drop=True)
    return df


# get csv from github
def read_github_csv(base_url, filename):
    url = raw_github_path(base_url, filename)
    return pd.read_csv(url)


# keep only cols listed if they exist = crash protection
def existing_cols(df, cols):
    return [c for c in cols if c in df.columns]


# stop if a variable required by the empirical design is missing
def require_columns(df, cols, label):
    missing = [c for c in cols if c not in df.columns]

    if missing:
        raise ValueError(
            f"missing {label} columns in {data_frame_file}: {', '.join(missing)}"
        )


# remove duplicate variable names while keeping the empirical ordering
def unique_list(values):
    return list(dict.fromkeys(values))


# creates the complete-case sample for one VAR
def create_var_sample(df, variables, common_dates=None, start=None, end=None):
    variables = unique_list(variables)
    var_df = df[["date"] + variables].copy()

    if start is not None:
        var_df = var_df[var_df["date"] >= start].copy()

    if end is not None:
        var_df = var_df[var_df["date"] <= end].copy()

    if common_dates is not None:
        var_df = var_df[var_df["date"].isin(common_dates)].copy()

    var_df = var_df.dropna(subset=variables).copy()
    var_df = var_df.sort_values("date").set_index("date")

    return var_df


# add model labels to each result table
def add_metadata(df, metadata):
    df = df.copy()

    for column, value in reversed(list(metadata.items())):
        df.insert(0, column, value)

    return df


# turn VAR coefficients into dataframe
def extract_var_coefficients(var_result):
    rows = []

    for equation in var_result.params.columns:
        for parameter in var_result.params.index:

            rows.append({
                "equation": equation,
                "parameter": parameter,
                "coef": var_result.params.loc[parameter, equation],
                "std_error": var_result.stderr.loc[parameter, equation],
                "t_stat": var_result.tvalues.loc[parameter, equation],
                "p_value": var_result.pvalues.loc[parameter, equation],
                "nobs": var_result.nobs,
                "lag_order": var_result.k_ar,
                "aic": var_result.aic,
                "bic": var_result.bic,
                "hqic": var_result.hqic,
                "fpe": var_result.fpe,
            })

    return pd.DataFrame(rows)



#%%------- HELPER FUNC: Lag selection
# select one to ten lags using a common comparison sample
"""
DESC:   Tests VAR lags 1–10
        Selects lowest AIC, BIC, HQIC or FPE
        Uses common sample for all lags
        Prevents sample differences from affecting selection
        Returns selected lag and full comparison table
"""
def select_var_lag(var_df, maxlags=10, criterion="bic", trend="c"):
    nobs = len(var_df)
    nvars = var_df.shape[1]
    model_data = var_df.reset_index(drop=True)

    # number of deterministic parameters in each equation
    ntrend = (
        len(trend)
        if trend.startswith("c")
        else 0
    )

    # exact Statsmodels limit for the largest estimable VAR
    max_estimable = (
        nobs
        - nvars
        - ntrend
    ) // (
        1
        + nvars
    )

    feasible_maxlags = min(
        maxlags,
        max_estimable
    )

    if feasible_maxlags < 1:
        raise ValueError(
            f"no fitting lag len for n={nobs}, variables={nvars}"
        )

    # Statsmodels compares all lag orders using the same observations
    lag_selection = VAR(model_data).select_order(
        maxlags=feasible_maxlags,
        trend=trend
    )

    # with a constant, Statsmodels reports criteria for lags 0 to maxlags
    minimum_reported_lag = (
        0
        if trend != "n"
        else 1
    )

    common_selection_nobs = (
        nobs
        - feasible_maxlags
    )

    rows = []

    # the empirical design considers lags 1 to 10, not lag zero
    for lag in range(1, feasible_maxlags + 1):
        criterion_index = (
            lag
            - minimum_reported_lag
        )

        number_of_parameters_per_equation = (
            nvars * lag
            + ntrend
        )

        rows.append({
            "candidate_lag": lag,
            "aic": lag_selection.ics[
                "aic"
            ][criterion_index],
            "bic": lag_selection.ics[
                "bic"
            ][criterion_index],
            "hqic": lag_selection.ics[
                "hqic"
            ][criterion_index],
            "fpe": lag_selection.ics[
                "fpe"
            ][criterion_index],
            "nobs": common_selection_nobs,
            "df_resid": (
                common_selection_nobs
                - number_of_parameters_per_equation
            ),
            "fit_error": np.nan,
        })

    lag_table = pd.DataFrame(rows)

    valid = lag_table.dropna(
        subset=[criterion]
    ).copy()

    if valid.empty:
        raise ValueError(
            f"Lag sel fail criterion: {criterion}"
        )

    selected_lag = int(
        valid.loc[
            valid[criterion].idxmin(),
            "candidate_lag"
        ]
    )

    # show what every criterion selects among lags 1 to maxlags
    for information_criterion in [
        "aic",
        "bic",
        "hqic",
        "fpe"
    ]:
        valid_ic = lag_table.dropna(
            subset=[information_criterion]
        )

        selected_column = (
            f"selected_by_{information_criterion}"
        )
        lag_table[selected_column] = False

        if not valid_ic.empty:
            selected_ic_lag = int(
                valid_ic.loc[
                    valid_ic[
                        information_criterion
                    ].idxmin(),
                    "candidate_lag"
                ]
            )

            lag_table[selected_column] = (
                lag_table["candidate_lag"]
                == selected_ic_lag
            )

    lag_table["selected_for_model"] = (
        lag_table["candidate_lag"]
        == selected_lag
    )
    lag_table["selection_criterion"] = criterion
    lag_table["maximum_lag_considered"] = (
        feasible_maxlags
    )
    lag_table["common_selection_sample"] = True

    return selected_lag, lag_table

#%%------- HELPER FUNC: ADF stationarity tests
# ADF stationarity helper
"""
DESC:   Test if there is "no trend" in data. Sort of redundant with this data...
        Most variables are already differences, returns, or log changes
"""

def run_adf_tests(df, variables, sample_label):
    rows = []

    for var in variables:
        series = pd.to_numeric(df[var], errors="coerce").dropna()

        try:
            adf_result = adfuller(
                series,
                regression="c",
                autolag="AIC"
            )

            rows.append({
                "sample": sample_label,
                "variable": var,
                "adf_stat": adf_result[0],
                "p_value": adf_result[1],
                "used_lag": adf_result[2],
                "nobs": adf_result[3],
                "critical_value_1pct": adf_result[4]["1%"],
                "critical_value_5pct": adf_result[4]["5%"],
                "critical_value_10pct": adf_result[4]["10%"],
                "stationary_at_10pct": adf_result[1] < 0.10,
                "stationary_at_5pct": adf_result[1] < 0.05,
                "stationary_at_1pct": adf_result[1] < 0.01,
                "error": np.nan,
            })

        except Exception as e:
            rows.append({
                "sample": sample_label,
                "variable": var,
                "adf_stat": np.nan,
                "p_value": np.nan,
                "used_lag": np.nan,
                "nobs": len(series),
                "critical_value_1pct": np.nan,
                "critical_value_5pct": np.nan,
                "critical_value_10pct": np.nan,
                "stationary_at_10pct": False,
                "stationary_at_5pct": False,
                "stationary_at_1pct": False,
                "error": str(e),
            })

    return pd.DataFrame(rows)

#%%------- HELPER FUNC: Granger tests
"""
DESC:   Tests predictive relationships in both directions
        Uses joint F-tests across all relevant lags
        Tests for individual variables and variable groups
        Creates p-values and significance indicators
        Version with test[S] is for joint tests
"""
# one joint Granger test
# tests whether all named causing lags are jointly zero in caused equation
def run_granger_test(
    var_result,
    causing_vars,
    caused_vars,
    direction,
    test_scope
):
    causing_vars = list(causing_vars)
    caused_vars = list(caused_vars)

    row = {
        "caused": " + ".join(caused_vars),
        "causing": " + ".join(causing_vars),
        "caused_variables": ", ".join(caused_vars),
        "causing_variables": ", ".join(causing_vars),
        "direction": direction,
        "test_scope": test_scope,
        "test": "granger_causality_f_test",
        "statistic": np.nan,
        "p_value": np.nan,
        "df": np.nan,
        "significant_10pct": False,
        "significant_5pct": False,
        "significant_1pct": False,
        "lag_order": var_result.k_ar,
        "nobs": var_result.nobs,
        "error": np.nan,
    }

    try:
        test = var_result.test_causality(
            caused=caused_vars,
            causing=causing_vars,
            kind="f"
        )

        row.update({
            "statistic": test.test_statistic,
            "p_value": test.pvalue,
            "df": str(test.df),
            "significant_10pct": test.pvalue < 0.10,
            "significant_5pct": test.pvalue < 0.05,
            "significant_1pct": test.pvalue < 0.01,
        })

    except Exception as e:
        row["error"] = str(e)

    return row

# run forward and reverse tests required for one system
def run_granger_tests(var_result, crypto_vars, outcome_vars):
    rows = []

    # individual crypto -> outcome tests
    for crypto_var in crypto_vars:
        for outcome_var in outcome_vars:
            rows.append(
                run_granger_test(
                    var_result=var_result,
                    causing_vars=[crypto_var],
                    caused_vars=[outcome_var],
                    direction="crypto_to_risk",
                    test_scope="individual"
                )
            )

    # joint family/full-system crypto -> outcome tests
    if len(crypto_vars) > 1:
        for outcome_var in outcome_vars:
            rows.append(
                run_granger_test(
                    var_result=var_result,
                    causing_vars=crypto_vars,
                    caused_vars=[outcome_var],
                    direction="crypto_to_risk",
                    test_scope="joint_crypto_group"
                )
            )

    # individual outcome -> crypto reverse tests
    for outcome_var in outcome_vars:
        for crypto_var in crypto_vars:
            rows.append(
                run_granger_test(
                    var_result=var_result,
                    causing_vars=[outcome_var],
                    caused_vars=[crypto_var],
                    direction="risk_to_crypto",
                    test_scope="individual"
                )
            )

    # joint outcomes -> crypto tests are additional feedback checks
    if len(outcome_vars) > 1:
        for crypto_var in crypto_vars:
            rows.append(
                run_granger_test(
                    var_result=var_result,
                    causing_vars=outcome_vars,
                    caused_vars=[crypto_var],
                    direction="risk_to_crypto",
                    test_scope="joint_risk_group"
                )
            )

    return pd.DataFrame(rows)

#%%------- HELPER FUNC: Impulse response functions
"""
DESC:   Calculates generalized impulse responses
        Uses one standard deviation crypto shock
        Does not depend on variable ordering (cause general)
        Measures daily and cumulative spread responses
        Returns values for horizons 0 to 20 in a table
"""


# generalized responses do not depend on contemporaneous variable ordering
# the shock is one residual standard deviation of crypto innovation
def generalized_irf_array(var_result, impulse, periods=20):
    variables = list(var_result.names)
    impulse_idx = variables.index(impulse)

    moving_average = np.asarray(
        var_result.ma_rep(maxn=periods)
    )
    residual_covariance = np.asarray(var_result.sigma_u)
    shock_sd = np.sqrt(residual_covariance[impulse_idx, impulse_idx])

    response_array = (
        moving_average
        @ residual_covariance[:, impulse_idx]
        / shock_sd
    )

    return response_array, shock_sd

# extracts period and cumulative generalized responses
# cumulative spread response is summed from horizons 1 to h
def extract_generalized_irf_table(
    var_result,
    impulses,
    responses,
    periods=20
):
    rows = []
    variables = list(var_result.names)

    for impulse in impulses:

        if impulse not in variables:
            continue

        response_array, shock_sd = generalized_irf_array(
            var_result=var_result,
            impulse=impulse,
            periods=periods
        )

        for response in responses:

            if response not in variables:
                continue

            response_idx = variables.index(response)
            period_response = response_array[:, response_idx]
            cumulative_response = np.concatenate([
                np.array([0.0]),
                np.cumsum(period_response[1:])
            ])

            for horizon in range(periods + 1):
                rows.append({
                    "horizon": horizon,
                    "impulse": impulse,
                    "response": response,
                    "generalized_irf": period_response[horizon],
                    "cumulative_response_h1_to_h": cumulative_response[horizon],
                    "shock_standard_deviation": shock_sd,
                    "shock_definition": "one_standard_deviation_reduced_form_innovation",
                    "cumulative_definition": "sum_of_spread_changes_from_horizon_1_to_h",
                    "lag_order": var_result.k_ar,
                    "nobs": var_result.nobs,
                })

    return pd.DataFrame(rows)


#%%------- HELPER FUNC: Stability and whiteness tests
"""
# DESC: Checks VAR stability and residual whiteness
        Tests residual autocorrelation at 5, 10 and 20 lags
        Uses adjusted Portmanteau tests
        Uses the 20 lag test as the main diagnostic
        Returns stability, test values, pass/fail results
"""
# VAR stability and multivariate residual autocorrelation diagnostics
def run_var_diagnostics(
    var_result,
    requested_whiteness_horizons=None
):
    if requested_whiteness_horizons is None:
        requested_whiteness_horizons = [5, 10, 20]

    is_stable = bool(var_result.is_stable(verbose=False))

    try:
        inverse_roots = np.abs(np.asarray(var_result.roots))
        minimum_inverse_root = float(np.min(inverse_roots))
        maximum_companion_root = float(1 / minimum_inverse_root)
    except Exception:
        minimum_inverse_root = np.nan
        maximum_companion_root = np.nan

    diagnostic_results = {
        "is_stable": is_stable,
        "stability_pass": is_stable,
        "minimum_inverse_root_modulus": minimum_inverse_root,
        "maximum_companion_root_modulus": maximum_companion_root,
        "whiteness_test": "adjusted_portmanteau",
    }

    # shorter horizons show whether rejection appears immediately
    # statsmodels requires each test horizon exceed fitted VAR lag
    for test_horizon in requested_whiteness_horizons:
        prefix = f"whiteness_{test_horizon}lag"

        if test_horizon <= var_result.k_ar:
            diagnostic_results.update({
                f"{prefix}_available": False,
                f"{prefix}_statistic": np.nan,
                f"{prefix}_p_value": np.nan,
                f"{prefix}_df": np.nan,
                f"{prefix}_pass_5pct": np.nan,
                f"{prefix}_error": (
                    "test horizon must exceed fitted VAR lag order"
                ),
            })
            continue

        try:
            whiteness = var_result.test_whiteness(
                nlags=test_horizon,
                signif=0.05,
                adjusted=True
            )

            diagnostic_results.update({
                f"{prefix}_available": True,
                f"{prefix}_statistic": whiteness.test_statistic,
                f"{prefix}_p_value": whiteness.pvalue,
                f"{prefix}_df": whiteness.df,
                f"{prefix}_pass_5pct": whiteness.pvalue >= 0.05,
                f"{prefix}_error": np.nan,
            })

        except Exception as e:
            diagnostic_results.update({
                f"{prefix}_available": False,
                f"{prefix}_statistic": np.nan,
                f"{prefix}_p_value": np.nan,
                f"{prefix}_df": np.nan,
                f"{prefix}_pass_5pct": np.nan,
                f"{prefix}_error": str(e),
            })

    # the 20 lag result = main residual diagnostic
    headline_prefix = "whiteness_20lag"
    headline_available = bool(
        diagnostic_results.get(
            f"{headline_prefix}_available",
            False
        )
    )
    headline_pass = bool(
        diagnostic_results.get(
            f"{headline_prefix}_pass_5pct",
            False
        )
    ) if headline_available else False

    diagnostic_results.update({
        "whiteness_test_lags": 20,
        "whiteness_statistic": diagnostic_results.get(
            f"{headline_prefix}_statistic",
            np.nan
        ),
        "whiteness_p_value": diagnostic_results.get(
            f"{headline_prefix}_p_value",
            np.nan
        ),
        "whiteness_df": diagnostic_results.get(
            f"{headline_prefix}_df",
            np.nan
        ),
        "residual_autocorrelation_pass_5pct": headline_pass,
        "residual_whiteness_warning": not headline_pass,
        "strict_diagnostics_pass": (
            is_stable
            and headline_pass
        ),
        # retained for compatibility with existing output files
        "diagnostics_pass": (
            is_stable
            and headline_pass
        ),
        "diagnostic_error": diagnostic_results.get(
            f"{headline_prefix}_error",
            np.nan
        ),
    })

    return diagnostic_results

#%%------- HELPER FUNC: Assembly Analysis
"""
DESC:   Estimates ONE complete VAR specification
        Selects the lag order using required criterion
        Runs stability, whiteness, Granger tests
        Calculates coefficients and impulse responses
        Returns the model and all result tables
"""

# estimates one VAR and creates outputs
def estimate_var_system(
    var_df,
    specification,
    analysis_group,
    sample_type,
    crypto_vars,
    outcome_vars,
    control_vars,
    lag_criterion
):
    if len(var_df) < minimum_observations:
        raise ValueError(
            f"Too few observations after dropna, n={len(var_df)}"
        )

    selected_lag, lag_table = select_var_lag(
        var_df=var_df,
        maxlags=maxlags,
        criterion=lag_criterion,
        trend=trend
    )

    print(
        f"\nRunning {specification}: "
        f"sample={sample_type}, n={len(var_df)}, "
        f"variables={var_df.shape[1]}, {lag_criterion} lag={selected_lag}"
    )

    model_data = var_df.reset_index(drop=True)
    
    # VAR
    var_result = VAR(model_data).fit(
        maxlags=selected_lag,
        trend=trend
    )

    # Stability and whiteness
    diagnostics = run_var_diagnostics(
        var_result=var_result,
        requested_whiteness_horizons=whiteness_test_horizons
    )

    # markers to distinguish
    metadata = {
        "specification": specification,
        "analysis_group": analysis_group,
        "sample_type": sample_type,
        "lag_criterion": lag_criterion,
    }

    # label lags
    lag_table = add_metadata(lag_table, metadata)
    
    # get coef --> the core VAR output
    coefficient_table = add_metadata(
        extract_var_coefficients(var_result),
        metadata
    )
    
    # Granger table
    granger_table = add_metadata(
        run_granger_tests(
            var_result=var_result,
            crypto_vars=crypto_vars,
            outcome_vars=outcome_vars
        ),
        metadata
    )
    
    # IRF table
    irf_table = add_metadata(
        extract_generalized_irf_table(
            var_result=var_result,
            impulses=crypto_vars,
            responses=outcome_vars,
            periods=irf_periods
        ),
        metadata
    )

    # create model summary
    model_info = {
        **metadata,
        "crypto_variable": (
            crypto_vars[0]
            if len(crypto_vars) == 1
            else np.nan
        ),
        "crypto_variables": ", ".join(crypto_vars),
        "outcome_variables": ", ".join(outcome_vars),
        "control_variables": ", ".join(control_vars),
        "variables": ", ".join(var_df.columns),
        "sample_start": var_df.index.min(),
        "sample_end": var_df.index.max(),
        "sample_rows_before_lags": len(var_df),
        "nobs": var_result.nobs,
        "number_of_variables": var_result.neqs,
        "lag_order": var_result.k_ar,
        "df_resid": var_result.df_resid,
        "aic": var_result.aic,
        "bic": var_result.bic,
        "hqic": var_result.hqic,
        "fpe": var_result.fpe,
        "log_likelihood": var_result.llf,
        **diagnostics,
    }

    print(
        f"Finished {specification}: "
        f"stable={diagnostics['is_stable']}, "
        f"whiteness p={diagnostics['whiteness_p_value']:.4f}"
    )

    return {
        "model": var_result,
        "sample": var_df,
        "model_info": pd.DataFrame([model_info]),
        "lag_selection": lag_table,
        "coefficients": coefficient_table,
        "granger": granger_table,
        "irf": irf_table,
    }


#%%------- HELPER FUNC: Predictive screening
"""
DESC:   Applies main predictive-screen requirements
        Corrects Granger p-values for multiple testing
        Uses Benjamini-Hochberg FDR method
        Checks response direction under BIC, AIC and HQIC
        Labels direction of cum spread response
        Returns FDR and directional-robustness results
"""

# apply Benjamini-Hochberg FDR within a pre-specified test family
# correct for multiple testing 
def apply_fdr_family(df, mask, family_label):
    valid_mask = mask & df["p_value"].notna()
    valid_index = df.index[valid_mask]

    if len(valid_index) == 0:
        return df

    reject, q_values, _, _ = multipletests(
        df.loc[valid_index, "p_value"].to_numpy(),
        alpha=0.10,
        method="fdr_bh"
    )

    df.loc[valid_index, "fdr_family"] = family_label
    df.loc[valid_index, "fdr_family_size"] = len(valid_index)
    df.loc[valid_index, "fdr_q_value"] = q_values
    df.loc[valid_index, "fdr_reject_10pct"] = reject

    return df

# response direction must agree under BIC and at least one alternative criterion
def direction_confirmed(bic_value, aic_value, hqic_value):
    if pd.isna(bic_value) or np.sign(bic_value) == 0:
        return False

    bic_sign = np.sign(bic_value)

    aic_match = (
        not pd.isna(aic_value)
        and np.sign(aic_value) == bic_sign
    )
    hqic_match = (
        not pd.isna(hqic_value)
        and np.sign(hqic_value) == bic_sign
    )

    return bool(aic_match or hqic_match)


# label the direction of the cumulative (spread) response
def response_direction_label(value):
    if pd.isna(value):
        return "missing"
    if value > 0:
        return "spread_widening"
    if value < 0:
        return "spread_compression"
    return "zero"


#%%------- HELPER FUNC: Block bootstrap
"""
# DESC: Creates moving-block bootstrap confidence bands
        Resamples consecutive blocks
        Generates and re-estimates artificial VAR datasets
        Recalculates generalized and cumulative impulse responses
        Returns 90% and 95% confidence bands for selected relationships
"""


# resample consecutive residual blocks to preserve remaining serial dependence
def moving_block_resample(
    residuals,
    target_length,
    block_length,
    rng
):
    if block_length < 1:
        raise ValueError("Bootstrap block length must be at least one")

    if block_length > len(residuals):
        raise ValueError(
            "Bootstrap block length cannot exceed residual sample length"
        )

    number_of_blocks = int(
        np.ceil(target_length / block_length)
    )
    maximum_start = len(residuals) - block_length

    block_starts = rng.integers(
        low=0,
        high=maximum_start + 1,
        size=number_of_blocks
    )

    sampled_blocks = [
        residuals[
            start:start + block_length
        ]
        for start in block_starts
    ]

    return np.vstack(sampled_blocks)[:target_length]

# moving-block residual bootstrap bands for selected generalized responses
def bootstrap_generalized_irf_bands(
    var_result,
    var_df,
    relationships,
    periods=20,
    reps=1000,
    seed=20260730,
    block_length=10
):
    relationships = list(relationships)
    variables = list(var_result.names)
    values = var_df.to_numpy(dtype=float)
    residuals = np.asarray(var_result.resid, dtype=float)
    residuals = residuals - residuals.mean(axis=0)
    coefficient_matrices = np.asarray(var_result.coefs, dtype=float)
    intercept = np.asarray(var_result.intercept, dtype=float)
    lag_order = var_result.k_ar
    rng = np.random.default_rng(seed)

    bootstrap_curves = {
        relationship: []
        for relationship in relationships
    }
    failed_draws = 0
    unstable_draws = 0

    for _ in range(reps):
        simulated = np.zeros_like(values)
        simulated[:lag_order] = values[:lag_order]
        
        # create block bootstrap sample
        boot_residuals = moving_block_resample(
            residuals=residuals,
            target_length=len(values) - lag_order,
            block_length=block_length,
            rng=rng
        )

        for t in range(lag_order, len(values)):
            fitted_value = intercept.copy()

            for lag in range(1, lag_order + 1):
                fitted_value += (
                    coefficient_matrices[lag - 1]
                    @ simulated[t - lag]
                )

            simulated[t] = (
                fitted_value
                + boot_residuals[t - lag_order]
            )

        try:
            boot_df = pd.DataFrame(
                simulated,
                columns=variables
            )
            boot_result = VAR(boot_df).fit(
                maxlags=lag_order,
                trend=trend
            )
            
            # skip unstable sims
            if not boot_result.is_stable(verbose=False):
                unstable_draws += 1
                continue

            for impulse, response in relationships:
                response_array, _ = generalized_irf_array(
                    var_result=boot_result,
                    impulse=impulse,
                    periods=periods
                )
                response_idx = variables.index(response)
                period_curve = response_array[:, response_idx]
                cumulative_curve = np.concatenate([
                    np.array([0.0]),
                    np.cumsum(period_curve[1:])
                ])
                bootstrap_curves[(impulse, response)].append(
                    np.column_stack([
                        period_curve,
                        cumulative_curve
                    ])
                )

        except Exception:
            failed_draws += 1

    rows = []

    for impulse, response in relationships:
        original_array, shock_sd = generalized_irf_array(
            var_result=var_result,
            impulse=impulse,
            periods=periods
        )
        response_idx = variables.index(response)
        original_period = original_array[:, response_idx]
        original_cumulative = np.concatenate([
            np.array([0.0]),
            np.cumsum(original_period[1:])
        ])

        curves = bootstrap_curves[(impulse, response)]

        if len(curves) > 0:
            curve_array = np.stack(curves)
            percentiles = np.percentile(
                curve_array,
                [2.5, 5.0, 95.0, 97.5],
                axis=0
            )
        else:
            percentiles = np.full(
                (4, periods + 1, 2),
                np.nan
            )

        for horizon in range(periods + 1):
            rows.append({
                "horizon": horizon,
                "impulse": impulse,
                "response": response,
                "generalized_irf": original_period[horizon],
                "generalized_irf_lower_95": percentiles[0, horizon, 0],
                "generalized_irf_lower_90": percentiles[1, horizon, 0],
                "generalized_irf_upper_90": percentiles[2, horizon, 0],
                "generalized_irf_upper_95": percentiles[3, horizon, 0],
                "cumulative_response_h1_to_h": original_cumulative[horizon],
                "cumulative_response_lower_95": percentiles[0, horizon, 1],
                "cumulative_response_lower_90": percentiles[1, horizon, 1],
                "cumulative_response_upper_90": percentiles[2, horizon, 1],
                "cumulative_response_upper_95": percentiles[3, horizon, 1],
                "shock_standard_deviation": shock_sd,
                "bootstrap_reps_requested": reps,
                "bootstrap_reps_successful": len(curves),
                "bootstrap_unstable_draws": unstable_draws,
                "bootstrap_failed_draws": failed_draws,
                "bootstrap_seed": seed,
                "bootstrap_method": "moving_block_residual",
                "bootstrap_block_length": block_length,
            })

    return pd.DataFrame(rows)


#%%------- IMPORT DATA FROM GITHUB

data_frame = read_github_csv(
    github_url,
    data_frame_file
)

# standardize date column and restrict to the paper sample
data_frame = standardize_date_column(data_frame)
data_frame = data_frame[
    data_frame["date"].between(sample_start, sample_end)
].copy()


#%%------- DEFINE VARIABLES
"""
DESC:   Defines and groups all variables used in the VAR models
        Separates credit outcomes, crypto predictors and baseline controls
        Defines additional controls used in the robustness specifications
        Confirms that every required variable exists in the dataset
"""

# main credit-spread outcomes
credit_vars = [
    "hy_spread_daily_chg",
    "ig_spread_daily_chg",
]

# stablecoin liquidity variables
stablecoin_predictors = [
    "stablecoin_supply_daily_log_chg",
    "stablecoin_exchange_netflow_scaled",
    "stablecoin_exchange_reserve_daily_log_chg",
]

# crypto derivatives and market activity
crypto_market_predictors = [
    "btc_eth_open_interest_daily_log_chg",
    "btc_eth_funding_rate_avg",
    "tradingVol_btc+eth_daily_log_chg",
]

all_crypto_predictors = (
    stablecoin_predictors
    + crypto_market_predictors
)

# general crypto-market controls
crypto_market_controls = [
    "btc_eth_daily_log_ret_avg",
    "btc_eth_realized_vol_7d",
]

# baseline macro-financial controls
macro_controls = [
    "term_spread_daily_chg",
    "usd_strength_daily_log_chg",
]

# controls included in every baseline VAR
baseline_controls = (
    crypto_market_controls
    + macro_controls
)

# publication-aligned weekly financial conditions robustness control
nfci_control = [
    "nfci_weekly_fill",
]

# S&P 500 and VIX enter the broader-risk robustness VARs only
broader_risk_vars = [
    "sp500_daily_log_ret",
    "vix_daily_log_chg",
]

# make missing required variables a visible error instead of changing the model
require_columns(
    data_frame,
    (
        credit_vars
        + all_crypto_predictors
        + baseline_controls
        + nfci_control
        + broader_risk_vars
    ),
    "VAR"
)


#%%------- CREATE COMMON SAMPLES
"""
DESC:   Creates common complete samples for each VAR group
        Ensures models within a group use identical observation dates
        Defines primary, broader-risk, restricted and NFCI samples
        Saves the common dates too ensure later models used common samples
"""


# main VARs use the same business-day sample across all six indicators
primary_common_variables = (
    all_crypto_predictors
    + credit_vars
    + baseline_controls
)
primary_common_df = create_var_sample(
    df=data_frame,
    variables=primary_common_variables
)
primary_common_dates = primary_common_df.index


# broader-risk systems also use one common sample where all variables exist
# ie +SP500 +VIX
broader_common_variables = (
    all_crypto_predictors
    + credit_vars
    + baseline_controls
    + broader_risk_vars
)
broader_common_df = create_var_sample(
    df=data_frame,
    variables=broader_common_variables
)
broader_common_dates = broader_common_df.index


# split-sample uses a common 2020-2023 sample
split_common_df = create_var_sample(
    df=data_frame,
    variables=primary_common_variables,
    start=sample_start,
    end=split_stage1_end
)
split_common_dates = split_common_df.index


# NFCI systems keep the same baseline variables and add the aligned weekly index
nfci_common_variables = (
    all_crypto_predictors
    + credit_vars
    + baseline_controls
    + nfci_control
)
nfci_common_df = create_var_sample(
    df=data_frame,
    variables=nfci_common_variables
)
nfci_common_dates = nfci_common_df.index


#%%------- RUN STATIONARITY CHECKS
"""
DESC:   Runs ADF stationarity tests for all VAR variables
        Tests both full available and primary common samples
        Combines all stationarity results into one output table
"""


stationarity_vars = unique_list(
    credit_vars
    + all_crypto_predictors
    + baseline_controls
    + nfci_control
    + broader_risk_vars
)


adf_full_available = run_adf_tests(
    df=data_frame,
    variables=stationarity_vars,
    sample_label="full_available_sample"
)

adf_primary_common = run_adf_tests(
    df=primary_common_df.reset_index(),
    variables=primary_common_variables,
    sample_label="primary_common_business_day_sample"
)

adf_results = pd.concat(
    [
        adf_full_available,
        adf_primary_common
    ],
    ignore_index=True
)


#%%------- RUN VAR ANALYSIS
"""
DESC:   Estimates all primary and robustness VAR specifications
        Runs BIC, AIC, HQIC, expanded-control and alternative-sample models
        Estimates individual, family and full-system VARs
        Stores fitted models, samples and all result tables
        Records failed specifications without stopping the analysis
"""


all_model_info = []
all_lag_selection = []
all_var_coefficients = []
all_granger_results = []
all_irf_results = []
all_errors = []

all_var_models = {}
all_var_samples = {}

# estimates a requested specification and keeps errors in a CSV
def run_and_store(
    var_df,
    specification,
    analysis_group,
    sample_type,
    crypto_vars,
    outcome_vars,
    control_vars,
    lag_criterion
):
    try:
        output = estimate_var_system(
            var_df=var_df,
            specification=specification,
            analysis_group=analysis_group,
            sample_type=sample_type,
            crypto_vars=crypto_vars,
            outcome_vars=outcome_vars,
            control_vars=control_vars,
            lag_criterion=lag_criterion
        )

        all_var_models[specification] = output["model"]
        all_var_samples[specification] = output["sample"]
        all_model_info.append(output["model_info"])
        all_lag_selection.append(output["lag_selection"])
        all_var_coefficients.append(output["coefficients"])
        all_granger_results.append(output["granger"])
        all_irf_results.append(output["irf"])

    except Exception as e:
        print(f"Skipped {specification}: {e}")
        all_errors.append({
            "specification": specification,
            "analysis_group": analysis_group,
            "sample_type": sample_type,
            "lag_criterion": lag_criterion,
            "error": str(e),
        })


# main BIC and alternative AIC/HQIC models
for crypto_var in all_crypto_predictors:
    baseline_variables = (
        [crypto_var]
        + credit_vars
        + baseline_controls
    )
    baseline_common_sample = create_var_sample(
        df=data_frame,
        variables=baseline_variables,
        common_dates=primary_common_dates
    )

    run_and_store(
        var_df=baseline_common_sample,
        specification=f"primary_bic_{crypto_var}",
        analysis_group="primary",
        sample_type="common_business_day",
        crypto_vars=[crypto_var],
        outcome_vars=credit_vars,
        control_vars=baseline_controls,
        lag_criterion="bic"
    )

    for alternative_criterion in ["aic", "hqic"]:
        run_and_store(
            var_df=baseline_common_sample,
            specification=(
                f"lag_robustness_{alternative_criterion}_{crypto_var}"
            ),
            analysis_group="lag_robustness",
            sample_type="common_business_day",
            crypto_vars=[crypto_var],
            outcome_vars=credit_vars,
            control_vars=baseline_controls,
            lag_criterion=alternative_criterion
        )


# predictor-specific maximum samples are coverage robustness check
for crypto_var in all_crypto_predictors:
    coverage_variables = (
        [crypto_var]
        + credit_vars
        + baseline_controls
    )
    coverage_sample = create_var_sample(
        df=data_frame,
        variables=coverage_variables
    )

    run_and_store(
        var_df=coverage_sample,
        specification=f"coverage_max_bic_{crypto_var}",
        analysis_group="coverage_robustness",
        sample_type="predictor_specific_maximum",
        crypto_vars=[crypto_var],
        outcome_vars=credit_vars,
        control_vars=baseline_controls,
        lag_criterion="bic"
    )


# broader-risk systems add S&P 500 + VIX
for crypto_var in all_crypto_predictors:
    broader_variables = (
        [crypto_var]
        + credit_vars
        + baseline_controls
        + broader_risk_vars
    )
    broader_sample = create_var_sample(
        df=data_frame,
        variables=broader_variables,
        common_dates=broader_common_dates
    )

    run_and_store(
        var_df=broader_sample,
        specification=f"broader_risk_bic_{crypto_var}",
        analysis_group="broader_risk_robustness",
        sample_type="broader_risk_common_business_day",
        crypto_vars=[crypto_var],
        outcome_vars=credit_vars,
        control_vars=baseline_controls + broader_risk_vars,
        lag_criterion="bic"
    )

# publication-aligned NFCI is additional robustness system
for crypto_var in all_crypto_predictors:
    nfci_variables = (
        [crypto_var]
        + credit_vars
        + baseline_controls
        + nfci_control
    )
    nfci_sample = create_var_sample(
        df=data_frame,
        variables=nfci_variables,
        common_dates=nfci_common_dates
    )

    run_and_store(
        var_df=nfci_sample,
        specification=f"nfci_bic_{crypto_var}",
        analysis_group="nfci_robustness",
        sample_type="nfci_common_business_day",
        crypto_vars=[crypto_var],
        outcome_vars=credit_vars,
        control_vars=baseline_controls + nfci_control,
        lag_criterion="bic"
    )

# stablecoin variables jointly
stablecoin_family_variables = (
    stablecoin_predictors
    + credit_vars
    + baseline_controls
)
stablecoin_family_sample = create_var_sample(
    df=data_frame,
    variables=stablecoin_family_variables,
    common_dates=primary_common_dates
)
run_and_store(
    var_df=stablecoin_family_sample,
    specification="family_bic_stablecoin_liquidity",
    analysis_group="family_robustness",
    sample_type="common_business_day",
    crypto_vars=stablecoin_predictors,
    outcome_vars=credit_vars,
    control_vars=baseline_controls,
    lag_criterion="bic"
)

# open interest and funding rates jointly
derivatives_predictors = [
    "btc_eth_open_interest_daily_log_chg",
    "btc_eth_funding_rate_avg",
]
derivatives_family_variables = (
    derivatives_predictors
    + credit_vars
    + baseline_controls
)
derivatives_family_sample = create_var_sample(
    df=data_frame,
    variables=derivatives_family_variables,
    common_dates=primary_common_dates
)
run_and_store(
    var_df=derivatives_family_sample,
    specification="family_bic_crypto_derivatives",
    analysis_group="family_robustness",
    sample_type="common_business_day",
    crypto_vars=derivatives_predictors,
    outcome_vars=credit_vars,
    control_vars=baseline_controls,
    lag_criterion="bic"
)

# full all-crypto VAR is appendix robustness only
full_var_variables = (
    all_crypto_predictors
    + credit_vars
    + baseline_controls
)
full_var_sample = create_var_sample(
    df=data_frame,
    variables=full_var_variables,
    common_dates=primary_common_dates
)
run_and_store(
    var_df=full_var_sample,
    specification="full_var_bic_all_crypto",
    analysis_group="full_var_appendix",
    sample_type="common_business_day",
    crypto_vars=all_crypto_predictors,
    outcome_vars=credit_vars,
    control_vars=baseline_controls,
    lag_criterion="bic"
)

# split-sample screen on 2020-2023
for crypto_var in all_crypto_predictors:
    split_variables = (
        [crypto_var]
        + credit_vars
        + baseline_controls
    )
    split_sample = create_var_sample(
        df=data_frame,
        variables=split_variables,
        common_dates=split_common_dates,
        start=sample_start,
        end=split_stage1_end
    )

    run_and_store(
        var_df=split_sample,
        specification=f"split_2020_2023_bic_{crypto_var}",
        analysis_group="split_sample_screen",
        sample_type="common_business_day_2020_2023",
        crypto_vars=[crypto_var],
        outcome_vars=credit_vars,
        control_vars=baseline_controls,
        lag_criterion="bic"
    )


#%%------- COMBINE VAR RESULTS
"""
DESC:   Combines results from all estimated VAR specifications
        Creates complete model, lag, coefficient, Granger and IRF tables
        Returns empty tables when no results are available
        Converts recorded model errors into a separate table
"""

def combine_or_empty(tables):
    if len(tables) > 0:
        return pd.concat(tables, ignore_index=True)
    return pd.DataFrame()

var_info = combine_or_empty(all_model_info)
lag_selection_results = combine_or_empty(all_lag_selection)
var_coefficients = combine_or_empty(all_var_coefficients)
granger_results = combine_or_empty(all_granger_results)
irf_results = combine_or_empty(all_irf_results)
error_results = pd.DataFrame(all_errors)


#%%------- APPLY MULTIPLE-TESTING ADJUSTMENTS
"""
DESC:   Applies FDR correction to the main Granger-test families
        Adjusts forward and reverse tests separately
        Adds family labels, adjusted q-values and 10% rejection indicators
"""

if not granger_results.empty:
    granger_results["fdr_family"] = pd.Series(
        pd.NA,
        index=granger_results.index,
        dtype="string"
    )
    granger_results["fdr_family_size"] = np.nan
    granger_results["fdr_q_value"] = np.nan
    granger_results["fdr_reject_10pct"] = False

    # twelve primary forward tests: six indicators x two credit spreads
    primary_forward_mask = (
        (granger_results["analysis_group"] == "primary")
        & (granger_results["lag_criterion"] == "bic")
        & (granger_results["direction"] == "crypto_to_risk")
        & (granger_results["test_scope"] == "individual")
        & (granger_results["caused"].isin(credit_vars))
    )
    granger_results = apply_fdr_family(
        df=granger_results,
        mask=primary_forward_mask,
        family_label="primary_forward_12_tests"
    )

    # reverse tests are adjusted as a separate family
    primary_reverse_mask = (
        (granger_results["analysis_group"] == "primary")
        & (granger_results["lag_criterion"] == "bic")
        & (granger_results["direction"] == "risk_to_crypto")
        & (granger_results["test_scope"] == "individual")
        & (granger_results["causing"].isin(credit_vars))
    )
    granger_results = apply_fdr_family(
        df=granger_results,
        mask=primary_reverse_mask,
        family_label="primary_reverse_12_tests"
    )

    # same pre-specified adjustment within the 2020-2023 screen
    split_forward_mask = (
        (granger_results["analysis_group"] == "split_sample_screen")
        & (granger_results["direction"] == "crypto_to_risk")
        & (granger_results["test_scope"] == "individual")
        & (granger_results["caused"].isin(credit_vars))
    )
    granger_results = apply_fdr_family(
        df=granger_results,
        mask=split_forward_mask,
        family_label="split_2020_2023_forward_12_tests"
    )

    split_reverse_mask = (
        (granger_results["analysis_group"] == "split_sample_screen")
        & (granger_results["direction"] == "risk_to_crypto")
        & (granger_results["test_scope"] == "individual")
        & (granger_results["causing"].isin(credit_vars))
    )
    granger_results = apply_fdr_family(
        df=granger_results,
        mask=split_reverse_mask,
        family_label="split_2020_2023_reverse_12_tests"
    )


#%%------- APPLY THE STAGE 1 SCREENING RULE
"""
DESC:   Applies the Stage 1 predictive screening rules
        Combines forward, reverse, FDR and model-diagnostic results
        Checks response direction under BIC, AIC and HQIC
        Residual whiteness as a warning, not an exclusion criterion
        Adds the 2020-2023 screening results
        Selects only relationships passing the screen for Stage 2
"""


if not granger_results.empty:
    
    # primary forward/reverse tests
    primary_forward = granger_results[
        primary_forward_mask
    ][
        [
            "specification",
            "causing",
            "caused",
            "statistic",
            "p_value",
            "fdr_q_value",
            "fdr_reject_10pct",
            "lag_order",
            "nobs",
            "error",
        ]
    ].copy()
    primary_forward = primary_forward.rename(columns={
        "specification": "primary_specification",
        "causing": "crypto_variable",
        "caused": "credit_variable",
        "statistic": "forward_granger_statistic",
        "p_value": "forward_granger_p_value",
        "fdr_q_value": "forward_granger_q_value",
        "fdr_reject_10pct": "forward_fdr_reject_10pct",
        "lag_order": "bic_lag_order",
        "nobs": "primary_nobs",
        "error": "forward_granger_error",
    })

    primary_reverse = granger_results[
        primary_reverse_mask
    ][
        [
            "causing",
            "caused",
            "statistic",
            "p_value",
            "fdr_q_value",
            "fdr_reject_10pct",
            "error",
        ]
    ].copy()
    primary_reverse = primary_reverse.rename(columns={
        "causing": "credit_variable",
        "caused": "crypto_variable",
        "statistic": "reverse_granger_statistic",
        "p_value": "reverse_granger_p_value",
        "fdr_q_value": "reverse_granger_q_value",
        "fdr_reject_10pct": "reverse_fdr_reject_10pct",
        "error": "reverse_granger_error",
    })

    screening_results = primary_forward.merge(
        primary_reverse,
        on=["crypto_variable", "credit_variable"],
        how="left"
    )


    # collect 20 day generalized response for each lag crit
    direction_irfs = irf_results[
        (
            irf_results["analysis_group"].isin([
                "primary",
                "lag_robustness"
            ])
        )
        & (irf_results["horizon"] == irf_periods)
        & (irf_results["impulse"].isin(all_crypto_predictors))
        & (irf_results["response"].isin(credit_vars))
    ][
        [
            "impulse",
            "response",
            "lag_criterion",
            "cumulative_response_h1_to_h",
        ]
    ].copy()

    direction_pivot = direction_irfs.pivot_table(
        index=["impulse", "response"],
        columns="lag_criterion",
        values="cumulative_response_h1_to_h",
        aggfunc="first"
    ).reset_index()
    direction_pivot = direction_pivot.rename(columns={
        "impulse": "crypto_variable",
        "response": "credit_variable",
        "bic": "bic_cumulative_response_20d",
        "aic": "aic_cumulative_response_20d",
        "hqic": "hqic_cumulative_response_20d",
    })

    screening_results = screening_results.merge(
        direction_pivot,
        on=["crypto_variable", "credit_variable"],
        how="left"
    )


    # model diagnostics and sample details for the primary BIC model
    primary_diagnostics = var_info[
        var_info["analysis_group"] == "primary"
    ][
        [
            "crypto_variable",
            "sample_start",
            "sample_end",
            "is_stable",
            "whiteness_p_value",
            "residual_autocorrelation_pass_5pct",
            "residual_whiteness_warning",
            "diagnostics_pass",
        ]
    ].copy()

    screening_results = screening_results.merge(
        primary_diagnostics,
        on="crypto_variable",
        how="left"
    )


    # alternative lag orders used for direction check
    lag_orders = var_info[
        var_info["analysis_group"].isin([
            "primary",
            "lag_robustness"
        ])
    ][
        [
            "crypto_variable",
            "lag_criterion",
            "lag_order",
        ]
    ].copy()
    lag_order_pivot = lag_orders.pivot_table(
        index="crypto_variable",
        columns="lag_criterion",
        values="lag_order",
        aggfunc="first"
    ).reset_index()
    lag_order_pivot = lag_order_pivot.rename(columns={
        "bic": "selected_lag_bic",
        "aic": "selected_lag_aic",
        "hqic": "selected_lag_hqic",
    })


    # check lag order based direction fits
    screening_results = screening_results.merge(
        lag_order_pivot,
        on="crypto_variable",
        how="left"
    )

    screening_results["direction_confirmed_by_alternative_lag"] = (
        screening_results.apply(
            lambda row: direction_confirmed(
                row.get("bic_cumulative_response_20d", np.nan),
                row.get("aic_cumulative_response_20d", np.nan),
                row.get("hqic_cumulative_response_20d", np.nan)
            ),
            axis=1
        )
    )
    screening_results["bic_response_direction"] = (
        screening_results["bic_cumulative_response_20d"]
        .apply(response_direction_label)
    )

    # main predictive screen
    screening_results["passes_predictive_screen"] = (
        screening_results["forward_fdr_reject_10pct"].fillna(False)
        & screening_results[
            "direction_confirmed_by_alternative_lag"
        ].fillna(False)
        & screening_results["is_stable"].fillna(False)
    )

    # stricter result (include whiteness pass/fail)
    screening_results["passes_strict_screen"] = (
        screening_results["passes_predictive_screen"]
        & screening_results[
            "residual_autocorrelation_pass_5pct"
        ].fillna(False)
    )
    screening_results[
        "selected_with_residual_whiteness_warning"
        ] = (
        screening_results["passes_predictive_screen"]
        & screening_results[
            "residual_whiteness_warning"
    ].fillna(True)
    )


    # relic cause too lazy to change code...
    screening_results["passes_primary_screen"] = (
        screening_results["passes_predictive_screen"]
    )

    # mark un-FDR-adjusted p-vals
    screening_results["raw_forward_significant_10pct"] = (
        screening_results["forward_granger_p_value"] < 0.10
    )
    screening_results["suggestive_raw_only"] = (
        screening_results["raw_forward_significant_10pct"]
        & ~screening_results["forward_fdr_reject_10pct"].fillna(False)
    )

    # reverse test flags
    screening_results["reverse_relationship_label"] = np.where(
        screening_results["reverse_fdr_reject_10pct"].fillna(False),
        "reverse_evidence_fdr_10pct",
        np.where(
            screening_results["reverse_granger_p_value"] < 0.10,
            "reverse_evidence_raw_10pct_only",
            "no_reverse_evidence_at_10pct"
        )
    )

    # overall classification
    screening_results["screening_classification"] = np.select(
    [
        screening_results["passes_strict_screen"],
        screening_results[
            "selected_with_residual_whiteness_warning"
        ],
        screening_results["suggestive_raw_only"],
    ],
        [
            "selected_without_diagnostic_warning",
            "selected_with_residual_whiteness_warning",
            "suggestive_raw_only",
        ],
        default="not_selected"
    )

    # Stage 2 uses the predictive screen
    screening_results["stage2_included"] = (
        screening_results["passes_predictive_screen"]
    )
    screening_results["stage2_inclusion_reason"] = np.select(
    [
        screening_results["passes_strict_screen"],
        screening_results[
            "selected_with_residual_whiteness_warning"
        ],
    ],
        [
            "passes_predictive_and_strict_screen",
            "passes_predictive_screen_with_residual_warning",
        ],
        default="not_selected"
    )
    
    
    # decided not to use fallback cause why test if I know nothing there...
    """
    if not screening_results["passes_predictive_screen"].any():
        fallback_mask = (
            screening_results["crypto_variable"]
            == "stablecoin_supply_daily_log_chg"
        )
        screening_results.loc[
            fallback_mask,
            "stage2_included"
        ] = True
        screening_results.loc[
            fallback_mask,
            "stage2_inclusion_reason"
        ] = "benchmark_fallback_no_predictive_pair_passed"
    """
    
    
    # split-sample results 
    split_forward = granger_results[
        split_forward_mask
    ][
        [
            "causing",
            "caused",
            "p_value",
            "fdr_q_value",
            "fdr_reject_10pct",
            "lag_order",
            "nobs",
        ]
    ].copy()
    split_forward = split_forward.rename(columns={
        "causing": "crypto_variable",
        "caused": "credit_variable",
        "p_value": "split_2020_2023_forward_p_value",
        "fdr_q_value": "split_2020_2023_forward_q_value",
        "fdr_reject_10pct": "split_2020_2023_forward_fdr_reject_10pct",
        "lag_order": "split_2020_2023_lag_order",
        "nobs": "split_2020_2023_nobs",
    })

    screening_results = screening_results.merge(
        split_forward,
        on=["crypto_variable", "credit_variable"],
        how="left"
    )



    screening_results = screening_results.sort_values(
        ["crypto_variable", "credit_variable"]
    ).reset_index(drop=True)

# error handling
else:
    screening_results = pd.DataFrame()

# show what goes to stage 2 (nothing...)
stage2_selected_relationships = (
    screening_results[
        screening_results["stage2_included"] == True
    ].copy()
    if not screening_results.empty
    else pd.DataFrame()
)


#%%------- BOOTSTRAP SELECTED GENERALIZED IRFS
"""
DESC:   Bootstraps IRFs only for relationships passing Stage 1
        Creates 90% and 95% confidence bands for selected responses
        Skips the bootstrap when no relationship passes the screen
        Returns a structured empty table when no bands are estimated
        A gigantique waste of time and a lesson in checking wether your data is correct.......
"""


selected_irf_bands = []

if not screening_results.empty:
    selected_for_bands = screening_results[
        screening_results["passes_predictive_screen"] == True
    ].copy()

    selected_crypto_vars = (
        selected_for_bands["crypto_variable"]
        .drop_duplicates()
        .tolist()
    )

    for model_number, crypto_var in enumerate(selected_crypto_vars):
        specification = f"primary_bic_{crypto_var}"

        if specification not in all_var_models:
            continue

        model_relationships = (
            selected_for_bands[
                selected_for_bands["crypto_variable"] == crypto_var
            ][["crypto_variable", "credit_variable"]]
            .itertuples(index=False, name=None)
        )
        model_relationships = list(model_relationships)

        print(
            f"\nBootstrapping generalized IRFs for {crypto_var}: "
            f"{bootstrap_reps} residual draws"
        )
        
        # bootstrap starts here --> slightly different seed per iter
        try:
            band_table = bootstrap_generalized_irf_bands(
                var_result=all_var_models[specification],
                var_df=all_var_samples[specification],
                relationships=model_relationships,
                periods=irf_periods,
                reps=bootstrap_reps,
                seed=bootstrap_seed + model_number,
                block_length=bootstrap_block_length
            )
            band_table.insert(0, "specification", specification)
            selected_irf_bands.append(band_table)

        except Exception as e:
            print(f"Bootstrap failed for {specification}: {e}")
            all_errors.append({
                "specification": specification,
                "analysis_group": "selected_irf_bootstrap",
                "sample_type": "common_business_day",
                "lag_criterion": "bic",
                "error": str(e),
            })

selected_irf_confidence_bands = combine_or_empty(
    selected_irf_bands
)

selected_irf_band_columns = [
    "specification",
    "horizon",
    "impulse",
    "response",
    "generalized_irf",
    "generalized_irf_lower_95",
    "generalized_irf_lower_90",
    "generalized_irf_upper_90",
    "generalized_irf_upper_95",
    "cumulative_response_h1_to_h",
    "cumulative_response_lower_95",
    "cumulative_response_lower_90",
    "cumulative_response_upper_90",
    "cumulative_response_upper_95",
    "shock_standard_deviation",
    "bootstrap_reps_requested",
    "bootstrap_reps_successful",
    "bootstrap_unstable_draws",
    "bootstrap_failed_draws",
    "bootstrap_seed",
    "bootstrap_method",
    "bootstrap_block_length",
]

if selected_irf_confidence_bands.empty:
    selected_irf_confidence_bands = pd.DataFrame(
        columns=selected_irf_band_columns
    )

error_columns = [
    "specification",
    "analysis_group",
    "sample_type",
    "lag_criterion",
    "error",
]
error_results = pd.DataFrame(
    all_errors,
    columns=error_columns
)


#%%------- CREATE COMPACT SUMMARY TABLES
"""
DESC:   Creates compact tables for reporting the main VAR results
        Separates forward and reverse primary Granger tests
        Collects model, stability and whiteness diagnostics
        Returns empty tables safely when results are unavailable
"""


if not granger_results.empty:
    summary_granger_crypto_to_risk = (
        granger_results[
            primary_forward_mask
        ]
        .sort_values(["causing", "caused"])
        .copy()
    )

    summary_granger_risk_to_crypto = (
        granger_results[
            primary_reverse_mask
        ]
        .sort_values(["caused", "causing"])
        .copy()
    )
else:
    summary_granger_crypto_to_risk = pd.DataFrame()
    summary_granger_risk_to_crypto = pd.DataFrame()

# one row per estimated model with the diagnostic fields used in screening
diagnostic_columns = [
    "specification",
    "analysis_group",
    "sample_type",
    "lag_criterion",
    "crypto_variables",
    "outcome_variables",
    "sample_start",
    "sample_end",
    "sample_rows_before_lags",
    "nobs",
    "number_of_variables",
    "lag_order",
    "df_resid",
    "is_stable",
    "stability_pass",
    "minimum_inverse_root_modulus",
    "maximum_companion_root_modulus",
    "whiteness_test",
    "whiteness_5lag_available",
    "whiteness_5lag_statistic",
    "whiteness_5lag_p_value",
    "whiteness_5lag_df",
    "whiteness_5lag_pass_5pct",
    "whiteness_5lag_error",
    "whiteness_10lag_available",
    "whiteness_10lag_statistic",
    "whiteness_10lag_p_value",
    "whiteness_10lag_df",
    "whiteness_10lag_pass_5pct",
    "whiteness_10lag_error",
    "whiteness_20lag_available",
    "whiteness_20lag_statistic",
    "whiteness_20lag_p_value",
    "whiteness_20lag_df",
    "whiteness_20lag_pass_5pct",
    "whiteness_20lag_error",
    "whiteness_test_lags",
    "whiteness_statistic",
    "whiteness_p_value",
    "whiteness_df",
    "residual_autocorrelation_pass_5pct",
    "residual_whiteness_warning",
    "strict_diagnostics_pass",
    "diagnostics_pass",
    "diagnostic_error",
]
var_diagnostics = (
    var_info[existing_cols(var_info, diagnostic_columns)].copy()
    if not var_info.empty
    else pd.DataFrame(columns=diagnostic_columns)
)


#%%------- SAVE CSV RESULTS

output_tables = {
    "var_stationarity_adf_tests.csv": adf_results,
    "var_model_info.csv": var_info,
    "var_lag_selection.csv": lag_selection_results,
    "var_diagnostics.csv": var_diagnostics,
    "var_coefficients.csv": var_coefficients,
    "var_granger_causality_results.csv": granger_results,
    "var_irf_values_crypto_to_risk.csv": irf_results,
    "var_primary_screening_results.csv": screening_results,
    "var_stage2_selected_relationships.csv": stage2_selected_relationships,
    "var_selected_irf_confidence_bands.csv": selected_irf_confidence_bands,
    "summary_granger_crypto_to_risk.csv": summary_granger_crypto_to_risk,
    "summary_granger_risk_to_crypto.csv": summary_granger_risk_to_crypto,
    "var_errors.csv": error_results,
}

for filename, table in output_tables.items():
    table.to_csv(
        output_dir / filename,
        index=False
    )

print("\nSaved CSV outputs:")
for filename in output_tables:
    print(output_dir / filename)

print(
    f"\nRelationships passing the predictive screen: "
    f"{int(screening_results['passes_predictive_screen'].sum()) if not screening_results.empty else 0}"
)
print(
    f"Selected relationships with a residual-whiteness warning: "
    f"{int(screening_results['selected_with_residual_whiteness_warning'].sum()) if not screening_results.empty else 0}"
)
print(
    f"Relationships also passing the strict diagnostic screen: "
    f"{int(screening_results['passes_strict_screen'].sum()) if not screening_results.empty else 0}"
)
print(
    f"Relationships included in Stage 2: "
    f"{len(stage2_selected_relationships)}"
)
