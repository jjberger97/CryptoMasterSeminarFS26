#%%------- IMPORT PACKAGES

import pandas as pd
import numpy as np
from urllib.parse import quote
from pathlib import Path
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller


#%%------- SETTINGS

# github path
github_url = "https://raw.githubusercontent.com/jjberger97/CryptoMasterSeminarFS26/main/Coding/code"
data_frame_file = "data_table.csv"

# create output folder for VAR results
output_dir = Path("var_results")
output_dir.mkdir(exist_ok=True)

# sample start
sample_start = pd.to_datetime("2020-01-01")

# maximum lag length considered in lag selection
maxlags = 10

# lag-selection rule used for the VAR
# alternatives: "aic", "bic", "hqic", "fpe"
# CAUSING 6 LAGS CURRENTLY - bic strong criterion
lag_selection_criterion = "bic"

# deterministic terms in VAR
# "c" = constant only
# i.e. forces intercept and allows for non-zero averages (not trends!)
trend = "c"


#%%------- HELPER FUNCTIONS

# creates github url
def raw_github_path(base_url, filename):
    return f"{base_url}/{quote(filename)}"

# makes dates under date col to dates and sort by date asc
def standardize_date_column(df):
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df

# get csv from github
def read_github_csv(base_url, filename):
    url = raw_github_path(base_url, filename)
    return pd.read_csv(url)

# keep only cols listed if they exist = crash protection
def existing_cols(df, cols):
    return [c for c in cols if c in df.columns]

# safe lag selection
def select_var_lag(var_df, maxlags=10, criterion="bic"):
    """
    Selects VAR lag length using the selected information criterion.
    Falls back to 1 lag if selection fails or returns invalid value.
    """

    # make sure maxlags is feasible relative to sample size and number of variables
    nobs = len(var_df)
    nvars = var_df.shape[1]
    
    # use at most the user-defined maxlags, but reduce it if the sample is too small
    feasible_maxlags = min(maxlags, max(1, int((nobs - 5) / (nvars + 1))))

    # if something impossible happens, use one lag...
    if feasible_maxlags < 1:
        return 1

    try:
        # create VAR from the selected dataframe
        model = VAR(var_df)
        
        # compare candidate lag lengths up to feasible_maxlags
        selected = model.select_order(maxlags=feasible_maxlags)
        
        # check what chooses what lag
        print(selected.summary())
        print("Selected by AIC:", selected.aic)
        print("Selected by BIC:", selected.bic)
        print("Selected by HQIC:", selected.hqic)
        print("Selected by FPE:", selected.fpe)

        # extract the lag length chosen by the selected criterion, e.g. BIC
        selected_lag = getattr(selected, criterion)

        # if the selected lag is invalid fall back to one
        if selected_lag is None or selected_lag < 1 or pd.isna(selected_lag):
            selected_lag = 1

        return int(selected_lag)

    except Exception as e:
        print(f"Lag selection failed: {e}")
        return 1

# ADF stationarity helper
# sanity check as most already stationary
# if non-stationary could mistake common upward trend as connection even though both naturally "go up"
def run_adf_tests(df, variables):
    """
    CHATGPT:
    Runs ADF tests for the variables used in a VAR sample.
    Since most variables are already returns/log changes
    is mainly sanity check
    """

    rows = []

    for var in variables:
        series = df[var].dropna() # drop all NA cause cant handle NAs

        try:
            adf_result = adfuller(series, autolag="AIC") # use AIC cause ADF sensitive to too few lags

            rows.append({
                "variable": var,
                "adf_stat": adf_result[0],
                "p_value": adf_result[1],
                "nobs": adf_result[3],
                "stationary_at_5pct": adf_result[1] < 0.05
            })

        except Exception as e:
            rows.append({
                "variable": var,
                "adf_stat": np.nan,
                "p_value": np.nan,
                "nobs": len(series),
                "stationary_at_5pct": False,
                "error": str(e)
            })

    return pd.DataFrame(rows)

# turn VAR coefficients into dataframe
def extract_var_coefficients(var_result, specification):
    """
    Extracts all VAR coefficients into a clean dataframe.
    """

    rows = []

    for equation in var_result.params.columns:
        for param in var_result.params.index:

            rows.append({
                "specification": specification, # which crpyto market var am i looking at or full model
                "equation": equation, # which dep coef belongs to
                "parameter": param, # which indep coef belongs to
                "coef": var_result.params.loc[param, equation],
                "t_stat": var_result.tvalues.loc[param, equation],
                "p_value": var_result.pvalues.loc[param, equation],
                "nobs": var_result.nobs,
                "lag_order": var_result.k_ar,
                "aic": var_result.aic,
                "bic": var_result.bic,
                "hqic": var_result.hqic,
            })

    return pd.DataFrame(rows)

# Granger causality tests
# Test wether past val of var A improves pred of B in var if all other lagged var ar already in
# Mathematically: are all laggs of predictor in question jointly zero
def run_granger_tests(var_result, specification, causing_vars, caused_vars):
    """
    CHATGPT
    Tests whether causing_vars Granger-cause each variable in caused_vars.
    """

    rows = []

    for caused in caused_vars: # the risk vars, "the dependents"
        for causing in causing_vars: # the cause, "the independent"

            if caused == causing: # dont need to know if var causes itself
                continue

            try:
                test = var_result.test_causality(
                    caused=caused, # which equation
                    causing=[causing], # which lags
                    kind="f" # perform f-test
                )

                rows.append({
                    "specification": specification,
                    "caused": caused,
                    "causing": causing,
                    "test": "granger_causality_f_test",
                    "statistic": test.test_statistic,
                    "p_value": test.pvalue,
                    "df": str(test.df),
                    "significant_10pct": test.pvalue < 0.10,
                    "significant_5pct": test.pvalue < 0.05,
                    "significant_1pct": test.pvalue < 0.01,
                    "lag_order": var_result.k_ar,
                    "nobs": var_result.nobs,
                })

            except Exception as e:
                rows.append({
                    "specification": specification,
                    "caused": caused,
                    "causing": causing,
                    "test": "granger_causality_f_test",
                    "statistic": np.nan,
                    "p_value": np.nan,
                    "df": np.nan,
                    "error": str(e),
                    "lag_order": var_result.k_ar,
                    "nobs": var_result.nobs,
                })

    return pd.DataFrame(rows)

# IRF extraction without plots
# tests what the response to a shock is: What happens to one var over time after shock to other
# important! one unit shocks! i.e. one unit movement here causes coef movement in other
# cant really interpret size w/o accounting for units, only direction
def extract_irf_table(var_result, specification, impulses, responses, periods=20):
    """
    CHATGPT
    Extracts impulse response values into a dataframe.
    No graphs are produced.
    """

    irf = var_result.irf(periods)

    rows = []

    variables = var_result.names

    for impulse in impulses:
        for response in responses:

            if impulse not in variables or response not in variables:
                continue

            impulse_idx = variables.index(impulse)
            response_idx = variables.index(response)

            for h in range(periods + 1):
                rows.append({
                    "specification": specification,
                    "horizon": h,
                    "impulse": impulse,
                    "response": response,
                    "irf": irf.irfs[h, response_idx, impulse_idx],
                    "lag_order": var_result.k_ar,
                    "nobs": var_result.nobs,
                })

    return pd.DataFrame(rows)


#%%------- IMPORT DATA FROM GITHUB

data_frame = read_github_csv(github_url, data_frame_file)

# standardize date column
data_frame = standardize_date_column(data_frame)

# restrict sample
data_frame = data_frame[data_frame["date"] >= sample_start].copy()


#%%------- DEFINE VARIABLES

# risk appetite / macro-financial variables
risk_vars = existing_cols(data_frame, [
    "sp500_daily_log_ret",
    "vix_daily_log_chg",
    "hy_spread_daily_chg",
    "ig_spread_daily_chg",
])

# stablecoin liquidity variables
# USING AGGREGATE SUPPLY CURRENTLY --------------------------------------------------!!!
stablecoin_predictors = existing_cols(data_frame, [
    "stablecoin_supply_daily_log_chg"
])

# crypto market activity variables
crypto_market_predictors = existing_cols(data_frame, [
    "stablecoin_exchange_netflow_scaled",
    "btc_eth_open_interest_daily_log_chg",
    "btc_eth_funding_rate_avg",
    "stablecoin_exchange_reserve_daily_log_chg",
    "tradingVol_btc+eth_daily_log_chg",
])

all_crypto_predictors = (
    stablecoin_predictors
    + crypto_market_predictors
)

# macro controls
# no NCFI weekly cause forces weekly lag
macro_controls = existing_cols(data_frame, [
    "term_spread_daily_chg",
    "usd_strength_daily_log_chg",
])


#%%------- RUN STATIONARITY CHECKS

stationarity_vars = risk_vars + all_crypto_predictors + macro_controls
stationarity_vars = list(dict.fromkeys(stationarity_vars))

adf_results = run_adf_tests(data_frame, stationarity_vars)

adf_results.to_csv(
    output_dir / "var_stationarity_adf_tests.csv",
    index=False
)

print("\nADF stationarity tests:")
print(adf_results)


#%%------- RUN VAR ANALYSIS

all_var_coefficients = []
all_granger_results = []
all_irf_results = []
all_var_info = []

all_var_models = {}
all_var_samples = {}

# Main approach:
# estimate one VAR per crypto liquidity/risk variable
# avoids an overly large daily VAR and gives cleaner interpretation
for crypto_var in all_crypto_predictors:

    var_list = [crypto_var] + risk_vars + macro_controls # build list of variables for model
    var_list = list(dict.fromkeys(var_list)) # sanity: remove dupl

    # keep only existing columns
    var_list = existing_cols(data_frame, var_list) # sanity: check all vars exist in df, drop missing

    # create VAR dataframe
    var_df = data_frame[["date"] + var_list].copy() # create new df with date + variables

    # drop missing values only for this VAR specification
    var_df = var_df.dropna(subset=var_list).copy() # drop NAs ---> IMPORTANT! each var has own set then -------------------------------!!!

    # set date index
    var_df = var_df.set_index("date")

    # skip if sample too small --> sanity, why not keep...
    if len(var_df) < 100:
        print(f"Skipped {crypto_var}: too few observations after dropna, n={len(var_df)}")
        continue

    # lag selection --> how many lags based on helper
    selected_lag = select_var_lag(
        var_df=var_df,
        maxlags=maxlags,
        criterion=lag_selection_criterion
    )

    # create name for current var --> important for later analysis
    specification = f"var_{crypto_var}"

    # print msg: which var currently being run
    print(
        f"\nRunning {specification}: "
        f"nobs={len(var_df)}, variables={len(var_list)}, selected_lag={selected_lag}"
    )

    try:
        # estimate VAR
        var_model = VAR(var_df)
        var_result = var_model.fit(
            maxlags=selected_lag,
            trend=trend
        )

        # store full model and sample
        all_var_models[specification] = var_result
        all_var_samples[specification] = var_df

        # model info
        all_var_info.append({
            "specification": specification,
            "crypto_variable": crypto_var,
            "variables": ", ".join(var_list),
            "nobs": var_result.nobs,
            "lag_order": var_result.k_ar,
            "aic": var_result.aic,
            "bic": var_result.bic,
            "hqic": var_result.hqic,
            "fpe": var_result.fpe,
            "is_stable": var_result.is_stable(verbose=False),
        })

        # coefficients
        coef_df = extract_var_coefficients(
            var_result=var_result,
            specification=specification
        )
        all_var_coefficients.append(coef_df)

        # Granger: crypto liquidity -> risk variables
        granger_crypto_to_risk = run_granger_tests(
            var_result=var_result,
            specification=specification,
            causing_vars=[crypto_var],
            caused_vars=risk_vars
        )
        granger_crypto_to_risk["direction"] = "crypto_to_risk"
        all_granger_results.append(granger_crypto_to_risk)

        # Granger: risk variables -> crypto liquidity
        granger_risk_to_crypto = run_granger_tests(
            var_result=var_result,
            specification=specification,
            causing_vars=risk_vars,
            caused_vars=[crypto_var]
        )
        granger_risk_to_crypto["direction"] = "risk_to_crypto"
        all_granger_results.append(granger_risk_to_crypto)

        # IRF crypto shock -> risk variables
        irf_df = extract_irf_table(
            var_result=var_result,
            specification=specification,
            impulses=[crypto_var],
            responses=risk_vars,
            periods=20
        )
        all_irf_results.append(irf_df)

        print(
            f"Finished {specification}: "
            f"lag={var_result.k_ar}, AIC={var_result.aic:.4f}, "
            f"BIC={var_result.bic:.4f}, stable={var_result.is_stable(verbose=False)}"
        )

    except Exception as e:
        print(f"Skipped {specification}: {e}")


#%%------- COMBINE AND SAVE VAR RESULTS

var_info = pd.DataFrame(all_var_info)

if len(all_var_coefficients) > 0:
    var_coefficients = pd.concat(all_var_coefficients, ignore_index=True)
else:
    var_coefficients = pd.DataFrame()

if len(all_granger_results) > 0:
    granger_results = pd.concat(all_granger_results, ignore_index=True)
else:
    granger_results = pd.DataFrame()

if len(all_irf_results) > 0:
    irf_results = pd.concat(all_irf_results, ignore_index=True)
else:
    irf_results = pd.DataFrame()

# save outputs
var_info.to_csv(
    output_dir / "var_model_info.csv",
    index=False
)

var_coefficients.to_csv(
    output_dir / "var_coefficients.csv",
    index=False
)

granger_results.to_csv(
    output_dir / "var_granger_causality_results.csv",
    index=False
)

irf_results.to_csv(
    output_dir / "var_irf_values_crypto_to_risk.csv",
    index=False
)

print("\nSaved:")
print(output_dir / "var_model_info.csv")
print(output_dir / "var_coefficients.csv")
print(output_dir / "var_granger_causality_results.csv")
print(output_dir / "var_irf_values_crypto_to_risk.csv")


#%%------- CREATE COMPACT SUMMARY TABLES

# Summary 1: Granger causality, crypto -> risk
if not granger_results.empty:

    crypto_to_risk_summary = (
        granger_results
        .query("direction == 'crypto_to_risk'")
        .sort_values(["causing", "caused"])
        .copy()
    )

    crypto_to_risk_summary.to_csv(
        output_dir / "summary_granger_crypto_to_risk.csv",
        index=False
    )

    print("\nCrypto -> Risk Granger summary:")
    print(
        crypto_to_risk_summary[
            [
                "specification",
                "causing",
                "caused",
                "p_value",
                "significant_10pct",
                "significant_5pct",
                "lag_order",
                "nobs"
            ]
        ]
    )

    # Summary 2: Risk -> crypto
    risk_to_crypto_summary = (
        granger_results
        .query("direction == 'risk_to_crypto'")
        .sort_values(["caused", "causing"])
        .copy()
    )

    risk_to_crypto_summary.to_csv(
        output_dir / "summary_granger_risk_to_crypto.csv",
        index=False
    )

    print("\nRisk -> Crypto Granger summary:")
    print(
        risk_to_crypto_summary[
            [
                "specification",
                "causing",
                "caused",
                "p_value",
                "significant_10pct",
                "significant_5pct",
                "lag_order",
                "nobs"
            ]
        ]
    )


#%%------- PLOT VAR DOT HEATMAP (BOTH DIRECTIONS)

import matplotlib.pyplot as plt


# Use results already created in this file
plot_granger = granger_results.copy()
plot_irf = irf_results.copy()


# PRETTY LABELS

predictor_labels = {
    "stablecoin_supply_daily_log_chg": "Stablecoin Supply",
    "stablecoin_exchange_netflow_scaled": "Exchange Netflow",
    "stablecoin_exchange_reserve_daily_log_chg": "Exchange Reserves",
    "btc_eth_open_interest_daily_log_chg": "Open Interest",
    "btc_eth_funding_rate_avg": "Funding Rate",
    "tradingVol_btc+eth_daily_log_chg": "BTC + ETH Volume",
}

dependent_labels = {
    "sp500_daily_log_ret": "S&P 500",
    "vix_daily_log_chg": "VIX",
    "hy_spread_daily_chg": "HY Spread",
    "ig_spread_daily_chg": "IG Spread",
}


# DISPLAY ORDER (raw variable names)

row_vars = [
    "stablecoin_supply_daily_log_chg",
    "stablecoin_exchange_netflow_scaled",
    "stablecoin_exchange_reserve_daily_log_chg",
    "btc_eth_open_interest_daily_log_chg",
    "btc_eth_funding_rate_avg",
    "tradingVol_btc+eth_daily_log_chg",
]

col_vars = [
    "sp500_daily_log_ret",
    "vix_daily_log_chg",
    "hy_spread_daily_chg",
    "ig_spread_daily_chg",
]

row_order = [predictor_labels[x] for x in row_vars]
col_order = [dependent_labels[x] for x in col_vars]


# CREATE FULL GRID OF CELLS

base_cells = pd.MultiIndex.from_product(
    [row_vars, col_vars],
    names=["crypto_var", "risk_var"]
).to_frame(index=False)

base_cells["predictor_label"] = base_cells["crypto_var"].map(predictor_labels)
base_cells["dependent_label"] = base_cells["risk_var"].map(dependent_labels)


#------------------------------------------------------------
# 1) MAIN DIRECTION: CRYPTO -> RISK
#------------------------------------------------------------

granger_c2r = plot_granger[
    plot_granger["direction"] == "crypto_to_risk"
][["causing", "caused", "p_value"]].copy()

granger_c2r = granger_c2r.rename(columns={
    "causing": "crypto_var",
    "caused": "risk_var",
    "p_value": "p_c2r"
})


#------------------------------------------------------------
# 2) REVERSE DIRECTION: RISK -> CRYPTO
#------------------------------------------------------------

granger_r2c = plot_granger[
    plot_granger["direction"] == "risk_to_crypto"
][["causing", "caused", "p_value"]].copy()

# In the graph cell, row = crypto variable, col = risk variable.
# But risk_to_crypto has causing = risk_var, caused = crypto_var.
granger_r2c = granger_r2c.rename(columns={
    "causing": "risk_var",
    "caused": "crypto_var",
    "p_value": "p_r2c"
})


#------------------------------------------------------------
# 3) IRF DIRECTION: CRYPTO SHOCK -> RISK RESPONSE
#------------------------------------------------------------

# Use cumulative IRF over horizons 1 to 20
irf_start_h = 1
irf_end_h = 20

irf_window = plot_irf[
    (plot_irf["horizon"] >= irf_start_h) &
    (plot_irf["horizon"] <= irf_end_h)
].copy()

irf_summary = (
    irf_window
    .groupby(["impulse", "response"])
    .agg(
        cum_irf=("irf", "sum"),
        avg_irf=("irf", "mean")
    )
    .reset_index()
)

irf_summary = irf_summary.rename(columns={
    "impulse": "crypto_var",
    "response": "risk_var"
})

#------------------------------------------------------------
# 3B) REVERSE IRF DIRECTION: RISK SHOCK -> CRYPTO RESPONSE
#------------------------------------------------------------

# For the small reverse markers, compute how the crypto variable
# responds after a shock to the risk variable.

reverse_irf_list = []

for crypto_var in all_crypto_predictors:

    specification = f"var_{crypto_var}"

    if specification not in all_var_models:
        continue

    reverse_irf_df = extract_irf_table(
        var_result=all_var_models[specification],
        specification=specification,
        impulses=risk_vars,
        responses=[crypto_var],
        periods=20
    )

    reverse_irf_list.append(reverse_irf_df)

if len(reverse_irf_list) > 0:
    reverse_irf_results = pd.concat(reverse_irf_list, ignore_index=True)
else:
    reverse_irf_results = pd.DataFrame()


# Use cumulative reverse IRF over horizons 1 to 20
reverse_irf_window = reverse_irf_results[
    (reverse_irf_results["horizon"] >= irf_start_h) &
    (reverse_irf_results["horizon"] <= irf_end_h)
].copy()

reverse_irf_summary = (
    reverse_irf_window
    .groupby(["impulse", "response"])
    .agg(
        cum_reverse_irf=("irf", "sum"),
        avg_reverse_irf=("irf", "mean")
    )
    .reset_index()
)

# In the graph cell:
# row = crypto variable
# column = risk variable
# reverse IRF has impulse = risk_var, response = crypto_var
reverse_irf_summary = reverse_irf_summary.rename(columns={
    "impulse": "risk_var",
    "response": "crypto_var"
})

#------------------------------------------------------------
# 4) MERGE EVERYTHING INTO ONE TABLE
#------------------------------------------------------------

dot_summary = base_cells.merge(
    granger_c2r,
    on=["crypto_var", "risk_var"],
    how="left"
)

dot_summary = dot_summary.merge(
    granger_r2c,
    on=["crypto_var", "risk_var"],
    how="left"
)

dot_summary = dot_summary.merge(
    irf_summary,
    on=["crypto_var", "risk_var"],
    how="left"
)

dot_summary = dot_summary.merge(
    reverse_irf_summary,
    on=["crypto_var", "risk_var"],
    how="left"
)


#------------------------------------------------------------
# 5) FORMATTING RULES
#------------------------------------------------------------

# Main dot size = crypto -> risk significance
def size_from_p_main(p):
    if pd.isna(p) or p >= 0.10:
        return 0
    elif p < 0.01:
        return 950
    elif p < 0.05:
        return 700
    else:
        return 430

# Main dot color = crypto -> risk significance
def color_from_p_main(p):
    if pd.isna(p) or p >= 0.10:
        return "white"
    elif p < 0.01:
        return "#2D373C"   # strong dark
    elif p < 0.05:
        return "#A5D7D2"   # medium teal
    else:
        return "#D2EBE9"   # light teal

# Reverse marker size = risk -> crypto significance
def size_from_p_reverse(p):
    if pd.isna(p) or p >= 0.10:
        return 0
    elif p < 0.01:
        return 240
    elif p < 0.05:
        return 200
    else:
        return 160

# Reverse marker color = risk -> crypto significance
def color_from_p_reverse(p):
    if pd.isna(p) or p >= 0.10:
        return "white"
    elif p < 0.01:
        return "#2D373C"
    elif p < 0.05:
        return "#6EA9A3"
    else:
        return "#A5D7D2"

# + / - from cumulative IRF
def direction_from_irf(x):
    if pd.isna(x):
        return ""
    elif x > 0:
        return "+"
    elif x < 0:
        return "−"
    else:
        return "0"

dot_summary["dot_size"] = dot_summary["p_c2r"].apply(size_from_p_main)
dot_summary["dot_color"] = dot_summary["p_c2r"].apply(color_from_p_main)

dot_summary["reverse_size"] = dot_summary["p_r2c"].apply(size_from_p_reverse)
dot_summary["reverse_color"] = dot_summary["p_r2c"].apply(color_from_p_reverse)

dot_summary["direction_label"] = dot_summary["cum_irf"].apply(direction_from_irf)
dot_summary["reverse_direction_label"] = dot_summary["cum_reverse_irf"].apply(direction_from_irf)


# POSITIONING

dot_summary["y_pos"] = dot_summary["predictor_label"].map(
    {name: i for i, name in enumerate(row_order)}
)

dot_summary["x_pos"] = dot_summary["dependent_label"].map(
    {name: i for i, name in enumerate(col_order)}
)


#------------------------------------------------------------
# 6) PLOT
#------------------------------------------------------------

plt.rcParams["font.family"] = "Palatino Linotype"

fig, ax = plt.subplots(figsize=(10, 5.9))


# Draw internal grid lines only
for x in range(1, len(col_order)):
    ax.vlines(
        x - 0.5,
        ymin=-0.5,
        ymax=len(row_order) - 0.5,
        colors="lightgray",
        linewidth=0.9,
        zorder=1
    )

for y in range(1, len(row_order)):
    ax.hlines(
        y - 0.5,
        xmin=-0.5,
        xmax=len(col_order) - 0.5,
        colors="lightgray",
        linewidth=0.9,
        zorder=1
    )


# Draw dots and reverse markers
for _, row in dot_summary.iterrows():

    dot_x = row["x_pos"]
    dot_y = row["y_pos"]

    # MAIN DOT: crypto -> risk
    if row["dot_size"] > 0:
        ax.scatter(
            dot_x,
            dot_y,
            s=row["dot_size"],
            facecolor=row["dot_color"],
            edgecolor="black",
            linewidth=1.0,
            zorder=3
        )

        # white text for darker dots
        text_color = "white" if row["p_c2r"] < 0.05 else "black"

        ax.text(
            dot_x,
            dot_y,
            row["direction_label"],
            ha="center",
            va="center",
            fontsize=13,
            fontweight="bold",
            color=text_color,
            zorder=4
        )

    # REVERSE MARKER: risk -> crypto
    # small hollow circle in upper-right corner of the cell
    # +/- inside marker = cumulative crypto response after risk-variable shock
    if row["reverse_size"] > 0:
    
        reverse_x = dot_x + 0.22
        reverse_y = dot_y - 0.22
    
        ax.scatter(
            reverse_x,
            reverse_y,
            s=row["reverse_size"],
            facecolor="white",
            edgecolor=row["reverse_color"],
            linewidth=1.4,
            zorder=5
        )
    
        ax.text(
            reverse_x,
            reverse_y,
            row["reverse_direction_label"],
            ha="center",
            va="center",
            fontsize=7.5,
            fontweight="bold",
            color="#2D373C",
            zorder=6
        )


# Axis formatting
ax.set_xticks(range(len(col_order)))
ax.set_xticklabels(col_order, fontsize=11)

ax.xaxis.tick_top()
ax.tick_params(axis="x", labeltop=True, labelbottom=False, length=0, pad=8)

ax.set_yticks(range(len(row_order)))
ax.set_yticklabels(row_order, fontsize=11)

ax.set_xlim(-0.5, len(col_order) - 0.5)
ax.set_ylim(len(row_order) - 0.5, -0.5)

ax.tick_params(length=0)

for spine in ax.spines.values():
    spine.set_visible(False)

plt.subplots_adjust(bottom=0.18)
plt.tight_layout(rect=[0.03, 0.16, 0.98, 0.96])


# Footnote
fig.text(
    0.5,
    0.03,
    "Main dot = crypto to risk Granger significance: dark <1%, medium <5%, light <10%\n"
    "Main +/- = cumulative risk-variable response after crypto-indicator shock\n"
    "Corner circle = risk to crypto Granger significance; +/- = cumulative crypto response after risk-variable shock\n"
    "Signs show raw predicted response. Risk-on = S&P 500 up, VIX down, HY/IG spreads down",
    ha="center",
    va="bottom",
    fontsize=7.5,
    color="#46505A",
    linespacing=1.25
)


# Save graph
plt.savefig(
    output_dir / "dot_heatmap_var_bidirectional.png",
    dpi=600,
    bbox_inches="tight"
)

plt.show()


# Save underlying plotted table
dot_summary.to_csv(
    output_dir / "dot_heatmap_var_bidirectional_summary.csv",
    index=False
)