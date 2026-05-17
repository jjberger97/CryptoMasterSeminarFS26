#%%------- IMPORT PACKAGES

import pandas as pd
import statsmodels.api as sm
from urllib.parse import quote
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


#%%------- SETTINGS

# github path
github_url = "https://raw.githubusercontent.com/jjberger97/CryptoMasterSeminarFS26/main/Coding/code"
data_frame_file = "data_table.csv"

# create output folder for regression results
output_dir = Path("regression_results")
output_dir.mkdir(exist_ok=True)

# predictive horizons in days
horizons = [1, 5, 20]

# sample start
sample_start = pd.to_datetime("2020-01-01")


#%%------- HELPER FUNCTIONS

# creates github url
def raw_github_path(base_url, filename):
    return f"{base_url}/{quote(filename)}"

# makes dates under date col to actual dates
def standardize_date_column(df):
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df

# get a csv from github
def read_github_csv(base_url, filename):
    url = raw_github_path(base_url, filename)
    print(f"Reading: {url}")
    return pd.read_csv(url)

# keep only col listed in target in the imported df
def existing_cols(df, cols):
    return [c for c in cols if c in df.columns]

# create forward sum for horizons --> need for pred x cummulative events until t+x
def forward_sum(series, h):
    return series.shift(-1).rolling(window=h).sum().shift(-(h - 1))
"""
CHATGPT EXP:
Creates future cumulative h-day value from t+1 to t+h.
For log returns/log changes:
    this equals the h-day log return/log change.
For spread changes:
    this equals the h-day cumulative spread change.
Example for h=5:
    y_fwd_5[t] = y[t+1] + y[t+2] + ... + y[t+5]
    
"""

"""
# OLD REGRESSION FUNCTION
# regression function
def run_predictive_regression(df, y_col, x_cols, control_cols=None, horizon=1, hac_lags=None):
    
    # if no controls, use empty list --> so functions w/o controls
    if control_cols is None:
        control_cols = []
    
    # hac lags allows regression errors to be correlated up to "horizon" periods apart
    # otherwise "normal" independence of observation assumption does not hold
    # makes sure std_errors are valid
    if hac_lags is None:
        hac_lags = max(1, horizon)

    # check wether all values needed are present    
    needed_cols = [y_col] + x_cols + control_cols
    missing = [c for c in needed_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    
    # create regression df
    reg_df = df[["date"] + needed_cols].copy()

    # future dependent variable --> cumulative measure
    reg_df[f"{y_col}_fwd_{horizon}d"] = forward_sum(reg_df[y_col], horizon)

    # drop missing values only for this specific regression
    # WATCH OUT! May need to change to keep analysis consistent --------------------------------------------- !!!
    reg_df = reg_df.dropna(subset=[f"{y_col}_fwd_{horizon}d"] + x_cols + control_cols)

    # build inputs for actual regression
    y = reg_df[f"{y_col}_fwd_{horizon}d"] # fut cum dependent var
    X = reg_df[x_cols + control_cols] # predictors + controls
    X = sm.add_constant(X) # add intercept

    # create OLS but with HAC adjusted std-errors
    model = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": hac_lags})
    
    # return results of the model and the exact dataframe used
    return model, reg_df
"""

# function takes one finished regression model and turns the important output into a table
def extract_results(model, dep_var, horizon, specification, x_cols):

    # prep list to store results
    rows = []
    
    # loops over all variables estimated in regression
    for var in model.params.index:
        if var == "const": # skip intercept
            continue

        rows.append({
            "dependent": dep_var, # the dependent....
            "horizon": horizon, # horizon specified
            "specification": specification, # tag to tells which results belong to same regression
            "variable": var, # var whose coef is being stored
            "is_main_predictor": var in x_cols, # mark if main explanatory or just confounder
            "coef": model.params[var], # stores estimated coefficient
            "t_stat": model.tvalues[var], # stores t-stat
            "p_value": model.pvalues[var], # ...
            "r_squared": model.rsquared, # ...
            "adj_r_squared": model.rsquared_adj, # ...
            "nobs": int(model.nobs), # number of observations
        })

    return pd.DataFrame(rows)

# account for NAs in dependents
def add_dependent_timing_variables(df, y_col, horizons):
    df = df.copy()
    df = df.sort_values("date").reset_index(drop=True)

    # work only on dates where the dependent variable exists
    y_df = df[["date", y_col]].dropna().copy()
    y_df = y_df.sort_values("date").reset_index(drop=True)

    # previous available dependent observation
    lagged_y_col = f"{y_col}_lag1"
    y_df[lagged_y_col] = y_df[y_col].shift(1)

    # future cumulative dependent variables over valid dependent observations
    for h in horizons:
        fwd_col = f"{y_col}_fwd_{h}d"
        y_df[fwd_col] = (
            y_df[y_col]
            .shift(-1)
            .rolling(window=h)
            .sum()
            .shift(-(h - 1))
        )

    # merge back to full dataframe by date
    timing_cols = ["date", lagged_y_col] + [
        f"{y_col}_fwd_{h}d" for h in horizons
    ]

    df = pd.merge(
        df,
        y_df[timing_cols],
        on="date",
        how="left"
    )

    return df

# regression function
def run_predictive_regression(df, y_col, x_cols, control_cols=None, horizon=1, hac_lags=None):
    
    # if no controls, use empty list --> so functions w/o controls
    if control_cols is None:
        control_cols = []
    
    # hac lags allows regression errors to be correlated up to "horizon" periods apart
    # otherwise "normal" independence of observation assumption does not hold
    # makes sure std_errors are valid
    if hac_lags is None:
        hac_lags = max(1, horizon)

    # COMMENT --------------------------------------------------------------------
    fwd_y_col = f"{y_col}_fwd_{horizon}d"

    # check wether all values needed are present 
    needed_cols = [fwd_y_col] + x_cols + control_cols
    missing = [c for c in needed_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    
    reg_df = df[["date"] + needed_cols].copy()

    # drop missing values only for this specific regression
    # WATCH OUT! May need to change to keep analysis consistent --------------------------------------------- !!!
    reg_df = reg_df.dropna(subset=needed_cols)

    # build inputs for actual regression
    y = reg_df[fwd_y_col]
    X = reg_df[x_cols + control_cols]
    X = sm.add_constant(X)

    # create OLS but with HAC adjusted std-errors
    model = sm.OLS(y, X).fit(
        cov_type="HAC",
        cov_kwds={"maxlags": hac_lags}
    )
    
    # return results of the model and the exact dataframe used
    return model, reg_df


#%%------- LOAD DATA FROM GITHUB

data_frame = read_github_csv(github_url, data_frame_file)

# standardize date column
data_frame = standardize_date_column(data_frame)

# Restrict sample
data_frame = data_frame[data_frame["date"] >= sample_start].copy()


#%%------- DEFINE DEPENDENT VARIABLES

dependent_vars = {
    "sp500": "sp500_daily_log_ret",
    "vix": "vix_daily_log_chg",
    "hy_spread": "hy_spread_daily_chg",
    "ig_spread": "ig_spread_daily_chg",
}



#%%------- DEFINE PREDICTORS

# stable coin mcap variables
# NOTE: CURRENTLY ONLY ALL SC SUPPLY, ADD SEPERATE HERE LATER -------------------------------------------- !!!
stablecoin_predictors = existing_cols(data_frame, [
    "stablecoin_supply_daily_log_chg"
])

# crypto market data
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


#%%------- DEFINE CONTROLS

macro_controls = existing_cols(data_frame, [
    "term_spread_daily_chg",
    "usd_strength_daily_log_chg",
    "nfci_weekly_fill"
])

additional_controls = existing_cols(data_frame, [
    
])

base_controls = macro_controls + additional_controls
base_controls = list(dict.fromkeys(base_controls))


#%%------- RUN PREDICTIVE REGRESSIONS
# regression engine --> run reg across each dep / each horizon / each model spec and store

# set up output collection
all_results = [] # store all from extract results
all_models = {} # store full models, touple key (dep_label, h, spec_name)
all_regression_samples = {} # store dataframes for each reg, touple key (dep_label, h, spec_name)

# loop through dependent variables
for dep_label, y_col in dependent_vars.items():

    # create lagged dependent variable once per dependent variable
    # controlls for persistance of dependent
    data_frame_dep = add_dependent_timing_variables(
    df=data_frame,
    y_col=y_col,
    horizons=horizons
    )

    lagged_y_col = f"{y_col}_lag1"

    # controls used for this dependent variable + add in lagged dep as cont
    controls = [lagged_y_col] + base_controls
    controls = existing_cols(data_frame, controls)

    # loop through prediction horizons
    for h in horizons:

        # list of regression specifications
        specifications = {}

        # 1) Standalone predictor regressions
        for x in all_crypto_predictors:
            specifications[f"standalone_{x}"] = [x]

        # 2) Stablecoin market-cap group -------------------------- WARNING! CURRENTLY SAME AS STANDALON AS ONLY TOTAL STABLECOIN SUPPLY !!!
        if len(stablecoin_predictors) > 0:
            specifications["group_stablecoin_mcaps"] = stablecoin_predictors

        # 3) Crypto market activity group
        if len(crypto_market_predictors) > 0:
            specifications["group_crypto_market"] = crypto_market_predictors

        # 4) Full specification with all available crypto predictors
        if len(all_crypto_predictors) > 0:
            specifications["full_all_crypto_predictors"] = all_crypto_predictors

        # run all specifications from above
        # spec_name = model name, x_cols = predictors used in model
        for spec_name, x_cols in specifications.items():

            # remove duplicated predictors while preserving order
            x_cols_clean = list(dict.fromkeys(x_cols))

            # remove controls that also appear as predictors, good for sanity but hopefully not necessary
            controls_clean = [c for c in controls if c not in x_cols_clean]
            
            # runs regression, skips and prints error if one does not work
            try:
                model, reg_df = run_predictive_regression(
                    df=data_frame_dep,
                    y_col=y_col,
                    x_cols=x_cols_clean,
                    control_cols=controls_clean,
                    horizon=h,
                    hac_lags=max(1, h),
                )

                # extract regression results
                res = extract_results(
                    model=model, # full model from regression
                    dep_var=dep_label, # 
                    horizon=h,
                    specification=spec_name,
                    x_cols=x_cols_clean,
                )

                # add actual dependent column name
                res["dependent_column"] = y_col

                # append results
                all_results.append(res)

                # store model and regression sample
                all_models[(dep_label, h, spec_name)] = model
                all_regression_samples[(dep_label, h, spec_name)] = reg_df

                print(
                    f"Finished: {dep_label}, h={h}, {spec_name}, "
                    f"nobs={int(model.nobs)}, adj.R2={model.rsquared_adj:.4f}"
                )

            except Exception as e:
                print(f"Skipped {dep_label}, h={h}, {spec_name}: {e}")


#%%------- COMBINE REGRESSION RESULTS

# combine all results into one dataframe
if len(all_results) > 0:
    regression_results = pd.concat(all_results, ignore_index=True)
else:
    regression_results = pd.DataFrame()

# add significance stars
def significance_stars(p):
    if p < 0.1:
        return "***"
    elif p < 0.5:
        return "**"
    elif p < 0.1:
        return "*"
    else:
        return ""

if len(regression_results) > 0:

    regression_results["significance"] = regression_results["p_value"].apply(significance_stars)

    regression_results["coef_formatted"] = (
        regression_results["coef"].round(4).astype(str)
        + regression_results["significance"]
    )

    # sort results
    regression_results = regression_results.sort_values(
        ["dependent", "horizon", "specification", "is_main_predictor", "variable"],
        ascending=[True, True, True, False, True]
    ).reset_index(drop=True)

    print("\nRegression results created.")
    print(regression_results.head())
    print(regression_results.info())

else:
    print("\nNo regression results were created.")


#%%------- SAVE FULL REGRESSION RESULTS

if len(regression_results) > 0:

    regression_results.to_csv(
        output_dir / "predictive_regression_full_results.csv",
        index=False
    )

    print("\nSaved full regression results:")
    print(output_dir / "predictive_regression_full_results.csv")


#%%------- SAVE MAIN PREDICTOR RESULTS ONLY

if len(regression_results) > 0:

    main_predictor_results = regression_results[
        regression_results["is_main_predictor"] == True
    ].copy()

    main_predictor_results.to_csv(
        output_dir / "predictive_regression_main_predictors_only.csv",
        index=False
    )

    print("\nSaved main predictor results only:")
    print(output_dir / "predictive_regression_main_predictors_only.csv")


#%%------- CREATE SUMMARY TABLES

if len(regression_results) > 0:

    # keep only main predictors for summary tables
    main_results = regression_results[
        regression_results["is_main_predictor"] == True
    ].copy()

    # coefficient table
    coef_table = main_results.pivot_table(
        index=["dependent", "specification", "variable"],
        columns="horizon",
        values="coef"
    )

    # t-stat table
    tstat_table = main_results.pivot_table(
        index=["dependent", "specification", "variable"],
        columns="horizon",
        values="t_stat"
    )

    # p-value table
    pval_table = main_results.pivot_table(
        index=["dependent", "specification", "variable"],
        columns="horizon",
        values="p_value"
    )

    # formatted coefficient table with stars
    formatted_coef_table = main_results.pivot_table(
        index=["dependent", "specification", "variable"],
        columns="horizon",
        values="coef_formatted",
        aggfunc="first"
    )

    # save summary tables
    coef_table.to_csv(output_dir / "predictive_regression_coefficients.csv")
    tstat_table.to_csv(output_dir / "predictive_regression_tstats.csv")
    pval_table.to_csv(output_dir / "predictive_regression_pvalues.csv")
    formatted_coef_table.to_csv(output_dir / "predictive_regression_coefficients_formatted.csv")

    print("\nSaved summary tables.")


#%%------- DISPLAY MOST SIGNIFICANT MAIN PREDICTOR RESULTS

if len(regression_results) > 0:

    significant_results = regression_results[
        regression_results["is_main_predictor"] == True
    ].copy()

    significant_results = significant_results.sort_values(
        "p_value"
    ).reset_index(drop=True)

    print("\nMost significant main predictor results:")
    print(
        significant_results[
            [
                "dependent",
                "horizon",
                "specification",
                "variable",
                "coef",
                "t_stat",
                "p_value",
                "significance",
                "adj_r_squared",
                "nobs"
            ]
        ].head(30)
    )


#%%------- SAVE RESULTS SIGNIFICANT AT 10%

if len(regression_results) > 0:

    significant_10 = regression_results[
        (regression_results["is_main_predictor"] == True) &
        (regression_results["p_value"] < 0.10)
    ].copy()

    significant_10 = significant_10.sort_values(
        ["dependent", "horizon", "specification", "p_value"]
    ).reset_index(drop=True)

    significant_10.to_csv(
        output_dir / "predictive_regression_significant_10pct.csv",
        index=False
    )

    print("\nSaved results significant at 10%:")
    print(output_dir / "predictive_regression_significant_10pct.csv")

    print("\nSignificant at 10%:")
    print(
        significant_10[
            [
                "dependent",
                "horizon",
                "specification",
                "variable",
                "coef",
                "t_stat",
                "p_value",
                "significance",
                "adj_r_squared",
                "nobs"
            ]
        ].head(30)
    )
    
 

#%%------- PLOT
# use the regression results already created in this file
plot_results = regression_results.copy()

# Keep only the actual crypto predictors, not controls
plot_results = plot_results[plot_results["is_main_predictor"] == True].copy()


# Since group_stablecoin_mcaps currently contains only stablecoin_supply_daily_log_chg,
# it duplicates the standalone stablecoin regression
plot_results = plot_results[
    plot_results["specification"] != "group_stablecoin_mcaps"
].copy()


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
    "sp500": "S&P 500",
    "vix": "VIX",
    "hy_spread": "HY Spread",
    "ig_spread": "IG Spread",
}

plot_results["predictor_label"] = plot_results["variable"].map(predictor_labels)
plot_results["dependent_label"] = plot_results["dependent"].map(dependent_labels)

# Drop rows that are not part of the visual
plot_results = plot_results.dropna(subset=["predictor_label", "dependent_label"]).copy()


# SIGNIFICANCE INDICATORS
plot_results["sig_10"] = (plot_results["p_value"] < 0.10).astype(int)
plot_results["sig_05"] = (plot_results["p_value"] < 0.05).astype(int)
plot_results["sig_01"] = (plot_results["p_value"] < 0.01).astype(int)


# AGGREGATE TO PREDICTOR x DEPENDENT LEVEL ---
# Keep only significant observations for coefficient summary
# This means avg/std are based only on results that are significant at p < 0.10
sig_plot_results = plot_results[plot_results["p_value"] < 0.10].copy()

# Main summary: significance counts and strongest p-value
dot_summary = (
    plot_results
    .groupby(["predictor_label", "dependent_label"])
    .agg(
        n_sig_10=("sig_10", "sum"),
        n_sig_05=("sig_05", "sum"),
        n_sig_01=("sig_01", "sum"),
        min_p=("p_value", "min"),
        total_tests=("p_value", "count")
    )
    .reset_index()
)

# Coefficient summary among significant observations only
coef_summary = (
    sig_plot_results
    .groupby(["predictor_label", "dependent_label"])
    .agg(
        avg_coef_sig=("coef", "mean"),
        std_coef_sig=("coef", "std")
    )
    .reset_index()
)

# Merge coefficient summary back into dot_summary
dot_summary = pd.merge(
    dot_summary,
    coef_summary,
    on=["predictor_label", "dependent_label"],
    how="left"
)

# Strongest significance found in each predictor-dependent cell
def star_label(p):
    if pd.isna(p):
        return ""
    elif p < 0.01:
        return "***"
    elif p < 0.05:
        return "**"
    elif p < 0.10:
        return "*"
    else:
        return ""

dot_summary["star_label"] = dot_summary["min_p"].apply(star_label)

# DEFINE DISPLAY ORDER
row_order = [
    "Stablecoin Supply",
    "Exchange Netflow",
    "Exchange Reserves",
    "Open Interest",
    "Funding Rate",
    "BTC + ETH Volume",
]
col_order = [
    "S&P 500",
    "VIX",
    "HY Spread",
    "IG Spread",
]

dot_summary["y_pos"] = dot_summary["predictor_label"].map(
    {name: i for i, name in enumerate(row_order)}
)

dot_summary["x_pos"] = dot_summary["dependent_label"].map(
    {name: i for i, name in enumerate(col_order)}
)

dot_summary = dot_summary.dropna(subset=["x_pos", "y_pos"]).copy()


# DOT SIZE RULE

# Dot size = number of significant results at 10% level
# This captures recurrence across horizons/specifications.
def size_from_count(n):
    if n == 0:
        return 0
    elif n == 1:
        return 260
    elif n == 2:
        return 520
    elif n == 3:
        return 860
    elif n == 4:
        return 1250
    else:
        return 1650

dot_summary["dot_size"] = dot_summary["n_sig_10"].apply(size_from_count)


# DOT COLOR RULE
def color_from_p(p):
    if pd.isna(p) or p >= 0.10:
        return "white"
    elif p < 0.01:
        return "#2D373C"
    elif p < 0.05:
        return "#A5D7D2"
    else:
        return "#D2EBE9"

dot_summary["dot_color"] = dot_summary["min_p"].apply(color_from_p)


# PLOT DOT HEATMAP

plt.rcParams["font.family"] = "Palatino Linotype"

fig, ax = plt.subplots(figsize=(10, 5.9))

# Draw only internal grid lines (no outer border)
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

# Draw dots: dot on right side of cell, text on left side
for _, row in dot_summary.iterrows():

    if row["dot_size"] > 0:

        # Center dot in cell
        dot_x = row["x_pos"]
        dot_y = row["y_pos"]

        ax.scatter(
            dot_x,
            dot_y,
            s=row["dot_size"],
            facecolor=row["dot_color"],
            edgecolor="black",
            linewidth=1.0,
            zorder=3
        )

        # Text color: white for darker dots, black otherwise
        text_color = "white" if row["min_p"] < 0.05 else "black"

        # Direction of average coefficient among significant results
        if row["avg_coef_sig"] > 0:
            direction_label = "+"
        elif row["avg_coef_sig"] < 0:
            direction_label = "−"   # proper minus sign
        else:
            direction_label = "0"

        ax.text(
            dot_x,
            dot_y,
            direction_label,
            ha="center",
            va="center",
            fontsize=13,
            fontweight="bold",
            color=text_color,
            zorder=4
        )

# Axis formatting
ax.set_xticks(range(len(col_order)))
ax.set_xticklabels(col_order, fontsize=11)

# Move column headers to the top
ax.xaxis.tick_top()
ax.tick_params(axis="x", labeltop=True, labelbottom=False, length=0, pad=8)

ax.set_yticks(range(len(row_order)))
ax.set_yticklabels(row_order, fontsize=11)

ax.set_xlim(-0.5, len(col_order) - 0.5)
ax.set_ylim(len(row_order) - 0.5, -0.5)

ax.tick_params(length=0)

for spine in ax.spines.values():
    spine.set_visible(False)

plt.subplots_adjust(bottom=0.16)

plt.tight_layout(rect=[0.03, 0.125, 0.98, 0.96])

fig.text(
    0.5,
    0.032,
    "Dot size = recurrence of significant results across horizons/specifications (p < 0.10)\n"
    "Dot color = strongest significance in cell: black < 1%, dark < 5%, light < 10%\n"
    "+/− = average coefficient direction among significant results\n"
    "Signs show raw predicted response. Risk-on = S&P 500 up, VIX down, HY/IG spreads down",
    ha="center",
    va="bottom",
    fontsize=7.8,
    color="#46505A",
    linespacing=1.25
)

# Save into the same regression_results folder you already use
plt.savefig(
    output_dir / "dot_heatmap_predictive_signals.png",
    dpi=600,
    bbox_inches="tight"
)

plt.show()