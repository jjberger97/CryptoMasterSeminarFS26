#%%------- IMPORT PACKAGES
import pandas as pd
import numpy as np
from urllib.parse import quote
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.transforms import blended_transform_factory


#%%------- IMPORT RAW DATA FROM GITHUB

# github folder
github_url = "https://raw.githubusercontent.com/jjberger97/CryptoMasterSeminarFS26/main/Coding/data"

# list of csv files to be added to data
files = {
    "sp500": "SP500 Daily Price.csv", 
    "vix": "VIX.csv",
    "hy_spread": "ICE BofA US High Yield Index Option-Adjusted Spread.csv",
    "ig_spread": "ICE BofA US Corporate Index Option-Adjusted Spread.csv",
    "term_spread": "10-Year Minus 2-Year Treasury Constant Maturity.csv",
    "usd_index": "Nominal Broad U.S. Dollar Index.csv",
    "nfci": "NCFI indicators.csv",
    "global_crypto_mcap": "GlobalCryptoMktCap.csv",
    "btc": "btc-usd-max.csv",
    "eth": "eth-usd-max.csv",
    "usdt": "usdt-usd-max.csv",
    "usdc": "usdc-usd-max.csv",
    "dai": "dai-usd-max.csv",
    "tusd": "tusd-usd-max.csv",
    # "fdusd": "fdusd-usd-max.csv", # removed due to low data availability
}

# import files into dict of data frames
data_raw = {}

for name, file in files.items():
    url = f"{github_url}/{quote(file)}"
    data_raw[name] = pd.read_csv(url)
    print(f"Imported: {name}")
    

#%%------- CLEAN DATA & ADJUST TO ANALYSIS

# create data dict for all clean datasets
data = {} 


#%% VIX - Dependent
# use daily log change closee

vix = data_raw["vix"].copy()

# only date and data
vix = vix[["DATE", "CLOSE"]].copy()

# convert date col
vix["date"] = pd.to_datetime(vix["DATE"], format="%m/%d/%Y").dt.date

# convert close to numeric
vix["CLOSE"] = pd.to_numeric(vix["CLOSE"], errors="coerce")

# sort 
vix = vix.sort_values("date").reset_index(drop=True)

# compute daily log change: ln(vix_t / vix_t-1)
vix["vix_daily_log_chg"] = np.log(vix["CLOSE"] / vix["CLOSE"].shift(1))

# only final format
vix = vix[["date", "vix_daily_log_chg"]].copy()

# store clean 
data["vix"] = vix


#%% S&P 500 - Dependent
# use daily log change

sp500 = data_raw["sp500"].copy()

# only date and data
sp500 = sp500[["observation_date", "SP500"]].copy()

# convert date col
sp500["date"] = pd.to_datetime(sp500["observation_date"]).dt.date

# convert snp500 value to numeric
sp500["SP500"] = pd.to_numeric(sp500["SP500"], errors="coerce")

# sort
sp500 = sp500.sort_values("date").reset_index(drop=True)

# compute daily log return: ln(sp500_t / sp500_t-1)
sp500["sp500_daily_log_ret"] = np.log(sp500["SP500"] / sp500["SP500"].shift(1))

# only final format
sp500 = sp500[["date", "sp500_daily_log_ret"]].copy()

# store clean 
data["sp500"] = sp500


#%% US HY Spread - Dependent
# create daily changes

hy_spread = data_raw["hy_spread"].copy()

# only date and data
hy_spread = hy_spread[["observation_date", "BAMLH0A0HYM2"]].copy()

# convert date col
hy_spread["date"] = pd.to_datetime(hy_spread["observation_date"]).dt.date

# convert spread to numeric
hy_spread["BAMLH0A0HYM2"] = pd.to_numeric(hy_spread["BAMLH0A0HYM2"], errors="coerce")

# sort 
hy_spread = hy_spread.sort_values("date").reset_index(drop=True)

# compute daily change: spread_t - spread_t-1
hy_spread["hy_spread_daily_chg"] = (hy_spread["BAMLH0A0HYM2"] - hy_spread["BAMLH0A0HYM2"].shift(1))

# only final format
hy_spread = hy_spread[["date", "hy_spread_daily_chg"]].copy()

# store clean 
data["hy_spread"] = hy_spread


#%% US IG Spread - Dependent
# create daily changes

ig_spread = data_raw["ig_spread"].copy()

# keep only date and data
ig_spread = ig_spread[["observation_date", "BAMLC0A0CM"]].copy()

# convert date col
ig_spread["date"] = pd.to_datetime(ig_spread["observation_date"]).dt.date

# convert spread to numeric
ig_spread["BAMLC0A0CM"] = pd.to_numeric(ig_spread["BAMLC0A0CM"], errors="coerce")

# sort 
ig_spread = ig_spread.sort_values("date").reset_index(drop=True)

# compute daily change: spread_t - spread_t-1
ig_spread["ig_spread_daily_chg"] = (ig_spread["BAMLC0A0CM"] - ig_spread["BAMLC0A0CM"].shift(1))

# keep only final
ig_spread = ig_spread[["date", "ig_spread_daily_chg"]].copy()

# store clean 
data["ig_spread"] = ig_spread


#%% USDT Market Cap - Independent
# create daily log changes

usdt = data_raw["usdt"].copy()

# keep only date and data
usdt = usdt[["snapped_at", "market_cap"]].copy()

# convert date col
usdt["date"] = pd.to_datetime(usdt["snapped_at"], utc=True).dt.date

# convert mcap to numeric
usdt["market_cap"] = pd.to_numeric(usdt["market_cap"], errors="coerce")

# sort
usdt = usdt.sort_values("date").reset_index(drop=True)

# compute daily log change: ln(mcap_t / mcap_t-1)
usdt["usdt_mcap_daily_log_chg"] = np.log(usdt["market_cap"] / usdt["market_cap"].shift(1))

# only final format
usdt = usdt[["date", "usdt_mcap_daily_log_chg"]].copy()

# store clean
data["usdt"] = usdt


#%% USDC Market Cap - Independent
# create daily log changes

usdc = data_raw["usdc"].copy()

# keep only date and data
usdc = usdc[["snapped_at", "market_cap"]].copy()

# convert date column 
usdc["date"] = pd.to_datetime(usdc["snapped_at"], utc=True).dt.date

# convert mcap to numeric
usdc["market_cap"] = pd.to_numeric(usdc["market_cap"], errors="coerce")

# sort
usdc = usdc.sort_values("date").reset_index(drop=True)

# compute daily log change: ln(mcap_t / mcap_t-1)
usdc["usdc_mcap_daily_log_chg"] = np.log(usdc["market_cap"] / usdc["market_cap"].shift(1))

# only final format
usdc = usdc[["date", "usdc_mcap_daily_log_chg"]].copy()

# store clean 
data["usdc"] = usdc


#%% DAI Market Cap - Independent
# create daily log changes

dai = data_raw["dai"].copy()

# keep only date and data
dai = dai[["snapped_at", "market_cap"]].copy()

# convert date col
dai["date"] = pd.to_datetime(dai["snapped_at"], utc=True).dt.date

# mcap to numeric
dai["market_cap"] = pd.to_numeric(dai["market_cap"], errors="coerce")

# sort
dai = dai.sort_values("date").reset_index(drop=True)

# compute daily log change: ln(mcap_t / mcap_t-1)
dai["dai_mcap_daily_log_chg"] = np.log(dai["market_cap"] / dai["market_cap"].shift(1))

# only final format
dai = dai[["date", "dai_mcap_daily_log_chg"]].copy()

# store clean
data["dai"] = dai


#%% TUSD Market Cap - Independent
# create daily log changes

tusd = data_raw["tusd"].copy()

# keep only date and data
tusd = tusd[["snapped_at", "market_cap"]].copy()

# convert date col
tusd["date"] = pd.to_datetime(tusd["snapped_at"], utc=True).dt.date

# convert to numeric
tusd["market_cap"] = pd.to_numeric(tusd["market_cap"], errors="coerce")

# sort 
tusd = tusd.sort_values("date").reset_index(drop=True)

# compute daily log change: ln(mcap_t / mcap_t-1)
tusd["tusd_mcap_daily_log_chg"] = np.log(tusd["market_cap"] / tusd["market_cap"].shift(1))

# only final format
tusd = tusd[["date", "tusd_mcap_daily_log_chg"]].copy()

# store clean 
data["tusd"] = tusd


#%% FDUSD Market Cap - Independent
"""
# create daily log changes REMOVED DUE TO LOW DATA AVAILABILITY

fdusd = data_raw["fdusd"].copy()

# keep only date and data
fdusd = fdusd[["snapped_at", "market_cap"]].copy()

# convert date column 
fdusd["date"] = pd.to_datetime(fdusd["snapped_at"], utc=True).dt.date

# convert market cap to numeric
fdusd["market_cap"] = pd.to_numeric(fdusd["market_cap"], errors="coerce")

# sort by date
fdusd = fdusd.sort_values("date").reset_index(drop=True)

# daily log change: ln(mcap_t / mcap_t-1)
fdusd["fdusd_mcap_daily_log_chg"] = np.log(fdusd["market_cap"] / fdusd["market_cap"].shift(1))

# keep only final
fdusd = fdusd[["date", "fdusd_mcap_daily_log_chg"]].copy()

# store clean 
data["fdusd"] = fdusd
"""

#%% BTC + ETH Trading Volume - Independent
# create daily log change of combined BTC and ETH volume

btc_volume = data_raw["btc"].copy()
eth_volume = data_raw["eth"].copy()

# keep only date and data
btc_volume = btc_volume[["snapped_at", "total_volume"]].copy()
eth_volume = eth_volume[["snapped_at", "total_volume"]].copy()

# convert date column 
btc_volume["date"] = pd.to_datetime(btc_volume["snapped_at"], utc=True).dt.date
eth_volume["date"] = pd.to_datetime(eth_volume["snapped_at"], utc=True).dt.date

# convert to numeric
btc_volume["btc_volume"] = pd.to_numeric(btc_volume["total_volume"], errors="coerce")
eth_volume["eth_volume"] = pd.to_numeric(eth_volume["total_volume"], errors="coerce")

# keep only date and eth+btc data
btc_volume = btc_volume[["date", "btc_volume"]].copy()
eth_volume = eth_volume[["date", "eth_volume"]].copy()

# merge BTC and ETH volume by date
trading_volume = pd.merge(
    btc_volume,
    eth_volume,
    on="date",
    how="outer"
)

# sort by date
trading_volume = trading_volume.sort_values("date").reset_index(drop=True)

# BTC + ETH trading volume
trading_volume["total_volume"] = (
    trading_volume["btc_volume"] +
    trading_volume["eth_volume"]
)

# compute daily log change: ln(volume_t / volume_t-1)
trading_volume["tradingVol_btc+eth_daily_log_chg"] = np.log(
    trading_volume["total_volume"] / trading_volume["total_volume"].shift(1)
)

# only final format
trading_volume = trading_volume[["date", "tradingVol_btc+eth_daily_log_chg"]].copy()

# clean version
data["trading_volume"] = trading_volume


#%% Term Spread - Confounder
# create daily changes

term_spread = data_raw["term_spread"].copy()

# keep only date and data
term_spread = term_spread[["observation_date", "T10Y2Y"]].copy()

# Convert date column to date only
term_spread["date"] = pd.to_datetime(term_spread["observation_date"]).dt.date

# spread to numeric
term_spread["T10Y2Y"] = pd.to_numeric(
    term_spread["T10Y2Y"],
    errors="coerce"
)

# sort by date
term_spread = term_spread.sort_values("date").reset_index(drop=True)

# compute daily change: spread_t - spread_t-1
term_spread["term_spread_daily_chg"] = (
    term_spread["T10Y2Y"] -
    term_spread["T10Y2Y"].shift(1)
)

# only final format
term_spread = term_spread[["date", "term_spread_daily_chg"]].copy()

# clean version
data["term_spread"] = term_spread


#%% Dollar Strength - Confounder
# create daily log changes

usd_index = data_raw["usd_index"].copy()

# only date and data
usd_index = usd_index[["observation_date", "DTWEXBGS"]].copy()

# convert date column
usd_index["date"] = pd.to_datetime(usd_index["observation_date"]).dt.date

# index value to numeric
usd_index["DTWEXBGS"] = pd.to_numeric(
    usd_index["DTWEXBGS"],
    errors="coerce"
)

# sort 
usd_index = usd_index.sort_values("date").reset_index(drop=True)

# Compute daily log change: ln(index_t / index_t-1)
usd_index["usd_strength_daily_log_chg"] = np.log(
    usd_index["DTWEXBGS"] / usd_index["DTWEXBGS"].shift(1)
)

# only final format
usd_index = usd_index[["date", "usd_strength_daily_log_chg"]].copy()

# clean version
data["usd_index"] = usd_index


#%% NFCI - Confounder
# weekly level, aligned by approximate publication date and forward-filled

nfci = data_raw["nfci"].copy()

# only date and value
nfci = nfci[["Friday_of_Week", "NFCI"]].copy()

# convert Friday-of-week date
nfci["week_ending_friday"] = pd.to_datetime(
    nfci["Friday_of_Week"],
    format="%m/%d/%Y",
    errors="coerce"
)

# convert NFCI to numeric
nfci["NFCI"] = pd.to_numeric(nfci["NFCI"], errors="coerce")

# drop unusable rows
nfci = nfci.dropna(subset=["week_ending_friday", "NFCI"]).copy()

# approximate publication date:
# NFCI is released on Wednesday and covers the previous Friday
nfci["date"] = nfci["week_ending_friday"] + pd.Timedelta(days=5)

# keep only relevant columns
nfci = nfci[["date", "NFCI"]].copy()

# sort and remove duplicate publication dates if any
nfci = (
    nfci
    .sort_values("date")
    .drop_duplicates(subset="date", keep="last")
    .reset_index(drop=True)
)

# create full daily calendar from first to last NFCI release
nfci_daily_calendar = pd.DataFrame({
    "date": pd.date_range(nfci["date"].min(), nfci["date"].max(), freq="D")
})

# merge weekly releases onto daily calendar
nfci = pd.merge(
    nfci_daily_calendar,
    nfci,
    on="date",
    how="left"
)

# forward-fill from release date onward
nfci["NFCI"] = nfci["NFCI"].ffill()

# convert date back to date-only format
nfci["date"] = nfci["date"].dt.date

# rename
nfci = nfci.rename(columns={"NFCI": "nfci_weekly_fill"})

# keep final format
nfci = nfci[["date", "nfci_weekly_fill"]].copy()

# store clean version
data["nfci"] = nfci


#%%------- MERGE CLEAN DATA FRAMES
# create merged dataset
data_table = None

for name, df in data.items():
    
    temp = df.copy()
    
    # make sure date is date-only
    temp["date"] = pd.to_datetime(temp["date"]).dt.date
    
    # safety check
    if temp.shape[1] != 2:
        raise ValueError(f"{name} has {temp.shape[1]} columns, expected 2.")
    
    # merge
    if data_table is None:
        data_table = temp
    else:
        data_table = pd.merge(
            data_table,
            temp,
            on="date",
            how="outer"
        )

# sort by date
data_table = data_table.sort_values("date").reset_index(drop=True)

# filter sample period
data_table = data_table[
    (data_table["date"] >= pd.to_datetime("2020-01-01").date()) &
    (data_table["date"] <= pd.to_datetime("2025-12-31").date())
].reset_index(drop=True)

# check 
print(data_table.head())
print(data_table.tail())
print(data_table.info())

# save as csv
output_path = "data_table.csv"

data_table.to_csv(
    output_path,
    index=False
)

#%%------- DATA AVAILABILITY GRAPH

# use Palatino Linotype
plt.rcParams["font.family"] = "Palatino Linotype"

# create full daily calendar for sample period
sample_start = pd.to_datetime("2020-01-01")
sample_end = pd.to_datetime("2025-12-31")

full_calendar = pd.DataFrame({
    "date": pd.date_range(sample_start, sample_end, freq="D").date
})

# merge onto full calendar to make sure every calendar day is counted
availability_df = pd.merge(
    full_calendar,
    data_table,
    on="date",
    how="left"
)

# total number of calendar days
total_days = len(availability_df)
print(f"Total days in sample: {total_days}")

# classify variables
crypto_vars = [
    "usdt_mcap_daily_log_chg",
    "usdc_mcap_daily_log_chg",
    "dai_mcap_daily_log_chg",
    "tusd_mcap_daily_log_chg",
    "tradingVol_btc+eth_daily_log_chg",
]

tradfi_vars = [
    "vix_daily_log_chg",
    "sp500_daily_log_ret",
    "hy_spread_daily_chg",
    "ig_spread_daily_chg",
    "term_spread_daily_chg",
    "usd_strength_daily_log_chg",
    "nfci_weekly_fill",
]

# combine in desired order
plot_vars = tradfi_vars + crypto_vars

# nicer names for plot
var_labels = {
    "vix_daily_log_chg": "VIX",
    "sp500_daily_log_ret": "S&P 500",
    "hy_spread_daily_chg": "HY Spread",
    "ig_spread_daily_chg": "IG Spread",
    "term_spread_daily_chg": "Term Spread",
    "usd_strength_daily_log_chg": "USD Index",
    "nfci_weekly_fill": "NFCI",
    "usdt_mcap_daily_log_chg": "USDT Market Cap",
    "usdc_mcap_daily_log_chg": "USDC Market Cap",
    "dai_mcap_daily_log_chg": "DAI Market Cap",
    "tusd_mcap_daily_log_chg": "TUSD Market Cap",
    "tradingVol_btc+eth_daily_log_chg": "BTC + ETH Volume",
}

# smaller subtext under each variable
var_subtexts = {
    "vix_daily_log_chg": "daily log change",
    "sp500_daily_log_ret": "daily log return",
    "hy_spread_daily_chg": "daily absolute change",
    "ig_spread_daily_chg": "daily absolute change",
    "term_spread_daily_chg": "daily absolute change",
    "usd_strength_daily_log_chg": "daily log change",
    "nfci_weekly_fill": "weekly level, forward-filled to daily frequency",
    "usdt_mcap_daily_log_chg": "daily log change in market cap",
    "usdc_mcap_daily_log_chg": "daily log change in market cap",
    "dai_mcap_daily_log_chg": "daily log change in market cap",
    "tusd_mcap_daily_log_chg": "daily log change in market cap",
    "tradingVol_btc+eth_daily_log_chg": "daily log change in BTC + ETH volume",
}

# calculate availability
availability = []

for var in plot_vars:
    available_days = availability_df[var].notna().sum()
    missing_days = total_days - available_days
    availability_pct = available_days / total_days * 100
    
    availability.append({
        "variable": var,
        "label": var_labels.get(var, var),
        "subtext": var_subtexts.get(var, ""),
        "available_days": available_days,
        "missing_days": missing_days,
        "availability_pct": availability_pct,
        "type": "Crypto" if var in crypto_vars else "Traditional finance"
    })

availability = pd.DataFrame(availability)

# colors
crypto_color = "#2D373C"
tradfi_color = "#A5D7D2"
missing_color = "#e0e0e0"
subtext_color = "#7a7a7a"

availability["color"] = np.where(
    availability["type"] == "Crypto",
    crypto_color,
    tradfi_color
)

# reverse order so first variable appears at top
availability_plot = availability.iloc[::-1].reset_index(drop=True)

# numeric y positions
y_pos = np.arange(len(availability_plot))

# plot
fig, ax = plt.subplots(figsize=(10, 6.8))

# available part
ax.barh(
    y_pos,
    availability_plot["available_days"],
    color=availability_plot["color"],
    edgecolor="none",
    height=0.46
)

# missing part, stacked to the right
ax.barh(
    y_pos,
    availability_plot["missing_days"],
    left=availability_plot["available_days"],
    color=missing_color,
    edgecolor="none",
    height=0.46
)

# remove default y-axis labels
ax.set_yticks([])

# custom y labels with subtext
label_transform = blended_transform_factory(ax.transAxes, ax.transData)

for i, row in availability_plot.iterrows():
    ax.text(
        -0.025,
        i + 0.10,
        row["label"],
        transform=label_transform,
        ha="right",
        va="center",
        fontsize=10.5,
        color="#1f1f1f"
    )
    
    ax.text(
        -0.025,
        i - 0.16,
        row["subtext"],
        transform=label_transform,
        ha="right",
        va="center",
        fontsize=8.2,
        color=subtext_color
    )

# add percentage labels
for i, row in availability_plot.iterrows():
    ax.text(
        total_days + 20,
        i,
        f"{row['availability_pct']:.1f}%",
        va="center",
        fontsize=9,
        color="#1f1f1f"
    )

# formatting
ax.set_xlim(0, total_days + 180)

# remove horizontal axis marks/ticks/grid
ax.set_xlabel("")
ax.set_xticks([])
ax.tick_params(axis="x", bottom=False, labelbottom=False)
ax.grid(False)

# remove unnecessary spines
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["bottom"].set_visible(False)
ax.spines["left"].set_visible(False)

# legend
tradfi_patch = mpatches.Patch(color=tradfi_color, label="Traditional finance")
crypto_patch = mpatches.Patch(color=crypto_color, label="Crypto")
missing_patch = mpatches.Patch(color=missing_color, label="Missing")

fig.legend(
    handles=[tradfi_patch, crypto_patch, missing_patch],
    loc="upper left",
    bbox_to_anchor=(0.055, 0.90),
    frameon=False,
    fontsize=7,
    handlelength=0.9,
    handleheight=0.6,
    handletextpad=0.4,
    borderpad=0.2,
    labelspacing=0.25
)

# add enough left margin for custom labels
fig.subplots_adjust(left=0.32, right=0.88, top=0.90, bottom=0.08)

# save graph
"""
fig.savefig(
    "data_availability_graph.png",
    dpi=600,
    bbox_inches="tight",
    facecolor="white"
)
"""

plt.show()