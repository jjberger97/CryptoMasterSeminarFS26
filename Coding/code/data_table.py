#%%------- IMPORT PACKAGES
import pandas as pd
import numpy as np
from urllib.parse import quote


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
# create daily log changes

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
# weekly level, converted to daily by forward filling

nfci = data_raw["nfci"].copy()

# only date and val
nfci = nfci[["Friday_of_Week", "NFCI"]].copy()

# make date column
nfci["date"] = pd.to_datetime(nfci["Friday_of_Week"], format="%m/%d/%Y").dt.date

# NFCI to numeric
nfci["NFCI"] = pd.to_numeric(nfci["NFCI"], errors="coerce")

# date sort
nfci = nfci.sort_values("date").reset_index(drop=True)

# set date index for expansion
nfci["date"] = pd.to_datetime(nfci["date"])
nfci = nfci.set_index("date")

# weekly observations to daily frequency
nfci = nfci.asfreq("D")

# forward fill weekly value to daily observations
nfci["NFCI"] = nfci["NFCI"].ffill()

# reset index
nfci = nfci.reset_index()

# convert back to date only
nfci["date"] = nfci["date"].dt.date

# rename data col
nfci = nfci.rename(columns={"NFCI": "ncfi_weekly_fill"})

# keep only final format
nfci = nfci[["date", "ncfi_weekly_fill"]].copy()

# clean version
data["nfci"] = nfci


#%%------- MERGE CLEAN DATA FRAMES
# create merged dataset
merged_full = None

for name, df in data.items():
    
    temp = df.copy()
    
    # make sure date is date-only
    temp["date"] = pd.to_datetime(temp["date"]).dt.date
    
    # safety check
    if temp.shape[1] != 2:
        raise ValueError(f"{name} has {temp.shape[1]} columns, expected 2.")
    
    # merge
    if merged_full is None:
        merged_full = temp
    else:
        merged_full = pd.merge(
            merged_full,
            temp,
            on="date",
            how="outer"
        )

# sort by date
merged_full = merged_full.sort_values("date").reset_index(drop=True)

# filter sample period
merged_full = merged_full[
    (merged_full["date"] >= pd.to_datetime("2020-01-01").date()) &
    (merged_full["date"] <= pd.to_datetime("2025-12-31").date())
].reset_index(drop=True)

# check 
print(merged_full.head())
print(merged_full.tail())
print(merged_full.info())