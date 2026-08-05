#%%------- IMPORT PACKAGES
import pandas as pd
import numpy as np
from urllib.parse import quote
#import matplotlib.pyplot as plt
#import matplotlib.patches as mpatches
#from matplotlib.transforms import blended_transform_factory
#import matplotlib.dates as mdates


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
    "stablecoin_netflow_erc20": "All Stablecoins(ERC20) Exchange Netflow (Total) - All Exchanges - Day.csv",
    "usdt_netflow_trc20": "Tether USD(TRC20) Exchange Netflow (Total) - All Exchanges - Day.csv",
    "stablecoin_reserve_erc20": "All Stablecoins(ERC20) Exchange Reserve - All Exchanges - Day.csv",
    "usdt_reserve_trc20": "Tether USD(TRC20) Exchange Reserve - All Exchanges - Day.csv",
    "btc_open_interest": "Bitcoin Open Interest - All Exchanges, All Symbol - Day.csv",
    "eth_open_interest": "Ethereum Open Interest - All Exchanges, All Symbol - Day.csv",
    "btc_funding_rate": "Bitcoin Funding Rates - All Exchanges - Day.csv",
    "eth_funding_rate": "Ethereum Funding Rates - All Exchanges - Day.csv",
    "stablecoin_supply_erc20": "All Stablecoins(ERC20) Total Supply - Day.csv",
    "usdt_supply_trc20": "Tether USD(TRC20) Supply Total - Day.csv"
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



#%% US HY Spread - Dependent
# create daily changes

hy_spread = data_raw["hy_spread"].copy()

# only date and data
hy_spread = hy_spread[["observation_date", "BAMLH0A0HYM2"]].copy()

# convert date col
hy_spread["date"] = pd.to_datetime(hy_spread["observation_date"]).dt.date

# convert spread to numeric
hy_spread["BAMLH0A0HYM2"] = pd.to_numeric(hy_spread["BAMLH0A0HYM2"], errors="coerce")

# drop NA and sort
hy_spread = (
    hy_spread
    .dropna(subset=["BAMLH0A0HYM2"])
    .sort_values("date")
    .reset_index(drop=True)
)

# compute daily change: spread_t - spread_t-1
hy_spread["hy_spread_daily_chg"] = hy_spread["BAMLH0A0HYM2"].diff()

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

# drop NA and sort
ig_spread = (
    ig_spread
    .dropna(subset=["BAMLC0A0CM"])
    .sort_values("date")
    .reset_index(drop=True)
)

# compute daily change: spread_t - spread_t-1
ig_spread["ig_spread_daily_chg"] = ig_spread["BAMLC0A0CM"].diff()

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

#%% Aggregate Stablecoin Supply - Independent
# daily log change of All Stablecoins(ERC20) supply + USDT TRON supply

# ERC20 aggregate stablecoin supply
stablecoin_supply_erc20 = data_raw["stablecoin_supply_erc20"].copy()

# keep only date and supply
stablecoin_supply_erc20 = stablecoin_supply_erc20[["Datetime", "Total Supply"]].copy()

# convert date column
stablecoin_supply_erc20["date"] = pd.to_datetime(
    stablecoin_supply_erc20["Datetime"],
    utc=True
).dt.date

# convert supply to numeric
stablecoin_supply_erc20["stablecoin_supply_erc20"] = pd.to_numeric(
    stablecoin_supply_erc20["Total Supply"],
    errors="coerce"
)

# keep final level
stablecoin_supply_erc20 = stablecoin_supply_erc20[
    ["date", "stablecoin_supply_erc20"]
].copy()


# USDT TRON supply
usdt_supply_trc20 = data_raw["usdt_supply_trc20"].copy()

# keep only date and supply
# note: this file uses "Supply Total", not "Total Supply"
usdt_supply_trc20 = usdt_supply_trc20[["Datetime", "Supply Total"]].copy()

# convert date column
usdt_supply_trc20["date"] = pd.to_datetime(
    usdt_supply_trc20["Datetime"],
    utc=True
).dt.date

# convert supply to numeric
usdt_supply_trc20["usdt_supply_trc20"] = pd.to_numeric(
    usdt_supply_trc20["Supply Total"],
    errors="coerce"
)

# keep final level
usdt_supply_trc20 = usdt_supply_trc20[
    ["date", "usdt_supply_trc20"]
].copy()


# merge ERC20 and TRON supply
stablecoin_supply_total = pd.merge(
    stablecoin_supply_erc20,
    usdt_supply_trc20,
    on="date",
    how="outer"
)

# sort from oldest to newest before computing log change
stablecoin_supply_total = stablecoin_supply_total.sort_values("date").reset_index(drop=True)

# aggregate stablecoin supply level
stablecoin_supply_total["stablecoin_supply_total"] = (
    stablecoin_supply_total["stablecoin_supply_erc20"] +
    stablecoin_supply_total["usdt_supply_trc20"]
)

# compute daily log change: ln(supply_t / supply_t-1)
stablecoin_supply_total["stablecoin_supply_daily_log_chg"] = np.log(
    stablecoin_supply_total["stablecoin_supply_total"] /
    stablecoin_supply_total["stablecoin_supply_total"].shift(1)
)

# only final format for analysis
stablecoin_supply_total_final = stablecoin_supply_total[
    ["date", "stablecoin_supply_daily_log_chg"]
].copy()

# store clean
data["stablecoin_supply"] = stablecoin_supply_total_final


#%% Exchange Netflow - Independent
# daily netflow level, scaled by aggregate stablecoin supply

# ERC20 stablecoin exchange netflow
stablecoin_netflow_erc20 = data_raw["stablecoin_netflow_erc20"].copy()

# convert date column
stablecoin_netflow_erc20["date"] = pd.to_datetime(
    stablecoin_netflow_erc20["Datetime"],
    utc=True
).dt.date

# convert netflow to numeric
stablecoin_netflow_erc20["stablecoin_netflow_erc20"] = pd.to_numeric(
    stablecoin_netflow_erc20["Exchange Netflow (Total)"],
    errors="coerce"
)

# keep final level
stablecoin_netflow_erc20 = stablecoin_netflow_erc20[
    ["date", "stablecoin_netflow_erc20"]
].copy()

# USDT TRON exchange netflow
usdt_netflow_trc20 = data_raw["usdt_netflow_trc20"].copy()

# convert date column
usdt_netflow_trc20["date"] = pd.to_datetime(
    usdt_netflow_trc20["Datetime"],
    utc=True
).dt.date

# convert netflow to numeric
usdt_netflow_trc20["usdt_netflow_trc20"] = pd.to_numeric(
    usdt_netflow_trc20["Exchange Netflow (Total)"],
    errors="coerce"
)

# keep final level
usdt_netflow_trc20 = usdt_netflow_trc20[
    ["date", "usdt_netflow_trc20"]
].copy()

# merge ERC20 and TRON netflows
exchange_netflow = pd.merge(
    stablecoin_netflow_erc20,
    usdt_netflow_trc20,
    on="date",
    how="outer"
)

# sort
exchange_netflow = exchange_netflow.sort_values("date").reset_index(drop=True)

# combined netflow level
exchange_netflow["stablecoin_exchange_netflow_total"] = (
    exchange_netflow["stablecoin_netflow_erc20"] +
    exchange_netflow["usdt_netflow_trc20"]
)

# merge aggregate stablecoin supply level for scaling
exchange_netflow = pd.merge(
    exchange_netflow,
    stablecoin_supply_total[["date", "stablecoin_supply_total"]],
    on="date",
    how="left"
)

# scale combined netflow by aggregate stablecoin supply
exchange_netflow["stablecoin_exchange_netflow_scaled"] = (
    exchange_netflow["stablecoin_exchange_netflow_total"] /
    exchange_netflow["stablecoin_supply_total"]
)

# only final format
exchange_netflow = exchange_netflow[
    ["date", "stablecoin_exchange_netflow_scaled"]
].copy()

# store clean
data["exchange_netflow"] = exchange_netflow


#%% Open Interest - Independent
# daily log change of combined BTC and ETH open interest

# BTC open interest
btc_open_interest = data_raw["btc_open_interest"].copy()

# convert date column
btc_open_interest["date"] = pd.to_datetime(
    btc_open_interest["Datetime"],
    utc=True
).dt.date

# convert open interest to numeric
btc_open_interest["btc_open_interest"] = pd.to_numeric(
    btc_open_interest["Open Interest"],
    errors="coerce"
)

# keep final level
btc_open_interest = btc_open_interest[
    ["date", "btc_open_interest"]
].copy()


# ETH open interest
eth_open_interest = data_raw["eth_open_interest"].copy()

# convert date column
eth_open_interest["date"] = pd.to_datetime(
    eth_open_interest["Datetime"],
    utc=True
).dt.date

# convert open interest to numeric
eth_open_interest["eth_open_interest"] = pd.to_numeric(
    eth_open_interest["Open Interest"],
    errors="coerce"
)

# keep final level
eth_open_interest = eth_open_interest[
    ["date", "eth_open_interest"]
].copy()


# merge BTC and ETH open interest
open_interest = pd.merge(
    btc_open_interest,
    eth_open_interest,
    on="date",
    how="outer"
)

# sort
open_interest = open_interest.sort_values("date").reset_index(drop=True)

# combined BTC + ETH open interest level
open_interest["btc_eth_open_interest_total"] = (
    open_interest["btc_open_interest"] +
    open_interest["eth_open_interest"]
)

# compute daily log change: ln(open_interest_t / open_interest_t-1)
open_interest["btc_eth_open_interest_daily_log_chg"] = np.log(
    open_interest["btc_eth_open_interest_total"] /
    open_interest["btc_eth_open_interest_total"].shift(1)
)

# only final format
open_interest = open_interest[
    ["date", "btc_eth_open_interest_daily_log_chg"]
].copy()

# store clean
data["open_interest"] = open_interest


#%% Funding Rates - Independent
# daily average of BTC and ETH funding rates

# BTC funding rate
btc_funding_rate = data_raw["btc_funding_rate"].copy()

# convert date column
btc_funding_rate["date"] = pd.to_datetime(
    btc_funding_rate["Datetime"],
    utc=True
).dt.date

# convert funding rate to numeric
btc_funding_rate["btc_funding_rate"] = pd.to_numeric(
    btc_funding_rate["Funding Rates"],
    errors="coerce"
)

# keep final level
btc_funding_rate = btc_funding_rate[
    ["date", "btc_funding_rate"]
].copy()


# ETH funding rate
eth_funding_rate = data_raw["eth_funding_rate"].copy()

# convert date column
eth_funding_rate["date"] = pd.to_datetime(
    eth_funding_rate["Datetime"],
    utc=True
).dt.date

# convert funding rate to numeric
eth_funding_rate["eth_funding_rate"] = pd.to_numeric(
    eth_funding_rate["Funding Rates"],
    errors="coerce"
)

# keep final level
eth_funding_rate = eth_funding_rate[
    ["date", "eth_funding_rate"]
].copy()


# merge BTC and ETH funding rates
funding_rates = pd.merge(
    btc_funding_rate,
    eth_funding_rate,
    on="date",
    how="outer"
)

# sort
funding_rates = funding_rates.sort_values("date").reset_index(drop=True)

# average BTC and ETH funding rates
funding_rates["btc_eth_funding_rate_avg"] = funding_rates[
    ["btc_funding_rate", "eth_funding_rate"]
].mean(axis=1)

# only final format
funding_rates = funding_rates[
    ["date", "btc_eth_funding_rate_avg"]
].copy()

# store clean
data["funding_rates"] = funding_rates


#%% Exchange Reserves - Independent
# daily log change of combined ERC20 stablecoin reserves + USDT TRON reserves

# ERC20 stablecoin exchange reserves
stablecoin_reserve_erc20 = data_raw["stablecoin_reserve_erc20"].copy()

# keep only date and reserve
stablecoin_reserve_erc20 = stablecoin_reserve_erc20[
    ["Datetime", "Exchange Reserve"]
].copy()

# convert date column
stablecoin_reserve_erc20["date"] = pd.to_datetime(
    stablecoin_reserve_erc20["Datetime"],
    utc=True
).dt.date

# convert reserve to numeric
stablecoin_reserve_erc20["stablecoin_reserve_erc20"] = pd.to_numeric(
    stablecoin_reserve_erc20["Exchange Reserve"],
    errors="coerce"
)

# keep final level
stablecoin_reserve_erc20 = stablecoin_reserve_erc20[
    ["date", "stablecoin_reserve_erc20"]
].copy()


# USDT TRON exchange reserves
usdt_reserve_trc20 = data_raw["usdt_reserve_trc20"].copy()

# keep only date and reserve
usdt_reserve_trc20 = usdt_reserve_trc20[
    ["Datetime", "Reserve"]
].copy()

# convert date column
usdt_reserve_trc20["date"] = pd.to_datetime(
    usdt_reserve_trc20["Datetime"],
    utc=True
).dt.date

# convert reserve to numeric
usdt_reserve_trc20["usdt_reserve_trc20"] = pd.to_numeric(
    usdt_reserve_trc20["Reserve"],
    errors="coerce"
)

# keep final level
usdt_reserve_trc20 = usdt_reserve_trc20[
    ["date", "usdt_reserve_trc20"]
].copy()


# merge ERC20 and TRON reserves
exchange_reserves = pd.merge(
    stablecoin_reserve_erc20,
    usdt_reserve_trc20,
    on="date",
    how="outer"
)

# sort
exchange_reserves = exchange_reserves.sort_values("date").reset_index(drop=True)

# combined exchange reserve level
exchange_reserves["stablecoin_exchange_reserve_total"] = (
    exchange_reserves["stablecoin_reserve_erc20"] +
    exchange_reserves["usdt_reserve_trc20"]
)

# compute daily log change: ln(reserve_t / reserve_t-1)
exchange_reserves["stablecoin_exchange_reserve_daily_log_chg"] = np.log(
    exchange_reserves["stablecoin_exchange_reserve_total"] /
    exchange_reserves["stablecoin_exchange_reserve_total"].shift(1)
)

# only final format
exchange_reserves = exchange_reserves[
    ["date", "stablecoin_exchange_reserve_daily_log_chg"]
].copy()

# store clean
data["exchange_reserves"] = exchange_reserves


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


#%% BTC + ETH Log Returns - Independent
# create average daily log return of BTC and ETH

btc_return = data_raw["btc"].copy()
eth_return = data_raw["eth"].copy()

# keep only date and price
btc_return = btc_return[["snapped_at", "price"]].copy()
eth_return = eth_return[["snapped_at", "price"]].copy()

# convert date column
btc_return["date"] = pd.to_datetime(
    btc_return["snapped_at"],
    utc=True
).dt.date

eth_return["date"] = pd.to_datetime(
    eth_return["snapped_at"],
    utc=True
).dt.date

# convert price to numeric
btc_return["btc_price"] = pd.to_numeric(
    btc_return["price"],
    errors="coerce"
)

eth_return["eth_price"] = pd.to_numeric(
    eth_return["price"],
    errors="coerce"
)

# keep final price levels
btc_return = btc_return[["date", "btc_price"]].copy()
eth_return = eth_return[["date", "eth_price"]].copy()

# merge BTC and ETH prices
crypto_returns = pd.merge(
    btc_return,
    eth_return,
    on="date",
    how="outer"
)

# sort
crypto_returns = crypto_returns.sort_values("date").reset_index(drop=True)

# compute individual daily log returns
crypto_returns["btc_daily_log_ret"] = np.log(
    crypto_returns["btc_price"] / crypto_returns["btc_price"].shift(1)
)

crypto_returns["eth_daily_log_ret"] = np.log(
    crypto_returns["eth_price"] / crypto_returns["eth_price"].shift(1)
)

# average BTC and ETH daily log returns
crypto_returns["btc_eth_daily_log_ret_avg"] = crypto_returns[
    ["btc_daily_log_ret", "eth_daily_log_ret"]
].mean(axis=1)

# only final format
crypto_returns = crypto_returns[
    ["date", "btc_eth_daily_log_ret_avg"]
].copy()

# store clean
data["crypto_returns"] = crypto_returns


#%% BTC + ETH Realized Volatility - Confounder
# create 7-day rolling realized volatility from average BTC + ETH daily log returns

crypto_realized_vol = data["crypto_returns"].copy()

# make sure sorted by date
crypto_realized_vol = crypto_realized_vol.sort_values("date").reset_index(drop=True)

# compute 7-day realized volatility
# not annualized; this is a rolling daily volatility control
crypto_realized_vol["btc_eth_realized_vol_7d"] = (
    crypto_realized_vol["btc_eth_daily_log_ret_avg"]
    .rolling(window=7, min_periods=7)
    .std()
)

# only final format
crypto_realized_vol = crypto_realized_vol[
    ["date", "btc_eth_realized_vol_7d"]
].copy()

# store clean
data["crypto_realized_vol"] = crypto_realized_vol




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

# drop NA and sort by date
term_spread = (
    term_spread
    .dropna(subset=["T10Y2Y"])
    .sort_values("date")
    .reset_index(drop=True)
)

# compute daily change: spread_t - spread_t-1
term_spread["term_spread_daily_chg"] = (
    term_spread["T10Y2Y"].diff()
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

# sort and drop NA
usd_index = (
    usd_index
    .dropna(subset=["DTWEXBGS"])
    .sort_values("date")
    .reset_index(drop=True)
)

# Compute daily log change: ln(index_t / index_t-1)
usd_index["usd_strength_daily_log_chg"] = np.log(usd_index["DTWEXBGS"]).diff()

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


#%% VIX - Dependent / Robust
# use daily log change at close

vix = data_raw["vix"].copy()

# only date and data
vix = vix[["DATE", "CLOSE"]].copy()

# convert date col
vix["date"] = pd.to_datetime(vix["DATE"], format="%m/%d/%Y").dt.date

# convert close to numeric
vix["CLOSE"] = pd.to_numeric(vix["CLOSE"], errors="coerce")

# sort and drop NA
vix = (
    vix
    .dropna(subset=["CLOSE"])
    .sort_values("date")
    .reset_index(drop=True)
)

# Compute daily log change: ln(vix_t / vix_t-1)
vix["vix_daily_log_chg"] = np.log(vix["CLOSE"]).diff()

# only final format
vix = vix[["date", "vix_daily_log_chg"]].copy()

# store clean 
data["vix"] = vix


#%% S&P 500 - Dependent / Robust
# use daily log change

sp500 = data_raw["sp500"].copy()

# only date and data
sp500 = sp500[["observation_date", "SP500"]].copy()

# convert date col
sp500["date"] = pd.to_datetime(sp500["observation_date"]).dt.date

# convert snp500 value to numeric
sp500["SP500"] = pd.to_numeric(sp500["SP500"], errors="coerce")

# sort and drop NA
sp500 = (
    sp500
    .dropna(subset=["SP500"])
    .sort_values("date")
    .reset_index(drop=True)
)

# Compute daily log return: ln(sp500_t / spr_t-1)
sp500["sp500_daily_log_ret"] = np.log(sp500["SP500"]).diff()

# only final format
sp500 = sp500[["date", "sp500_daily_log_ret"]].copy()

# store clean 
data["sp500"] = sp500




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

# replace infinite values from log changes with missing values - dont think there are any but CHATGPT is insisting and why not...
data_table = data_table.replace([np.inf, -np.inf], np.nan)

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

"""
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
dependent_vars = [
    "hy_spread_daily_chg",
    "ig_spread_daily_chg",
]

crypto_vars = [
    "stablecoin_supply_daily_log_chg",
    "stablecoin_exchange_netflow_scaled",
    "stablecoin_exchange_reserve_daily_log_chg",
    "btc_eth_open_interest_daily_log_chg",
    "btc_eth_funding_rate_avg",
    "tradingVol_btc+eth_daily_log_chg",
    "btc_eth_daily_log_ret_avg",
    "btc_eth_realized_vol_7d",
    "usdt_mcap_daily_log_chg",
    "usdc_mcap_daily_log_chg",
    "dai_mcap_daily_log_chg",
    "tusd_mcap_daily_log_chg",
]

tradfi_vars = [
    "vix_daily_log_chg",
    "sp500_daily_log_ret",
    "term_spread_daily_chg",
    "usd_strength_daily_log_chg",
    "nfci_weekly_fill",
]

# combine in desired order
plot_vars = dependent_vars + tradfi_vars + crypto_vars

# nicer names for plot
var_labels = {
    "vix_daily_log_chg": "VIX",
    "sp500_daily_log_ret": "S&P 500",
    "hy_spread_daily_chg": "HY Spread",
    "ig_spread_daily_chg": "IG Spread",
    "term_spread_daily_chg": "Term Spread",
    "usd_strength_daily_log_chg": "USD Index",
    "nfci_weekly_fill": "NFCI",
    "stablecoin_supply_daily_log_chg": "Stablecoin Supply",
    "stablecoin_exchange_netflow_scaled": "Exchange Netflow",
    "stablecoin_exchange_reserve_daily_log_chg": "Exchange Reserves",
    "btc_eth_open_interest_daily_log_chg": "Open Interest",
    "btc_eth_funding_rate_avg": "Funding Rates",
    "tradingVol_btc+eth_daily_log_chg": "BTC + ETH Volume",
    "btc_eth_daily_log_ret_avg": "BTC + ETH Return",
    "btc_eth_realized_vol_7d": "BTC + ETH Volatility",
    "usdt_mcap_daily_log_chg": "USDT Market Cap",
    "usdc_mcap_daily_log_chg": "USDC Market Cap",
    "dai_mcap_daily_log_chg": "DAI Market Cap",
    "tusd_mcap_daily_log_chg": "TUSD Market Cap",
}

# smaller subtext under each variable
var_subtexts = {
    "vix_daily_log_chg": "daily log change",
    "sp500_daily_log_ret": "daily log return",
    "hy_spread_daily_chg": "daily absolute change",
    "ig_spread_daily_chg": "daily absolute change",
    "term_spread_daily_chg": "daily absolute change",
    "usd_strength_daily_log_chg": "daily log change",
    "nfci_weekly_fill": "weekly level, aligned to publication date and forward-filled",
    "stablecoin_supply_daily_log_chg": "daily log change in ERC20 stablecoins + USDT TRON supply",
    "stablecoin_exchange_netflow_scaled": "daily netflow scaled by aggregate stablecoin supply",
    "stablecoin_exchange_reserve_daily_log_chg": "daily log change in ERC20 + TRON exchange reserves",
    "btc_eth_open_interest_daily_log_chg": "daily log change in BTC + ETH open interest",
    "btc_eth_funding_rate_avg": "daily average of BTC + ETH funding rates",
    "tradingVol_btc+eth_daily_log_chg": "daily log change in BTC + ETH volume",
    "btc_eth_daily_log_ret_avg": "equal-weighted average daily log return",
    "btc_eth_realized_vol_7d": "7-day rolling standard deviation of BTC + ETH average log return",
    "usdt_mcap_daily_log_chg": "daily log change in market cap",
    "usdc_mcap_daily_log_chg": "daily log change in market cap",
    "dai_mcap_daily_log_chg": "daily log change in market cap",
    "tusd_mcap_daily_log_chg": "daily log change in market cap",
}

# calculate availability
availability = []

for var in plot_vars:
    available_days = availability_df[var].notna().sum()
    missing_days = total_days - available_days
    availability_pct = available_days / total_days * 100
    
    if var in dependent_vars:
        var_type = "Dependents"
    elif var in crypto_vars:
        var_type = "Crypto"
    else:
        var_type = "Traditional finance"

    availability.append({
        "variable": var,
        "label": var_labels.get(var, var),
        "subtext": var_subtexts.get(var, ""),
        "available_days": available_days,
        "missing_days": missing_days,
        "availability_pct": availability_pct,
        "type": var_type
    })

availability = pd.DataFrame(availability)

# colors
dependent_color = "#2D373C"
crypto_color = "#D2EBE9"
tradfi_color = "#A5D7D2"
missing_color = "#e0e0e0"
subtext_color = "#7a7a7a"

availability["color"] = np.select(
    [
        availability["type"] == "Dependents",
        availability["type"] == "Crypto",
        availability["type"] == "Traditional finance",
    ],
    [
        dependent_color,
        crypto_color,
        tradfi_color,
    ],
    default=missing_color
)

# reverse order so first variable appears at top
availability_plot = availability.iloc[::-1].reset_index(drop=True)

# numeric y positions
y_pos = np.arange(len(availability_plot))

# plot
fig, ax = plt.subplots(figsize=(12, 8.6))

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
    
    label_weight = "bold" if row["type"] == "Dependents" else "normal"
    
    ax.text(
        -0.025,
        i + 0.14,
        row["label"],
        transform=label_transform,
        ha="right",
        va="center",
        fontsize=9.8,
        fontweight=label_weight,
        color="#1f1f1f"
    )
    
    ax.text(
        -0.025,
        i - 0.20,
        row["subtext"],
        transform=label_transform,
        ha="right",
        va="center",
        fontsize=7.6,
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
dependent_patch = mpatches.Patch(color=dependent_color, label="Dependents")
tradfi_patch = mpatches.Patch(color=tradfi_color, label="Traditional finance")
crypto_patch = mpatches.Patch(color=crypto_color, label="Crypto")
missing_patch = mpatches.Patch(color=missing_color, label="Missing")

legend = fig.legend(
    handles=[dependent_patch, tradfi_patch, crypto_patch, missing_patch],
    loc="upper left",
    bbox_to_anchor=(0.055, 0.91),
    frameon=False,
    fontsize=10,
    handlelength=1.2,
    handleheight=1.0,
    handletextpad=0.5,
    borderpad=0.3,
    labelspacing=0.4
)

# add enough left margin for custom labels
fig.subplots_adjust(left=0.35, right=0.97, top=0.93, bottom=0.08)

# save graph

fig.savefig(
    "data_availability_graph.png",
    dpi=600,
    bbox_inches="tight",
    facecolor="white"
)


plt.show()


#%%------- DATA AVAILABILITY TIMELINE GRAPH

# only used in presentation, not in paper


# use Palatino Linotype
plt.rcParams["font.family"] = "Palatino Linotype"

# create full daily calendar for extended availability period
sample_start = pd.to_datetime("2015-01-01")
sample_end = pd.to_datetime("2025-12-31")

full_calendar = pd.DataFrame({
    "date": pd.date_range(sample_start, sample_end, freq="D")
})

# make sure data_table date is datetime
availability_source = data_table.copy()
availability_source["date"] = pd.to_datetime(availability_source["date"])

# merge onto full calendar to make sure every calendar day is counted
availability_df = pd.merge(
    full_calendar,
    availability_source,
    on="date",
    how="left"
)

# classify variables
crypto_vars = [
    "stablecoin_supply_daily_log_chg",
    "stablecoin_exchange_netflow_scaled",
    "stablecoin_exchange_reserve_daily_log_chg",
    "btc_eth_open_interest_daily_log_chg",
    "btc_eth_funding_rate_avg",
    "tradingVol_btc+eth_daily_log_chg",
    "usdt_mcap_daily_log_chg",
    "usdc_mcap_daily_log_chg",
    "dai_mcap_daily_log_chg",
    "tusd_mcap_daily_log_chg",
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

    "stablecoin_supply_daily_log_chg": "Stablecoin Supply",
    "stablecoin_exchange_netflow_scaled": "Exchange Netflow",
    "stablecoin_exchange_reserve_daily_log_chg": "Exchange Reserves",
    "btc_eth_open_interest_daily_log_chg": "Open Interest",
    "btc_eth_funding_rate_avg": "Funding Rates",
    "tradingVol_btc+eth_daily_log_chg": "BTC + ETH Volume",
    "usdt_mcap_daily_log_chg": "USDT Market Cap",
    "usdc_mcap_daily_log_chg": "USDC Market Cap",
    "dai_mcap_daily_log_chg": "DAI Market Cap",
    "tusd_mcap_daily_log_chg": "TUSD Market Cap",
}

# smaller subtext under each variable
var_subtexts = {
    "vix_daily_log_chg": "daily log change",
    "sp500_daily_log_ret": "daily log return",
    "hy_spread_daily_chg": "daily absolute change",
    "ig_spread_daily_chg": "daily absolute change",
    "term_spread_daily_chg": "daily absolute change",
    "usd_strength_daily_log_chg": "daily log change",
    "nfci_weekly_fill": "weekly level, aligned to publication date and forward-filled",

    "stablecoin_supply_daily_log_chg": "daily log change in ERC20 stablecoins + USDT TRON supply",
    "stablecoin_exchange_netflow_scaled": "daily netflow scaled by aggregate stablecoin supply",
    "stablecoin_exchange_reserve_daily_log_chg": "daily log change in ERC20 + TRON exchange reserves",
    "btc_eth_open_interest_daily_log_chg": "daily log change in BTC + ETH open interest",
    "btc_eth_funding_rate_avg": "daily average of BTC + ETH funding rates",
    "tradingVol_btc+eth_daily_log_chg": "daily log change in BTC + ETH volume",
    "usdt_mcap_daily_log_chg": "daily log change in market cap",
    "usdc_mcap_daily_log_chg": "daily log change in market cap",
    "dai_mcap_daily_log_chg": "daily log change in market cap",
    "tusd_mcap_daily_log_chg": "daily log change in market cap",
}

# colors
available_color = "#2D373C"
missing_color = "#EAEBEC"
subtext_color = "#7a7a7a"
main_text_color = "#1f1f1f"

# reverse order so first variable appears at top
plot_vars_reversed = plot_vars[::-1]

# convert dates to matplotlib date numbers
x_dates = mdates.date2num(availability_df["date"])
bar_width = 1.0

# plot
fig, ax = plt.subplots(figsize=(12, 8.6))

# draw one timeline row per variable
for i, var in enumerate(plot_vars_reversed):
    
    # availability indicator
    is_available = availability_df[var].notna().astype(int)
    
    # draw missing background for full sample
    ax.bar(
        x_dates,
        np.full(len(x_dates), 0.46),
        bottom=i - 0.23,
        width=bar_width,
        color=missing_color,
        edgecolor="none",
        align="center"
    )
    
    # draw available days
    ax.bar(
        x_dates[is_available == 1],
        np.full(is_available.sum(), 0.46),
        bottom=i - 0.23,
        width=bar_width,
        color=available_color,
        edgecolor="none",
        align="center"
    )

# remove default y labels
ax.set_yticks([])

# custom y labels with subtext
label_transform = blended_transform_factory(ax.transAxes, ax.transData)

for i, var in enumerate(plot_vars_reversed):
    ax.text(
        -0.018,
        i + 0.14,
        var_labels.get(var, var),
        transform=label_transform,
        ha="right",
        va="center",
        fontsize=9.8,
        color=main_text_color
    )
    
    ax.text(
        -0.018,
        i - 0.20,
        var_subtexts.get(var, ""),
        transform=label_transform,
        ha="right",
        va="center",
        fontsize=7.4,
        color=subtext_color
    )

# x-axis formatting: show only years
ax.set_xlim(
    mdates.date2num(sample_start),
    mdates.date2num(sample_end)
)

ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

# remove tick marks but keep year labels
ax.tick_params(
    axis="x",
    length=0,
    labelsize=8,
    colors=main_text_color
)

# optional: light vertical lines at year changes
for year in range(2015, 2027):
    ax.axvline(
        mdates.date2num(pd.to_datetime(f"{year}-01-01")),
        color="#f0f0f0",
        linewidth=0.7,
        zorder=0
    )

# formatting
ax.set_ylim(-0.7, len(plot_vars_reversed) - 0.3)
ax.set_xlabel("")
ax.grid(False)

# remove spines
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["bottom"].set_visible(False)
ax.spines["left"].set_visible(False)

# legend
available_patch = mpatches.Patch(color=available_color, label="Available")
missing_patch = mpatches.Patch(color=missing_color, label="Missing")

legend = fig.legend(
    handles=[available_patch, missing_patch],
    loc="upper left",
    bbox_to_anchor=(0.045, 0.90),
    frameon=False,
    fontsize=10,          # larger text
    handlelength=1.2,     # larger color box width
    handleheight=1.0,     # larger color box height
    handletextpad=0.5,
    borderpad=0.3,
    labelspacing=0.4
)

for text in legend.get_texts():
    text.set_color(main_text_color)

# margins
fig.subplots_adjust(left=0.35, right=0.97, top=0.93, bottom=0.08)

# save graph

fig.savefig(
    "data_availability_timeline_graph.png",
    dpi=600,
    bbox_inches="tight",
    facecolor="white"
)



plt.show()

"""