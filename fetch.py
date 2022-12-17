# !usr/bin/python3.10

import argparse
import numpy as np
import pandas as pd
import pandas_ta as ta
from datetime import datetime
from keys import API_KEY, API_SECRET
from scipy.signal import find_peaks
from binance import Client
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
import utils
import test_utils
import constants
import warnings 
warnings.filterwarnings("ignore")


parser = argparse.ArgumentParser(description='Description of your program')
parser.add_argument('-p', dest='plot', action='store_true', help="Plots the indicators")
parser.add_argument('-c', dest='contingency', action='store_true', help="Prepares the contingency tables through the intervals")
parser.add_argument('-er', dest='ema_ribb', action='store_true', help="Includes the EMA Ribbon to the plot, to show must include -p first")
parser.add_argument('-bb', dest='bbands', action='store_true', help="Includes the Bolinger Bands to the plot, to show must include -p first")
parser.add_argument('-rs', dest='rsi', action='store_true', help="Includes the RSI to the plot, to show must include -p first")
parser.add_argument('-md', dest='macd', action='store_true', help="Includes the MACD to the plot, to show must include -p first")
parser.add_argument('-pall', dest='pall', action='store_true', help="Plots all indicators, to show must include -p first")
parser.add_argument('-test', dest='test', action='store_true', help="Plots the indicators")
args = parser.parse_args()


EMA_RIBBON = [12, 21, 34, 42, 55, 62]
EMA_RIBBON_COLORS = ["#ff4242", "#ff4266", "#ff4288", "#ff42aa", "#ff42bb", "#ff42cc"]
BBANDS_NAMES = ["low", "mid", "high", 'bandwith', "percent"]
MACD_NAMES = ["macd", "hist", "sig"]
REDUCED = ["OpenTime", "Open","High", "Low", "Close", "Volume"]

# use your own API_KEY and API_SECRET key
client = Client(api_key=API_KEY, api_secret=API_SECRET)              # Calling the Binance api

# Currency pairs
pairs = ["ADAEUR", "BTCEUR", "ETHEUR"]                               
# One day lenght of one minute market data
klines = client.get_historical_klines(pairs[0], Client.KLINE_INTERVAL_1MINUTE, "1 day ago UTC") # One day lenght of one minute market data

columns = ["OpenTime", "Open","High", "Low", "Close", "Volume", "CloseTime", 
           "QuoteAssetVol", "#Trades", "TakerBaseVol", "TakerQuoteVol", "Ignore"]

# candlestick data to dataframe
df = pd.DataFrame(klines, columns=columns)
df = df.loc[:, REDUCED]

for i in REDUCED:
    df[f"{i}"] = df[f"{i}"].astype(float)

# timestamp to date
df["Date"] = df["OpenTime"].apply(lambda x: datetime.fromtimestamp(int(x/1000)).strftime("%d/%m/%Y, %H:%M:%S"))

# contingency table and chi_squared statistics
if args.contingency:
    contingency = pd.DataFrame()
    for i in constants.intervals:
        contlines = client.get_historical_klines(pairs[0], i, "1 day ago UTC") # One day lenght of one minute market data
        table = pd.DataFrame(contlines, columns=columns)
        table = table.loc[:, ["Open","Close" ]]
        ret = utils.contingency_table(table["Open"], table["Close"])
        contingency[f"{i}"] = ret

    print(utils.contingency_over_intervals(contingency))
    
# preparing indicators
indicators = pd.DataFrame()

if args.ema_ribb or args.pall:
    for i in EMA_RIBBON:
        indicators[f'{i}'] = df.Close.ewm(span=int(i)).mean()

# rsi
if args.rsi or args.pall:
    indicators["rsi"] = ta.rsi(df["Close"])
    # find the rsi signal peaks
    peaks, _ = find_peaks(indicators["rsi"].to_numpy())
    peak_pos = np.zeros(len(df["Date"]))
    peak_pos[peaks] = indicators["rsi"].iloc[peaks]

# bollinger
if args.bbands or args.pall:
    bollingers = ta.bbands(df["Close"], length=20, std=2, append=True)
    bollingers.columns = BBANDS_NAMES
    nan_indices = np.where(bollingers['low'].notnull())[0]
    length = nan_indices[0]
    bollingers.loc[:length] = np.flip(bollingers[length:length+length+1].to_numpy(), axis=0)

# macd
if args.macd or args.pall:
    macd = ta.macd(df["Close"])
    macd.columns = MACD_NAMES 


# put the test cases inside of the statement 
if args.test:
    print("This flag for testing a case")
    print(test_utils.candle_mean(df["High"].to_numpy(), df["Low"].to_numpy()))
    

if args.plot or args.pall:

    pio.templates.default = "plotly_dark"
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True)

    fig.add_trace(go.Candlestick(name="CAND", x=df['Date'],
                                         open=df['Open'],
                                         high=df['High'],
                                         low=df['Low'],
                                         close=df['Close']), row=1, col=1)
    
    if args.bbands or args.pall:
        fig.add_trace(go.Scatter(name="BUP", x=df["Date"], y=bollingers["high"], line=dict(color="rgba(131, 165, 152, .5)", width=1)),
                    row=1, col=1)
        fig.add_trace(go.Scatter(name="BDO", x=df["Date"], y=bollingers["low"], line=dict(color="rgba(131, 165, 152, .5)", width=1), fill="tonexty"),
                    row=1, col=1)

    if args.ema_ribb or args.pall:
        for e, i in enumerate([x for x in indicators.columns if x.isnumeric()]):
            fig.add_trace(go.Scatter(name=f"EMA{i}", x=df["Date"], y=indicators[f"{i}"], 
                                    line=dict(color=EMA_RIBBON_COLORS[e], width=1)),
                    row=1, col=1)

    
    if args.rsi or args.pall:
        fig.add_trace(go.Scatter(name="RSI", x=df["Date"], y=indicators["rsi"],line=dict(color="#ebdbb2")), row=2, col=1)
        fig.add_trace(go.Scatter(name="RSIPEAKS", mode="markers",x=df["Date"], y=peak_pos, marker=dict(size=5, color="#d65d0e")), row=2, col=1)
    
    if args.macd or args.pall:
        fig.add_trace(go.Bar(name="HIST", x=df["Date"], y=macd["hist"]), row=3, col=1)    # I couldn't find the easier way will check later
        fig.add_trace(go.Scatter(name="MACD", x=df["Date"], y=macd["macd"]), row=3, col=1)
        fig.add_trace(go.Scatter(name="SIG", x=df["Date"], y=macd["sig"]), row=3, col=1)


    fig.update_layout(xaxis_rangeslider_visible=False)
    fig.show()
