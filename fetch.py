# !usr/bin/python3.10

import numpy as np
from scipy.signal import find_peaks
from scipy.interpolate import interp1d
from binance import Client
from keys import API_KEY, API_SECRET
from datetime import datetime
import plotly.graph_objects as go
import plotly.io as pio
import utils
from plotly.subplots import make_subplots
import pandas as pd
import pandas_ta as ta
import warnings 

warnings.filterwarnings("ignore")

# use your own API_KEY and API_SECRET key
client = Client(api_key=API_KEY, api_secret=API_SECRET)              # Calling the Binance api
prices = client.get_all_tickers()                                    # Get the Price data

# Currency pairs
pairs = ["ADAEUR", "BTCEUR", "ETHEUR"]                               
# One day lenght of one minute market data
klines = client.get_historical_klines(pairs[0], Client.KLINE_INTERVAL_1MINUTE, "1 day ago UTC") # One day lenght of one minute market data

columns = ["OpenTime", "Open","High", "Low", "Close", "Volume", "CloseTime", 
           "QuoteAssetVol", "#Trades", "TakerBaseVol", "TakerQuoteVol", "Ignore"]

# candlestick data to dataframe
df = pd.DataFrame(klines, columns=columns)
# timestamp to date
df["Date"] = df["OpenTime"].apply(lambda x: datetime.fromtimestamp(int(x/1000)).strftime("%d/%m/%Y, %H:%M:%S"))

EMA_RIBBON = [12, 21, 34, 42, 55, 62]
EMA_RIBBON_COLORS = ["#ff4242", "#ff4266", "#ff4288", "#ff42aa", "#ff42bb", "#ff42cc"]
BBANDS_NAMES = ["low", "mid", "high", 'bandwith', "percent"]
MACD_NAMES = ["macd", "hist", "sig"]

# preparing indicators
indicators = pd.DataFrame()
for i in EMA_RIBBON:
    indicators[f'{i}'] = df.Close.ewm(span=int(i)).mean()

df["Open"] = df["Open"].astype(float)
df["Close"] = df["Close"].astype(float)
# rsi
indicators["rsi"] = ta.rsi(df["Close"])
# bollinger
bollingers = ta.bbands(df["Close"], length=20, std=2, append=True)
bollingers.columns = BBANDS_NAMES
nan_indices = np.where(bollingers['low'].notnull())[0]
length = nan_indices[0]
bollingers.loc[:length] = np.flip(bollingers[length:length+length+1].to_numpy(), axis=0)
# macd
macd = ta.macd(df["Close"])
macd.columns = MACD_NAMES 

# find the rsi signal peaks
peaks, _ = find_peaks(indicators["rsi"].to_numpy())
peak_pos = np.zeros(len(df["Date"]))
peak_pos[peaks] = indicators["rsi"].iloc[peaks]

regressions = pd.DataFrame()
reg_news = pd.DataFrame()

window = 144
spans = [10, 15, 20]

# utils.histogram(df["Open"], is_plot=True) #try this
# utils.sample_histogram(df["Open"], is_plot=True) #try this

"""
find the way to plot those regression lines
select the peaks in a different way
there is an artefact at the end of the regression, interpolating it making the plot false
"""
for j in spans:
    regressions[f"REG{j}"] = np.zeros(len(peaks))
    for i in range(0, len(peaks), len(peaks)//j):
        if i + window > len(peaks):
            window = len(peaks) - i
            regressions[f"REG{j}"].iloc[i: i+window] = utils.regression_line(peaks[i:i+window], indicators["rsi"].iloc[peaks[i:i+window]])
            break
        else: regressions[f"REG{j}"].iloc[i: i+window] = utils.regression_line(peaks[i:i+window], indicators["rsi"].iloc[peaks[i:i+window]])

    f = interp1d(np.arange(0,len(regressions[f"REG{j}"][:-2]), 1), regressions[f"REG{j}"][:-2])
    reg_news[f"REG{j}"] = f(np.linspace(0, len(regressions[f"REG{j}"][:-2])-1, len(df["Date"])))


pio.templates.default = "plotly_dark"

fig = make_subplots(rows=3, cols=1, shared_xaxes=True)

for e, i in enumerate([x for x in indicators.columns if x.isnumeric()]):
    fig.add_trace(go.Scatter(name=f"EMA{i}", x=df["Date"], y=indicators[f"{i}"], 
                            line=dict(color=EMA_RIBBON_COLORS[e], width=1)),
              row=1, col=1)

fig.add_trace(go.Scatter(name="BUP", x=df["Date"], y=bollingers["high"], line=dict(color="rgba(131, 165, 152, .5)", width=1)),
              row=1, col=1)
fig.add_trace(go.Scatter(name="BDO", x=df["Date"], y=bollingers["low"], line=dict(color="rgba(131, 165, 152, .5)", width=1), fill="tonexty"),
              row=1, col=1)
fig.add_trace(go.Candlestick(name="CAND", x=df['Date'],
                                     open=df['Open'],
                                     high=df['High'],
                                     low=df['Low'],
                                     close=df['Close']), row=1, col=1)
fig.update_layout(xaxis_rangeslider_visible=False)
fig.add_trace(go.Scatter(name="RSI", x=df["Date"], y=indicators["rsi"],line=dict(color="#ebdbb2")), row=2, col=1)

# for e, i in enumerate(reg_news.columns):
#     fig.add_trace(go.Scatter(name=f"{i}", x=df["Date"], y=reg_news[f"{i}"], 
#                             line=dict(color=EMA_RIBBON_COLORS[e], width=1)), row=2, col=1)

fig.add_trace(go.Scatter(name="RSIPEAKS", mode="markers",x=df["Date"], y=peak_pos, marker=dict(size=5, color="#d65d0e")), row=2, col=1)
fig.add_trace(go.Bar(name="HIST", x=df["Date"], y=macd["hist"]), row=3, col=1)    # I couldn't find the easier way will check later
fig.add_trace(go.Scatter(name="MACD", x=df["Date"], y=macd["macd"]), row=3, col=1)
fig.add_trace(go.Scatter(name="SIG", x=df["Date"], y=macd["sig"]), row=3, col=1)


fig.show()
