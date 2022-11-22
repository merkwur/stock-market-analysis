# !usr/bin/python3.10

import numpy as np
from scipy.signal import find_peaks
from binance import Client, ThreadedWebsocketManager, ThreadedDepthCacheManager
from keys import API_KEY, API_SECRET
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import pandas_ta as ta


client = Client(api_key=API_KEY, api_secret=API_SECRET)              # Calling the Binance api
prices = client.get_all_tickers()                                    # Get the Price data

# Currency pairs
pairs = ["ADATRY", "BTCTRY", "ETHTRY"]                               
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
indicators = pd.DataFrame(columns=EMA_RIBBON)
for i in EMA_RIBBON:
    indicators[f'{i}'] = df.Close.ewm(span=int(i)).mean()


df["Close"] = df["Close"].astype(float)
indicators["rsi"] = ta.rsi(df["Close"])
bollingers = ta.bbands(df["Close"], length=20, std=2, append=True)
bollingers.columns = BBANDS_NAMES
nan_indices = np.where(bollingers['low'].notnull())[0]
length = nan_indices[0]
bollingers.loc[:length] = np.flip(bollingers[length:length+length+1].to_numpy(), axis=0)
macd = ta.macd(df["Close"])
macd.columns = MACD_NAMES 

# find the rsi signal peaks
peaks, _ = find_peaks(indicators["rsi"].to_numpy())
peak_pos = np.zeros(len(df["Date"]))
peak_pos[peaks] = indicators["rsi"].loc[peaks]


# plot the data
fig = make_subplots(rows=3, cols=1, shared_xaxes=True)

fig.add_trace(go.Scatter(x=df["Date"], y=indicators["12"], 
                        line=dict(color=EMA_RIBBON_COLORS[0], width=1)),
              row=1, col=1)
fig.add_trace(go.Scatter(x=df["Date"], y=indicators["21"],
                        line=dict(color=EMA_RIBBON_COLORS[1], width=1)),
              row=1, col=1)
fig.add_trace(go.Scatter(x=df["Date"], y=indicators["34"], 
                        line=dict(color=EMA_RIBBON_COLORS[2], width=1)),
              row=1, col=1)
fig.add_trace(go.Scatter(x=df["Date"], y=indicators["42"],
                        line=dict(color=EMA_RIBBON_COLORS[3], width=1)),
              row=1, col=1)
fig.add_trace(go.Scatter(x=df["Date"], y=indicators["55"], 
                        line=dict(color=EMA_RIBBON_COLORS[4], width=1)),
              row=1, col=1)
fig.add_trace(go.Scatter(x=df["Date"], y=indicators["62"],
                        line=dict(color=EMA_RIBBON_COLORS[5], width=1)),
              row=1, col=1)
fig.add_trace(go.Scatter(x=df["Date"], y=bollingers["high"], line=dict(color="#ebb00e", width=1)),
              row=1, col=1)
fig.add_trace(go.Scatter(x=df["Date"], y=bollingers["low"], line=dict(color="#ebb00e", width=1), fill="tonexty"),
              row=1, col=1)
fig.add_trace(go.Candlestick(x=df['Date'],
                                     open=df['Open'],
                                     high=df['High'],
                                     low=df['Low'],
                                     close=df['Close']), row=1, col=1)
fig.update_layout(xaxis_rangeslider_visible=False)

fig.add_trace(go.Scatter(x=df["Date"], y=indicators["rsi"]), row=2, col=1)
fig.add_trace(go.Scatter(mode="markers",x=df["Date"], y=peak_pos, marker=dict(size=5, color="#ff4242")), row=2, col=1)
fig.add_trace(go.Bar(x=df["Date"], y=macd["hist"]), row=3, col=1)
fig.add_trace(go.Scatter(x=df["Date"], y=macd["macd"]), row=3, col=1)
fig.add_trace(go.Scatter(x=df["Date"], y=macd["sig"]), row=3, col=1)


fig.show()








