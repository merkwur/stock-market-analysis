import numpy as np

intervals = ['1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h']
ud = ["green_candle", "red_candle"]
weights = np.array([1, 1/3, 1/5, 1/15, 1/30, 1/60, 1/120, 1/240])