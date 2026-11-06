# Flatten yfinance column names if necessary
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

# Feature 1: Daily Return
df["Daily_Return"] = df["Close"].pct_change()

# Feature 2: Trading Volume
df["Trading_Volume"] = df["Volume"]

# Feature 3: 5-Day Momentum
df["Momentum_5D"] = df["Close"].pct_change(periods=5)

# Feature 4: 5-Day Volatility
df["Volatility_5D"] = df["Daily_Return"].rolling(window=5).std()

# Display the calculated features
features = [
    "Daily_Return",
    "Trading_Volume",
    "Momentum_5D",
    "Volatility_5D"
]

print(df[features].head(10))
