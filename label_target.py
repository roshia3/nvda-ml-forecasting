# Calculate next trading day's closing price
df["Next_Day_Close"] = df["Close"].shift(-1)

# Create binary target labels
df["Target"] = (
    df["Next_Day_Close"] > df["Close"]
).astype(int)

# Remove rows with missing features or next-day prices
df = df.dropna(subset=features + ["Next_Day_Close"])

# Display the processed observations
print(df[features + ["Target"]].head(10))

# Check target distribution
print("\nTarget distribution:")
print(df["Target"].value_counts())
